"""
DocuMind — AI Document Intelligence.

FastAPI application providing a RAG-powered document Q&A system.
Upload PDFs, ask natural language questions, get answers with citations.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import AnswerResponse, QuestionRequest, SessionInfo, UploadResponse
from rag_pipeline import RAGPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
CLEANUP_INTERVAL_SECONDS: int = 15 * 60       # 15 minutes

# ──────────────────────────────────────────────────────────────────────
# Application state
# ──────────────────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None
_cleanup_task: Optional[asyncio.Task] = None


async def _session_cleanup_loop() -> None:
    """Background task that periodically cleans up expired sessions."""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            if _pipeline:
                _pipeline.cleanup_expired_sessions()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Session cleanup error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Startup: initialise the RAG pipeline (loads the embedding model).
    Shutdown: cancel the background cleanup task.
    """
    global _pipeline, _cleanup_task

    logger.info("Starting DocuMind — initialising RAG pipeline...")
    _pipeline = RAGPipeline()
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())
    logger.info("DocuMind ready — accepting requests")

    yield

    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    logger.info("DocuMind shut down")


def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline instance.

    Raises:
        RuntimeError: If the pipeline has not been initialised.
    """
    if _pipeline is None:
        raise RuntimeError("RAG pipeline not initialised")
    return _pipeline


# ──────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind — AI Document Intelligence",
    description="Chat with your documents. Get answers with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: lock down to an explicit allowlist via CORS_ALLOWED_ORIGINS
# (comma-separated). Wildcard + credentials is a browser no-op and a
# security smell, so disable credentials when no allowlist is given.
_cors_raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors_raw:
    _allowed_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    _allow_credentials = True
else:
    _allowed_origins = ["*"]
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────────────────────────────
# Routes — pages
# ──────────────────────────────────────────────────────────────────────


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the single-page chat application."""
    return FileResponse("static/index.html")


# ──────────────────────────────────────────────────────────────────────
# Routes — session management
# ──────────────────────────────────────────────────────────────────────


@app.post("/session/create")
async def create_session() -> JSONResponse:
    """Create a new chat session.

    Returns:
        JSON with session_id.
    """
    pipeline = get_pipeline()
    session_id = pipeline.create_session()
    return JSONResponse({"session_id": session_id})


@app.get("/session/{session_id}")
async def get_session(session_id: str) -> SessionInfo:
    """Get information about an existing session.

    Args:
        session_id: The session identifier.

    Returns:
        SessionInfo with uploaded files and message count.
    """
    pipeline = get_pipeline()
    try:
        session = pipeline.get_session(session_id)
        return session.to_session_info()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    """Delete a session and all its data.

    Args:
        session_id: The session identifier.

    Returns:
        JSON confirmation message.
    """
    pipeline = get_pipeline()
    pipeline.clear_session(session_id)
    return JSONResponse({"message": "Session cleared successfully."})


# ──────────────────────────────────────────────────────────────────────
# Routes — document upload
# ──────────────────────────────────────────────────────────────────────


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload a PDF document for indexing.

    Args:
        session_id: The session to add the document to.
        file: The PDF file to upload (max 20 MB).

    Returns:
        UploadResponse with processing results.
    """
    pipeline = get_pipeline()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Read and validate file size
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    # Process the document
    try:
        result = await pipeline.ingest_document(
            session_id, file_bytes, file.filename,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error during upload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the document.",
        )


# ──────────────────────────────────────────────────────────────────────
# Routes — chat / Q&A
# ──────────────────────────────────────────────────────────────────────


@app.post("/chat/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Ask a question about the uploaded documents.

    Args:
        request: QuestionRequest with question, session_id, and optional top_k.

    Returns:
        AnswerResponse with answer, sources, and confidence score.
    """
    pipeline = get_pipeline()

    try:
        result = await pipeline.answer_question(
            session_id=request.session_id,
            question=request.question,
            top_k=request.top_k,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Rate limit or LLM service errors
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error during Q&A: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the answer.",
        )


# ──────────────────────────────────────────────────────────────────────
# Routes — health check
# ──────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint.

    Returns:
        JSON with service status, model info, and session count.
    """
    pipeline = get_pipeline()
    return JSONResponse({
        "status": "ok",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama-3.1-8b-instant",
        "active_sessions": len(pipeline.sessions),
    })

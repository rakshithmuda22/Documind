# DocuMind — AI Document Intelligence

> Chat with your documents. Get answers with citations. Powered by RAG.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange.svg)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-purple.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/YOUR_USERNAME/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/documind/actions)

<!-- Add a screenshot or GIF of the app here -->
<!-- ![DocuMind Demo](docs/demo.gif) -->

---

## Features

- **Multi-Document Upload** — Upload multiple PDFs and query across all of them
- **Source Citations** — Every answer includes exact page numbers and text excerpts
- **Conversation Memory** — Follow-up questions work naturally with context
- **Confidence Scoring** — See how confident the AI is in each answer (High/Medium/Low)
- **Smart Chunking** — Paragraph-aware text splitting with overlap for better retrieval
- **Follow-Up Detection** — Automatically rewrites vague follow-ups into standalone queries
- **100% Free** — No paid APIs required (Groq free tier + local embeddings)
- **One-Click Deploy** — Deploys to Hugging Face Spaces with zero config

---

## How It Works

```
                         DocuMind RAG Architecture

  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐
  │  Upload   │────>│ PDF Processor │────>│  Smart Chunker  │
  │  PDF(s)   │     │  (PyMuPDF)   │     │ (500 tok, 50    │
  └──────────┘     └──────────────┘     │  tok overlap)   │
                                         └────────┬────────┘
                                                   │
                                                   v
                                         ┌─────────────────┐
                                         │   Embeddings     │
                                         │ (all-MiniLM-L6)  │
                                         │  384-dim vectors  │
                                         └────────┬────────┘
                                                   │
                                                   v
                                         ┌─────────────────┐
                                         │    ChromaDB      │
                                         │  (in-memory,     │
                                         │   per-session)   │
                                         └─────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐
  │   Ask a   │────>│  Embed Query  │────>│ Similarity Search│
  │ Question  │     │              │     │  (top 5 chunks)  │
  └──────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   v
                                         ┌─────────────────┐
                                         │   Groq LLM       │
                                         │ (LLaMA 3.1 8B)   │
                                         │ + chat history    │
                                         └────────┬────────┘
                                                   │
                                                   v
                                         ┌─────────────────┐
                                         │  Answer + Cited   │
                                         │  Sources + Score  │
                                         └─────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI | Async REST API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim text embeddings, runs locally |
| Vector Store | ChromaDB (ephemeral) | In-memory similarity search |
| LLM | Groq (LLaMA 3.1 8B Instant) | Fast, free inference |
| PDF Parsing | PyMuPDF (fitz) | Robust text extraction |
| Frontend | Vanilla HTML/CSS/JS | Single-file chat UI |
| Deployment | Docker + HF Spaces | Free hosting |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/documind.git
cd documind
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) and create an API key.

### 3. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Run the app

```bash
uvicorn main:app --reload --port 7860
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

> **Note:** The embedding model (~80 MB) downloads automatically on first launch. This takes ~30 seconds — you'll see "Loading embedding model..." in the logs.

---

## Running with Docker

```bash
docker-compose up --build
```

The app will be available at [http://localhost:7860](http://localhost:7860).

---

## Running Tests

```bash
pytest tests/ -v
```

All LLM calls are mocked — no API key needed for tests.

---

## Deploy to Hugging Face Spaces (Free)

1. Create an account at [huggingface.co](https://huggingface.co)
2. Create a new Space: **New Space > Docker > Blank**
3. Connect your GitHub repository (or push directly)
4. Add your `GROQ_API_KEY` as a **Space Secret** (Settings > Variables and Secrets)
5. The Space auto-deploys on every push

---

## Project Structure

```
documind/
├── main.py                 # FastAPI app with routes and lifecycle
├── rag_pipeline.py         # RAG orchestrator and session management
├── embeddings.py           # Sentence-transformer embedding service
├── vector_store.py         # ChromaDB per-session vector store
├── llm_service.py          # Groq LLM client with retry logic
├── pdf_processor.py        # PDF extraction and smart chunking
├── prompts.py              # All LLM prompt templates
├── models.py               # Pydantic models and data structures
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── pytest.ini
├── Dockerfile              # Multi-stage build for HF Spaces
├── docker-compose.yml
├── .github/workflows/ci.yml
├── static/
│   └── index.html          # Single-file chat UI (dark/light mode)
└── tests/
    ├── test_pdf_processor.py
    ├── test_vector_store.py
    └── test_rag_pipeline.py
```

---

## Interview-Ready: How I Built This

### Why build RAG from scratch instead of using LangChain?

I built every component manually so I can explain exactly what happens at each step. In an interview, I can walk through the entire pipeline: PDF parsing, chunking strategy, embedding generation, vector search, prompt engineering, and answer generation — without hiding behind a framework.

### Chunking strategy and why chunk size matters

Documents are split into ~500-token chunks with 50-token overlap. Chunking by paragraph first preserves semantic boundaries. The overlap ensures that information split across chunk boundaries is still retrievable. Too-small chunks lose context; too-large chunks dilute relevance in similarity search.

### Why sentence-transformers over OpenAI embeddings?

The all-MiniLM-L6-v2 model runs locally with zero API cost, provides 384-dimensional embeddings, and has no rate limits. For a portfolio demo, this eliminates the dependency on a paid service while delivering strong semantic search quality.

### Per-session ChromaDB collections

Each user session gets an isolated ChromaDB collection. This ensures complete data isolation between users, simplifies cleanup (just delete the collection), and prevents cross-contamination of search results.

### Follow-up question rewriting

When a user asks "Tell me more about that," the system detects it as a follow-up, reads the conversation history, and rewrites it as a standalone query like "Tell me more about the methodology described in section 3." This dramatically improves retrieval quality for conversational interactions.

### Confidence scoring

The LLM self-rates confidence (0.0-1.0) based on how well the retrieved context answers the question. This helps users gauge answer reliability and know when to look at the source documents directly.

---

## License

MIT

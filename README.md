# Hybrid RAG PDF Question Answering System For Research Papers

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using hybrid retrieval, reranking, and streaming LLM responses.

---

## Features

- PDF ingestion pipeline
- Recursive chunking with overlap
- Dense semantic retrieval using embeddings
- Sparse retrieval using BM25
- MMR-based diversity retrieval
- Cross-encoder reranking
- Whole-paper questions (summaries, key findings, top-N observations) via map-reduce over cached, page-tagged notes
- Source attribution with metadata
- Streaming and synchronous responses
- FastAPI integration
- Built-in web UI with streaming chat
- Dockerized deployment

---

## Architecture

```text
PDFs
 ↓
Chunking + Metadata
 ↓
Embeddings
 ↓
Chroma Vector DB
 ↓
Hybrid Retrieval (Dense + BM25 + MMR)
 ↓
CrossEncoder Reranker
 ↓
Context Builder
 ↓
LLM
 ↓
Answer + Sources
```

---

## Tech Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- Sentence Transformers
- BM25
- PyMuPDF
- Docker

---

## Retrieval Pipeline

### Dense Retrieval
Uses sentence-transformer embeddings for semantic similarity search.

### Sparse Retrieval
Uses BM25 for exact keyword matching.

### Fusion
Dense and sparse results are merged with Reciprocal Rank Fusion (RRF).

### MMR
Ensures retrieval diversity and reduces redundant chunks.

### Reranking
Cross-encoder reranking improves final retrieval precision.

### Whole-paper questions
Questions like "summarise the paper" or "top 5 observations" need the whole document, not
the top few chunks. They are detected by intent and answered by map-reduce:

1. **Map** – the paper is read in ~4.5k-token sections and condensed into page-tagged notes.
   Notes are cached in `data/notes/` and pre-built in the background after upload (and at
   startup for any paper without them), so later summaries skip this step.
2. **Reduce** – the question is answered from the notes of the full paper, with page citations.

Generic requests go to the most recently uploaded paper; naming a paper, or asking about
"all papers", targets those instead. Specific questions still use chunk retrieval.

---

## Installation

### Clone repository

```bash
git clone https://github.com/ananyayadav0/ResearchAssistant.git
cd ResearchAssistant
```

### Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### LLM backend

By default the app uses a local [Ollama](https://ollama.com) model:

```bash
ollama pull llama3
ollama serve
```

To use Groq instead, set `USE_GROQ=true` and `GROQ_API_KEY` in `.env`.

---

## Running Locally

```bash
uvicorn app.main:app --reload
```

Web UI: http://localhost:8000 — a sidebar lists every uploaded paper; pick one (or "All papers") to scope your questions. Upload by clicking or dragging a PDF onto the sidebar. Each paper keeps its own conversation, and answers stream in with page citations.

Interactive API docs: http://localhost:8000/docs

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/` | GET | – | Web UI |
| `/health` | GET | – | Health check |
| `/upload` | POST | multipart `file` (PDF) | Ingest a PDF, returns its `document_id` |
| `/documents` | GET | – | Uploaded papers, newest first |
| `/ask` | POST | `{"query": "...", "document_id": "..."}` | Answer + sources as JSON (`document_id` optional; omit to search all papers) |
| `/ask-stream` | POST | same as `/ask` | Server-sent events: `status`, `sources`, `token`..., `done` |

```bash
curl -F "file=@paper.pdf" http://localhost:8000/upload
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "What is the main contribution?"}'
curl -N -X POST http://localhost:8000/ask-stream -H "Content-Type: application/json" -d '{"query": "Summarise the method"}'
```

---

## Docker Setup

### Build image

```bash
docker build -t rag-app .
```

### Run container

The container talks to Ollama on the host via `host.docker.internal`.

```bash
docker run -p 8000:8000 rag-app
```

### Persist ChromaDB

```bash
docker run -p 8000:8000 \
-v $(pwd)/chroma_db:/app/chroma_db \
-v $(pwd)/data:/app/data \
rag-app
```

---

## Future Improvements

- Token-aware context management
- OCR support for scanned PDFs
- Redis caching
- Managed vector DB (Pinecone / Weaviate)
- Observability and metrics
- Async ingestion pipeline

---

## Key Learnings

- Hybrid retrieval improves recall significantly
- Reranking is critical for precision
- Streaming improves perceived latency
- Metadata-aware retrieval improves traceability

---

## Example Query Flow

```text
User Query
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Context Construction
   ↓
LLM Generation
   ↓
Answer + Sources
```

---

## Author

Ananya Yadav

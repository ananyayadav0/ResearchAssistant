# Hybrid RAG PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using hybrid retrieval, reranking, and streaming LLM responses.

---

## Features

- PDF ingestion pipeline
- Recursive chunking with overlap
- Dense semantic retrieval using embeddings
- Sparse retrieval using BM25
- MMR-based diversity retrieval
- Cross-encoder reranking
- Source attribution with metadata
- Streaming and synchronous responses
- FastAPI integration
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

### MMR
Ensures retrieval diversity and reduces redundant chunks.

### Reranking
Cross-encoder reranking improves final retrieval precision.

---

## Installation

### Clone repository

```bash
git clone <your_repo_url>
cd <repo_name>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Locally

```bash
uvicorn app:app --reload
```

API:
- `/ask`
- `/stream`

---

## Docker Setup

### Build image

```bash
docker build -t rag-app .
```

### Run container

```bash
docker run -p 8000:8000 rag-app
```

### Persist ChromaDB

```bash
docker run -p 8000:8000 \
-v $(pwd)/chroma_db:/app/chroma_db \
rag-app
```

---

## Future Improvements

- Token-aware context management
- OCR support for scanned PDFs
- Redis caching
- Managed vector DB (Pinecone / Weaviate)
- Score fusion for hybrid retrieval
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

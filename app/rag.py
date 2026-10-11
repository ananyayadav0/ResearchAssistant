# rag.py

import asyncio
import os
import re
import threading
from collections import Counter

import numpy as np
import torch
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from starlette.concurrency import run_in_threadpool

from app.config import MIN_RERANK_SCORE, RERANKER_MODEL, get_llm
from app.store import vectordb
from app.summarize import (
    ALL_DOCS,
    build_reduce_prompt,
    documents_from_chunks,
    get_notes,
    has_notes,
    is_generic,
    is_global_query,
    mentions_document,
    sources_for,
)

# 1. Models (init once)
reranker = CrossEncoder(RERANKER_MODEL)
#A CrossEncoder is a re-ranking model that scores the relevance of a query-document pair by jointly encoding them, unlike embedding models that encode them separately. It is used after vector retrieval to improve precision by reordering top-k results based on deeper semantic understanding.

dense_retriever = vectordb.as_retriever(
    search_type="mmr", #Maximal Marginal Relevance - diverse + relevant results. MMR retrieval selects documents by balancing relevance to the query and diversity among results. It first retrieves a larger candidate set (fetch_k) using similarity search, then iteratively selects the top k documents that maximize both relevance and novelty, reducing redundancy in retrieved context.
    search_kwargs={"k": 8, "fetch_k": 20}
)

# 2. BM25 index, rebuilt at startup and after every ingest
_bm25_lock = threading.Lock()
bm25 = None
bm25_docs: list[Document] = []


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def init_bm25():
    global bm25, bm25_docs

    data = vectordb.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
        if text
    ]

    with _bm25_lock:
        bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None
        bm25_docs = docs

    print(f"BM25 index built over {len(docs)} chunks")


def doc_key_of(doc: Document) -> str:
    return doc.metadata.get("file_hash") or doc.metadata.get("source", "unknown")


# 3. Hybrid retrieval with Reciprocal Rank Fusion
# document_id (a file_hash) limits retrieval to one paper; None searches all papers
def hybrid_retrieve(query, k=10, rrf_k=60, document_id=None):
    if document_id:
        dense_docs = vectordb.max_marginal_relevance_search(
            query, k=8, fetch_k=20, filter={"file_hash": document_id}
        )
    else:
        dense_docs = dense_retriever.invoke(query)

    with _bm25_lock:
        index, corpus = bm25, bm25_docs

    sparse_docs = []
    if index is not None:
        scores = index.get_scores(tokenize(query))
        if document_id:
            scores = np.where([doc_key_of(d) == document_id for d in corpus], scores, 0)
        top_idx = np.argsort(scores)[::-1][:k]
        sparse_docs = [corpus[i] for i in top_idx if scores[i] > 0]

    # RRF: score each chunk by 1/(rrf_k + rank) in every list it appears in,
    # so chunks found by both retrievers rise to the top
    fused: dict[str, float] = {}
    by_text: dict[str, Document] = {}
    for ranked in (dense_docs, sparse_docs):
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_text.setdefault(key, doc)

    ordered = sorted(fused, key=fused.get, reverse=True)
    return [by_text[key] for key in ordered[:k]]


# 4. Reranking
def rerank(query, docs, top_k=5):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    # Sigmoid maps the raw ms-marco logits to 0-1 so the threshold is meaningful
    scores = reranker.predict(
        pairs,
        batch_size=16,
        activation_fn=torch.nn.Sigmoid()
    )

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc for doc, score in ranked[:top_k]
        if score >= MIN_RERANK_SCORE
    ]


# 5. Context builder
def build_context(docs, max_chars=3000):
    context = []
    sources = []

    total = 0
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        if context and total + len(text) > max_chars:
            break

        source_id = len(sources) + 1
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")

        context.append(f"[{source_id}] ({source}, page {page})\n{text}")
        total += len(text)

        sources.append({
            "id": source_id,
            "source": source,
            "page": page
        })

    return "\n\n".join(context), sources


# 6. Prompt
PROMPT = """You are a research assistant.

Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Provide a clear, structured answer. Cite the context passages you use with their
bracketed numbers, e.g. [1] or [2][3].

Context:
{context}

Question:
{question}

Answer:
"""

NO_CONTEXT_ANSWER = "I don't know. No relevant passages were found in the uploaded documents."


def retrieve_context(query: str, document_id=None):
    docs = hybrid_retrieve(query, document_id=document_id)
    print(f"Retrieved {len(docs)} docs before reranking")

    docs = rerank(query, docs)
    print(f"Reranked to {len(docs)} docs")

    return build_context(docs)


# 7. Query routing: chunk retrieval for specific questions, whole-paper
# map-reduce (app/summarize.py) for summaries / key findings / top-N lists
def list_documents() -> list[dict]:
    """Indexed papers, most recently uploaded first."""
    with _bm25_lock:
        corpus = bm25_docs
    docs = documents_from_chunks(corpus)
    return sorted(
        (
            {
                "id": key,
                "source": doc["source"],
                "pages": doc["pages"],
                "uploaded_at": doc["uploaded_at"],
                "notes_ready": has_notes(key),
            }
            for key, doc in docs.items()
        ),
        key=lambda d: d["uploaded_at"],
        reverse=True,
    )


def select_documents(query: str, document_id=None):
    with _bm25_lock:
        corpus = bm25_docs
    docs = documents_from_chunks(corpus)
    if document_id:
        return [(document_id, docs[document_id])] if document_id in docs else []
    if len(docs) <= 1 or ALL_DOCS.search(query):
        return list(docs.items())

    # A paper named in the question, e.g. "summarise attention-is-all-you-need"
    q = query.lower()
    named = [
        (key, doc) for key, doc in docs.items()
        if os.path.splitext(doc["source"])[0].lower().replace("-", " ").replace("_", " ") in q.replace("-", " ").replace("_", " ")
    ]
    if named:
        return named

    # "Summarise the paper" with several papers loaded: the most recent upload
    if is_generic(query):
        latest = max(docs.items(), key=lambda kv: kv[1]["uploaded_at"])
        return [latest]

    # Otherwise the paper that dominates retrieval for this question
    hits = Counter(doc_key_of(d) for d in hybrid_retrieve(query))
    best = hits.most_common(1)[0][0] if hits else next(iter(docs))
    return [(best, docs[best])]


def plan(query: str, document_id=None) -> dict:
    if is_global_query(query):
        targets = select_documents(query, document_id)
        if targets:
            print(f"Global query over: {[doc['source'] for _, doc in targets]}")
            return {"mode": "global", "targets": targets, "sources": sources_for(targets)}

    context, sources = retrieve_context(query, document_id)

    # Broad questions about "the paper" (e.g. its strengths) that match no single chunk
    if not context and mentions_document(query):
        targets = select_documents(query, document_id)
        if targets:
            print(f"No chunk matched; falling back to whole-paper notes")
            return {"mode": "global", "targets": targets, "sources": sources_for(targets)}

    return {"mode": "rag", "context": context, "sources": sources}


def warm_notes(doc_key: str):
    """Pre-build a paper's notes after upload so its first summary is fast."""
    with _bm25_lock:
        corpus = bm25_docs
    doc = documents_from_chunks(corpus).get(doc_key)
    if not doc:
        return
    try:
        get_notes(doc_key, doc, on_progress=print)
        print(f"Notes ready for {doc['source']}")
    except Exception as e:
        # Not fatal: notes are rebuilt on the first whole-paper question
        print(f"Could not pre-build notes for {doc['source']}: {e}")


def warm_all_notes():
    """At startup, build notes for any indexed paper that doesn't have them yet."""
    with _bm25_lock:
        corpus = bm25_docs
    for key, doc in documents_from_chunks(corpus).items():
        if not has_notes(key):
            warm_notes(key)


# 8. Main RAG
def ask(query: str, document_id=None):
    print(f"\n Query: {query}")

    p = plan(query, document_id)
    if p["mode"] == "global":
        notes = [get_notes(key, doc) for key, doc in p["targets"]]
        prompt = build_reduce_prompt(query, p["targets"], notes)
    else:
        if not p["context"]:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}
        prompt = PROMPT.format(context=p["context"], question=query)

    response = get_llm().invoke(prompt)
    answer = getattr(response, "content", str(response))

    return {"answer": answer, "sources": p["sources"]}


# 9. Streaming
# Yields events: {"type": "status" | "sources" | "token", ...}
async def stream_answer(query: str, document_id=None):
    yield {"type": "status", "message": "Searching papers…"}
    p = await run_in_threadpool(plan, query, document_id)
    yield {"type": "sources", "sources": p["sources"]}

    if p["mode"] == "global":
        loop = asyncio.get_running_loop()
        progress: asyncio.Queue = asyncio.Queue()

        def on_progress(message):
            loop.call_soon_threadsafe(progress.put_nowait, message)

        notes = []
        for key, doc in p["targets"]:
            yield {"type": "status", "message": f"Reading {doc['source']}…"}
            task = asyncio.ensure_future(run_in_threadpool(get_notes, key, doc, on_progress))
            while not task.done() or not progress.empty():
                try:
                    message = await asyncio.wait_for(progress.get(), timeout=0.5)
                    yield {"type": "status", "message": message}
                except asyncio.TimeoutError:
                    pass
            notes.append(task.result())

        prompt = build_reduce_prompt(query, p["targets"], notes)
        yield {"type": "status", "message": "Writing answer…"}
    else:
        if not p["context"]:
            yield {"type": "token", "content": NO_CONTEXT_ANSWER}
            return
        prompt = PROMPT.format(context=p["context"], question=query)

    async for chunk in get_llm().astream(prompt):
        yield {"type": "token", "content": getattr(chunk, "content", str(chunk))}

    #The ask function is a synchronous RAG pipeline that returns the complete response after retrieval and generation, while stream_answer is an asynchronous generator that streams tokens incrementally from the LLM, improving user experience by reducing perceived latency without changing the underlying computation.

# rag.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

from app.config import get_llm

DB_DIR = "chroma_db"

# 1. Models (init once)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#A CrossEncoder is a re-ranking model that scores the relevance of a query-document pair by jointly encoding them, unlike embedding models that encode them separately. It is used after vector retrieval to improve precision by reordering top-k results based on deeper semantic understanding.
llm = get_llm()

# 2. Load vector DB
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

dense_retriever = vectordb.as_retriever(
    search_type="mmr", #Maximal Marginal Relevance - diverse + relevant results. MMR retrieval selects documents by balancing relevance to the query and diversity among results. It first retrieves a larger candidate set (fetch_k) using similarity search, then iteratively selects the top k documents that maximize both relevance and novelty, reducing redundancy in retrieved context.
    search_kwargs={"k": 8, "fetch_k": 20}
)

# 3. Build BM25 index (lazy)
bm25 = None
bm25_docs = []

def init_bm25():
    global bm25, bm25_docs
    
    docs = vectordb.get()["documents"]
    if not docs:
        return
    
    tokenized = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_docs = docs  # Keep as list of strings

# 4. Hybrid retrieval
def hybrid_retrieve(query, k=10):
    dense_docs = dense_retriever.invoke(query)
    
    sparse_docs = []
    if bm25:
        scores = bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:k]
        sparse_docs = [bm25_docs[i] for i in top_idx]
    
    # Get metadata mapping from ChromaDB
    all_data = vectordb.get()
    text_to_metadata = {
        text: meta 
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    }
    
    # Merge results
    seen = set()
    merged = []
    
    for d in dense_docs:
        if d.page_content not in seen:
            merged.append(d)
            seen.add(d.page_content)
    
    for text in sparse_docs:
        if text not in seen:
            metadata = text_to_metadata.get(text, {"source": "unknown", "page": "N/A"})
            # Create document with proper metadata
            merged.append(Document(page_content=text, metadata=metadata))
            seen.add(text)
    
    return merged[:k]

# 5. Reranking
def rerank(query, docs, top_k=5):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(
        pairs,
        batch_size=16
    )

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc for doc, score in ranked[:top_k]
        if score > 0.2  # optional threshold
    ]

# 6. Context builder
def build_context(docs, max_chars=3000):
    context = []
    sources = []

    total = 0
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue

        if total + len(text) > max_chars:
            break

        context.append(text)
        total += len(text)

        sources.append({
            "id": i,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A")
        })

    return "\n\n".join(context), sources

# 7. Prompt
PROMPT = """You are a research assistant.

Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Provide a clear, structured answer.

Context:
{context}

Question:
{question}

Answer:
"""

# 8. Main RAG
def ask(query: str):
    print(f"\n Query: {query}")
    
    docs = hybrid_retrieve(query)
    print(f"Retrieved {len(docs)} docs before reranking")
    
    docs = rerank(query, docs)
    print(f"Reranked to {len(docs)} docs")
    
    context, sources = build_context(docs)
    print(f"Sources with metadata: {sources}")  # Debug output
    
    # Check if we have good sources
    if sources and sources[0]["source"] == "unknown":
        print("WARNING: Missing metadata! Check ingest pipeline.")
    
    prompt = PROMPT.format(context=context, question=query)
    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response))
    
    return {"answer": answer, "sources": sources}

# 9. Streaming
async def stream_answer(query: str):
    docs = hybrid_retrieve(query)
    docs = rerank(query, docs)

    context, sources = build_context(docs)

    prompt = PROMPT.format(
        context=context,
        question=query
    )

    async for chunk in llm.astream(prompt):
        yield getattr(chunk, "content", str(chunk))

    yield f"\n\n[SOURCES]: {sources}"

    #The ask function is a synchronous RAG pipeline that returns the complete response after retrieval and generation, while stream_answer is an asynchronous generator that streams tokens incrementally from the LLM, improving user experience by reducing perceived latency without changing the underlying computation.
    
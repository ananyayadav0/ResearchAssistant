import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DB_DIR = os.getenv("DB_DIR", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Reranker scores are passed through a sigmoid, so this is a 0-1 relevance cutoff
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.001"))

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Ollama's default context is smaller than llama3's 8k and silently truncates long prompts
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


@lru_cache(maxsize=1)
def get_llm():
    # Created lazily so the API can start (and ingest) before the LLM is reachable
    if USE_GROQ:
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, temperature=0)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX, temperature=0
    )

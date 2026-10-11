from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import DB_DIR, EMBEDDING_MODEL

# Shared by ingestion and retrieval so the embedding model is loaded once
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)

import hashlib
import os
import re

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.store import vectordb

MIN_CHUNK_CHARS = 30

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


def clean_text(text: str) -> str:
    # Re-join words hyphenated across line breaks, then collapse whitespace
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(file_path: str, source_name: str | None = None) -> int:
    """Chunk a PDF and upsert it into the vector store. Returns the chunk count.

    Chunk ids are derived from the file's content hash, so re-uploading the
    same PDF overwrites its chunks instead of duplicating them.
    """
    source = source_name or os.path.basename(file_path)
    doc_hash = file_hash(file_path)

    texts, metadatas, ids = [], [], []

    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if not isinstance(page_text, str) or not page_text.strip():
                continue

            chunks = splitter.split_text(clean_text(page_text))
            for i, chunk in enumerate(chunks):
                if len(chunk) < MIN_CHUNK_CHARS:
                    continue
                texts.append(chunk)
                metadatas.append({
                    "source": source,
                    "page": page_num,
                    "chunk_index": i,
                    "file_path": file_path,
                    "file_hash": doc_hash,
                })
                ids.append(f"{doc_hash[:16]}-p{page_num}-c{i}")

    if not texts:
        raise ValueError("No text extracted from PDF (it may be scanned or image-only)")

    print(f"Ingesting {len(texts)} chunks from {source}")

    # langchain_chroma upserts by id and persists automatically
    vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(texts)

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import fitz
import os

DB_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest_pdf(file_path):
    doc = fitz.open(file_path)
    texts = []
    metadatas = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        if isinstance(page_text, str) and page_text.strip():
            chunks = splitter.split_text(page_text)

            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    "source": os.path.basename(file_path),
                    "page": page_num,
                    "file_path": file_path
                })

    if not texts:
        raise ValueError("No text extracted from PDF")
    print("Ingesting chunks:", len(texts))
    print("Metadata sample:", metadatas[:2])

    # Append or create DB
    if os.path.exists(DB_DIR):
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        vectordb.add_texts(texts=texts, metadatas=metadatas)
    else:
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=DB_DIR
        )

    vectordb.persist()
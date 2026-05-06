from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import shutil
import uuid
import os

from app.ingest import ingest_pdf
from app.rag import ask, stream_answer
from typing import Optional

app = FastAPI()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def is_valid_pdf(filename: Optional[str], content_type: Optional[str]) -> bool:
    if filename is None or content_type is None:
        return False

    return filename.lower().endswith(".pdf") and content_type == "application/pdf"

# 1. Upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not is_valid_pdf(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="Invalid PDF")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(DATA_DIR, f"{file_id}.pdf")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_pdf(file_path)

    return {"message": "File processed", "file_id": file_id}

# 2. Non-streaming RAG
@app.post("/ask")
async def ask_question(query: str):
    try:
        result = ask(query)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Streaming RAG (SSE)
@app.post("/ask-stream")
async def ask_stream(query: str):

    async def event_generator():
        try:
            async for chunk in stream_answer(query):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# 4. Health check
@app.get("/")
def root():
    return {"status": "RAG API running"}
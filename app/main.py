import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import DATA_DIR
from app.ingest import file_hash, ingest_pdf
from app.rag import ask, init_bm25, list_documents, stream_answer, warm_all_notes, warm_notes

os.makedirs(DATA_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(init_bm25)
    # Daemon thread: pre-builds whole-paper notes without delaying startup
    threading.Thread(target=warm_all_notes, daemon=True).start()
    yield


app = FastAPI(title="Research Assistant RAG", lifespan=lifespan)


class Question(BaseModel):
    query: str = Field(min_length=1)
    # Limit the answer to one paper (an id from GET /documents); omit to search all papers
    document_id: str | None = None


def is_valid_pdf(filename: str | None, head: bytes) -> bool:
    # Check the file signature rather than trusting the client's content type
    return bool(filename) and filename.lower().endswith(".pdf") and head.startswith(b"%PDF-")


# 1. Upload endpoint
# Plain `def` endpoints run in FastAPI's threadpool, so slow embedding /
# LLM calls don't block the event loop for other requests.
@app.post("/upload")
def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    head = file.file.read(5)
    if not is_valid_pdf(file.filename, head):
        raise HTTPException(status_code=400, detail="Invalid PDF")
    file.file.seek(0)

    tmp_path = os.path.join(DATA_DIR, f".upload-{os.getpid()}-{id(file)}.pdf")
    with open(tmp_path, "wb") as buffer:
        while block := file.file.read(1 << 20):
            buffer.write(block)

    # Name stored files by content hash so duplicate uploads map to one file
    doc_hash = file_hash(tmp_path)
    file_id = doc_hash[:16]
    file_path = os.path.join(DATA_DIR, f"{file_id}.pdf")
    os.replace(tmp_path, file_path)

    try:
        chunks = ingest_pdf(file_path, source_name=file.filename)
    except ValueError as e:
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    init_bm25()
    # Build whole-paper notes after responding, so summaries are fast later
    background_tasks.add_task(warm_notes, doc_hash)

    return {
        "message": "File processed",
        "file_id": file_id,
        "document_id": doc_hash,
        "chunks": chunks,
    }


# 2. Non-streaming RAG
@app.post("/ask")
def ask_question(body: Question):
    try:
        return ask(body.query, body.document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. Streaming RAG (SSE)
# Each event is a JSON object: {"type": "status" | "sources" | "token" | "error" | "done", ...}
@app.post("/ask-stream")
async def ask_stream(body: Question):

    def sse(payload: dict) -> str:
        # JSON-encoding escapes newlines inside tokens, which would otherwise break SSE framing
        return f"data: {json.dumps(payload)}\n\n"

    async def event_generator():
        try:
            async for event in stream_answer(body.query, body.document_id):
                yield sse(event)
        except Exception as e:
            yield sse({"type": "error", "detail": str(e)})

        yield sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# 4. Uploaded papers, newest first
@app.get("/documents")
def documents():
    return list_documents()


# 5. Health check
@app.get("/health")
def health():
    return {"status": "RAG API running"}


# 6. Web UI
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

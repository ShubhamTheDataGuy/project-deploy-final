from __future__ import annotations
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exception.custom_exception import DocumentPortalException

# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(title="MultiDocChat", version="0.1.0")



# CORS (optional for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Simple in-memory chat history
# ----------------------------
SESSIONS: Dict[str, List[dict]] = {}

# ----------------------------
# Paths for cleanup
# ----------------------------
DATA_DIR = Path("data")
FAISS_DIR = Path("faiss_index")


def cleanup_old_files(max_age_minutes: int = 2):
    """
    Delete session folders older than max_age_minutes from data/ and faiss_index/.
    """
    now = time.time()
    max_age = max_age_minutes * 60  # Convert minutes to seconds

    for folder in [DATA_DIR, FAISS_DIR]:
        if not folder.exists():
            continue
        for item in folder.iterdir():
            if item.is_dir():
                age = now - item.stat().st_mtime
                if age > max_age:
                    try:
                        shutil.rmtree(item)
                        print(f"[Cleanup] Removed old session folder: {item}")
                    except Exception as e:
                        print(f"[Cleanup] Failed to remove {item}: {e}")


# ----------------------------
# Adapters
# ----------------------------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


# ----------------------------
# Models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Trigger cleanup before processing new upload (2-minute test mode)
        cleanup_old_files(max_age_minutes=2)

        # Wrap FastAPI files to preserve filename/ext and provide a read buffer
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        # Ingest and index
        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id

        # Save, load, split, embed, and write FAISS index with MMR
        ingestor.built_retriver(
            uploaded_files=wrapped_files,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Initialize empty history for this session
        SESSIONS[session_id] = []

        return UploadResponse(session_id=session_id, indexed=True, message="Indexing complete with MMR")

    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()

    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Build RAG and load retriever from persisted FAISS with MMR
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"faiss_index/{session_id}"
        rag.load_retriever_from_faiss(
            index_path=index_path,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Use simple in-memory history and convert to BaseMessage list
        simple = SESSIONS.get(session_id, [])
        lc_history = []
        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        answer = rag.invoke(message, chat_history=lc_history)

        # Update history
        simple.append({"role": "user", "content": message})
        simple.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = simple

        return ChatResponse(answer=answer)

    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# ----------------------------
# Uvicorn entrypoint
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)

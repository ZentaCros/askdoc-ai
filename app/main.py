"""
AskDoc AI - FastAPI Backend
Enterprise document Q&A chatbot powered by RAG + LLM
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

from app.rag import DocumentStore, build_rag_prompt

load_dotenv()

app = FastAPI(
    title="AskDoc AI",
    description="Enterprise document Q&A chatbot powered by RAG + LLM",
    version="1.0.0"
)

# Global document store
doc_store = DocumentStore()

# Groq API config
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: int
    document: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "document_loaded": bool(doc_store.chunks),
        "chunks": len(doc_store.chunks),
        "filename": doc_store.filename
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document for Q&A."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        contents = await file.read()
        num_chunks = doc_store.ingest_pdf(contents, file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": num_chunks,
            "message": f"Document processed into {num_chunks} searchable chunks"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about the uploaded document."""
    if not doc_store.chunks:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured"
        )

    # Retrieve relevant chunks
    relevant_chunks = doc_store.search(request.message, top_k=3)

    # Build RAG prompt
    prompt = build_rag_prompt(request.message, relevant_chunks)

    # Call Groq API
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are AskDoc AI, a precise document assistant. Answer questions based only on the provided document context."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                }
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"LLM API error: {response.text}"
            )

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return ChatResponse(
            response=answer,
            sources=len(relevant_chunks),
            document=doc_store.filename
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("static/index.html")

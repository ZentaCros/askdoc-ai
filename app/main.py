"""
AskDoc AI - FastAPI Backend
Enterprise document Q&A chatbot powered by RAG + Multimodal LLM
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
from dotenv import load_dotenv

from app.rag import DocumentStore, build_rag_prompt
from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_VISION_MODEL,
    MAX_IMAGE_EXTRACTION_PER_DOC,
)

load_dotenv()

app = FastAPI(
    title="AskDoc AI",
    description="Enterprise document Q&A chatbot powered by RAG + LLM",
    version="1.2.0"
)

# Global document store
doc_store = DocumentStore()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    response: str
    sources: int
    document: str
    is_fallback: bool = False


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "document_loaded": bool(doc_store.chunks),
        "chunks": len(doc_store.chunks),
        "images_analyzed": doc_store.images_analyzed,
        "filename": doc_store.filename,
        "model": GROQ_MODEL,
        "vision_model": GROQ_VISION_MODEL,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document for Q&A, including visual/image elements."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        contents = await file.read()
        num_chunks, num_images = await doc_store.ingest_pdf(
            contents,
            file.filename,
            api_key=GROQ_API_KEY,
            vision_model=GROQ_VISION_MODEL,
            max_images=MAX_IMAGE_EXTRACTION_PER_DOC,
        )

        msg = f"Document processed into {num_chunks} searchable chunks"
        if num_images > 0:
            msg += f" (including {num_images} visual elements analyzed via Vision AI)"

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": num_chunks,
            "images": num_images,
            "message": msg
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about the uploaded document with multi-turn conversation memory."""
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

    # Extract user history to expand follow-up questions/pronouns
    user_queries = [
        h.content for h in (request.history or []) if h.role == "user"
    ]

    # Retrieve relevant chunks with smart conversational fallback
    relevant_chunks, is_fallback = doc_store.search(
        request.message,
        top_k=3,
        history_messages=user_queries
    )

    # Build prompt
    prompt = build_rag_prompt(request.message, relevant_chunks, is_fallback=is_fallback)

    # Build conversation payload for the LLM
    llm_messages = [
        {
            "role": "system",
            "content": (
                "You are AskDoc AI, an intelligent, discerning, and friendly document assistant. "
                "You maintain conversation context, remember previous turns, and evaluate documents thoughtfully "
                "based on the provided context."
            ),
        }
    ]

    # Include recent turns of conversation history (up to last 6 messages)
    if request.history:
        for item in request.history[-6:]:
            role = "assistant" if item.role in ["assistant", "bot"] else "user"
            llm_messages.append({"role": role, "content": item.content})

    # Final user prompt with retrieved document context
    llm_messages.append({"role": "user", "content": prompt})

    # Call Groq API
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": llm_messages,
                    "temperature": 0.4,
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
            document=doc_store.filename,
            is_fallback=is_fallback
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

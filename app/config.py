"""
AskDoc AI - Configuration
Loads environment variables for LLM API access.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "")

# Chunk settings
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks

"""
AskDoc AI - Configuration
Loads environment variables for LLM API access.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "")

# Chunk settings
# Using 1200 characters preserves section coherence and headers
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_IMAGE_EXTRACTION_PER_DOC = int(os.getenv("MAX_IMAGE_EXTRACTION_PER_DOC", "5"))

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

API_TOKEN = os.getenv("API_TOKEN", "")
RATE_LIMIT = os.getenv("RATE_LIMIT", "5/minute")

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "info")
CHUNK_SIZE = 200  # words
TOP_K = 3

# Embedding model (any sentence-transformers model name)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# FAISS index strategy: flat | ivf | hnsw
INDEX_TYPE = os.getenv("INDEX_TYPE", "flat")

# Vector store backend: faiss | qdrant
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")

# Directory where the FAISS index is persisted between restarts
INDEX_PERSIST_PATH = os.getenv("INDEX_PERSIST_PATH", os.path.join(os.path.dirname(__file__), "..", "faiss_store"))

# Qdrant (only used when VECTOR_DB=qdrant)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")

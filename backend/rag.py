import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import DOCS_DIR, CHUNK_SIZE, TOP_K

_model = SentenceTransformer("all-MiniLM-L6-v2")
_index: faiss.IndexFlatL2 | None = None
_chunks: list[str] = []


def _load_and_chunk_docs() -> list[str]:
    chunks = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, encoding="utf-8") as f:
            words = f.read().split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
    return chunks


def build_index() -> None:
    global _index, _chunks
    _chunks = _load_and_chunk_docs()
    if not _chunks:
        raise RuntimeError(f"No .txt files found in {DOCS_DIR}")
    vectors = _model.encode(_chunks, convert_to_numpy=True).astype("float32")
    dim = vectors.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(vectors)


def retrieve(query: str) -> list[str]:
    if _index is None:
        raise RuntimeError("Index not built. Call build_index() first.")
    q_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = _index.search(q_vec, TOP_K)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return (
        "Answer the question using ONLY the context below.\n"
        'If the answer is not in the context, say "I don\'t know".\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

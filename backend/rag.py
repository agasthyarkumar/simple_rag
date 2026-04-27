import os
import glob
import time
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    DOCS_DIR, CHUNK_SIZE, TOP_K,
    INDEX_TYPE, VECTOR_DB, INDEX_PERSIST_PATH, EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_model = SentenceTransformer(EMBEDDING_MODEL)
_index: faiss.Index | None = None
_chunks: list[dict] = []   # each entry: {"text": str, "source": str}
_qdrant_client = None      # initialized lazily when VECTOR_DB="qdrant"

_INDEX_FILE = os.path.join(INDEX_PERSIST_PATH, "index.faiss")
_CHUNKS_FILE = os.path.join(INDEX_PERSIST_PATH, "chunks.npy")


# ── document loading ──────────────────────────────────────────────────────────

def _load_and_chunk_docs() -> list[dict]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            words = f.read().split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if chunk.strip():
                chunks.append({"text": chunk, "source": source})
    return chunks


# ── vector utilities ──────────────────────────────────────────────────────────

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize so dot-product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype("float32")


def _encode(texts: list[str]) -> np.ndarray:
    vecs = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    return _normalize(vecs)


# ── FAISS index factory ───────────────────────────────────────────────────────

def _make_faiss_index(dim: int, n: int) -> faiss.Index:
    """
    Create the configured FAISS index type.

    flat  → IndexFlatL2  : exact brute-force L2 search; always 100% recall;
                           O(n) per query — best for small corpora (<50 k).
    ivf   → IndexIVFFlat : partitions vectors into nlist Voronoi cells; at
                           query time visits only the nearest `nprobe` cells.
                           Requires training. Fast on large corpora (>10 k).
    hnsw  → IndexHNSWFlat: hierarchical navigable small-world graph; sub-linear
                           query time with high recall; no training required.
                           High memory usage but very fast.
    """
    if INDEX_TYPE == "ivf":
        # nlist: number of Voronoi cells. Rule of thumb: sqrt(n), capped at 256.
        nlist = max(1, min(int(np.sqrt(n)), n // 4, 256))
        if n < 30 * nlist:
            logger.warning(
                f"IVFFlat: only {n} vectors for nlist={nlist}. "
                f"Optimal clustering needs ~{30 * nlist}+ vectors. "
                "Retrieval quality may be lower than IndexFlatL2."
            )
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        logger.info(f"IVFFlat index created | nlist={nlist}")
        return index

    elif INDEX_TYPE == "hnsw":
        # M: connections per node. Higher → better recall, more memory.
        M = 32
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = 200  # graph build quality
        logger.info(f"HNSWFlat index created | M={M}")
        return index

    else:  # "flat" or any unknown value → safe default
        if INDEX_TYPE not in ("flat", ""):
            logger.warning(f"Unknown INDEX_TYPE='{INDEX_TYPE}', using IndexFlatL2")
        logger.info("IndexFlatL2 (exact search)")
        return faiss.IndexFlatL2(dim)


# ── FAISS persistence ─────────────────────────────────────────────────────────

def _save_faiss_index() -> None:
    os.makedirs(INDEX_PERSIST_PATH, exist_ok=True)
    faiss.write_index(_index, _INDEX_FILE)
    np.save(_CHUNKS_FILE, np.array(_chunks, dtype=object))
    logger.info(f"FAISS index persisted → {INDEX_PERSIST_PATH}")


def _load_faiss_index() -> bool:
    global _index, _chunks
    if not (os.path.exists(_INDEX_FILE) and os.path.exists(_CHUNKS_FILE)):
        return False
    try:
        _index = faiss.read_index(_INDEX_FILE)
        _chunks = np.load(_CHUNKS_FILE, allow_pickle=True).tolist()
        logger.info(f"Loaded persisted FAISS index: {len(_chunks)} chunks")
        return True
    except Exception as exc:
        logger.warning(f"Could not load persisted index ({exc}), rebuilding…")
        return False


# ── Qdrant integration ────────────────────────────────────────────────────────

def _build_qdrant_index(vectors: np.ndarray) -> None:
    global _qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from config import QDRANT_URL, QDRANT_COLLECTION

    _qdrant_client = QdrantClient(url=QDRANT_URL)
    dim = vectors.shape[1]

    _qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": c["text"], "source": c["source"]},
        )
        for i, c in enumerate(_chunks)
    ]
    _qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info(f"Qdrant collection built | collection={QDRANT_COLLECTION} | vectors={len(points)}")


def _retrieve_qdrant(q_vec: np.ndarray) -> list[dict]:
    from config import QDRANT_COLLECTION
    results = _qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=q_vec[0].tolist(),
        limit=TOP_K,
    )
    return [{"text": r.payload["text"], "source": r.payload["source"]} for r in results]


# ── public API ────────────────────────────────────────────────────────────────

def build_index() -> None:
    """Build (or load from disk) the vector index for all documents in DOCS_DIR."""
    global _index, _chunks
    logger.info(f"build_index() | INDEX_TYPE={INDEX_TYPE} | VECTOR_DB={VECTOR_DB} | MODEL={EMBEDDING_MODEL}")

    # Fast path: load persisted FAISS index (skip if using Qdrant)
    if VECTOR_DB == "faiss" and _load_faiss_index():
        return

    _chunks = _load_and_chunk_docs()
    if not _chunks:
        raise RuntimeError(f"No .txt files found in {DOCS_DIR}")

    logger.info(f"Encoding {len(_chunks)} chunks…")
    t0 = time.perf_counter()
    vectors = _encode([c["text"] for c in _chunks])
    logger.info(f"Encoding done in {time.perf_counter() - t0:.2f}s")

    if VECTOR_DB == "qdrant":
        _build_qdrant_index(vectors)
        return

    # ── FAISS build ──
    n, dim = vectors.shape
    _index = _make_faiss_index(dim, n)

    if not _index.is_trained:
        try:
            _index.train(vectors)
            logger.info(f"{INDEX_TYPE} training complete")
        except Exception as exc:
            logger.warning(f"Training failed ({exc}), falling back to IndexFlatL2")
            _index = faiss.IndexFlatL2(dim)

    _index.add(vectors)
    logger.info(f"FAISS index ready | vectors={_index.ntotal} | dim={dim}")
    _save_faiss_index()


def retrieve_with_metadata(query: str) -> list[dict]:
    """Return top-k chunks as [{"text": str, "source": str}]."""
    if not query.strip():
        return []

    t0 = time.perf_counter()
    q_vec = _encode([query])

    if VECTOR_DB == "qdrant":
        hits = _retrieve_qdrant(q_vec)
        logger.info(f"Qdrant retrieval | chunks={len(hits)} | {time.perf_counter()-t0:.3f}s")
        return hits

    if _index is None:
        raise RuntimeError("Index not built. Call build_index() first.")

    # Set search-time accuracy parameters
    if hasattr(_index, "nprobe"):          # IVFFlat: cells to visit per query
        _index.nprobe = min(10, _index.nlist)
    if hasattr(_index, "hnsw"):            # HNSWFlat: beam width during search
        _index.hnsw.efSearch = 128

    k = min(TOP_K, _index.ntotal)
    _, indices = _index.search(q_vec, k)
    hits = [_chunks[i] for i in indices[0] if 0 <= i < len(_chunks)]
    logger.info(f"FAISS ({INDEX_TYPE}) retrieval | chunks={len(hits)} | {time.perf_counter()-t0:.3f}s")
    return hits


def retrieve(query: str) -> list[str]:
    """Return plain text chunks — kept for backward compatibility."""
    return [c["text"] for c in retrieve_with_metadata(query)]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return (
        "Answer the question using ONLY the context below.\n"
        'If the answer is not in the context, say "I don\'t know".\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

import json
import time as _time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime, timedelta

import rag
import llm_provider
from config import API_TOKEN, RATE_LIMIT, INDEX_TYPE, VECTOR_DB, EMBEDDING_MODEL

# ── rate-limit state (in-memory, per IP) ────────────────────────────────────
_rate_limit_count = int(RATE_LIMIT.split("/")[0])
_rate_windows: dict[str, list[datetime]] = defaultdict(list)


def _check_rate_limit(ip: str) -> None:
    now = datetime.utcnow()
    window = timedelta(minutes=1)
    _rate_windows[ip] = [t for t in _rate_windows[ip] if now - t < window]
    if len(_rate_windows[ip]) >= _rate_limit_count:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_windows[ip].append(now)


# ── auth ─────────────────────────────────────────────────────────────────────
security = HTTPBearer()


def _verify_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> None:
    if API_TOKEN and creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.build_index()
    yield


app = FastAPI(title="Simple RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    context: list[str]        # plain text chunks (backward-compatible)
    sources: list[str] = []   # source filename for each chunk (parallel to context)


# ── helpers ───────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": len(rag._chunks)}


@app.get("/index-info")
async def index_info():
    """Return metadata about the active retrieval configuration."""
    return {
        "index_type": INDEX_TYPE,
        "vector_db": VECTOR_DB,
        "embedding_model": EMBEDDING_MODEL,
        "chunks_indexed": len(rag._chunks),
    }


@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    _: None = Depends(_verify_token),
):
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    hits = rag.retrieve_with_metadata(req.question)
    chunks = [h["text"] for h in hits]
    sources = [h["source"] for h in hits]

    prompt = rag.build_prompt(req.question, chunks)
    answer = await llm_provider.call_llm(prompt)

    return QueryResponse(answer=answer, context=chunks, sources=sources)


@app.post("/query/pipeline")
async def query_pipeline(
    req: QueryRequest,
    request: Request,
    _: None = Depends(_verify_token),
):
    """
    Streaming SSE endpoint that emits one event per pipeline stage:
      fastapi → embedding → search → retrieval → llm → done

    Each event: data: {"step": "<name>", "status": "running"|"done", ...metadata}
    """
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate():
        # Stage 1: FastAPI received the request
        yield _sse({"step": "fastapi", "status": "done"})

        # Stage 2: Embed the query
        yield _sse({"step": "embedding", "status": "running"})
        t0 = _time.perf_counter()
        q_vec = rag.encode_query(req.question)
        embed_ms = round((_time.perf_counter() - t0) * 1000)
        yield _sse({
            "step": "embedding", "status": "done",
            "ms": embed_ms, "dim": int(q_vec.shape[1]),
        })

        # Stage 3: Vector search (FAISS or Qdrant)
        yield _sse({"step": "search", "status": "running"})
        t0 = _time.perf_counter()
        hits = rag.search_encoded(q_vec)
        search_ms = round((_time.perf_counter() - t0) * 1000)
        yield _sse({
            "step": "search", "status": "done",
            "ms": search_ms, "chunks": len(hits),
            "db": VECTOR_DB, "index_type": INDEX_TYPE,
        })

        chunks = [h["text"] for h in hits]
        sources = [h["source"] for h in hits]

        # Stage 4: Retrieval context assembled
        yield _sse({"step": "retrieval", "status": "done", "sources": sources})

        # Stage 5: LLM generates the answer
        yield _sse({"step": "llm", "status": "running"})
        t0 = _time.perf_counter()
        prompt = rag.build_prompt(req.question, chunks)
        answer = await llm_provider.call_llm(prompt)
        llm_ms = round((_time.perf_counter() - t0) * 1000)

        # Final event carries the full result
        yield _sse({
            "step": "done",
            "llm_ms": llm_ms,
            "answer": answer,
            "context": chunks,
            "sources": sources,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

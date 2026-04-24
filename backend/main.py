from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime, timedelta

import rag
import llm_provider
from config import API_TOKEN, RATE_LIMIT

# ── rate-limit state (in-memory, per IP) ────────────────────────────────────
_rate_limit_count = int(RATE_LIMIT.split("/")[0])  # e.g. 5
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
    context: list[str]


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": len(rag._chunks)}


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

    chunks = rag.retrieve(req.question)
    prompt = rag.build_prompt(req.question, chunks)
    answer = await llm_provider.call_llm(prompt)

    return QueryResponse(answer=answer, context=chunks)

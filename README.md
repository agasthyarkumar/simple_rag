# Simple RAG

A minimal but production-ready **Retrieval-Augmented Generation** app for learning and experimentation.

Ask questions, get answers grounded in your own documents — no hallucination.

---

## What it does

1. Chunks your `.txt` documents into overlapping word windows
2. Embeds every chunk with **SentenceTransformers** (`all-MiniLM-L6-v2`)
3. Indexes the embeddings in **FAISS** (or **Qdrant**)
4. At query time: embeds your question → vector search → top-k chunks → LLM prompt
5. Streams each pipeline stage live to the UI so you can watch the data flow in real time

---

## Features

| Area | Detail |
|---|---|
| **Live pipeline UI** | Each stage lights up as it runs: FastAPI → Embed → Search → Retrieval → LLM |
| **Multiple FAISS indexes** | `flat` (exact), `ivf` (clustered), `hnsw` (graph-based ANN) — swap with one env var |
| **Qdrant support** | Switch to a real vector database via Docker — same query interface |
| **Index persistence** | FAISS index saved to disk; reloads instantly on restart, no re-encoding |
| **Chunk metadata** | Every chunk tracks its source file; shown as color-coded badges in the UI |
| **Configurable** | Embedding model, index type, vector DB, chunk size, top-k — all via `.env` |
| **Auth + rate limiting** | Bearer token auth, per-IP rate limiting (requests/minute) |

---

## Project structure

```
simple_rag/
├── backend/
│   ├── main.py           # FastAPI app — /query, /query/pipeline (SSE), /index-info
│   ├── rag.py            # Chunking, embedding, FAISS/Qdrant index, retrieval
│   ├── llm_provider.py   # Groq async client
│   ├── config.py         # All settings (reads from .env)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx        # Chat UI + live pipeline visualization
│       └── styles.css
├── info/                  # Drop your .txt documents here
├── faiss_store/           # Auto-created — persisted FAISS index
├── .env                   # Your secrets and config (copy from .env.example)
├── start                  # One-command startup (Linux/Mac)
└── start.bat              # One-command startup (Windows)
```

---

## Quick start

### 1. Copy and fill in the env file

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key
API_TOKEN=any_secret_string
```

### 2. Add your documents

Drop any `.txt` files into the `info/` directory. The backend chunks and indexes them on startup.

### 3. Start everything

**Linux / Mac**
```bash
./start
```

**Windows**
```bat
start.bat
```

This installs dependencies, starts the FastAPI backend on `http://localhost:8000` and the Vite frontend on `http://localhost:5173`.

---

## Configuration

All options live in `.env`. The backend reads them at startup.

### Core

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq model ID |
| `API_TOKEN` | — | Bearer token for `/query`; leave empty to disable auth |
| `RATE_LIMIT` | `5/minute` | Max requests per IP per minute |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any `sentence-transformers` model |
| `CHUNK_SIZE` | `200` | Words per chunk |
| `TOP_K` | `3` | Chunks retrieved per query |

### Vector index

| Variable | Default | Options | Description |
|---|---|---|---|
| `INDEX_TYPE` | `flat` | `flat` `ivf` `hnsw` | FAISS index strategy (see below) |
| `VECTOR_DB` | `faiss` | `faiss` `qdrant` | Where vectors are stored |
| `INDEX_PERSIST_PATH` | `./faiss_store` | any path | Where the FAISS index is saved to disk |

### Qdrant (only needed when `VECTOR_DB=qdrant`)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `rag_docs` | Collection name |

---

## FAISS index types

Change `INDEX_TYPE` in `.env`, delete `faiss_store/`, and restart to rebuild.

### `flat` — IndexFlatL2 (default)
Exact brute-force L2 search. Compares the query against every vector.
- **Recall**: 100% — always finds the true nearest neighbors
- **Speed**: O(n) per query — fine up to ~50k vectors
- **Training**: none required
- **Best for**: small corpora, benchmarking, when correctness matters most

### `ivf` — IndexIVFFlat
Partitions vectors into Voronoi cells (clusters). At query time, searches only the nearest `nprobe` cells instead of all vectors.
- **Recall**: ~95–99% (tunable via `nprobe`)
- **Speed**: much faster than flat on large datasets (>10k vectors)
- **Training**: required — needs enough vectors to cluster
- **Best for**: large document collections where sub-millisecond queries matter

### `hnsw` — IndexHNSWFlat
Builds a hierarchical navigable small-world graph. Traverses the graph to find approximate nearest neighbors.
- **Recall**: ~99% with default settings
- **Speed**: sub-linear query time, very fast even on large datasets
- **Training**: none required
- **Memory**: higher than flat or ivf
- **Best for**: production workloads where speed and recall both matter

---

## Switching to Qdrant

Qdrant is a purpose-built vector database. Use it when you want persistence, filtering, or a production-grade search backend.

### 1. Update `.env`

```env
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_docs
```

### 2. Restart with `./start`

```bash
./start
```

The start script detects `VECTOR_DB=qdrant` in your `.env` and automatically starts (or restarts) the Qdrant Docker container before launching the backend. On first run it pulls the image; subsequent runs just call `docker start qdrant`.

If Docker is unavailable, start Qdrant manually first:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

On startup the backend retries the Qdrant connection up to 5 times (2 s apart), then uploads all chunk vectors. The header badge will show `flat · qdrant · N chunks`.

### Switching back to FAISS

```env
VECTOR_DB=faiss
```

Restart — the persisted `faiss_store/` index loads instantly.

---

## API

All endpoints are served by FastAPI on `http://localhost:8000`.

### `POST /query`

Standard request/response query.

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{"question": "what is a linked list?"}'
```

```json
{
  "answer": "A linked list is a linear data structure...",
  "context": ["chunk text 1", "chunk text 2", "chunk text 3"],
  "sources": ["linkedlist.txt", "arrays.txt", "tree.txt"]
}
```

### `POST /query/pipeline`

Server-Sent Events stream. Emits one event per pipeline stage, then the final result.

```
data: {"step": "fastapi",   "status": "done"}
data: {"step": "embedding", "status": "running"}
data: {"step": "embedding", "status": "done", "ms": 17, "dim": 384}
data: {"step": "search",    "status": "running"}
data: {"step": "search",    "status": "done", "ms": 1, "chunks": 3, "db": "qdrant", "index_type": "flat"}
data: {"step": "retrieval", "status": "done", "sources": ["linkedlist.txt", "arrays.txt"]}
data: {"step": "llm",       "status": "running"}
data: {"step": "done",      "llm_ms": 980, "answer": "...", "context": [...], "sources": [...]}
```

This is the endpoint the UI uses. The pipeline visualization is driven entirely by these events.

### `GET /index-info`

```json
{
  "index_type": "flat",
  "vector_db": "qdrant",
  "embedding_model": "all-MiniLM-L6-v2",
  "chunks_indexed": 4
}
```

### `GET /health`

```json
{"status": "ok", "chunks_indexed": 4}
```

---

## How the pipeline UI works

When you send a question, the frontend opens a streaming connection to `/query/pipeline` and renders each SSE event in real time:

```
[FastAPI] → [Embed] → [Search] → [Retrieval] → [LLM]
  ✓           ✓  17ms   ✓  1ms      ✓             ✓  980ms
```

- **Gray** = pending
- **Blue pulsing** = currently running
- **Green ✓** = done, with timing/metadata below

Once all stages complete, the pipeline card turns green and the answer + color-coded source chunks appear beneath it.

---

## Adding documents

Drop any `.txt` file into `info/` and restart the backend. The index is rebuilt automatically.

If you're using FAISS, delete `faiss_store/` first to force a full rebuild:

```bash
rm -rf faiss_store/
cd backend && uvicorn main:app --reload
```

If you're using Qdrant, the collection is recreated on every startup automatically.

---

## Dependencies

**Backend** (`backend/requirements.txt`)
- `fastapi`, `uvicorn` — API server
- `sentence-transformers` — embedding model
- `faiss-cpu` — vector index
- `groq` — LLM client
- `qdrant-client` — optional, only used when `VECTOR_DB=qdrant`

**Frontend**
- React 18, Vite — no UI framework, plain CSS

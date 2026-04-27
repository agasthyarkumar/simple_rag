import { useState, useRef, useEffect, Fragment } from "react";

// ── pipeline step definitions (order matters — left to right) ─────────────────
const STEP_DEFS = [
  {
    id: "fastapi",
    label: "FastAPI",
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8M12 17v4" />
      </svg>
    ),
  },
  {
    id: "embedding",
    label: "Embed",
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    ),
  },
  {
    id: "search",
    label: "Search",
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v6c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
        <path d="M3 11v6c0 1.66 4.03 3 9 3s9-1.34 9-3v-6" />
      </svg>
    ),
  },
  {
    id: "retrieval",
    label: "Retrieval",
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><polyline points="13 2 13 9 20 9" />
      </svg>
    ),
  },
  {
    id: "llm",
    label: "LLM",
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z" />
      </svg>
    ),
  },
];

const INITIAL_STEPS = Object.fromEntries(STEP_DEFS.map(s => [s.id, { status: "pending" }]));

// ── color palette for source file badges ──────────────────────────────────────
const FILE_PALETTE = [
  { bg: "#EFF6FF", border: "#93C5FD", text: "#1D4ED8" },
  { bg: "#F0FDF4", border: "#86EFAC", text: "#166534" },
  { bg: "#FFF7ED", border: "#FDBA74", text: "#C2410C" },
  { bg: "#FDF4FF", border: "#D8B4FE", text: "#7E22CE" },
  { bg: "#FFF1F2", border: "#FDA4AF", text: "#9F1239" },
  { bg: "#F0FDFA", border: "#5EEAD4", text: "#0F766E" },
];

function fileColor(filename) {
  let hash = 0;
  for (let i = 0; i < filename.length; i++)
    hash = (hash * 31 + filename.charCodeAt(i)) % FILE_PALETTE.length;
  return FILE_PALETTE[hash];
}

// ── API ───────────────────────────────────────────────────────────────────────
async function fetchIndexInfo() {
  try {
    const res = await fetch("/index-info");
    return res.ok ? res.json() : null;
  } catch { return null; }
}

async function streamPipeline(question, onEvent) {
  const res = await fetch("/query/pipeline", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith("data: ")) {
        try { onEvent(JSON.parse(line.slice(6))); } catch { /* skip malformed */ }
      }
    }
  }
}

// ── components ────────────────────────────────────────────────────────────────

function IndexBadge({ info }) {
  if (!info) return null;
  const DESCRIPTIONS = {
    flat: "IndexFlatL2 — exact brute-force, 100% recall",
    ivf:  "IndexIVFFlat — clustered, fast on large datasets",
    hnsw: "IndexHNSWFlat — graph-based ANN, sub-linear query time",
  };
  return (
    <div className="index-badge" title={DESCRIPTIONS[info.index_type] ?? info.index_type}>
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v6c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
        <path d="M3 11v6c0 1.66 4.03 3 9 3s9-1.34 9-3v-6" />
      </svg>
      <span>{info.index_type}</span>
      <span className="index-badge-sep">·</span>
      <span>{info.vector_db}</span>
      <span className="index-badge-sep">·</span>
      <span>{info.chunks_indexed} chunks</span>
    </div>
  );
}

function PipelineStep({ def, step }) {
  const { status, detail } = step;
  return (
    <div className={`pl-step pl-step--${status}`}>
      <div className="pl-dot">
        {status === "done"    && <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>}
        {status === "running" && <div className="pl-spinner" />}
        {status === "pending" && def.icon}
      </div>
      <span className="pl-label">{def.label}</span>
      {detail && <span className="pl-detail">{detail}</span>}
    </div>
  );
}

function Pipeline({ steps }) {
  const allDone = STEP_DEFS.every(d => steps[d.id]?.status === "done");
  return (
    <div className={`pipeline ${allDone ? "pipeline--done" : ""}`}>
      <div className="pipeline-title">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
        </svg>
        RAG Pipeline
      </div>
      <div className="pipeline-track">
        {STEP_DEFS.map((def, i) => (
          <Fragment key={def.id}>
            {i > 0 && (
              <div className={`pl-arrow ${steps[def.id]?.status !== "pending" || steps[STEP_DEFS[i-1].id]?.status === "done" ? "pl-arrow--active" : ""}`}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
                </svg>
              </div>
            )}
            <PipelineStep def={def} step={steps[def.id] ?? { status: "pending" }} />
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function SourceChunks({ chunks, sources = [] }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(o => !o)}>
        <span className="sources-icon">{open ? "▾" : "▸"}</span>
        {open ? "Hide" : "Show"} {chunks.length} retrieved chunk{chunks.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div className="sources-list">
          {chunks.map((c, i) => {
            const color = sources[i] ? fileColor(sources[i]) : null;
            return (
              <div key={i} className="source-card">
                <span className="source-num">{i + 1}</span>
                <div className="source-content">
                  {sources[i] && (
                    <span className="source-file" style={color ? { background: color.bg, borderColor: color.border, color: color.text } : {}}>
                      {sources[i]}
                    </span>
                  )}
                  <p>{c}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "pipeline") {
    return (
      <div className="msg-row msg-assistant">
        <div className="avatar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
          </svg>
        </div>
        <div className="msg-body">
          <Pipeline steps={msg.steps} />
          {msg.result && (
            <>
              <div className="bubble">{msg.result.answer}</div>
              {msg.result.context?.length > 0 && (
                <SourceChunks chunks={msg.result.context} sources={msg.result.sources} />
              )}
            </>
          )}
        </div>
      </div>
    );
  }

  const isUser = msg.role === "user";
  return (
    <div className={`msg-row ${isUser ? "msg-user" : "msg-assistant"}`}>
      {!isUser && (
        <div className="avatar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
          </svg>
        </div>
      )}
      <div className="msg-body">
        <div className="bubble">{msg.text}</div>
        {!isUser && msg.context?.length > 0 && (
          <SourceChunks chunks={msg.context} sources={msg.sources} />
        )}
      </div>
    </div>
  );
}

// ── main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [indexInfo, setIndexInfo] = useState(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => { fetchIndexInfo().then(setIndexInfo); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  async function send(text) {
    const question = (text ?? input).trim();
    if (!question || loading) return;

    setInput("");
    setError("");
    setMessages(prev => [...prev, { role: "user", text: question }]);
    setLoading(true);
    textareaRef.current?.focus();

    const msgId = `pipeline-${Date.now()}`;
    setMessages(prev => [...prev, {
      id: msgId, role: "pipeline",
      steps: { ...INITIAL_STEPS },
      result: null,
    }]);

    try {
      await streamPipeline(question, (event) => {
        setMessages(prev => {
          const idx = prev.findIndex(m => m.id === msgId);
          if (idx === -1) return prev;

          const msgs = [...prev];
          const pipe = { ...msgs[idx], steps: { ...msgs[idx].steps } };

          if (event.step === "done") {
            pipe.steps.llm = { status: "done", detail: `${event.llm_ms}ms` };
            pipe.result = { answer: event.answer, context: event.context, sources: event.sources };
          } else if (event.status === "running") {
            pipe.steps[event.step] = { status: "running" };
          } else if (event.status === "done") {
            let detail = null;
            if (event.step === "fastapi")   detail = null;
            if (event.step === "embedding") detail = `${event.dim} dim · ${event.ms}ms`;
            if (event.step === "search")    detail = `${event.chunks} chunks · ${event.ms}ms · ${event.index_type}/${event.db}`;
            if (event.step === "retrieval") detail = event.sources?.join(", ");
            pipe.steps[event.step] = { status: "done", detail };
          }

          msgs[idx] = pipe;
          return msgs;
        });
      });
    } catch (e) {
      setError(e.message);
      setMessages(prev => prev.filter(m => m.id !== msgId));
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="shell">
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          </div>
          <span className="header-title">Simple RAG</span>
        </div>
        <div className="header-right">
          <IndexBadge info={indexInfo} />
          <span className="model-badge">llama-3.3-70b · Groq</span>
        </div>
      </header>

      <main className="chat">
        {isEmpty ? (
          <div className="empty">
            <div className="empty-headline">What do you want to know?</div>
            <p className="empty-sub">Answers are grounded in your documents — no guessing.</p>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => <Message key={msg.id ?? i} msg={msg} />)}
            {error && <div className="error-banner">{error}</div>}
          </>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="footer">
        <div className="input-wrap">
          <textarea
            ref={textareaRef}
            rows={2}
            placeholder="Ask anything…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading}
          />
          <button className="send-btn" onClick={() => send()} disabled={loading || !input.trim()} aria-label="Send">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
        <p className="footer-hint">Enter to send · Shift+Enter for new line</p>
      </footer>
    </div>
  );
}

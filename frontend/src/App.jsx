import { useState, useRef, useEffect } from "react";

// const EXAMPLES = [
//   "What is RAG and how does it work?",
//   "How do embeddings capture meaning?",
//   "What is FAISS and why is it fast?",
//   "How does SentenceTransformers work?",
// ];
const EXAMPLES = [
];
async function askQuestion(question) {
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

function SourceChunks({ chunks, sources = [] }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="sources-icon">{open ? "▾" : "▸"}</span>
        {open ? "Hide" : "Show"} {chunks.length} retrieved chunk{chunks.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div className="sources-list">
          {chunks.map((c, i) => (
            <div key={i} className="source-card">
              <span className="source-num">{i + 1}</span>
              <div className="source-content">
                {sources[i] && <span className="source-file">{sources[i]}</span>}
                <p>{c}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`msg-row ${isUser ? "msg-user" : "msg-assistant"}`}>
      {!isUser && (
        <div className="avatar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" /><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
          </svg>
        </div>
      )}
      <div className="msg-body">
        <div className="bubble">{msg.text}</div>
        {!isUser && msg.context?.length > 0 && <SourceChunks chunks={msg.context} sources={msg.sources} />}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="msg-row msg-assistant">
      <div className="avatar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" /><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
        </svg>
      </div>
      <div className="msg-body">
        <div className="bubble typing">
          <span /><span /><span />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const question = (text ?? input).trim();
    if (!question || loading) return;

    setInput("");
    setError("");
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setLoading(true);
    textareaRef.current?.focus();

    try {
      const data = await askQuestion(question);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, context: data.context, sources: data.sources ?? [] },
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="shell">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </div>
          <span className="header-title">Simple RAG</span>
        </div>
        <div className="header-right">
          <span className="model-badge">llama-3.3-70b · Groq</span>
        </div>
      </header>

      {/* Chat */}
      <main className="chat">
        {isEmpty ? (
          <div className="empty">
            <div className="empty-headline">What do you want to know?</div>
            <p className="empty-sub">Answers are grounded in your documents — no guessing.</p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} className="example-pill" onClick={() => send(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            {loading && <TypingIndicator />}
            {error && <div className="error-banner">{error}</div>}
          </>
        )}
        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="footer">
        <div className="input-wrap">
          <textarea
            ref={textareaRef}
            rows={2}
            placeholder="Ask anything…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={() => send()}
            disabled={loading || !input.trim()}
            aria-label="Send"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
        <p className="footer-hint">Enter to send · Shift+Enter for new line</p>
      </footer>
    </div>
  );
}

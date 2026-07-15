'use client';

import { useState, useEffect, useRef } from "react";

export default function Home() {
  // ─── Settings State ───
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000");
  const [apiKey, setApiKey] = useState("");
  const [settingsResult, setSettingsResult] = useState(null);

  // ─── Navigation State ───
  const [activePanel, setActivePanel] = useState("chat");

  // ─── API Health State ───
  const [apiStatus, setApiStatus] = useState("Connecting…");
  const [vectorsCount, setVectorsCount] = useState(0);
  const [redisConnected, setRedisConnected] = useState(false);
  const [postgresConnected, setPostgresConnected] = useState(false);
  const [openaiApiKeySet, setOpenaiApiKeySet] = useState(false);
  const [apiKeyAuthEnabled, setApiKeyAuthEnabled] = useState(false);

  // ─── Chat State ───
  const [messages, setMessages] = useState([]);
  const [queryInput, setQueryInput] = useState("");
  const [isQueryLoading, setIsQueryLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // ─── Ingestion State ───
  const [docId, setDocId] = useState("");
  const [docText, setDocText] = useState("");
  const [docSource, setDocSource] = useState("");
  const [ingestResult, setIngestResult] = useState(null);
  const [isIngesting, setIsIngesting] = useState(false);

  // ─── Stats State ───
  const [statsHours, setStatsHours] = useState(24);
  const [statsData, setStatsData] = useState(null);
  const [statsError, setStatsError] = useState(null);

  // ─── Live Health State ───
  const [circuitBreakers, setCircuitBreakers] = useState(null);
  const [healthError, setHealthError] = useState(null);

  // ─── Load Saved Settings ───
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedUrl = localStorage.getItem("neurorag_url");
      const savedKey = localStorage.getItem("neurorag_key");
      if (savedUrl) setBaseUrl(savedUrl);
      if (savedKey) setApiKey(savedKey);
    }
  }, []);

  // ─── Status Checking ───
  const checkStatus = async (currentUrl = baseUrl, currentKey = apiKey) => {
    try {
      const headers = { "Content-Type": "application/json" };
      if (currentKey) headers["X-API-Key"] = currentKey;

      const res = await fetch(currentUrl + "/health", { headers });
      if (!res.ok) throw new Error("HTTP Error");
      const h = await res.json();

      setApiStatus("API Online");
      setVectorsCount(h.faiss_vectors ?? 0);
      setRedisConnected(h.redis_connected);
      setPostgresConnected(h.postgres_connected);

      // Check env endpoint
      const envRes = await fetch(currentUrl + "/env-check", { headers });
      if (envRes.ok) {
        const env = await envRes.json();
        setOpenaiApiKeySet(env.OPENAI_API_KEY || env.GEMINI_API_KEY || false);
        setApiKeyAuthEnabled(env.NEURORAG_API_KEY || false);
      }
    } catch {
      setApiStatus("API Offline");
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(() => checkStatus(), 30000);
    return () => clearInterval(interval);
  }, [baseUrl, apiKey]);

  // ─── Auto Scroll Chat ───
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isQueryLoading]);

  // ─── API Wrapper ───
  const apiCall = async (path, opts = {}) => {
    const headers = { "Content-Type": "application/json" };
    if (apiKey) headers["X-API-Key"] = apiKey;
    const res = await fetch(baseUrl + path, { headers, ...opts });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errorText || "Unknown error"}`);
    }
    return res.json();
  };

  // ─── Chat Actions ───
  const sendQuery = async (queryText = queryInput) => {
    const cleanQuery = queryText.trim();
    if (!cleanQuery || isQueryLoading) return;

    setIsQueryLoading(true);
    setQueryInput("");

    // Append user message
    setMessages(prev => [...prev, { role: "user", content: cleanQuery }]);

    try {
      const data = await apiCall("/query", {
        method: "POST",
        body: JSON.stringify({ query: cleanQuery }),
      });
      setMessages(prev => [...prev, { role: "ai", content: data.answer, meta: data }]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: "ai",
          content: `❌ Error: ${err.message}\n\nPlease check if the API is running and the auth configuration matches in Settings.`,
        },
      ]);
    } finally {
      setIsQueryLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  const setQuery = (q) => {
    setQueryInput(q);
  };

  // ─── Ingestion Actions ───
  const ingestDocument = async () => {
    if (!docId.trim() || !docText.trim()) {
      setIngestResult({ status: "error", message: "Document ID and text are required." });
      return;
    }

    setIsIngesting(true);
    setIngestResult({ status: "info", message: "Ingesting document…" });

    try {
      const data = await apiCall("/ingest", {
        method: "POST",
        body: JSON.stringify({
          documents: [
            {
              id: docId.trim(),
              text: docText.trim(),
              metadata: { source: docSource.trim() || "nextjs-upload" },
            },
          ],
        }),
      });
      setIngestResult({
        status: "success",
        message: `✅ Ingested successfully. ${data.doc_count} document mapped to ${data.chunks_indexed} chunks indexed.`,
      });
      setDocId("");
      setDocText("");
      setDocSource("");
      checkStatus();
    } catch (err) {
      setIngestResult({ status: "error", message: `❌ Ingestion failed: ${err.message}` });
    } finally {
      setIsIngesting(false);
    }
  };

  const seedDefault = () => {
    setIngestResult({
      status: "info",
      message: "ℹ️ To seed the default dataset, run: python scripts/seed_data.py in the terminal.",
    });
  };

  // ─── Stats Actions ───
  const loadStats = async (hours = statsHours) => {
    setStatsHours(hours);
    setStatsData(null);
    setStatsError(null);

    try {
      const data = await apiCall(`/stats?hours=${hours}`);
      setStatsData(data);
    } catch (err) {
      setStatsError(err.message);
    }
  };

  // ─── Live Health Actions ───
  const loadHealth = async () => {
    setHealthError(null);
    setCircuitBreakers(null);
    checkStatus();

    try {
      const cbData = await apiCall("/circuit-breaker/status");
      setCircuitBreakers(cbData);
    } catch (err) {
      setHealthError(err.message);
    }
  };

  // ─── Settings Actions ───
  const saveSettings = () => {
    const cleanUrl = baseUrl.trim().replace(/\/$/, "");
    setBaseUrl(cleanUrl);
    localStorage.setItem("neurorag_url", cleanUrl);
    localStorage.setItem("neurorag_key", apiKey.trim());
    setSettingsResult({ status: "success", message: "✅ Settings saved successfully." });
    checkStatus(cleanUrl, apiKey.trim());
  };

  const testConnection = async () => {
    const cleanUrl = baseUrl.trim().replace(/\/$/, "");
    setSettingsResult({ status: "info", message: "Connecting…" });
    try {
      const headers = { "Content-Type": "application/json" };
      if (apiKey) headers["X-API-Key"] = apiKey;
      const res = await fetch(cleanUrl + "/health", { headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const h = await res.json();
      setSettingsResult({
        status: "success",
        message: `✅ Connected — API v${h.version}, ${h.faiss_vectors} vectors found.`,
      });
      checkStatus(cleanUrl, apiKey);
    } catch (err) {
      setSettingsResult({ status: "error", message: `❌ Connection failed: ${err.message}` });
    }
  };

  return (
    <div className="app">
      {/* ─── Top Bar ─── */}
      <header className="topbar">
        <span className="logo">⚡ NeuroRAG</span>
        <span className="tag">GEMINI 2.5 FLASH</span>
        <div className="status-bar">
          <span
            className={`dot ${
              apiStatus === "API Online"
                ? "green"
                : apiStatus === "Connecting…"
                ? "yellow"
                : "red"
            }`}
          ></span>
          <span
            style={{
              color:
                apiStatus === "API Online"
                  ? "var(--green)"
                  : apiStatus === "Connecting…"
                  ? "var(--yellow)"
                  : "var(--red)",
              fontSize: "12px",
            }}
          >
            {apiStatus}
          </span>
          <span style={{ color: "var(--border)" }}>|</span>
          <span style={{ color: "var(--muted)", fontSize: "12px" }}>
            {vectorsCount} vectors
          </span>
        </div>
      </header>

      {/* ─── Sidebar Navigation ─── */}
      <nav className="sidebar">
        <div className="sidebar-section">Navigation</div>
        <div
          className={`nav-item ${activePanel === "chat" ? "active" : ""}`}
          onClick={() => setActivePanel("chat")}
        >
          <span className="nav-icon">💬</span> Chat
        </div>
        <div
          className={`nav-item ${activePanel === "ingest" ? "active" : ""}`}
          onClick={() => setActivePanel("ingest")}
        >
          <span className="nav-icon">📥</span> Ingest Documents
        </div>
        <div
          className={`nav-item ${activePanel === "stats" ? "active" : ""}`}
          onClick={() => {
            setActivePanel("stats");
            loadStats();
          }}
        >
          <span className="nav-icon">📊</span> Query Stats
        </div>
        <div
          className={`nav-item ${activePanel === "health" ? "active" : ""}`}
          onClick={() => {
            setActivePanel("health");
            loadHealth();
          }}
        >
          <span className="nav-icon">🔍</span> System Health
        </div>
        <div
          className={`nav-item ${activePanel === "settings" ? "active" : ""}`}
          onClick={() => {
            setActivePanel("settings");
            setSettingsResult(null);
          }}
        >
          <span className="nav-icon">⚙️</span> Settings
        </div>
      </nav>

      {/* ─── Main Contents Panel ─── */}
      <main className="main">
        {/* ─── Chat Panel ─── */}
        <div className={`panel ${activePanel === "chat" ? "active" : ""}`}>
          <div className="chat-header">
            <h2>Gemini Chat Interface</h2>
            <p>
              Ask queries — the self-healing RAG pipeline retrieves, generates,
              and critiques responses automatically.
            </p>
          </div>

          <div className="messages">
            {messages.length === 0 ? (
              <div className="welcome">
                <h3>👋 Welcome to NeuroRAG</h3>
                <p>
                  Ask anything about RAG systems, LLMs, retrieval, embeddings,
                  or system architecture. The multi-agent pipeline powered by
                  Gemini 2.5 Flash will resolve queries.
                </p>
                <div className="chips">
                  <div
                    className="chip"
                    onClick={() =>
                      setQuery("What is retrieval-augmented generation and how does it work?")
                    }
                  >
                    What is RAG?
                  </div>
                  <div
                    className="chip"
                    onClick={() =>
                      setQuery("How does hybrid retrieval combine BM25 and vector search?")
                    }
                  >
                    Hybrid retrieval
                  </div>
                  <div
                    className="chip"
                    onClick={() =>
                      setQuery("How does the self-healing loop improve answer quality?")
                    }
                  >
                    Self-healing loop
                  </div>
                  <div
                    className="chip"
                    onClick={() =>
                      setQuery("What is the circuit breaker pattern and when does it trigger?")
                    }
                  >
                    Circuit breaker
                  </div>
                </div>
              </div>
            ) : (
              messages.map((m, idx) => (
                <div key={idx} className={`msg ${m.role}`}>
                  <div className={`avatar ${m.role === "user" ? "user" : "ai"}`}>
                    {m.role === "user" ? "👤" : "🤖"}
                  </div>
                  <div className="bubble">
                    {m.content}
                    {m.meta && (
                      <div className="meta">
                        <span
                          className={`badge ${
                            m.meta.confidence >= 0.8
                              ? "conf"
                              : m.meta.confidence >= 0.5
                              ? "loops"
                              : "error"
                          }`}
                        >
                          ⚡ Conf: {(m.meta.confidence * 100).toFixed(0)}%
                        </span>
                        <span className="badge loops">🔄 Loops: {m.meta.loops}</span>
                        <span className="badge latency">⏱ {m.meta.latency_ms}ms</span>
                        {m.meta.from_memory_cache && (
                          <span className="badge cache">📦 Cached</span>
                        )}
                        {m.meta.insufficient_context && (
                          <span className="badge insuff">⚠️ Insufficient Context</span>
                        )}
                      </div>
                    )}
                    {m.meta?.citations && m.meta.citations.length > 0 && (
                      <div className="citations">
                        📎 Sources:{" "}
                        {m.meta.citations.map((c, i) => (
                          <span key={i}>{c}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isQueryLoading && (
              <div className="msg">
                <div className="avatar ai">🤖</div>
                <div className="bubble">
                  <div className="thinking">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <div className="input-row">
              <textarea
                className="query-input"
                placeholder="Ask a question…"
                rows={1}
                value={queryInput}
                onChange={e => setQueryInput(e.target.value)}
                onKeyDown={handleKey}
              ></textarea>
              <button
                className="send-btn"
                disabled={isQueryLoading || !queryInput.trim()}
                onClick={() => sendQuery()}
              >
                <span>Send</span> <span>↗</span>
              </button>
            </div>
            <div className="quick-prompts">
              <span
                className="qp"
                onClick={() =>
                  setQuery("What is cross-encoder reranking and why is it better than bi-encoders?")
                }
              >
                Reranking
              </span>
              <span
                className="qp"
                onClick={() => setQuery("Explain FAISS indexing and how NeuroRAG uses it.")}
              >
                FAISS indexing
              </span>
              <span
                className="qp"
                onClick={() => setQuery("How does Redis caching improve RAG system performance?")}
              >
                Redis caching
              </span>
              <span
                className="qp"
                onClick={() => setQuery("What embedding model does NeuroRAG use and why?")}
              >
                Embeddings
              </span>
            </div>
          </div>
        </div>

        {/* ─── Ingest Panel ─── */}
        <div className={`panel ${activePanel === "ingest" ? "active" : ""}`}>
          <div className="panel-inner">
            <div>
              <div className="panel-title">📥 Ingest Documents</div>
              <div className="panel-subtitle">
                Add documents to the knowledge base. They will be chunked, embedded,
                and indexed automatically.
              </div>
            </div>

            <div className="ingest-form">
              <div className="form-row">
                <label className="form-label">Document ID</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. my-document-001"
                  value={docId}
                  onChange={e => setDocId(e.target.value)}
                />
              </div>
              <div className="form-row">
                <label className="form-label">Document Text</label>
                <textarea
                  className="form-input"
                  placeholder="Paste the full text of your document here…"
                  value={docText}
                  onChange={e => setDocText(e.target.value)}
                ></textarea>
              </div>
              <div className="form-row">
                <label className="form-label">Source (optional)</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. internal-wiki, pdf-upload"
                  value={docSource}
                  onChange={e => setDocSource(e.target.value)}
                />
              </div>
              <div style={{ display: "flex", gap: "10px" }}>
                <button
                  className="btn btn-primary"
                  disabled={isIngesting || !docId || !docText}
                  onClick={ingestDocument}
                >
                  Ingest Document
                </button>
                <button className="btn btn-secondary" onClick={seedDefault}>
                  Seed Default Data
                </button>
              </div>
              {ingestResult && (
                <div className={`alert ${ingestResult.status}`}>
                  {ingestResult.message}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ─── Stats Panel ─── */}
        <div className={`panel ${activePanel === "stats" ? "active" : ""}`}>
          <div className="panel-inner">
            <div>
              <div className="panel-title">📊 Query Statistics</div>
              <div className="panel-subtitle">Last {statsHours} hours</div>
            </div>
            {statsError ? (
              <div className="alert error">
                Failed to load stats: {statsError}
                <br />
                <small>Stats require PostgreSQL database connection.</small>
              </div>
            ) : statsData ? (
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-label">Total Queries</div>
                  <div className="stat-value" style={{ color: "var(--accent)" }}>
                    {statsData.total_queries ?? 0}
                  </div>
                  <div className="stat-sub">last {statsHours}h</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Avg Confidence</div>
                  <div className="stat-value" style={{ color: "var(--green)" }}>
                    {((statsData.avg_confidence || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="stat-sub">critic score</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">p95 Latency</div>
                  <div className="stat-value" style={{ color: "var(--yellow)" }}>
                    {parseFloat(statsData.p95_latency_ms || 0).toFixed(0)}ms
                  </div>
                  <div className="stat-sub">SLO target ≤ 1500ms</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Avg Loops</div>
                  <div className="stat-value" style={{ color: "var(--purple)" }}>
                    {parseFloat(statsData.avg_loops || 0).toFixed(2)}
                  </div>
                  <div className="stat-sub">self-healing iterations</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Hallucination Rate</div>
                  <div
                    className="stat-value"
                    style={{
                      color:
                        (statsData.hallucination_rate || 0) > 0.1
                          ? "var(--red)"
                          : "var(--green)",
                    }}
                  >
                    {((statsData.hallucination_rate || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="stat-sub">target &lt; 10%</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Cache Hit Rate</div>
                  <div className="stat-value" style={{ color: "var(--accent)" }}>
                    {((statsData.cache_hit_rate || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="stat-sub">Upstash Redis</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Avg Faithfulness</div>
                  <div className="stat-value" style={{ color: "var(--green)" }}>
                    {((statsData.avg_faithfulness || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="stat-sub">target ≥ 85%</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Insufficient Rate</div>
                  <div className="stat-value" style={{ color: "var(--muted)" }}>
                    {((statsData.insufficient_rate || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="stat-sub">no-context responses</div>
                </div>
              </div>
            ) : (
              <div className="stat-card">
                <div className="stat-label">Loading stats…</div>
              </div>
            )}
            <div style={{ display: "flex", gap: "8px" }}>
              <button className="btn btn-secondary" onClick={() => loadStats(1)}>
                1h
              </button>
              <button className="btn btn-secondary" onClick={() => loadStats(6)}>
                6h
              </button>
              <button className="btn btn-secondary" onClick={() => loadStats(24)}>
                24h
              </button>
              <button className="btn btn-secondary" onClick={() => loadStats(168)}>
                7d
              </button>
            </div>
          </div>
        </div>

        {/* ─── Health Panel ─── */}
        <div className={`panel ${activePanel === "health" ? "active" : ""}`}>
          <div className="panel-inner">
            <div>
              <div className="panel-title">🔍 System Health</div>
              <div className="panel-subtitle">
                Live status of all NeuroRAG cloud components
              </div>
            </div>
            <div className="health-grid">
              <div className="health-card">
                <span className="health-icon">🚀</span>
                <div>
                  <div className="health-label">FastAPI on Render</div>
                  <div
                    className={`health-status ${
                      apiStatus === "API Online" ? "ok" : "down"
                    }`}
                  >
                    {apiStatus === "API Online" ? "Online ✓" : "Offline ❌"}
                  </div>
                </div>
              </div>
              <div className="health-card">
                <span className="health-icon">🧠</span>
                <div>
                  <div className="health-label">FAISS Vector DB</div>
                  <div className={`health-status ${vectorsCount > 0 ? "ok" : "warn"}`}>
                    {vectorsCount} vectors
                  </div>
                </div>
              </div>
              <div className="health-card">
                <span className="health-icon">📦</span>
                <div>
                  <div className="health-label">Upstash Redis</div>
                  <div className={`health-status ${redisConnected ? "ok" : "warn"}`}>
                    {redisConnected ? "Connected ✓" : "Offline — caching disabled"}
                  </div>
                </div>
              </div>
              <div className="health-card">
                <span className="health-icon">🗄️</span>
                <div>
                  <div className="health-label">Supabase Postgres</div>
                  <div className={`health-status ${postgresConnected ? "ok" : "warn"}`}>
                    {postgresConnected ? "Connected ✓" : "Offline — logging disabled"}
                  </div>
                </div>
              </div>
              <div className="health-card">
                <span className="health-icon">🔑</span>
                <div>
                  <div className="health-label">Gemini API Key</div>
                  <div className={`health-status ${openaiApiKeySet ? "ok" : "down"}`}>
                    {openaiApiKeySet ? "Configured ✓" : "Missing — queries will fail"}
                  </div>
                </div>
              </div>
              <div className="health-card">
                <span className="health-icon">🛡️</span>
                <div>
                  <div className="health-label">API Key Auth</div>
                  <div className={`health-status ${apiKeyAuthEnabled ? "ok" : "warn"}`}>
                    {apiKeyAuthEnabled ? "Enabled ✓" : "Bypassed (dev mode)"}
                  </div>
                </div>
              </div>
            </div>
            <div>
              <button className="btn btn-secondary" onClick={loadHealth}>
                ↻ Refresh Status
              </button>
            </div>
            <div>
              <div className="panel-title" style={{ fontSize: "14px", marginBottom: "10px" }}>
                LLM Circuit Breaker Status
              </div>
              {healthError ? (
                <div className="alert error">Circuit status unavailable: {healthError}</div>
              ) : circuitBreakers ? (
                <div className="alert info">
                  Generator Client: <strong>{circuitBreakers.generator}</strong> &nbsp;|&nbsp;
                  Critic Client: <strong>{circuitBreakers.critic}</strong>
                </div>
              ) : (
                <div className="alert info">Loading breaker states…</div>
              )}
            </div>
          </div>
        </div>

        {/* ─── Settings Panel ─── */}
        <div className={`panel ${activePanel === "settings" ? "active" : ""}`}>
          <div className="panel-inner">
            <div>
              <div className="panel-title">⚙️ Settings</div>
              <div className="panel-subtitle">
                Configure API connection and authentication
              </div>
            </div>
            <div className="settings-form">
              <div className="settings-note">
                💡 Settings are saved to your local browser storage. Your credentials are only
                used locally to authenticate endpoint requests.
              </div>
              <div className="form-row">
                <label className="form-label">FastAPI Base URL</label>
                <input
                  type="text"
                  className="form-input"
                  value={baseUrl}
                  onChange={e => setBaseUrl(e.target.value)}
                  placeholder="http://localhost:8000"
                />
              </div>
              <div className="form-row">
                <label className="form-label">API Key (X-API-Key)</label>
                <input
                  type="password"
                  className="form-input"
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  placeholder="Your NEURORAG_API_KEY value"
                />
              </div>
              <div style={{ display: "flex", gap: "10px" }}>
                <button className="btn btn-primary" onClick={saveSettings}>
                  Save Settings
                </button>
                <button className="btn btn-secondary" onClick={testConnection}>
                  Test Connection
                </button>
              </div>
              {settingsResult && (
                <div className={`alert ${settingsResult.status}`}>
                  {settingsResult.message}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

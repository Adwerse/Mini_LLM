import { useEffect, useMemo, useState } from "react";
import { API_BASE_URL, generateText, getHealth, getModelInfo } from "./api.js";

const DEFAULT_PROMPT = "Once upon a time";

function formatValue(value, fallback = "unknown") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function StatusPill({ ok, label }) {
  return <span className={`status-pill ${ok ? "ok" : "warn"}`}>{label}</span>;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [startupError, setStartupError] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxNewTokens, setMaxNewTokens] = useState(50);
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.9);
  const [generatedText, setGeneratedText] = useState("");
  const [generationMeta, setGenerationMeta] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadBackendState() {
      try {
        const [healthResponse, infoResponse] = await Promise.all([
          getHealth(),
          getModelInfo(),
        ]);
        if (!cancelled) {
          setHealth(healthResponse);
          setModelInfo(infoResponse);
          setStartupError("");
        }
      } catch (requestError) {
        if (!cancelled) {
          setStartupError(requestError.message);
        }
      }
    }

    loadBackendState();
    return () => {
      cancelled = true;
    };
  }, []);

  const backendOnline = health?.status === "ok";
  const checkpointReady = Boolean(modelInfo?.checkpoint_loaded);
  const canGenerate = useMemo(() => {
    return prompt.trim().length > 0 && !isLoading && backendOnline;
  }, [backendOnline, isLoading, prompt]);

  async function handleGenerate(event) {
    event.preventDefault();
    setError("");
    setGeneratedText("");
    setGenerationMeta(null);

    if (!prompt.trim()) {
      setError("Prompt must not be empty.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await generateText({
        prompt,
        maxNewTokens,
        temperature,
        topP,
      });
      setGeneratedText(response.generated_text);
      setGenerationMeta(response.model_info);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="header-band">
        <div>
          <p className="eyebrow">Educational Mini LLM</p>
          <h1>SimpleLLM Inference Console</h1>
          <p className="lead">
            A small PyTorch transformer served through a FastAPI backend.
          </p>
        </div>
        <div className="api-target">
          <span>API</span>
          <code>{API_BASE_URL}</code>
        </div>
      </section>

      <section className="status-grid" aria-label="Backend and model status">
        <div className="metric-panel">
          <span className="metric-label">Backend</span>
          <StatusPill ok={backendOnline} label={backendOnline ? "online" : "offline"} />
          {startupError && <p className="panel-note error-text">{startupError}</p>}
        </div>
        <div className="metric-panel">
          <span className="metric-label">Checkpoint</span>
          <StatusPill ok={checkpointReady} label={checkpointReady ? "loaded" : "not loaded"} />
        </div>
        <div className="metric-panel">
          <span className="metric-label">Device</span>
          <strong>{formatValue(modelInfo?.device)}</strong>
        </div>
        <div className="metric-panel">
          <span className="metric-label">Parameters</span>
          <strong>{formatValue(modelInfo?.parameters)}</strong>
        </div>
      </section>

      <section className="workspace">
        <form className="control-surface" onSubmit={handleGenerate}>
          <div className="section-title">
            <h2>Prompt</h2>
            <span>{formatValue(modelInfo?.tokenizer, "tokenizer pending")}</span>
          </div>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Enter a prompt..."
            rows={9}
          />

          <div className="controls-grid">
            <label className="field">
              <span>Max tokens</span>
              <input
                type="number"
                min="1"
                max="256"
                value={maxNewTokens}
                onChange={(event) => setMaxNewTokens(event.target.value)}
              />
            </label>

            <label className="field slider-field">
              <span>Temperature</span>
              <div className="slider-row">
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                />
                <input
                  type="number"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                />
              </div>
            </label>

            <label className="field slider-field">
              <span>Top-p</span>
              <div className="slider-row">
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(event) => setTopP(event.target.value)}
                />
                <input
                  type="number"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(event) => setTopP(event.target.value)}
                />
              </div>
            </label>
          </div>

          <button className="generate-button" type="submit" disabled={!canGenerate}>
            {isLoading ? "Generating..." : "Generate"}
          </button>
        </form>

        <section className="output-surface" aria-live="polite">
          <div className="section-title">
            <h2>Generated Output</h2>
            <span>max context {formatValue(modelInfo?.max_seq_len, "-")}</span>
          </div>

          {error && <div className="message error-message">{error}</div>}

          <div className={`output-box ${generatedText ? "has-output" : ""}`}>
            {isLoading && <span className="muted">Waiting for model response...</span>}
            {!isLoading && generatedText && <pre>{generatedText}</pre>}
            {!isLoading && !generatedText && !error && (
              <span className="muted">Output will appear here after generation.</span>
            )}
          </div>

          {generationMeta && (
            <div className="run-meta">
              <span>device: {formatValue(generationMeta.device)}</span>
              <span>checkpoint: {formatValue(generationMeta.checkpoint, "none")}</span>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

type Health = { status: string; version: string };
type Meta = {
  app_version: string;
  schema_version: number;
  counts: Record<string, number>;
  data_dir: string;
};

type Status = "connecting" | "ok" | "error";

function App() {
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const [meta, setMeta] = useState<Meta | null>(null);

  // Learn the sidecar base URL: poll the command (in case it's already ready)
  // and also listen for the readiness handshake event from the Rust shell.
  useEffect(() => {
    let cancelled = false;
    invoke<string | null>("sidecar_url").then((url) => {
      if (!cancelled && url) setBaseUrl(url);
    });
    const unlisten = listen<string>("sidecar-ready", (event) => {
      if (!cancelled) setBaseUrl(event.payload);
    });
    return () => {
      cancelled = true;
      unlisten.then((fn) => fn());
    };
  }, []);

  // Once we have a base URL, probe /health and load /meta.
  useEffect(() => {
    if (!baseUrl) return;
    let cancelled = false;
    (async () => {
      try {
        const health: Health = await (await fetch(`${baseUrl}/health`)).json();
        if (cancelled) return;
        setStatus(health.status === "ok" ? "ok" : "error");
        const m: Meta = await (await fetch(`${baseUrl}/meta`)).json();
        if (!cancelled) setMeta(m);
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl]);

  const dot = status === "ok" ? "#22c55e" : status === "error" ? "#ef4444" : "#eab308";
  const label =
    status === "ok" ? "Sidecar connected" : status === "error" ? "Sidecar unreachable" : "Connecting…";

  return (
    <main className="container">
      <h1>Iris</h1>
      <div className="row" style={{ alignItems: "center", gap: "0.5rem" }}>
        <span
          style={{ width: 12, height: 12, borderRadius: "50%", background: dot, display: "inline-block" }}
        />
        <span>{label}</span>
      </div>
      <p>{baseUrl ? <code>{baseUrl}</code> : "waiting for sidecar handshake…"}</p>
      {meta && (
        <ul style={{ textAlign: "left" }}>
          <li>app version: {meta.app_version}</li>
          <li>schema version: {meta.schema_version}</li>
          <li>photos indexed: {meta.counts.photos ?? 0}</li>
          <li>data dir: <code>{meta.data_dir}</code></li>
        </ul>
      )}
    </main>
  );
}

export default App;

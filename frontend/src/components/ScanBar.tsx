import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type { Api, IngestStatus } from "../lib/api";

const TERMINAL = new Set(["done", "error", "canceled"]);

export function ScanBar({ api, onComplete }: { api: Api; onComplete: () => void }) {
  const [path, setPath] = useState("");
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };
  useEffect(() => stopPolling, []);

  const startPolling = () => {
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      try {
        const s = await api.status();
        setStatus(s);
        if (!s.running && s.job && TERMINAL.has(s.job.state)) {
          stopPolling();
          setBusy(false);
          onComplete(); // refresh the grid once the run finishes (avoids scroll resets)
        }
      } catch {
        // keep polling
      }
    }, 500);
  };

  const runScan = async (rawTarget: string) => {
    const target = rawTarget.trim();
    if (!target) return;
    setError(null);
    setBusy(true);
    try {
      await api.addRoot(target);
      await api.scan();
      startPolling();
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const chooseFolder = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Choose a photo folder",
      });
      if (typeof selected === "string") {
        setPath(selected);
        await runScan(selected);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const job = status?.job ?? null;
  const pct = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="scanbar">
      <button className="scan-btn" disabled={busy} onClick={() => void chooseFolder()}>
        Choose folder…
      </button>
      <input
        className="scan-input"
        placeholder="…or type an absolute path (~ supported)"
        value={path}
        disabled={busy}
        onChange={(e) => setPath(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") void runScan(path);
        }}
      />
      <button className="scan-btn" disabled={busy || !path.trim()} onClick={() => void runScan(path)}>
        {busy ? "Scanning…" : "Add & Scan"}
      </button>
      {job && (
        <div className="scan-progress">
          <div className="scan-bar-track">
            <div className="scan-bar-fill" style={{ width: `${pct}%` }} />
          </div>
          <span className="scan-progress-label">
            {job.state === "running" ? `${job.done}/${job.total}` : job.state}
            {job.errored > 0 ? ` · ${job.errored} skipped` : ""}
          </span>
        </div>
      )}
      {error && <span className="scan-error">{error}</span>}
    </div>
  );
}

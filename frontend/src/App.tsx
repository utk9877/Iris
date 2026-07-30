import { useEffect, useMemo, useState } from "react";
import { useSidecar } from "./hooks/useSidecar";
import { makeApi } from "./lib/api";
import { PhotoGrid } from "./components/PhotoGrid";
import { ScanBar } from "./components/ScanBar";
import "./App.css";
import "./styles.css";

function App() {
  const { baseUrl, connected } = useSidecar();
  const api = useMemo(() => (baseUrl ? makeApi(baseUrl) : null), [baseUrl]);
  const [reloadKey, setReloadKey] = useState(0);
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    if (!api) return;
    let cancelled = false;
    api
      .count()
      .then((c) => {
        if (!cancelled) setCount(c.count);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [api, reloadKey]);

  if (!api) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>Iris</h1>
          <span className="status">connecting…</span>
        </header>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Iris</h1>
        <span className={`dot ${connected ? "ok" : "bad"}`} />
        <span className="status">{connected ? "connected" : "unreachable"}</span>
        <span className="count">{count ?? "—"} photos</span>
      </header>
      <ScanBar api={api} onComplete={() => setReloadKey((k) => k + 1)} />
      <PhotoGrid api={api} reloadKey={reloadKey} />
    </div>
  );
}

export default App;

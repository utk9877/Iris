import { useEffect, useMemo, useState } from "react";
import { useSidecar } from "./hooks/useSidecar";
import { makeApi } from "./lib/api";
import { PhotoGrid } from "./components/PhotoGrid";
import { ScanBar } from "./components/ScanBar";
import { SearchBar, type SearchState } from "./components/SearchBar";
import "./App.css";
import "./styles.css";

function App() {
  const { baseUrl, connected } = useSidecar();
  const api = useMemo(() => (baseUrl ? makeApi(baseUrl) : null), [baseUrl]);
  const [reloadKey, setReloadKey] = useState(0);
  const [count, setCount] = useState<number | null>(null);
  const [search, setSearch] = useState<SearchState>(null);

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
      <SearchBar api={api} onResults={setSearch} />
      {search && (
        <div className="search-meta">
          {search.items.length} result{search.items.length === 1 ? "" : "s"} · tier: {search.tier}
        </div>
      )}
      <PhotoGrid api={api} reloadKey={reloadKey} searchItems={search ? search.items : null} />
    </div>
  );
}

export default App;

import { useState } from "react";
import type { Api, SearchItem } from "../lib/api";

export type SearchState = { items: SearchItem[]; tier: string } | null;

export function SearchBar({
  api,
  onResults,
}: {
  api: Api;
  onResults: (state: SearchState) => void;
}) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    const q = query.trim();
    if (!q) {
      onResults(null);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await api.search(q);
      onResults({ items: r.items, tier: r.tier });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      onResults({ items: [], tier: "error" });
    } finally {
      setBusy(false);
    }
  };

  const clear = () => {
    setQuery("");
    setError(null);
    onResults(null);
  };

  return (
    <div className="searchbar">
      <input
        className="search-input"
        placeholder="Search your photos… e.g. “a dog on a beach”"
        value={query}
        disabled={busy}
        onChange={(e) => setQuery(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") void submit();
        }}
      />
      <button className="scan-btn" disabled={busy || !query.trim()} onClick={() => void submit()}>
        {busy ? "Searching…" : "Search"}
      </button>
      {query && (
        <button className="scan-btn" onClick={clear}>
          Clear
        </button>
      )}
      {error && <span className="scan-error">{error}</span>}
    </div>
  );
}

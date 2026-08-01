import { useState } from "react";
import type { Api, AppliedFilters, SearchFilters, SearchItem } from "../lib/api";

export type SearchState = {
  items: SearchItem[];
  tier: string;
  applied?: AppliedFilters | null;
  placeError?: string | null;
} | null;

const toEpoch = (d: string, endOfDay: boolean): number | null =>
  d ? new Date(`${d}T${endOfDay ? "23:59:59" : "00:00:00"}`).getTime() / 1000 : null;

const isoDaysAgo = (days: number): string =>
  new Date(Date.now() - days * 86400_000).toISOString().slice(0, 10);
const isoJan1 = (year: number): string => `${year}-01-01`;

export function SearchBar({
  api,
  onResults,
}: {
  api: Api;
  onResults: (state: SearchState) => void;
}) {
  const [query, setQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [place, setPlace] = useState("");
  const [radius, setRadius] = useState("25");
  const [hasText, setHasText] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeFilters = (): SearchFilters | undefined => {
    const f: SearchFilters = {};
    if (from) f.date_from = toEpoch(from, false);
    if (to) f.date_to = toEpoch(to, true);
    if (place.trim()) {
      f.place = place.trim();
      const r = Number(radius);
      if (Number.isFinite(r) && r > 0) f.radius_km = r;
    }
    if (hasText) f.has_text = true;
    return Object.keys(f).length ? f : undefined;
  };

  const submit = async () => {
    const q = query.trim();
    const filters = activeFilters();
    if (!q && !filters) {
      onResults(null);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await api.search(q, filters);
      onResults({ items: r.items, tier: r.tier, applied: r.applied, placeError: r.place_error });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      onResults({ items: [], tier: "error" });
    } finally {
      setBusy(false);
    }
  };

  const clear = () => {
    setQuery("");
    setFrom("");
    setTo("");
    setPlace("");
    setHasText(false);
    setError(null);
    onResults(null);
  };

  const quickYear = (year: number) => {
    setFrom(isoJan1(year));
    setTo(`${year}-12-31`);
    setShowFilters(true);
  };
  const quickRecent = (days: number) => {
    setFrom(isoDaysAgo(days));
    setTo(isoDaysAgo(0));
    setShowFilters(true);
  };

  const now = new Date().getFullYear();

  return (
    <div className="searchbar-wrap">
      <div className="searchbar">
        <input
          className="search-input"
          placeholder="Search… e.g. “a dog on a beach”, “boarding pass 2024”, “last summer”"
          value={query}
          disabled={busy}
          onChange={(e) => setQuery(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
          }}
        />
        <button
          className={`scan-btn${showFilters ? " on" : ""}`}
          onClick={() => setShowFilters((s) => !s)}
          title="Date & location filters"
        >
          Filters
        </button>
        <button className="scan-btn" disabled={busy} onClick={() => void submit()}>
          {busy ? "Searching…" : "Search"}
        </button>
        {(query || from || to || place) && (
          <button className="scan-btn" onClick={clear}>
            Clear
          </button>
        )}
      </div>

      {showFilters && (
        <div className="search-filters">
          <label>
            From <input type="date" value={from} onChange={(e) => setFrom(e.currentTarget.value)} />
          </label>
          <label>
            To <input type="date" value={to} onChange={(e) => setTo(e.currentTarget.value)} />
          </label>
          <div className="chips">
            <button onClick={() => quickYear(now)}>{now}</button>
            <button onClick={() => quickYear(now - 1)}>{now - 1}</button>
            <button onClick={() => quickRecent(30)}>Last 30 days</button>
          </div>
          <label className="loc">
            Location
            <input
              type="text"
              placeholder="e.g. Paris"
              value={place}
              onChange={(e) => setPlace(e.currentTarget.value)}
            />
          </label>
          <label>
            within
            <input
              type="number"
              className="radius"
              min={1}
              value={radius}
              onChange={(e) => setRadius(e.currentTarget.value)}
            />
            km
          </label>
          <label className="cbx">
            <input
              type="checkbox"
              checked={hasText}
              onChange={(e) => setHasText(e.currentTarget.checked)}
            />
            has text
          </label>
        </div>
      )}
      {error && <span className="scan-error">{error}</span>}
    </div>
  );
}

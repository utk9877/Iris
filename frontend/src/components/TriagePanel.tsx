import { useCallback, useEffect, useState } from "react";
import type { Api, Group, TriageItem, TriagePreset, TriageResponse } from "../lib/api";

function fmtSpan(start: number | null, end: number | null): string {
  if (start == null) return "";
  const d = (t: number) => new Date(t * 1000).toLocaleDateString();
  const s = d(start);
  const e = end != null ? d(end) : s;
  return s === e ? s : `${s} – ${e}`;
}

const SCORE_KEYS: { key: keyof TriageItem; label: string }[] = [
  { key: "quality", label: "Q" },
  { key: "aesthetic", label: "A" },
  { key: "representativeness", label: "R" },
  { key: "subject", label: "S" },
];

/** Trip triage: pick an event, switch presets, review the ranked shortlist (ARCHITECTURE §7). */
export function TriagePanel({ api }: { api: Api }) {
  const [events, setEvents] = useState<Group[]>([]);
  const [presets, setPresets] = useState<TriagePreset[]>([]);
  const [eventId, setEventId] = useState<number | null>(null);
  const [preset, setPreset] = useState<string>("story-ready");
  const [result, setResult] = useState<TriageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [verdict, setVerdict] = useState<Record<number, "keep" | "reject">>({});

  useEffect(() => {
    void (async () => {
      const [evs, ps] = await Promise.all([api.groups("event"), api.triagePresets()]);
      setEvents(evs);
      setPresets(ps);
      if (evs.length > 0) setEventId((cur) => cur ?? evs[0].id);
    })();
  }, [api]);

  const runTriage = useCallback(async () => {
    if (eventId == null) return;
    setLoading(true);
    setVerdict({});
    try {
      setResult(await api.triage({ group_id: eventId, preset }));
    } finally {
      setLoading(false);
    }
  }, [api, eventId, preset]);

  useEffect(() => {
    void runTriage();
  }, [runTriage]);

  const toggleVerdict = useCallback((id: number, val: "keep" | "reject") => {
    setVerdict((m) => {
      const next = { ...m };
      if (next[id] === val) delete next[id];
      else next[id] = val;
      return next;
    });
  }, []);

  const activePreset = presets.find((p) => p.name === preset);
  const invert = activePreset?.invert ?? false;

  return (
    <div className="people-panel">
      <div className="people-bar triage-bar">
        <label>
          Event{" "}
          <select
            className="scan-input triage-select"
            value={eventId ?? ""}
            onChange={(e) => setEventId(Number(e.target.value))}
          >
            {events.length === 0 && <option value="">no events yet</option>}
            {events.map((ev) => (
              <option key={ev.id} value={ev.id}>
                {(ev.key ?? (fmtSpan(ev.start_at, ev.end_at) || `event #${ev.id}`)) +
                  ` · ${ev.size} photos`}
              </option>
            ))}
          </select>
        </label>
        <nav className="view-tabs">
          {presets.map((p) => (
            <button
              key={p.name}
              className={preset === p.name ? "on" : ""}
              title={p.description}
              onClick={() => setPreset(p.name)}
            >
              {p.name}
            </button>
          ))}
        </nav>
        <span className="status">
          {loading
            ? "ranking…"
            : result
              ? `${result.items.length} of ${result.scope_size} · ${result.considered} considered`
              : ""}
        </span>
      </div>

      {activePreset && <p className="triage-hint">{activePreset.description}</p>}

      {!loading && result && result.items.length === 0 ? (
        <p className="grid-empty">Nothing to show for this event.</p>
      ) : (
        <div className="people-grid triage-grid">
          {result?.items.map((it, rank) => {
            const v = verdict[it.id];
            return (
              <div
                key={it.id}
                className={`person-card triage-card${v === "reject" ? " rejected" : ""}${
                  v === "keep" ? " kept" : ""
                }`}
              >
                <div className="person-cover">
                  {it.has_thumb ? (
                    <img src={api.thumbUrl(it.id)} alt={it.filename} loading="lazy" />
                  ) : (
                    <span className="cell-fallback">{it.filename}</span>
                  )}
                  <span className="group-badge">#{rank + 1}</span>
                  {invert && it.redundant && <span className="triage-flag">dup</span>}
                </div>
                <div className="triage-scores">
                  {SCORE_KEYS.map(({ key, label }) => (
                    <span key={label} className="triage-bar" title={`${label} ${it[key]}`}>
                      <span className="triage-bar-label">{label}</span>
                      <span className="triage-bar-track">
                        <span
                          className="triage-bar-fill"
                          style={{ width: `${Math.round((it[key] as number) * 100)}%` }}
                        />
                      </span>
                    </span>
                  ))}
                </div>
                <div className="triage-reason">{it.reason}</div>
                <div className="triage-actions">
                  <button
                    className={`link-btn${v === "keep" ? " on" : ""}`}
                    onClick={() => toggleVerdict(it.id, "keep")}
                  >
                    ✓ keep
                  </button>
                  <button
                    className={`link-btn${v === "reject" ? " on" : ""}`}
                    onClick={() => toggleVerdict(it.id, "reject")}
                  >
                    ✕ reject
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

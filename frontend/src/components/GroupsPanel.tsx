import { useCallback, useEffect, useState } from "react";
import type { Api, Group, GroupDetail, GroupKind } from "../lib/api";

const KINDS: { key: GroupKind; label: string; hint: string }[] = [
  { key: "event", label: "Events", hint: "Trips & days, split on time/location gaps" },
  { key: "burst", label: "Bursts", hint: "Rapid same-camera sequences" },
  { key: "near_dup", label: "Near-duplicates", hint: "Near-identical frames" },
  { key: "semantic", label: "Themes", hint: "Visually similar photos" },
];

function fmtSpan(start: number | null, end: number | null): string {
  if (start == null) return "";
  const d = (t: number) => new Date(t * 1000).toLocaleDateString();
  const s = d(start);
  const e = end != null ? d(end) : s;
  return s === e ? s : `${s} – ${e}`;
}

export function GroupsPanel({ api }: { api: Api }) {
  const [kind, setKind] = useState<GroupKind>("event");
  const [groups, setGroups] = useState<Group[]>([]);
  const [open, setOpen] = useState<GroupDetail | null>(null);
  const [loading, setLoading] = useState(false);

  const reload = useCallback(async () => {
    setLoading(true);
    setOpen(null);
    try {
      setGroups(await api.groups(kind));
    } finally {
      setLoading(false);
    }
  }, [api, kind]);

  useEffect(() => {
    void reload();
  }, [reload]);

  if (open) {
    const g = open.group;
    return (
      <div className="people-detail">
        <div className="people-bar">
          <button className="scan-btn" onClick={() => setOpen(null)}>
            ← Back to groups
          </button>
          <strong>{g.key ?? `${g.kind} #${g.id}`}</strong>
          <span className="status">
            {g.size} photos{g.start_at ? ` · ${fmtSpan(g.start_at, g.end_at)}` : ""}
          </span>
        </div>
        <div className="person-photos">
          {open.photos.map((p) => (
            <img key={p.id} src={api.thumbUrl(p.id)} alt={p.filename} loading="lazy" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="people-panel">
      <div className="people-bar">
        <nav className="view-tabs">
          {KINDS.map((k) => (
            <button
              key={k.key}
              className={kind === k.key ? "on" : ""}
              title={k.hint}
              onClick={() => setKind(k.key)}
            >
              {k.label}
            </button>
          ))}
        </nav>
        <span className="status">
          {loading ? "loading…" : `${groups.length} ${kind === "semantic" ? "themes" : "groups"}`}
        </span>
      </div>
      {!loading && groups.length === 0 ? (
        <p className="grid-empty">
          No {KINDS.find((k) => k.key === kind)?.label.toLowerCase()} yet — scan a folder first.
        </p>
      ) : (
        <div className="people-grid">
          {groups.map((group) => (
            <div
              key={group.id}
              className="person-card"
              onClick={async () => setOpen(await api.group(group.id))}
              role="button"
            >
              <div className="person-cover" title="View photos">
                {group.rep_photo_id != null ? (
                  <img src={api.thumbUrl(group.rep_photo_id)} alt={group.key ?? "group"} />
                ) : (
                  <span className="cell-fallback">no cover</span>
                )}
                <span className="group-badge">{group.size}</span>
              </div>
              <div className="person-meta">
                <span>{group.key ?? (fmtSpan(group.start_at, group.end_at) || `#${group.id}`)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";
import type { Api, Person, PersonDetail } from "../lib/api";

function PersonCard({
  api,
  person,
  selected,
  onToggle,
  onOpen,
  onChanged,
}: {
  api: Api;
  person: Person;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
  onChanged: () => void;
}) {
  const [name, setName] = useState(person.label ?? "");

  const save = async () => {
    const label = name.trim();
    if (label === (person.label ?? "")) return;
    await api.renamePerson(person.id, label || null);
    onChanged();
  };

  return (
    <div className={`person-card${selected ? " sel" : ""}`}>
      <div className="person-cover" onClick={onOpen} title="View photos">
        {person.rep_photo_id != null ? (
          <img src={api.thumbUrl(person.rep_photo_id)} alt={person.label ?? "person"} />
        ) : (
          <span className="cell-fallback">no cover</span>
        )}
        <input
          type="checkbox"
          className="person-check"
          checked={selected}
          onClick={(e) => e.stopPropagation()}
          onChange={onToggle}
        />
      </div>
      <input
        className="person-name"
        placeholder="Add name…"
        value={name}
        onChange={(e) => setName(e.currentTarget.value)}
        onBlur={() => void save()}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
        }}
      />
      <div className="person-meta">
        <span>{person.size} faces</span>
        <button
          className="link-btn"
          onClick={async () => {
            await api.splitPerson(person.id);
            onChanged();
          }}
        >
          split
        </button>
      </div>
    </div>
  );
}

export function PeoplePanel({ api }: { api: Api }) {
  const [people, setPeople] = useState<Person[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [open, setOpen] = useState<PersonDetail | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    setPeople(await api.people());
    setSelected(new Set());
  }, [api]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const toggle = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const merge = async () => {
    setBusy(true);
    try {
      await api.mergePeople([...selected]);
      await reload();
    } finally {
      setBusy(false);
    }
  };

  const recluster = async () => {
    setBusy(true);
    try {
      await api.recluster();
      await reload();
    } finally {
      setBusy(false);
    }
  };

  if (open) {
    return (
      <div className="people-detail">
        <div className="people-bar">
          <button className="scan-btn" onClick={() => setOpen(null)}>
            ← Back to people
          </button>
          <strong>{open.person.label ?? "Unknown person"}</strong>
          <span className="status">{open.person.size} faces</span>
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
        <button className="scan-btn" disabled={busy} onClick={() => void recluster()}>
          Re-cluster
        </button>
        <button
          className="scan-btn"
          disabled={busy || selected.size < 2}
          onClick={() => void merge()}
        >
          Merge selected ({selected.size})
        </button>
        <span className="status">{people.length} people</span>
      </div>
      {people.length === 0 ? (
        <p className="grid-empty">No people yet — scan a folder with faces.</p>
      ) : (
        <div className="people-grid">
          {people.map((person) => (
            <PersonCard
              key={person.id}
              api={api}
              person={person}
              selected={selected.has(person.id)}
              onToggle={() => toggle(person.id)}
              onOpen={async () => setOpen(await api.person(person.id))}
              onChanged={() => void reload()}
            />
          ))}
        </div>
      )}
    </div>
  );
}

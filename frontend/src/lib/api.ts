// Typed client for the Iris sidecar HTTP API (ARCHITECTURE §8).

export type Meta = {
  app_version: string;
  schema_version: number;
  counts: Record<string, number>;
  data_dir: string;
};

export type Photo = {
  id: number;
  filename: string;
  sort_at: number | null;
  taken_at: number | null;
  width: number | null;
  height: number | null;
  has_thumb: boolean;
};

export type PhotoPage = { items: Photo[]; next_cursor: string | null };

export type SearchItem = Photo & { score: number };
export type SearchResponse = { tier: string; items: SearchItem[] };

export type Person = {
  id: number;
  label: string | null;
  pinned: number;
  size: number;
  rep_face_id: number | null;
  rep_photo_id: number | null;
};
export type PersonDetail = { person: Person; photos: Photo[] };

export type Job = {
  id: number;
  kind: string;
  state: string;
  total: number;
  done: number;
  errored: number;
  finished_at: number | null;
};

export type IngestStatus = { running: boolean; job: Job | null };

export type Api = ReturnType<typeof makeApi>;

export function makeApi(baseUrl: string) {
  async function json<T>(path: string, init?: RequestInit): Promise<T> {
    const resp = await fetch(`${baseUrl}${path}`, init);
    if (!resp.ok) {
      let detail = "";
      try {
        const body = (await resp.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = `: ${body.detail}`;
      } catch {
        // non-JSON error body
      }
      throw new Error(`${resp.status}${detail}`);
    }
    return (await resp.json()) as T;
  }

  return {
    baseUrl,
    meta: () => json<Meta>("/meta"),
    count: () => json<{ count: number }>("/photos/count"),
    photos: (cursor: string | null, limit = 200) =>
      json<PhotoPage>(
        `/photos?limit=${limit}` + (cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""),
      ),
    addRoot: (path: string) =>
      json<{ id: number; path: string }>("/library/roots", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ path }),
      }),
    scan: () => json<{ job_id: number; running: boolean }>("/ingest/scan", { method: "POST" }),
    status: () => json<IngestStatus>("/ingest/status"),
    search: (query: string) =>
      json<SearchResponse>("/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, limit: 200 }),
      }),
    people: () => json<Person[]>("/people"),
    person: (id: number) => json<PersonDetail>(`/people/${id}`),
    renamePerson: (id: number, label: string | null) =>
      json<Person>(`/people/${id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ label }),
      }),
    mergePeople: (ids: number[]) =>
      json<Person>("/people/merge", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ids }),
      }),
    splitPerson: (id: number) =>
      json<{ clusters: number }>(`/people/${id}/split`, { method: "POST" }),
    recluster: () => json<{ people: number }>("/faces/recluster", { method: "POST" }),
    thumbUrl: (id: number) => `${baseUrl}/thumb/${id}`,
  };
}

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

export type PhotoLocation = {
  id: number;
  path: string;
  dir: string;
  filename: string;
  exists: boolean;
};

export type SearchItem = Photo & { score: number };

export type SearchFilters = {
  date_from?: number | null;
  date_to?: number | null;
  place?: string | null;
  radius_km?: number | null;
  has_text?: boolean | null;
};

export type AppliedFilters = {
  date_label: string | null;
  date_from: number | null;
  date_to: number | null;
  place_label: string | null;
  radius_km: number | null;
  text_query: string | null;
};

export type SearchResponse = {
  tier: string;
  items: SearchItem[];
  applied?: AppliedFilters | null;
  place_error?: string | null;
};

export type Person = {
  id: number;
  label: string | null;
  pinned: number;
  size: number;
  rep_face_id: number | null;
  rep_photo_id: number | null;
};
export type PersonDetail = { person: Person; photos: Photo[] };

export type GroupKind = "event" | "burst" | "near_dup" | "semantic";

export type Group = {
  id: number;
  kind: string;
  key: string | null;
  rep_photo_id: number | null;
  size: number;
  score: number | null;
  start_at: number | null;
  end_at: number | null;
};
export type GroupDetail = { group: Group; photos: Photo[] };

export type Tag = { id: number; name: string; kind: string; count?: number | null };

export type OcrRegion = {
  bx: number;
  by: number;
  bw: number;
  bh: number;
  conf: number;
  text: string;
};
export type OcrResponse = { photo_id: number; text: string; regions: OcrRegion[] };

export type TriagePreset = {
  name: string;
  description: string;
  weights: Record<string, number>;
  lam: number;
  invert: boolean;
};

export type TriageItem = Photo & {
  score: number;
  quality: number;
  aesthetic: number;
  representativeness: number;
  subject: number;
  reason: string;
  redundant: boolean;
};

export type TriageResponse = {
  preset: string;
  scope_size: number;
  considered: number;
  items: TriageItem[];
};

export type TriageRequest = {
  group_id?: number | null;
  date_from?: number | null;
  date_to?: number | null;
  preset?: string | null;
  limit?: number;
};

export type CacheStat = { bytes: number; cap_bytes: number | null; over_cap: boolean };
export type StorageStats = {
  thumbs: CacheStat;
  previews: CacheStat;
  embeddings: CacheStat;
  database: CacheStat;
};
export type CompactStat = { before: number; after: number; reclaimed: number };
export type CompactResponse = { clip: CompactStat; faces: CompactStat };

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
    search: (query: string, filters?: SearchFilters) =>
      json<SearchResponse>("/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, limit: 200, filters: filters ?? null }),
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
    groups: (kind: GroupKind, limit = 200) =>
      json<Group[]>(`/groups?kind=${kind}&limit=${limit}`),
    group: (id: number) => json<GroupDetail>(`/groups/${id}`),
    tags: () => json<Tag[]>("/tags"),
    photoTags: (id: number) => json<Tag[]>(`/photos/${id}/tags`),
    addPhotoTag: (id: number, name: string) =>
      json<Tag>(`/photos/${id}/tags`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }),
      }),
    removePhotoTag: (id: number, tagId: number) =>
      json<{ ok: boolean }>(`/photos/${id}/tags/${tagId}`, { method: "DELETE" }),
    photoOcr: (id: number) => json<OcrResponse>(`/photos/${id}/ocr`),
    triagePresets: () => json<TriagePreset[]>("/triage/presets"),
    triage: (req: TriageRequest) =>
      json<TriageResponse>("/triage", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ limit: 60, ...req }),
      }),
    photoLocation: (id: number) => json<PhotoLocation>(`/photos/${id}/location`),
    revealPhoto: (id: number) =>
      json<{ ok: boolean; path: string }>(`/photos/${id}/reveal`, { method: "POST" }),
    storage: () => json<StorageStats>("/maintenance/storage"),
    compact: () => json<CompactResponse>("/maintenance/compact", { method: "POST" }),
    thumbUrl: (id: number) => `${baseUrl}/thumb/${id}`,
    previewUrl: (id: number) => `${baseUrl}/preview/${id}`,
  };
}

import { useCallback, useEffect, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { Api, Photo } from "../lib/api";

const CELL = 168;
const GAP = 8;

/** Keyset-paginated photo list that grows as you scroll (ARCHITECTURE §9). */
function useInfinitePhotos(api: Api, reloadKey: number) {
  const [items, setItems] = useState<Photo[]>([]);
  const stateRef = useRef({ cursor: null as string | null, hasMore: true, loading: false });

  const loadMore = useCallback(async () => {
    const s = stateRef.current;
    if (s.loading || !s.hasMore) return;
    s.loading = true;
    try {
      const page = await api.photos(s.cursor);
      s.cursor = page.next_cursor;
      s.hasMore = page.next_cursor !== null;
      setItems((prev) => prev.concat(page.items));
    } catch {
      // transient failure — a later scroll will retry
    } finally {
      s.loading = false;
    }
  }, [api]);

  useEffect(() => {
    stateRef.current = { cursor: null, hasMore: true, loading: false };
    setItems([]);
    void loadMore();
  }, [reloadKey, loadMore]);

  return { items, loadMore };
}

function Cell({ api, photo }: { api: Api; photo: Photo }) {
  const [failed, setFailed] = useState(false);
  const showThumb = photo.has_thumb && !failed;
  return (
    <div className="cell" title={photo.filename}>
      {showThumb ? (
        <img
          src={api.thumbUrl(photo.id)}
          loading="lazy"
          alt={photo.filename}
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="cell-fallback">{photo.filename}</span>
      )}
    </div>
  );
}

export function PhotoGrid({
  api,
  reloadKey,
  searchItems,
}: {
  api: Api;
  reloadKey: number;
  searchItems?: Photo[] | null;
}) {
  const parentRef = useRef<HTMLDivElement>(null);
  const infinite = useInfinitePhotos(api, reloadKey);
  const searching = searchItems != null;
  const items = searching ? searchItems : infinite.items;
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = parentRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const cols = Math.max(1, Math.floor((width + GAP) / (CELL + GAP)));
  const rowCount = Math.ceil(items.length / cols);

  const rowVirtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => CELL + GAP,
    overscan: 6,
  });

  const virtualRows = rowVirtualizer.getVirtualItems();
  useEffect(() => {
    if (searching) return; // search results are a fixed set, no pagination
    const last = virtualRows[virtualRows.length - 1];
    if (last && last.index >= rowCount - 3) void infinite.loadMore();
  }, [virtualRows, rowCount, infinite.loadMore, searching]);

  return (
    <div ref={parentRef} className="grid-scroll">
      {items.length === 0 ? (
        <p className="grid-empty">
          {searching ? "No matches." : "No photos yet — add a folder above and scan."}
        </p>
      ) : (
        <div style={{ height: rowVirtualizer.getTotalSize(), position: "relative" }}>
          {virtualRows.map((vr) => {
            const start = vr.index * cols;
            const rowItems = items.slice(start, start + cols);
            return (
              <div
                key={vr.key}
                className="grid-row"
                style={{
                  transform: `translateY(${vr.start}px)`,
                  height: CELL,
                  gridTemplateColumns: `repeat(${cols}, 1fr)`,
                  gap: GAP,
                }}
              >
                {rowItems.map((p) => (
                  <Cell key={p.id} api={api} photo={p} />
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

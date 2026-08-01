import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type SidecarState = { baseUrl: string | null; connected: boolean };

/** Resolve the sidecar base URL (via handshake) and probe its health. */
export function useSidecar(): SidecarState {
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // Dev fallback: when running the frontend in a plain browser (Vite, no Tauri),
    // `invoke` is unavailable — fall back to a fixed sidecar URL so the UI can be
    // exercised against a manually-run sidecar. Ignored in the packaged Tauri app.
    const devUrl = (import.meta.env.VITE_SIDECAR_URL as string | undefined) ?? null;
    invoke<string | null>("sidecar_url")
      .then((url) => {
        if (!cancelled && url) setBaseUrl(url);
        else if (!cancelled && devUrl) setBaseUrl(devUrl);
      })
      .catch(() => {
        if (!cancelled && devUrl) setBaseUrl(devUrl);
      });
    const unlisten = listen<string>("sidecar-ready", (event) => {
      if (!cancelled) setBaseUrl(event.payload);
    });
    return () => {
      cancelled = true;
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    if (!baseUrl) return;
    let cancelled = false;
    fetch(`${baseUrl}/health`)
      .then((r) => {
        if (!cancelled) setConnected(r.ok);
      })
      .catch(() => {
        if (!cancelled) setConnected(false);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl]);

  return { baseUrl, connected };
}

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
    invoke<string | null>("sidecar_url").then((url) => {
      if (!cancelled && url) setBaseUrl(url);
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

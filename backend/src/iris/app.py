"""FastAPI application factory.

On startup: create the app data directory tree, open the database, apply pending
migrations, then emit the sidecar readiness handshake the Tauri shell waits for.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iris import __version__
from iris.api.ingest import router as ingest_router
from iris.api.library import router as library_router
from iris.api.photos import router as photos_router
from iris.api.system import router as system_router
from iris.config import Settings, get_settings
from iris.db import apply_migrations, connect
from iris.ingest.orchestrator import IngestManager

# Origins the Tauri webview uses. In `tauri dev` the frontend is served by Vite at
# localhost:1420 and calls the sidecar cross-origin; in a packaged build the webview
# origin is the `tauri://` custom scheme. Both must be allowed to fetch the sidecar.
_DEV_ORIGINS = ["http://localhost:1420", "http://127.0.0.1:1420"]
_TAURI_ORIGIN_REGEX = r"^(tauri://localhost|https?://tauri\.localhost)$"

# The Tauri shell scans the sidecar's stdout for this exact prefix, then parses the
# trailing JSON to learn the base URL. Keep the token stable across versions.
READY_PREFIX = "IRIS_SIDECAR_READY "


def _emit_ready(host: str, port: int, schema_version: int) -> None:
    """Print the readiness handshake. Called only once the socket is truly bound."""
    payload = {
        "host": host,
        "port": port,
        "base_url": f"http://{host}:{port}",
        "app_version": __version__,
        "schema_version": schema_version,
    }
    print(READY_PREFIX + json.dumps(payload), flush=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Pass explicit settings in tests; else use env-derived."""
    resolved = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved.ensure_dirs()
        conn = connect(resolved.db_path, busy_timeout_ms=resolved.sqlite_busy_timeout_ms)
        try:
            schema_version = apply_migrations(conn)
        finally:
            conn.close()
        app.state.settings = resolved
        app.state.schema_version = schema_version
        app.state.ingest = IngestManager(resolved)
        yield

    app = FastAPI(title="Iris", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_DEV_ORIGINS,
        allow_origin_regex=_TAURI_ORIGIN_REGEX,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_router)
    app.include_router(library_router)
    app.include_router(ingest_router)
    app.include_router(photos_router)
    return app


def main() -> None:
    """Console entrypoint: run the sidecar with uvicorn.

    The handshake is emitted from a watcher thread only after the socket is bound
    and serving, and it reports the *actual* bound port — so ``--port 0`` (ephemeral)
    works and no false "ready" is printed if the bind fails.
    """
    import argparse
    import os
    import threading
    import time

    import uvicorn

    parser = argparse.ArgumentParser(prog="iris-sidecar")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    # CLI flags override env for this process (settings are env-derived).
    if args.host:
        os.environ["IRIS_HOST"] = args.host
    if args.port is not None:  # note: --port 0 is valid (ephemeral) and falsy
        os.environ["IRIS_PORT"] = str(args.port)
    if args.data_dir:
        os.environ["IRIS_DATA_DIR"] = args.data_dir
    get_settings.cache_clear()

    settings = get_settings()
    app = create_app(settings)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
    server = uvicorn.Server(config)

    def announce() -> None:
        while not server.started:
            time.sleep(0.05)
        port: int = settings.port
        servers = getattr(server, "servers", [])
        if servers and servers[0].sockets:
            port = int(servers[0].sockets[0].getsockname()[1])
        schema_version = int(getattr(app.state, "schema_version", 0))
        _emit_ready(settings.host, port, schema_version)

    threading.Thread(target=announce, daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:  # pragma: no cover - signal path
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()

"""Frozen-binary entrypoint for the PyInstaller sidecar bundle (ARCHITECTURE §10, Phase 6).

PyInstaller freezes a *script*, not a console-script entry point, so this thin wrapper
calls :func:`iris.app.main`. Keep it dependency-free beyond the app itself.
"""

from __future__ import annotations

from iris.app import main

if __name__ == "__main__":
    main()

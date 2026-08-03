"""Reveal an original file in the OS file manager (Finder / Explorer / xdg).

The sidecar runs on the user's own machine, so it — not the browser — is what can pop
open Finder at a photo's real location (useful for sharing/posting the original). This is
a **read-only** action: it never moves, renames, or writes the source file (CLAUDE.md
non-destructive constraint). Commands are invoked with argument lists (never a shell), and
the path always comes from the indexed DB row, not from user input.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def reveal_in_file_manager(path: Path) -> None:
    """Open the platform file manager with ``path`` selected/highlighted.

    macOS reveals the file in Finder; Windows selects it in Explorer; other platforms
    open the containing folder (no portable "select file" primitive). Raises on a missing
    launcher binary or a non-zero exit so the caller can surface the failure.
    """
    if sys.platform == "darwin":
        args = ["open", "-R", str(path)]
    elif sys.platform == "win32":
        args = ["explorer", f"/select,{path}"]
    else:
        args = ["xdg-open", str(path.parent)]
    subprocess.run(args, check=True)  # noqa: S603 - fixed argv, no shell, DB-sourced path

# PyInstaller spec for the Iris FastAPI sidecar (ARCHITECTURE §10 risk #3, Phase 6).
#
# Produces a `--onedir` bundle (dist/iris-sidecar/) — chosen over `--onefile` because the
# ML stack (onnxruntime + CoreML, insightface, opencv, pyobjc) unpacks large native
# libraries; onedir avoids the per-launch extraction cost and the temp-dir dylib-loading
# quirks that break CoreML. The Tauri shell spawns the executable inside that folder.
#
# Build:  cd backend && uv run --extra ml pyinstaller iris-sidecar.spec --noconfirm
# The heavy, native, or data-carrying deps are pulled in wholesale via collect_all; the
# optional-extra ones are wrapped so a lean build still succeeds (they just won't be in it).

from PyInstaller.utils.hooks import collect_all

datas: list = []
binaries: list = []
hiddenimports: list = [
    "iris.app",
    # uvicorn's workers/loops/protocols are imported by string name at runtime.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# Always-present (core deps).
_REQUIRED = ["iris", "uvicorn", "fastapi", "pydantic", "PIL", "pillow_heif", "blake3", "numpy"]
# Optional extras — include when installed, skip cleanly otherwise.
_OPTIONAL = [
    "onnxruntime", "hnswlib", "transformers", "tokenizers", "huggingface_hub",
    "insightface", "cv2", "ocrmac", "objc", "geonamescache",
]

for pkg in _REQUIRED:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in _OPTIONAL:
    try:
        d, b, h = collect_all(pkg)
    except Exception as exc:  # not installed in this build — fine
        print(f"[iris-sidecar.spec] skipping optional dep {pkg!r}: {exc}")
        continue
    datas += d
    binaries += b
    hiddenimports += h

block_cipher = None

a = Analysis(
    ["packaging/entrypoint.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="iris-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts CoreML/onnxruntime dylibs — keep it off
    console=True,  # the sidecar prints the IRIS_SIDECAR_READY handshake on stdout
    target_arch="arm64",  # Apple Silicon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="iris-sidecar",
)

//! Tauri shell entrypoint.
//!
//! On startup we launch the Python FastAPI sidecar and learn its base URL by
//! reading the `IRIS_SIDECAR_READY {json}` handshake line from its stdout, then
//! expose that URL to the frontend (via the `sidecar_url` command and a
//! `sidecar-ready` event). In dev the sidecar is launched with `uv`; production
//! packaging (Phase 6) will replace this with a bundled PyInstaller binary via
//! Tauri's `externalBin` sidecar mechanism.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Emitter, Manager, State};

const READY_PREFIX: &str = "IRIS_SIDECAR_READY ";
// 0 = ephemeral: the OS picks a free port, so we never collide with a stale
// sidecar. The real bound port comes back in the handshake. Override with IRIS_PORT.
const DEFAULT_PORT: u16 = 0;
const READY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct SidecarState {
    base_url: Mutex<Option<String>>,
    child: Mutex<Option<Child>>,
}

/// Current sidecar base URL, or `None` until the readiness handshake arrives.
#[tauri::command]
fn sidecar_url(state: State<SidecarState>) -> Option<String> {
    state.base_url.lock().unwrap().clone()
}

fn spawn_sidecar() -> std::io::Result<Child> {
    let port: u16 = std::env::var("IRIS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    // Escape hatch for custom launch commands (CI, alternate envs).
    if let Ok(cmd) = std::env::var("IRIS_SIDECAR_CMD") {
        return Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();
    }

    let backend_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("backend");

    // Prefer the venv console script directly: this child *is* uvicorn, so
    // `child.kill()` on exit actually stops it. `uv run` would insert a wrapper
    // process and leave an orphaned grandchild holding the port. Fall back to
    // `uv run` if the venv isn't built yet.
    let venv_bin = backend_dir.join(".venv").join("bin").join("iris-sidecar");
    let mut command = if venv_bin.exists() {
        Command::new(venv_bin)
    } else {
        let mut fallback = Command::new("uv");
        fallback.args(["run", "iris-sidecar"]);
        fallback
    };
    command
        .args(["--port", &port.to_string()])
        .current_dir(&backend_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
}

/// Launch the sidecar and block (on a background thread) until it reports ready.
fn start_sidecar(app: &tauri::AppHandle) {
    let mut child = match spawn_sidecar() {
        Ok(child) => child,
        Err(err) => {
            eprintln!("[iris] failed to spawn sidecar: {err}");
            return;
        }
    };
    let Some(stdout) = child.stdout.take() else {
        eprintln!("[iris] sidecar stdout was not piped");
        return;
    };

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let mut sent = false;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if !sent {
                if let Some(json) = line.strip_prefix(READY_PREFIX) {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
                        if let Some(url) = value.get("base_url").and_then(|u| u.as_str()) {
                            let _ = tx.send(url.to_string());
                            sent = true;
                        }
                    }
                }
            }
            println!("[iris-sidecar] {line}"); // forward logs; keep the pipe drained
        }
    });

    let state = app.state::<SidecarState>();
    match rx.recv_timeout(READY_TIMEOUT) {
        Ok(url) => {
            println!("[iris] sidecar ready at {url}");
            *state.base_url.lock().unwrap() = Some(url.clone());
            let _ = app.emit("sidecar-ready", url);
        }
        Err(_) => eprintln!("[iris] timed out waiting for sidecar readiness"),
    }
    *state.child.lock().unwrap() = Some(child);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(SidecarState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            // Don't block the UI thread on startup / the readiness timeout.
            std::thread::spawn(move || start_sidecar(&handle));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_url])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(mut child) = app_handle
                    .state::<SidecarState>()
                    .child
                    .lock()
                    .unwrap()
                    .take()
                {
                    let _ = child.kill(); // don't leave an orphaned sidecar
                }
            }
        });
}

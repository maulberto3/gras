//! The process-wide console sink ceiling.

use pyo3::prelude::*;

/// The engine's console sink is process-wide, so it can be installed exactly once: the FIRST `RaceEngine` in a process sets the ceiling.
pub(crate) static LOGGING: std::sync::Once = std::sync::Once::new();

/// Send Rust log text through Python's current `sys.stdout`; Jupyter then has a
/// single UTF-8 stream, in the same order as the cell's own prints.
fn python_stdout(text: &str) -> bool {
    Python::attach(|py| -> PyResult<()> {
        let stdout = py.import("sys")?.getattr("stdout")?;
        stdout.call_method1("write", (text,))?;
        stdout.call_method0("flush")?;
        // `run()` holds the GIL for the whole race. Give ipykernel's IOPub
        // thread a chance to forward the line we just queued; the closure
        // touches no engine state, so this is just a brief detach + reacquire.
        py.detach(std::thread::yield_now);
        Ok(())
    })
    .is_ok()
}

/// Route the engine's console output through Python. Safe to call more than
/// once: the first installation wins.
pub(crate) fn install_console_sink() {
    gras::engine::reporting::logging::set_console_sink(python_stdout);
}

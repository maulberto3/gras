//! Logging: the process-wide `tracing` subscriber the engine installs, and
//! the three sinks its single event stream fans out to.
//!
//! ```text
//! engine event ─┬─ console (stderr)  the human lines, filtered by `LogLevel`
//!               ├─ frame   (stdout)  the `Minimal` vitals table, redrawn
//!               └─ telemetry (file)  <run_dir>/telemetry.jsonl, opt-in
//! ```
//!
//! The file sink can't be aimed at `init` — the run dir does not exist yet — so
//! it BUFFERS the start-up block (race init / search space / build stamp) and
//! flushes it in place when the engine settles the run dir. Until it settles,
//! the telemetry layer is inert: a run that never traces pays nothing.

use std::fmt::Debug;
use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::filter::{FilterExt, filter_fn};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

use super::config::LogLevel;

/// Target the `Minimal` vitals frame rides on: the console layer skips it, the
/// frame layer renders it and nothing else.
pub const FRAME_TARGET: &str = "gras::frame";

/// Telemetry sidecar name inside the run directory.
const TELEMETRY_FILE: &str = "telemetry.jsonl";

/// The one telemetry sink per process — armed by `init`, aimed by the engine.
static SINK: OnceLock<Sink> = OnceLock::new();

/// Start-up window: buffering, because the run dir doesn't exist yet.
static PENDING: AtomicBool = AtomicBool::new(false);

/// The file is open and the sink streams into it (set once, never cleared).
static ARMED: AtomicBool = AtomicBool::new(false);

/// Cap on the start-up buffer: a caller that never settles the sink must not
/// be able to grow it without bound.
const MAX_STARTUP_BYTES: usize = 1 << 20;

/// Install the subscriber for a run: console + frame, plus the telemetry sink
/// (inert until [`settle_run_dir`] decides whether the run wants the file).
///
/// `filter_override` is a caller-supplied verbosity (the examples'
/// `--log-level`); `RUST_LOG` beats both. Returns false when a subscriber is
/// already installed (a second engine in one process) — the first one owns the
/// sinks, and calling this twice is not an error.
pub fn init(level: LogLevel, filter_override: Option<&str>) -> bool {
    let base = filter_override.unwrap_or_else(|| level.env_filter());
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| base.to_string());

    PENDING.store(true, Ordering::Relaxed);

    // `Minimal` mutes the info lines that would compete with the frame (its
    // `env_filter` is `warn`) — the frame rides a different target, so it
    // still gets through, which is exactly what the mode means.
    let console = fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_level(false)
        .with_filter(
            EnvFilter::new(&filter).and(filter_fn(|m: &Metadata<'_>| m.target() != FRAME_TARGET)),
        );
    let frame =
        FrameLayer::new().with_filter(filter_fn(|m: &Metadata<'_>| m.target() == FRAME_TARGET));
    // The record, not a view: every gras event lands in the file regardless of
    // the console verbosity, and no other crate's chatter does. Gated on the
    // settle state so an untraced run serializes nothing at all.
    let telemetry = fmt::layer()
        .with_writer(SINK.get_or_init(Sink::new).clone())
        .with_ansi(false)
        .json()
        .with_filter(filter_fn(|m: &Metadata<'_>| {
            m.target().starts_with("gras")
                && (ARMED.load(Ordering::Relaxed) || PENDING.load(Ordering::Relaxed))
        }));

    let installed = tracing_subscriber::registry()
        .with(console)
        .with(frame)
        .with(telemetry)
        .try_init()
        .is_ok();
    if !installed {
        // Someone else owns the subscriber: our layers never run, so the
        // start-up window would just buffer for nobody.
        PENDING.store(false, Ordering::Relaxed);
    }
    installed
}

/// Emit one `Minimal` vitals frame. The frame layer draws it; the console
/// layer ignores it.
pub(crate) fn frame(text: &str) {
    tracing::info!(target: FRAME_TARGET, frame = text);
}

/// Settle the telemetry sink now that the run dir exists: attach
/// `<run_dir>/telemetry.jsonl` when the run asked for the trace file, flush the
/// buffered start-up block into it, and otherwise close the window by dropping
/// that buffer. Called by the engine right after it writes `engine.json`.
pub fn settle_run_dir(run_dir: &Path, trace_file: bool) {
    PENDING.store(false, Ordering::Relaxed);
    let sink = SINK.get_or_init(Sink::new);
    if !trace_file {
        sink.lock().buffered.clear();
        return;
    }

    let path = run_dir.join(TELEMETRY_FILE);
    let outcome = {
        let mut telemetry = sink.lock();
        // Attach once: a second call would truncate the file under the run.
        if telemetry.file.is_some() {
            return;
        }
        File::create(&path).map(|mut file| {
            let _ = file.write_all(telemetry.buffered.as_bytes());
            let _ = file.flush();
            telemetry.buffered.clear();
            telemetry.file = Some(file);
            telemetry.path = Some(path.clone());
        })
    };

    // Reported outside the lock: the telemetry layer writes through this very
    // sink, so emitting while holding it would deadlock. A failed open leaves
    // the sink disarmed — no file, no half-written "telemetry" in the run dir.
    match outcome {
        Ok(()) => ARMED.store(true, Ordering::Relaxed),
        Err(e) => tracing::warn!("telemetry {} unavailable: {e}", path.display()),
    }
}

/// The telemetry file, shared by the layer (as its writer) and the engine
/// (which only learns the run dir after start-up).
#[derive(Clone)]
struct Sink(Arc<Mutex<Telemetry>>);

/// Buffered-until-attached state behind a [`Sink`].
struct Telemetry {
    /// Lines emitted before the run dir existed, in order.
    buffered: String,
    file: Option<File>,
    path: Option<PathBuf>,
}

impl Sink {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Telemetry {
            buffered: String::new(),
            file: None,
            path: None,
        })))
    }

    /// Poison-proof lock: a panicking event must not kill the log stream.
    fn lock(&self) -> std::sync::MutexGuard<'_, Telemetry> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut telemetry = self.lock();
        match telemetry.file.as_mut() {
            Some(file) => file.write(buf),
            None => {
                if telemetry.buffered.len() < MAX_STARTUP_BYTES {
                    telemetry.buffered.push_str(&String::from_utf8_lossy(buf));
                }
                Ok(buf.len())
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut telemetry = self.lock();
        match telemetry.file.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Sink {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Renders `Minimal` frames. On a terminal the previous frame is erased and
/// redrawn in place, so the table holds one screen position instead of
/// scrolling; off a terminal it prints plainly (no escape codes in a file).
struct FrameLayer {
    /// Lines the last frame occupied, for the cursor-up count.
    drawn: Mutex<usize>,
}

impl FrameLayer {
    fn new() -> Self {
        Self {
            drawn: Mutex::new(0),
        }
    }
}

impl<S: Subscriber> Layer<S> for FrameLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = FrameField::default();
        event.record(&mut fields);
        let Some(text) = fields.0 else {
            return;
        };

        let mut out = io::stdout().lock();
        let mut drawn = self.drawn.lock().unwrap_or_else(|e| e.into_inner());
        if *drawn > 0 && out.is_terminal() {
            // Cursor up over the previous frame, then clear everything below.
            let _ = write!(out, "\x1b[{}A\x1b[0J", *drawn);
        }
        let _ = writeln!(out, "{text}");
        let _ = out.flush();
        *drawn = text.lines().count();
    }
}

/// Extracts the `frame` field back out of an event.
#[derive(Default)]
struct FrameField(Option<String>);

impl Visit for FrameField {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "frame" {
            self.0 = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "frame" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

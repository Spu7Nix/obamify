#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool};
#[cfg(not(target_arch = "wasm32"))]
pub mod drawing_process;
mod genetic;
mod optimal;
pub mod util;

#[cfg(target_arch = "wasm32")]
pub mod worker;

fn _debug_print(s: String) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&s.into());
    #[cfg(not(target_arch = "wasm32"))]
    println!("{}", s);
}

use crate::app::calculate::util::Algorithm;
use crate::app::{
    calculate::util::{GenerationSettings, ProgressSink},
    preset::{Preset, UnprocessedPreset},
};
use serde::{Deserialize, Serialize};

#[inline(always)]
fn heuristic(
    apos: (u16, u16),
    bpos: (u16, u16),
    a: (u8, u8, u8),
    b: (u8, u8, u8),
    color_weight: i64,
    spatial_weight: i64,
) -> i64 {
    let spatial = (apos.0 as i64 - bpos.0 as i64).pow(2) + (apos.1 as i64 - bpos.1 as i64).pow(2);
    let color = (a.0 as i64 - b.0 as i64).pow(2)
        + (a.1 as i64 - b.1 as i64).pow(2)
        + (a.2 as i64 - b.2 as i64).pow(2);
    color * color_weight + (spatial * spatial_weight).pow(2)
}

#[derive(Serialize, Deserialize)]
pub enum ProgressMsg {
    Progress(f32),
    UpdatePreview {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    UpdateAssignments(Vec<usize>),
    Done(Preset), // result directory
    Error(String),
    Cancelled,
}

impl ProgressMsg {
    pub fn typ(&self) -> &'static str {
        match self {
            ProgressMsg::Progress(_) => "progress",
            ProgressMsg::UpdatePreview { .. } => "update_preview",
            ProgressMsg::UpdateAssignments(_) => "update_assignments",
            ProgressMsg::Done(_) => "done",
            ProgressMsg::Error(_) => "error",
            ProgressMsg::Cancelled => "cancelled",
        }
    }
}

fn make_new_img(source_pixels: &[(u8, u8, u8)], assignments: &[usize], sidelen: u32) -> Vec<u8> {
    let mut img = vec![0; (sidelen * sidelen * 3) as usize];
    for (target_idx, source_idx) in assignments.iter().enumerate() {
        let (r, g, b) = source_pixels[*source_idx];
        let base = target_idx * 3;
        img[base] = r;
        img[base + 1] = g;
        img[base + 2] = b;
    }
    img
}

const SWAPS_PER_GENERATION_PER_PIXEL: usize = 128;

/// How many threads the native solvers may use: all cores by default, or whatever
/// `OBAMIFY_THREADS` says (`1` forces the single-threaded code paths).
#[cfg(not(target_arch = "wasm32"))]
fn worker_threads() -> usize {
    std::env::var("OBAMIFY_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn process<S: ProgressSink + Send>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    // A private pool, so the solvers' worker count doesn't depend on (or fight with)
    // anything else in the app. `install` runs the closure *inside* the pool, which
    // makes the solvers' fork/join calls much cheaper than entering from outside.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_threads())
        .thread_name(|i| format!("obamify-worker-{i}"))
        .build()?;

    // Box<dyn Error> isn't Send, so only a String crosses the pool boundary.
    pool.install(move || {
        match settings.algorithm {
            Algorithm::Optimal => optimal::process_optimal(unprocessed, settings, tx, cancel),
            Algorithm::Genetic => genetic::process_genetic(unprocessed, settings, tx, cancel),
        }
        .map_err(|e| e.to_string())
    })
    .map_err(Into::into)
}

#[cfg(target_arch = "wasm32")]
pub fn process<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
) -> Result<(), Box<dyn std::error::Error>> {
    // The web build runs in a single worker without shared-memory threads.
    match settings.algorithm {
        Algorithm::Optimal => optimal::process_optimal(unprocessed, settings, tx),
        Algorithm::Genetic => genetic::process_genetic(unprocessed, settings, tx),
    }
}

//! The "genetic" algorithm: repeated random pair swaps that are kept if they lower
//! the total cost, with the allowed swap distance shrinking every generation.
//!
//! Native builds run each generation on several threads. All threads share one
//! grid of cells, and a swap only touches its two cells, so each cell carries a
//! tiny spin lock: a thread `try_lock`s both cells of a candidate swap and simply
//! skips the attempt if either is busy. Skipping is harmless for a randomised
//! search and, with thousands of cells and a handful of threads, rare.
//!
//! Because the threads interleave freely, parallel runs are not bit-for-bit
//! reproducible (the single-threaded path still is).

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering},
};

use super::util::{self, GenerationSettings, ProgressSink};
use super::{ProgressMsg, SWAPS_PER_GENERATION_PER_PIXEL, heuristic, make_new_img};
use crate::app::preset::{Preset, UnprocessedPreset};

const SEED: u64 = 12345;

/// Picks the partner cell for `apos`, exactly like the original: a uniformly random
/// offset within `max_dist` in both axes, clamped to the image.
#[inline(always)]
fn pick_partner(rng: &mut frand::Rand, apos: usize, side: usize, max_dist: u32) -> (u16, u16) {
    // 32-bit division: noticeably cheaper than usize on many CPUs, and this is the hot path.
    let (ax, ay) = ((apos as u32 % side as u32) as i16, (apos as u32 / side as u32) as i16);
    let d = max_dist as i16;
    let bx = (ax + rng.gen_range(-d..(d + 1))).clamp(0, side as i16 - 1) as u16;
    let by = (ay + rng.gen_range(-d..(d + 1))).clamp(0, side as i16 - 1) as u16;
    (bx, by)
}

/// What to do after a generation: keep going, or hand back the result.
fn finished(max_dist: u32, swaps_made: usize) -> bool {
    max_dist < 4 && swaps_made < 10
}

fn next_max_dist(max_dist: u32) -> u32 {
    (max_dist as f32 * 0.99).max(2.0) as u32
}

/// Shared by both paths: report one finished generation, or the final result.
/// Returns true when the run is over.
fn report<S: ProgressSink>(
    tx: &mut S,
    unprocessed: &UnprocessedPreset,
    settings: &GenerationSettings,
    source_pixels: &[(u8, u8, u8)],
    assignments: Vec<usize>,
    max_dist: u32,
    swaps_made: usize,
) -> bool {
    if finished(max_dist, swaps_made) {
        tx.send(ProgressMsg::Done(Preset {
            inner: UnprocessedPreset {
                name: unprocessed.name.clone(),
                width: settings.sidelen,
                height: settings.sidelen,
                source_img: source_pixels
                    .iter()
                    .flat_map(|(r, g, b)| [*r, *g, *b])
                    .collect(),
            },
            assignments,
        }));
        return true;
    }
    tx.send(ProgressMsg::UpdatePreview {
        width: settings.sidelen,
        height: settings.sidelen,
        data: make_new_img(source_pixels, &assignments, settings.sidelen),
    });
    tx.send(ProgressMsg::Progress(
        1.0 - max_dist as f32 / settings.sidelen as f32,
    ));
    false
}

// ---------------------------------------------------------------------------
// Sequential path (wasm, or a single core). Same logic as before.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Pixel {
    src_x: u16,
    src_y: u16,
    rgb: (u8, u8, u8),
    h: i64, // current heuristic value
}

fn process_sequential<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_img = image::ImageBuffer::from_vec(
        unprocessed.width,
        unprocessed.height,
        unprocessed.source_img.clone(),
    )
    .unwrap();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;
    let side = settings.sidelen as usize;
    let prox = settings.proximity_importance;

    let mut pixels = source_pixels
        .iter()
        .enumerate()
        .map(|(i, &rgb)| {
            let x = (i % side) as u16;
            let y = (i / side) as u16;
            let h = heuristic((x, y), (x, y), rgb, target_pixels[i], weights[i], prox);
            Pixel {
                src_x: x,
                src_y: y,
                rgb,
                h,
            }
        })
        .collect::<Vec<_>>();

    let mut rng = frand::Rand::with_seed(SEED);
    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * pixels.len();
    let mut max_dist = settings.sidelen;

    loop {
        let mut swaps_made = 0;
        for _ in 0..swaps_per_generation {
            let apos = rng.gen_range(0..pixels.len() as u32) as usize;
            let (ax, ay) = ((apos as u32 % side as u32) as u16, (apos as u32 / side as u32) as u16);
            let (bx, by) = pick_partner(&mut rng, apos, side, max_dist);
            let bpos = by as usize * side + bx as usize;

            let (pa, pb) = (pixels[apos], pixels[bpos]);
            let a_on_b = heuristic((pa.src_x, pa.src_y), (bx, by), pa.rgb, target_pixels[bpos], weights[bpos], prox);
            let b_on_a = heuristic((pb.src_x, pb.src_y), (ax, ay), pb.rgb, target_pixels[apos], weights[apos], prox);

            if (pa.h - b_on_a) + (pb.h - a_on_b) > 0 {
                pixels.swap(apos, bpos);
                pixels[apos].h = b_on_a;
                pixels[bpos].h = a_on_b;
                swaps_made += 1;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        if cancel.load(Ordering::Relaxed) {
            tx.send(ProgressMsg::Cancelled);
            return Ok(());
        }

        let assignments = pixels
            .iter()
            .map(|p| p.src_y as usize * side + p.src_x as usize)
            .collect::<Vec<_>>();
        if report(tx, &unprocessed, &settings, &source_pixels, assignments, max_dist, swaps_made) {
            return Ok(());
        }
        max_dist = next_max_dist(max_dist);
    }
}

// ---------------------------------------------------------------------------
// Parallel path (native, more than one thread).
// ---------------------------------------------------------------------------

/// One grid cell: which source pixel currently sits here, and its cost here.
/// The data fields are relaxed atomics only so the type is `Sync` without
/// `unsafe`; the `lock` (acquire/release) is what actually orders accesses.
#[cfg(not(target_arch = "wasm32"))]
struct Cell {
    lock: AtomicBool,
    /// Source position of the pixel in this cell: `x << 16 | y`.
    src: AtomicU32,
    h: AtomicI64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Cell {
    #[inline(always)]
    fn try_lock(&self) -> bool {
        // Test first so a contended cell doesn't take the cache line exclusively.
        !self.lock.load(Ordering::Relaxed)
            && self
                .lock
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
    }

    #[inline(always)]
    fn unlock(&self) {
        self.lock.store(false, Ordering::Release);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct Shared<'a> {
    side: usize,
    prox: i64,
    cells: Vec<Cell>,
    source: &'a [(u8, u8, u8)],
    target: &'a [(u8, u8, u8)],
    weights: &'a [i64],
}

#[cfg(not(target_arch = "wasm32"))]
impl Shared<'_> {
    /// If swapping cells `a` and `b` lowers the total cost, the two new per-cell
    /// costs `(b's pixel at a, a's pixel at b)`. Reads the cells as they are now;
    /// only exact while the caller holds both locks.
    #[inline(always)]
    fn improvement(
        &self,
        ca: &Cell,
        cb: &Cell,
        (ax, ay): (u16, u16),
        (bx, by): (u16, u16),
        apos: usize,
        bpos: usize,
    ) -> Option<(i64, i64)> {
        let side = self.side;
        let (sa, sb) = (ca.src.load(Ordering::Relaxed), cb.src.load(Ordering::Relaxed));
        let (ha, hb) = (ca.h.load(Ordering::Relaxed), cb.h.load(Ordering::Relaxed));
        let (sax, say) = ((sa >> 16) as u16, (sa & 0xFFFF) as u16);
        let (sbx, sby) = ((sb >> 16) as u16, (sb & 0xFFFF) as u16);
        let rgb_a = self.source[say as usize * side + sax as usize];
        let rgb_b = self.source[sby as usize * side + sbx as usize];

        let a_on_b = heuristic((sax, say), (bx, by), rgb_a, self.target[bpos], self.weights[bpos], self.prox);
        let b_on_a = heuristic((sbx, sby), (ax, ay), rgb_b, self.target[apos], self.weights[apos], self.prox);
        ((ha - b_on_a) + (hb - a_on_b) > 0).then_some((b_on_a, a_on_b))
    }

    #[inline(always)]
    fn improves(&self, ca: &Cell, cb: &Cell, a: (u16, u16), b: (u16, u16), apos: usize, bpos: usize) -> bool {
        self.improvement(ca, cb, a, b, apos, bpos).is_some()
    }

    /// Try `attempts` random swaps; returns how many were made.
    fn run(&self, rng: &mut frand::Rand, attempts: usize, max_dist: u32) -> usize {
        let side = self.side;
        let mut swaps_made = 0;
        for _ in 0..attempts {
            let apos = rng.gen_range(0..self.cells.len() as u32) as usize;
            let (bx, by) = pick_partner(rng, apos, side, max_dist);
            let bpos = by as usize * side + bx as usize;
            if apos == bpos {
                continue; // swapping a cell with itself never improves anything
            }

            let (ax, ay) = ((apos as u32 % side as u32) as u16, (apos as u32 / side as u32) as u16);
            let (ca, cb) = (&self.cells[apos], &self.cells[bpos]);

            // 1. Optimistic look, no locks. Most attempts don't improve anything, and
            //    for those this is all we do. The loads may be stale or even mix two
            //    moments in time (another thread may be mid-swap); that can only make
            //    us miss a swap or ask for a lock we then don't need, because step 2
            //    re-checks everything under the lock.
            if !self.improves(ca, cb, (ax, ay), (bx, by), apos, bpos) {
                continue;
            }

            // 2. Looks good: take both cells (or give up if someone else has one)
            //    and check again on data nobody else can change.
            let (lo, hi) = if apos < bpos { (apos, bpos) } else { (bpos, apos) };
            if !self.cells[lo].try_lock() {
                continue;
            }
            if !self.cells[hi].try_lock() {
                self.cells[lo].unlock();
                continue;
            }
            if let Some((b_on_a, a_on_b)) = self.improvement(ca, cb, (ax, ay), (bx, by), apos, bpos) {
                let (sa, sb) = (ca.src.load(Ordering::Relaxed), cb.src.load(Ordering::Relaxed));
                ca.src.store(sb, Ordering::Relaxed);
                ca.h.store(b_on_a, Ordering::Relaxed);
                cb.src.store(sa, Ordering::Relaxed);
                cb.h.store(a_on_b, Ordering::Relaxed);
                swaps_made += 1;
            }
            self.cells[hi].unlock();
            self.cells[lo].unlock();
        }
        swaps_made
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn process_parallel<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    cancel: &AtomicBool,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use rayon::prelude::*;

    let source_img = image::ImageBuffer::from_vec(
        unprocessed.width,
        unprocessed.height,
        unprocessed.source_img.clone(),
    )
    .unwrap();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;
    let side = settings.sidelen as usize;
    let prox = settings.proximity_importance;

    let cells = source_pixels
        .iter()
        .enumerate()
        .map(|(i, &rgb)| {
            let (x, y) = ((i % side) as u16, (i / side) as u16);
            Cell {
                lock: AtomicBool::new(false),
                src: AtomicU32::new((x as u32) << 16 | y as u32),
                h: AtomicI64::new(heuristic((x, y), (x, y), rgb, target_pixels[i], weights[i], prox)),
            }
        })
        .collect::<Vec<_>>();
    let shared = Shared {
        side,
        prox,
        cells,
        source: &source_pixels,
        target: &target_pixels,
        weights: &weights,
    };

    let n = source_pixels.len();
    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * n;
    let mut max_dist = settings.sidelen;
    let mut generation: u64 = 0;

    loop {
        // Split the generation's attempts evenly; each task has its own RNG stream.
        let swaps_made: usize = (0..threads)
            .into_par_iter()
            .map(|t| {
                let attempts = swaps_per_generation / threads
                    + if t < swaps_per_generation % threads { 1 } else { 0 };
                let seed = SEED
                    .wrapping_add(generation.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add(t as u64);
                shared.run(&mut frand::Rand::with_seed(seed), attempts, max_dist)
            })
            .sum();
        generation += 1;

        if cancel.load(Ordering::Relaxed) {
            tx.send(ProgressMsg::Cancelled);
            return Ok(());
        }

        let assignments = shared
            .cells
            .iter()
            .map(|c| {
                let s = c.src.load(Ordering::Relaxed);
                (s & 0xFFFF) as usize * side + (s >> 16) as usize
            })
            .collect::<Vec<_>>();
        if report(tx, &unprocessed, &settings, &source_pixels, assignments, max_dist, swaps_made) {
            return Ok(());
        }
        max_dist = next_max_dist(max_dist);
    }
}

// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn process_genetic<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let threads = rayon::current_num_threads();
    // Tiny images aren't worth the coordination.
    if threads > 1 && settings.sidelen >= 32 {
        process_parallel(unprocessed, settings, tx, &cancel, threads)
    } else {
        process_sequential(unprocessed, settings, tx, &cancel)
    }
}

#[cfg(target_arch = "wasm32")]
pub fn process_genetic<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
) -> Result<(), Box<dyn std::error::Error>> {
    process_sequential(unprocessed, settings, tx)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use uuid::Uuid;

    fn run(threads: usize, side: u32) -> Vec<usize> {
        let mut rng = frand::Rand::with_seed(3);
        let (w, h) = (48u32, 48u32);
        let img: Vec<u8> = (0..w * h * 3).map(|_| rng.gen_range(0..256u32) as u8).collect();
        let unprocessed = UnprocessedPreset {
            name: "t".into(),
            width: w,
            height: h,
            source_img: img,
        };
        let mut settings = GenerationSettings::default(Uuid::nil(), "t".into());
        settings.sidelen = side;

        let out = Arc::new(Mutex::new(None));
        let out2 = out.clone();
        let mut sink = move |m| {
            if let ProgressMsg::Done(p) = m {
                *out2.lock().unwrap() = Some(p.assignments);
            }
        };
        let cancel = AtomicBool::new(false);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            if threads > 1 {
                process_parallel(unprocessed, settings, &mut sink, &cancel, threads).unwrap();
            } else {
                process_sequential(unprocessed, settings, &mut sink, &cancel).unwrap();
            }
        });
        let result = out.lock().unwrap().take().unwrap();
        result
    }

    /// However the threads interleave, every source pixel must end up in exactly one
    /// cell: no pixel duplicated, none lost.
    #[test]
    fn parallel_run_yields_a_permutation() {
        let side = 24;
        for _ in 0..3 {
            let mut a = run(4, side);
            a.sort_unstable();
            assert!(a.iter().copied().eq(0..(side * side) as usize));
        }
    }

    /// Parallel and sequential runs should land at about the same quality.
    #[test]
    fn parallel_quality_matches_sequential() {
        let side = 24u32;
        let cost = |assign: &[usize]| -> i64 {
            // Rebuild the problem to score it with the shared heuristic.
            let mut rng = frand::Rand::with_seed(3);
            let img: Vec<u8> = (0..48 * 48 * 3).map(|_| rng.gen_range(0..256u32) as u8).collect();
            let mut settings = GenerationSettings::default(Uuid::nil(), "t".into());
            settings.sidelen = side;
            let src = image::ImageBuffer::from_vec(48, 48, img).unwrap();
            let (sp, tp, wt) = util::get_images(src, &settings).unwrap();
            let s = side as usize;
            assign
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    heuristic(
                        ((c % s) as u16, (c / s) as u16),
                        ((i % s) as u16, (i / s) as u16),
                        sp[c],
                        tp[i],
                        wt[i],
                        settings.proximity_importance,
                    )
                })
                .sum()
        };
        let (seq, par) = (cost(&run(1, side)), cost(&run(4, side)));
        let ratio = par as f64 / seq as f64;
        assert!((0.97..1.03).contains(&ratio), "seq={seq} par={par} ratio={ratio}");
    }
}

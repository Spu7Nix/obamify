//! The "optimal" algorithm: an exact minimum-cost assignment between source and
//! target cells (the Hungarian method, in its shortest-augmenting-path form).
//!
//! Compared to the previous inlined `kuhn_munkres`, this version
//!
//! * does a **single pass** over the columns per search step (cost evaluation,
//!   slack update and arg-min fused) instead of three, by keeping the dual
//!   variables *lazy* (they are only brought up to date once per augmentation),
//! * stores the columns in struct-of-arrays form and hoists everything that only
//!   depends on the row out of the inner loop, so the hot loop has no division,
//!   modulo or bounds-check-heavy indexing,
//! * needs no O(n²) initialisation pass (all costs are >= 0, so zero duals are
//!   already feasible),
//! * checks for cancellation after every row instead of every 100th, and
//! * on native targets, splits the per-step scan across a rayon thread pool.
//!
//! The result is still an exact optimum. Ties are broken towards the lowest
//! column index both sequentially and in parallel, so the output does not depend
//! on how many threads are used.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool, atomic::Ordering};

use super::util::{self, GenerationSettings, ProgressSink};
use super::{ProgressMsg, make_new_img};
use crate::app::preset::{Preset, UnprocessedPreset};

const INF: i64 = i64::MAX / 4;
/// Marks "no row" / "no column" / "the root of the search tree".
const NONE: u32 = u32::MAX;

/// Below this many columns, splitting a step across threads costs more than it saves.
#[cfg(not(target_arch = "wasm32"))]
const MIN_COLS_FOR_PARALLEL: usize = 4096;
/// Smallest slice of columns handed to one thread.
#[cfg(not(target_arch = "wasm32"))]
const MIN_CHUNK: usize = 1024;

/// Everything that is fixed for the whole run. Rows are target cells, columns are
/// source cells. The cost of putting source cell `j` at target cell `i` is
/// `heuristic(..)`, which we minimise.
struct Problem<'a> {
    side: usize,
    spatial_weight: i64,
    target: &'a [(u8, u8, u8)],
    weights: &'a [i64],
    // Source cells, struct-of-arrays so the hot loop streams through memory.
    col_x: Vec<i32>,
    col_y: Vec<i32>,
    col_r: Vec<i32>,
    col_g: Vec<i32>,
    col_b: Vec<i32>,
}

/// The parts of a row that the inner loop needs.
#[derive(Clone, Copy)]
struct RowCtx {
    x: i32,
    y: i32,
    r: i32,
    g: i32,
    b: i32,
    color_weight: i64,
}

impl<'a> Problem<'a> {
    fn new(
        source: &[(u8, u8, u8)],
        target: &'a [(u8, u8, u8)],
        weights: &'a [i64],
        side: usize,
        spatial_weight: i64,
    ) -> Self {
        let n = source.len();
        let mut p = Problem {
            side,
            spatial_weight,
            target,
            weights,
            col_x: Vec::with_capacity(n),
            col_y: Vec::with_capacity(n),
            col_r: Vec::with_capacity(n),
            col_g: Vec::with_capacity(n),
            col_b: Vec::with_capacity(n),
        };
        for (j, &(r, g, b)) in source.iter().enumerate() {
            p.col_x.push((j % side) as i32);
            p.col_y.push((j / side) as i32);
            p.col_r.push(r as i32);
            p.col_g.push(g as i32);
            p.col_b.push(b as i32);
        }
        p
    }

    #[inline]
    fn row_ctx(&self, row: usize) -> RowCtx {
        let (r, g, b) = self.target[row];
        RowCtx {
            x: (row % self.side) as i32,
            y: (row / self.side) as i32,
            r: r as i32,
            g: g as i32,
            b: b as i32,
            color_weight: self.weights[row],
        }
    }

}

/// Same value as `heuristic(..)` for (row, column), with the row-dependent parts
/// already in `c`. Kept in i64 so results are bit-identical to the old code.
#[inline(always)]
fn cost_of(c: &RowCtx, spatial_weight: i64, x: i32, y: i32, r: i32, g: i32, b: i32) -> i64 {
    let dx = (c.x - x) as i64;
    let dy = (c.y - y) as i64;
    let dr = (c.r - r) as i64;
    let dg = (c.g - g) as i64;
    let db = (c.b - b) as i64;
    let spatial = dx * dx + dy * dy;
    let color = dr * dr + dg * dg + db * db;
    color * c.color_weight + (spatial * spatial_weight).pow(2)
}

/// Relax the columns `start .. start + minv.len()` against one new row.
///
/// For every column not yet in the search tree this lowers `minv[j]` (and records
/// where it came from in `way[j]`) if this row offers a cheaper reduced cost, and
/// returns the smallest `(minv, column)` seen, preferring the lowest column on ties.
///
/// `offset` is `acc - u[row]`; see the lazy-dual notes in `solve`.
#[inline]
fn relax_range(
    prob: &Problem<'_>,
    ctx: &RowCtx,
    offset: i64,
    prev_col: u32,
    v: &[i64],
    used: &[bool],
    start: usize,
    minv: &mut [i64],
    way: &mut [u32],
) -> (i64, usize) {
    // Re-slice everything to exactly this chunk so the loop has no bounds checks.
    let len = minv.len();
    let end = start + len;
    let (cx, cy) = (&prob.col_x[start..end], &prob.col_y[start..end]);
    let (cr, cg, cb) = (
        &prob.col_r[start..end],
        &prob.col_g[start..end],
        &prob.col_b[start..end],
    );
    let (v, used, way) = (&v[start..end], &used[start..end], &mut way[..len]);
    let sw = prob.spatial_weight;

    let mut best = (INF, usize::MAX);
    for k in 0..len {
        if used[k] {
            continue;
        }
        let cur = cost_of(ctx, sw, cx[k], cy[k], cr[k], cg[k], cb[k]) - v[k] + offset;
        if cur < minv[k] {
            minv[k] = cur;
            way[k] = prev_col;
        }
        // Strict `<` keeps the first (lowest) column among equal minima.
        if minv[k] < best.0 {
            best = (minv[k], start + k);
        }
    }
    best
}

/// Scratch state for one run of the solver, plus the (optional) thread count.
struct Solver<'a> {
    prob: &'a Problem<'a>,
    /// Column potentials (only brought up to date at the end of each augmentation).
    v: Vec<i64>,
    /// Row potentials.
    u: Vec<i64>,
    col_row: Vec<u32>,
    row_col: Vec<u32>,
    // Per-search scratch.
    minv: Vec<i64>,
    way: Vec<u32>,
    used: Vec<bool>,
    #[cfg(not(target_arch = "wasm32"))]
    chunk: usize, // 0 = run sequentially
}

impl<'a> Solver<'a> {
    fn new(prob: &'a Problem<'a>, n_rows: usize) -> Self {
        let n_cols = prob.col_x.len();
        #[cfg(not(target_arch = "wasm32"))]
        let chunk = {
            let threads = rayon::current_num_threads();
            if threads > 1 && n_cols >= MIN_COLS_FOR_PARALLEL {
                // Round up to a multiple of 64 so chunks don't share cache lines.
                (n_cols.div_ceil(threads).max(MIN_CHUNK)).next_multiple_of(64)
            } else {
                0
            }
        };
        Solver {
            prob,
            v: vec![0; n_cols],
            u: vec![0; n_rows],
            col_row: vec![NONE; n_cols],
            row_col: vec![NONE; n_rows],
            minv: vec![INF; n_cols],
            way: vec![NONE; n_cols],
            used: vec![false; n_cols],
            #[cfg(not(target_arch = "wasm32"))]
            chunk,
        }
    }

    /// One relaxation step over every column, in parallel when worthwhile.
    fn relax(&mut self, ctx: &RowCtx, offset: i64, prev_col: u32) -> (i64, usize) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.chunk > 0 {
            use rayon::prelude::*;
            let chunk = self.chunk;
            let prob = self.prob;
            let v = &self.v[..];
            let used = &self.used[..];
            return self
                .minv
                .par_chunks_mut(chunk)
                .zip(self.way.par_chunks_mut(chunk))
                .enumerate()
                .map(|(ci, (minv, way))| {
                    relax_range(prob, ctx, offset, prev_col, v, used, ci * chunk, minv, way)
                })
                // Lexicographic min on (value, column) == what the sequential scan picks.
                .reduce(|| (INF, usize::MAX), |a, b| if b < a { b } else { a });
        }

        relax_range(
            self.prob,
            ctx,
            offset,
            prev_col,
            &self.v,
            &self.used,
            0,
            &mut self.minv,
            &mut self.way,
        )
    }

    /// Find a minimum-cost augmenting path from the free row `root` and apply it.
    ///
    /// This is Dijkstra over reduced costs `c(i,j) - u[i] - v[j]`. The textbook
    /// version shifts the potentials of every column on each step; instead we keep
    /// `acc`, the total shift so far, and store `minv[j]` as an absolute value
    /// (`true_minv[j] + acc`). A column's potential then only needs fixing once,
    /// at the end, by the amount of shift that happened after it joined the tree.
    fn augment(&mut self, root: usize) {
        self.minv.fill(INF);
        self.used.fill(false);

        // Columns that joined the tree, with the value of `acc` when they did.
        let mut visited: Vec<(usize, i64)> = Vec::new();

        let mut acc: i64 = 0;
        let mut row = root;
        let mut prev_col = NONE;
        let end_col = loop {
            let ctx = self.prob.row_ctx(row);
            let offset = acc - self.u[row];
            let (m, j) = self.relax(&ctx, offset, prev_col);
            debug_assert!(j != usize::MAX, "no unused column left to relax");

            acc = m; // everything in the tree is shifted by (m - acc_old)
            if self.col_row[j] == NONE {
                break j; // free column: augmenting path found
            }
            self.used[j] = true;
            visited.push((j, acc));
            row = self.col_row[j] as usize;
            prev_col = j as u32;
        };

        // Bring the potentials of the tree up to date.
        self.u[root] += acc;
        for &(j, a) in &visited {
            let shift = acc - a;
            self.v[j] -= shift;
            self.u[self.col_row[j] as usize] += shift;
        }

        // Flip the matching along the path end_col <- ... <- root.
        let mut j = end_col;
        loop {
            let prev = self.way[j];
            let r = if prev == NONE {
                root
            } else {
                self.col_row[prev as usize] as usize
            };
            self.col_row[j] = r as u32;
            self.row_col[r] = j as u32;
            if prev == NONE {
                break;
            }
            j = prev as usize;
        }
    }
}

pub fn process_optimal<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_img = image::ImageBuffer::from_vec(
        unprocessed.width,
        unprocessed.height,
        unprocessed.source_img.clone(),
    )
    .unwrap();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;

    let side = settings.sidelen as usize;
    let nx = target_pixels.len();
    let prob = Problem::new(
        &source_pixels,
        &target_pixels,
        &weights,
        side,
        settings.proximity_importance,
    );
    assert!(nx <= prob.col_x.len(), "more target cells than source cells");

    let mut solver = Solver::new(&prob, nx);

    for root in 0..nx {
        #[cfg(not(target_arch = "wasm32"))]
        if cancel.load(Ordering::Relaxed) {
            tx.send(ProgressMsg::Cancelled);
            return Ok(());
        }

        solver.augment(root);

        if root % 100 == 0 {
            tx.send(ProgressMsg::Progress(root as f32 / nx as f32));

            // Rows after `root` are not matched yet; show them as column 0 like before.
            let assignments = solver
                .row_col
                .iter()
                .map(|&c| if c == NONE { 0 } else { c as usize })
                .collect::<Vec<_>>();
            tx.send(ProgressMsg::UpdatePreview {
                width: settings.sidelen,
                height: settings.sidelen,
                data: make_new_img(&source_pixels, &assignments, settings.sidelen),
            });
        }
    }

    let assignments = solver
        .row_col
        .iter()
        .map(|&c| c as usize)
        .collect::<Vec<_>>();

    tx.send(ProgressMsg::Done(Preset {
        inner: UnprocessedPreset {
            name: unprocessed.name,
            width: settings.sidelen,
            height: settings.sidelen,
            source_img: source_pixels
                .into_iter()
                .flat_map(|(r, g, b)| [r, g, b])
                .collect(),
        },
        assignments,
    }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::calculate::heuristic;
    use pathfinding::prelude::{Matrix, kuhn_munkres};

    type Rgb = (u8, u8, u8);

    fn random_instance(side: usize, seed: u64) -> (Vec<Rgb>, Vec<Rgb>, Vec<i64>) {
        let mut rng = frand::Rand::with_seed(seed);
        let mut px = |n: usize| -> Vec<Rgb> {
            (0..n)
                .map(|_| {
                    (
                        rng.gen_range(0..256u32) as u8,
                        rng.gen_range(0..256u32) as u8,
                        rng.gen_range(0..256u32) as u8,
                    )
                })
                .collect()
        };
        let n = side * side;
        let (source, target) = (px(n), px(n));
        let weights = (0..n).map(|_| rng.gen_range(1..256u32) as i64).collect();
        (source, target, weights)
    }

    fn total_cost(
        assignment: &[usize],
        source: &[Rgb],
        target: &[Rgb],
        weights: &[i64],
        side: usize,
        prox: i64,
    ) -> i64 {
        assignment
            .iter()
            .enumerate()
            .map(|(row, &col)| {
                heuristic(
                    ((row % side) as u16, (row / side) as u16),
                    ((col % side) as u16, (col / side) as u16),
                    target[row],
                    source[col],
                    weights[row],
                    prox,
                )
            })
            .sum()
    }

    fn solve(prob: &Problem, n: usize, chunk: Option<usize>) -> Vec<usize> {
        let mut solver = Solver::new(prob, n);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(c) = chunk {
            solver.chunk = c;
        }
        let _ = chunk;
        for root in 0..n {
            solver.augment(root);
        }
        solver.row_col.iter().map(|&c| c as usize).collect()
    }

    fn is_permutation(a: &[usize]) -> bool {
        let mut seen = vec![false; a.len()];
        a.iter().all(|&c| c < a.len() && !std::mem::replace(&mut seen[c], true))
    }

    /// Same optimum as the old pathfinding-based implementation.
    #[test]
    fn matches_reference_kuhn_munkres() {
        for (side, prox) in [(2, 13), (3, 13), (5, 1), (6, 13), (8, 40), (10, 0)] {
            for seed in 0..4 {
                let (source, target, weights) = random_instance(side, seed * 31 + side as u64);
                let n = side * side;
                let prob = Problem::new(&source, &target, &weights, side, prox);
                let ours = solve(&prob, n, None);
                assert!(is_permutation(&ours));

                let matrix = Matrix::from_vec(
                    n,
                    n,
                    (0..n * n)
                        .map(|k| {
                            let (row, col) = (k / n, k % n);
                            -heuristic(
                                ((row % side) as u16, (row / side) as u16),
                                ((col % side) as u16, (col / side) as u16),
                                target[row],
                                source[col],
                                weights[row],
                                prox,
                            )
                        })
                        .collect(),
                )
                .unwrap();
                let (best, _) = kuhn_munkres(&matrix);

                assert_eq!(
                    total_cost(&ours, &source, &target, &weights, side, prox),
                    -best,
                    "side={side} prox={prox} seed={seed}"
                );
            }
        }
    }

    /// Splitting the scan over threads must not change the answer at all.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_scan_is_identical_to_sequential() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for side in [7, 12, 20] {
            let (source, target, weights) = random_instance(side, 99 + side as u64);
            let n = side * side;
            let prob = Problem::new(&source, &target, &weights, side, 13);
            let seq = solve(&prob, n, None);
            for chunk in [1, 5, 16, 64] {
                let par = pool.install(|| solve(&prob, n, Some(chunk)));
                assert_eq!(seq, par, "side={side} chunk={chunk}");
            }
        }
    }

    /// Rectangular case from the old code's assert: fewer rows than columns is fine.
    #[test]
    fn fewer_rows_than_columns() {
        let side = 4;
        let (source, target, weights) = random_instance(side, 5);
        let prob = Problem::new(&source, &target[..10], &weights, side, 13);
        let a = solve(&prob, 10, None);
        assert!(a.iter().all(|&c| c < 16));
        let mut s = a.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 10);
    }
}

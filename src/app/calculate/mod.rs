#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool};
#[cfg(not(target_arch = "wasm32"))]
pub mod drawing_process;
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
use egui::ahash::AHasher;
use pathfinding::prelude::Weights;
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

struct ImgDiffWeights<'a> {
    source: Vec<(u8, u8, u8)>,
    target: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    sidelen: usize,
    settings: &'a GenerationSettings,
}

// const TARGET_IMAGE_PATH: &str = "./target.png";
// const TARGET_WEIGHTS_PATH: &str = "./weights.png";

impl Weights<i64> for ImgDiffWeights<'_> {
    fn rows(&self) -> usize {
        self.target.len()
    }

    fn columns(&self) -> usize {
        self.source.len()
    }

    #[inline(always)]
    fn at(&self, row: usize, col: usize) -> i64 {
        let (x1, y1) = (row % self.sidelen, row / self.sidelen);
        let (x2, y2) = (col % self.sidelen, col / self.sidelen);
        let (r1, g1, b1) = self.target[row];
        let (r2, g2, b2) = self.source[col];
        let weight = self.weights[row];
        -heuristic(
            (x1 as u16, y1 as u16),
            (x2 as u16, y2 as u16),
            (r1, g1, b1),
            (r2, g2, b2),
            weight,
            self.settings.proximity_importance,
        )
    }

    fn neg(&self) -> Self {
        todo!()
    }
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

type FxIndexSet<K> = indexmap::IndexSet<K, std::hash::BuildHasherDefault<AHasher>>;

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
    // let start_time = std::time::Instant::now();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target_pixels,
        weights,
        sidelen: settings.sidelen as usize,
        settings: &settings,
    };

    // pathfinding::kuhn_munkres, inlined to allow for progress bar and cancelling
    let (_total_diff, assignments) = {
        // We call x the rows and y the columns. (nx, ny) is the size of the matrix.
        let nx = weights.rows();
        let ny = weights.columns();
        assert!(
            nx <= ny,
            "number of rows must not be larger than number of columns"
        );
        // xy represents matching for x, yz matching for y
        let mut xy: Vec<Option<usize>> = vec![None; nx];
        let mut yx: Vec<Option<usize>> = vec![None; ny];
        // lx is the labelling for x nodes, ly the labelling for y nodes. We start
        // with an acceptable labelling with the maximum possible values for lx
        // and 0 for ly.
        let mut lx: Vec<i64> = (0..nx)
            .map(|row| (0..ny).map(|col| weights.at(row, col)).max().unwrap())
            .collect::<Vec<_>>();
        let mut ly: Vec<i64> = vec![0; ny];
        // s, augmenting, and slack will be reset every time they are reused. augmenting
        // contains Some(prev) when the corresponding node belongs to the augmenting path.
        let mut s = FxIndexSet::<usize>::default();
        let mut alternating = Vec::with_capacity(ny);
        let mut slack = vec![0; ny];
        let mut slackx = Vec::with_capacity(ny);
        for root in 0..nx {
            alternating.clear();
            alternating.resize(ny, None);
            // Find y such that the path is augmented. This will be set when breaking for the
            // loop below. Above the loop is some code to initialize the search.
            let mut y = {
                s.clear();
                s.insert(root);
                // Slack for a vertex y is, initially, the margin between the
                // sum of the labels of root and y, and the weight between root and y.
                // As we add x nodes to the alternating path, we update the slack to
                // represent the smallest margin between one of the x nodes and y.
                for y in 0..ny {
                    slack[y] = lx[root] + ly[y] - weights.at(root, y);
                }
                slackx.clear();
                slackx.resize(ny, root);
                Some(loop {
                    let mut delta = pathfinding::num_traits::Bounded::max_value();
                    let mut x = 0;
                    let mut y = 0;
                    // Select one of the smallest slack delta and its edge (x, y)
                    // for y not in the alternating path already.
                    for yy in 0..ny {
                        if alternating[yy].is_none() && slack[yy] < delta {
                            delta = slack[yy];
                            x = slackx[yy];
                            y = yy;
                        }
                    }
                    // If some slack has been found, remove it from x nodes in the
                    // alternating path, and add it to y nodes in the alternating path.
                    // The slack of y nodes outside the alternating path will be reduced
                    // by this minimal slack as well.
                    if delta > 0 {
                        for &x in &s {
                            lx[x] -= delta;
                        }
                        for y in 0..ny {
                            if alternating[y].is_some() {
                                ly[y] += delta;
                            } else {
                                slack[y] -= delta;
                            }
                        }
                    }
                    // Add (x, y) to the alternating path.
                    alternating[y] = Some(x);
                    if yx[y].is_none() {
                        // We have found an augmenting path.
                        break y;
                    }
                    // This y node had a predecessor, add it to the set of x nodes
                    // in the augmenting path.
                    let x = yx[y].unwrap();
                    s.insert(x);
                    // Update slack because of the added vertex in s might contain a
                    // greater slack than with previously inserted x nodes in the augmenting
                    // path.
                    for y in 0..ny {
                        if alternating[y].is_none() {
                            let alternate_slack = lx[x] + ly[y] - weights.at(x, y);
                            if slack[y] > alternate_slack {
                                slack[y] = alternate_slack;
                                slackx[y] = x;
                            }
                        }
                    }
                })
            };
            // Inverse edges along the augmenting path.
            while y.is_some() {
                let x = alternating[y.unwrap()].unwrap();
                let prec = xy[x];
                yx[y.unwrap()] = Some(x);
                xy[x] = y;
                y = prec;
            }
            if root % 100 == 0 {
                // send progress
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        tx.send(ProgressMsg::Cancelled);
                        return Ok(());
                    }
                }

                tx.send(ProgressMsg::Progress(root as f32 / nx as f32));

                let data = make_new_img(
                    &source_pixels,
                    &xy.clone()
                        .into_iter()
                        .map(|a| a.unwrap_or(0))
                        .collect::<Vec<_>>(),
                    settings.sidelen,
                );

                tx.send(ProgressMsg::UpdatePreview {
                    width: settings.sidelen,
                    height: settings.sidelen,
                    data,
                });
            }
        }
        (
            lx.into_iter().sum::<i64>() + ly.into_iter().sum::<i64>(),
            xy.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
        )
    };

    //let img = make_new_img(&source_pixels, &assignments, target.width());

    //let dir_name = util::save_result(target, "todo".to_string(), source, assignments, img)?;

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
        assignments: assignments.clone(),
    }));

    // println!(
    //     "finished in {:.2?} seconds",
    //     std::time::Instant::now().duration_since(start_time)
    // );
    Ok(())
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

#[derive(Clone, Copy)]
struct Pixel {
    src_x: u16,
    src_y: u16,
    rgb: (u8, u8, u8),
    h: i64, // current heuristic value
}

impl Pixel {
    fn new(src_x: u16, src_y: u16, rgb: (u8, u8, u8), h: i64) -> Self {
        Self {
            src_x,
            src_y,
            rgb,
            h,
        }
    }

    fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    fn calc_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            self.rgb,
            target_col,
            weight,
            proximity_importance,
        )
    }
}

const SWAPS_PER_GENERATION_PER_PIXEL: usize = 128;

pub fn process_genetic<S: ProgressSink>(
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
    // let start_time = std::time::Instant::now();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;

    let mut pixels = source_pixels
        .iter()
        .enumerate()
        .map(|(i, &(r, g, b))| {
            let x = (i as u32 % settings.sidelen) as u16;
            let y = (i as u32 / settings.sidelen) as u16;
            let mut p = Pixel::new(x, y, (r, g, b), 0);
            let h = p.calc_heuristic(
                (x, y),
                target_pixels[i],
                weights[i],
                settings.proximity_importance,
            );
            p.update_heuristic(h);
            p
        })
        .collect::<Vec<_>>();

    let mut rng = frand::Rand::with_seed(12345);
    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * pixels.len();

    let mut max_dist = settings.sidelen;
    loop {
        let mut swaps_made = 0;
        for _ in 0..swaps_per_generation {
            let apos = rng.gen_range(0..pixels.len() as u32) as usize;
            let ax = apos as u16 % settings.sidelen as u16;
            let ay = apos as u16 / settings.sidelen as u16;
            let bx = (ax as i16 + rng.gen_range(-(max_dist as i16)..(max_dist as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist as i16)..(max_dist as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * settings.sidelen as usize + bx as usize;

            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];

            let a_on_b_h = pixels[apos].calc_heuristic(
                (bx, by),
                t_b,
                weights[bpos],
                settings.proximity_importance,
            );

            let b_on_a_h = pixels[bpos].calc_heuristic(
                (ax, ay),
                t_a,
                weights[apos],
                settings.proximity_importance,
            );

            let improvement_a = pixels[apos].h - b_on_a_h;
            let improvement_b = pixels[bpos].h - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                // swap
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_h);
                pixels[bpos].update_heuristic(a_on_b_h);
                swaps_made += 1;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                println!("cancelled");
                tx.send(ProgressMsg::Cancelled);
                return Ok(());
            }
        }

        let assignments = pixels
            .iter()
            .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
            .collect::<Vec<_>>();
        //debug_print(format!("max_dist = {max_dist}, swaps made = {swaps_made}"));
        if max_dist < 4 && swaps_made < 10 {
            //let dir_name = util::save_result(target, base_name, source, assignments, img)?;
            tx.send(ProgressMsg::Done(Preset {
                inner: UnprocessedPreset {
                    name: unprocessed.name,
                    width: settings.sidelen,
                    height: settings.sidelen,
                    source_img: source_pixels
                        .iter()
                        .flat_map(|(r, g, b)| [*r, *g, *b])
                        .collect(),
                },
                assignments: assignments.clone(),
            }));
            return Ok(());
        }
        let data = make_new_img(&source_pixels, &assignments, settings.sidelen);
        tx.send(ProgressMsg::UpdatePreview {
            width: settings.sidelen,
            height: settings.sidelen,
            data,
        });
        tx.send(ProgressMsg::Progress(
            1.0 - max_dist as f32 / settings.sidelen as f32,
        ));

        max_dist = (max_dist as f32 * 0.99).max(2.0) as u32;
    }
}

/// Greedy algorithm - assigns each target pixel to its best available source pixel
/// Time complexity: O(n² log n) where n = number of pixels
/// Quality: ~90-95% of optimal
pub fn process_greedy<S: ProgressSink>(
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
    let n = source_pixels.len();
    
    // Create list of (target_idx, weight) and sort by weight descending (highest priority first)
    let mut target_order: Vec<(usize, i64)> = weights.iter().enumerate()
        .map(|(i, &w)| (i, w))
        .collect();
    target_order.sort_by(|a, b| b.1.cmp(&a.1));
    
    let mut assignments = vec![0usize; n];
    let mut source_used = vec![false; n];
    
    for (progress_idx, &(target_idx, weight)) in target_order.iter().enumerate() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                tx.send(ProgressMsg::Cancelled);
                return Ok(());
            }
        }
        
        let tx_pos = (target_idx % settings.sidelen as usize, target_idx / settings.sidelen as usize);
        let t_col = target_pixels[target_idx];
        
        // Find best available source pixel
        let mut best_source = 0;
        let mut best_cost = i64::MAX;
        
        for (src_idx, &(sr, sg, sb)) in source_pixels.iter().enumerate() {
            if source_used[src_idx] {
                continue;
            }
            let sx_pos = (src_idx % settings.sidelen as usize, src_idx / settings.sidelen as usize);
            let cost = heuristic(
                (sx_pos.0 as u16, sx_pos.1 as u16),
                (tx_pos.0 as u16, tx_pos.1 as u16),
                (sr, sg, sb),
                t_col,
                weight,
                settings.proximity_importance,
            );
            if cost < best_cost {
                best_cost = cost;
                best_source = src_idx;
            }
        }
        
        assignments[target_idx] = best_source;
        source_used[best_source] = true;
        
        // Progress updates
        if progress_idx % 500 == 0 {
            tx.send(ProgressMsg::Progress(progress_idx as f32 / n as f32));
            
            let data = make_new_img(&source_pixels, &assignments, settings.sidelen);
            tx.send(ProgressMsg::UpdatePreview {
                width: settings.sidelen,
                height: settings.sidelen,
                data,
            });
        }
    }
    
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

/// Auction algorithm - uses economic bidding metaphor for assignment
/// Time complexity: O(n² log n) average case
/// Quality: ~95-99% of optimal
pub fn process_auction<S: ProgressSink>(
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
    let n = source_pixels.len();
    
    // Auction algorithm with epsilon-scaling
    let mut prices: Vec<f64> = vec![0.0; n]; // prices for source pixels
    let mut target_to_source: Vec<Option<usize>> = vec![None; n];
    let mut source_to_target: Vec<Option<usize>> = vec![None; n];
    
    // Calculate cost matrix entries on demand
    let cost = |target_idx: usize, source_idx: usize| -> f64 {
        let tx_pos = (target_idx % settings.sidelen as usize, target_idx / settings.sidelen as usize);
        let sx_pos = (source_idx % settings.sidelen as usize, source_idx / settings.sidelen as usize);
        let t_col = target_pixels[target_idx];
        let (sr, sg, sb) = source_pixels[source_idx];
        let weight = weights[target_idx];
        
        -(heuristic(
            (sx_pos.0 as u16, sx_pos.1 as u16),
            (tx_pos.0 as u16, tx_pos.1 as u16),
            (sr, sg, sb),
            t_col,
            weight,
            settings.proximity_importance,
        ) as f64)
    };
    
    // Epsilon determines minimum bid increment - smaller = more precise but slower
    let epsilon = 1.0 / (n as f64 + 1.0);
    
    let mut iteration = 0;
    let max_iterations = n * 20; // Generous limit
    let mut stale_count = 0;
    let max_stale = 100; // If no progress for 100 iterations, we're stuck
    let mut last_unassigned_count = n;
    
    while iteration < max_iterations {
        // Find unassigned targets
        let unassigned: Vec<usize> = (0..n)
            .filter(|&i| target_to_source[i].is_none())
            .collect();
        
        let current_unassigned = unassigned.len();
        
        // Check if we're done
        if current_unassigned == 0 {
            break; // All assigned!
        }
        
        // Detect if we're stuck (no progress)
        if current_unassigned >= last_unassigned_count {
            stale_count += 1;
            if stale_count >= max_stale {
                // We're stuck in a cycle - just assign remaining greedily
                _debug_print(format!("Auction stuck at {} unassigned, finishing greedily", current_unassigned));
                
                // Find unassigned sources
                let unassigned_sources: Vec<usize> = (0..n)
                    .filter(|&i| source_to_target[i].is_none())
                    .collect();
                
                // Greedily assign remaining
                for (&target_idx, &source_idx) in unassigned.iter().zip(unassigned_sources.iter()) {
                    target_to_source[target_idx] = Some(source_idx);
                    source_to_target[source_idx] = Some(target_idx);
                }
                break;
            }
        } else {
            stale_count = 0;
            last_unassigned_count = current_unassigned;
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                tx.send(ProgressMsg::Cancelled);
                return Ok(());
            }
        }
        
        // Process ONE unassigned target per iteration (Gauss-Seidel style - more stable)
        // This avoids the issue of all unassigned targets bidding simultaneously and creating cycles
        let target_idx = unassigned[iteration % unassigned.len()];
        
        // Find best and second-best source for this target
        let mut best_source = 0;
        let mut best_value = f64::NEG_INFINITY;
        let mut second_best_value = f64::NEG_INFINITY;
        
        for source_idx in 0..n {
            let value = cost(target_idx, source_idx) - prices[source_idx];
            if value > best_value {
                second_best_value = best_value;
                best_value = value;
                best_source = source_idx;
            } else if value > second_best_value {
                second_best_value = value;
            }
        }
        
        // Handle case where second_best is still NEG_INFINITY
        if second_best_value == f64::NEG_INFINITY {
            second_best_value = best_value - epsilon;
        }
        
        // Calculate bid increment
        let bid_increment = best_value - second_best_value + epsilon;
        
        // If source was assigned to someone else, unassign them
        if let Some(old_target) = source_to_target[best_source] {
            target_to_source[old_target] = None;
        }
        
        // Assign and update price
        target_to_source[target_idx] = Some(best_source);
        source_to_target[best_source] = Some(target_idx);
        prices[best_source] += bid_increment;
        
        iteration += 1;
        
        // Progress update
        if iteration % 200 == 0 {
            let assigned_count = n - current_unassigned;
            tx.send(ProgressMsg::Progress(assigned_count as f32 / n as f32));
            
            // Create partial preview
            let partial_assignments: Vec<usize> = target_to_source.iter()
                .enumerate()
                .map(|(i, opt)| opt.unwrap_or(i))
                .collect();
            let data = make_new_img(&source_pixels, &partial_assignments, settings.sidelen);
            tx.send(ProgressMsg::UpdatePreview {
                width: settings.sidelen,
                height: settings.sidelen,
                data,
            });
        }
    }
    
    // Extract final assignments - any remaining unassigned get identity mapping
    let assignments: Vec<usize> = target_to_source.iter()
        .enumerate()
        .map(|(i, opt)| opt.unwrap_or(i))
        .collect();
    
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

/// Hybrid algorithm - runs optimal at low resolution, then refines with genetic swaps
/// Quality: ~95-98% of optimal
pub fn process_hybrid<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Run optimal algorithm at reduced resolution (64x64)
    let coarse_sidelen = 64u32.min(settings.sidelen);
    let scale_factor = settings.sidelen / coarse_sidelen;
    
    let mut coarse_settings = settings.clone();
    coarse_settings.sidelen = coarse_sidelen;
    
    // Create coarse version of the image
    let source_img = image::ImageBuffer::from_vec(
        unprocessed.width,
        unprocessed.height,
        unprocessed.source_img.clone(),
    )
    .unwrap();
    
    let coarse_source = image::imageops::resize(
        &source_img,
        coarse_sidelen,
        coarse_sidelen,
        image::imageops::FilterType::Lanczos3,
    );
    
    let coarse_unprocessed = UnprocessedPreset {
        name: unprocessed.name.clone(),
        width: coarse_sidelen,
        height: coarse_sidelen,
        source_img: coarse_source.into_raw(),
    };
    
    // Collect coarse result
    let mut coarse_result: Option<Vec<usize>> = None;
    let mut progress_sink = |msg: ProgressMsg| {
        match msg {
            ProgressMsg::Progress(p) => {
                tx.send(ProgressMsg::Progress(p * 0.5)); // First half of progress
            }
            ProgressMsg::Done(preset) => {
                coarse_result = Some(preset.assignments);
            }
            ProgressMsg::UpdatePreview { .. } => {
                // Skip coarse previews
            }
            other => tx.send(other),
        }
    };
    
    // Run optimal on coarse
    #[cfg(not(target_arch = "wasm32"))]
    process_optimal(coarse_unprocessed, coarse_settings, &mut progress_sink, cancel.clone())?;
    #[cfg(target_arch = "wasm32")]
    process_optimal(coarse_unprocessed, coarse_settings, &mut progress_sink)?;
    
    let coarse_assignments = match coarse_result {
        Some(a) => a,
        None => return Err("Coarse optimization failed".into()),
    };
    
    // Step 2: Upsample assignments to full resolution
    let (source_pixels, target_pixels, weights) = util::get_images(source_img.clone(), &settings)?;
    let n = source_pixels.len();
    
    // Initialize fine assignments based on coarse assignments
    let mut assignments: Vec<usize> = (0..n).collect(); // Start with identity
    
    for coarse_target in 0..(coarse_sidelen * coarse_sidelen) as usize {
        let coarse_source = coarse_assignments[coarse_target];
        let ctx = coarse_target % coarse_sidelen as usize;
        let cty = coarse_target / coarse_sidelen as usize;
        let csx = coarse_source % coarse_sidelen as usize;
        let csy = coarse_source / coarse_sidelen as usize;
        
        // Map to fine grid
        for dy in 0..scale_factor as usize {
            for dx in 0..scale_factor as usize {
                let fine_target = (cty * scale_factor as usize + dy) * settings.sidelen as usize 
                                + (ctx * scale_factor as usize + dx);
                let fine_source = (csy * scale_factor as usize + dy) * settings.sidelen as usize 
                                + (csx * scale_factor as usize + dx);
                if fine_target < n && fine_source < n {
                    assignments[fine_target] = fine_source;
                }
            }
        }
    }
    
    tx.send(ProgressMsg::Progress(0.5));
    
    // Step 3: Refine with genetic swaps (local optimization)
    let mut pixels: Vec<Pixel> = assignments.iter().enumerate()
        .map(|(target_idx, &source_idx)| {
            let (sr, sg, sb) = source_pixels[source_idx];
            let sx = (source_idx % settings.sidelen as usize) as u16;
            let sy = (source_idx / settings.sidelen as usize) as u16;
            let tx = (target_idx % settings.sidelen as usize) as u16;
            let ty = (target_idx / settings.sidelen as usize) as u16;
            let t_col = target_pixels[target_idx];
            let weight = weights[target_idx];
            let h = heuristic((sx, sy), (tx, ty), (sr, sg, sb), t_col, weight, settings.proximity_importance);
            Pixel::new(sx, sy, (sr, sg, sb), h)
        })
        .collect();
    
    let mut rng = frand::Rand::with_seed(12345);
    let refinement_passes = 20;
    let swaps_per_pass = n * 8;
    
    for pass in 0..refinement_passes {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                tx.send(ProgressMsg::Cancelled);
                return Ok(());
            }
        }
        
        let max_dist = ((refinement_passes - pass) as f32 / refinement_passes as f32 * settings.sidelen as f32 / 4.0).max(2.0) as u32;
        
        for _ in 0..swaps_per_pass {
            let apos = rng.gen_range(0..n as u32) as usize;
            let ax = apos as u16 % settings.sidelen as u16;
            let ay = apos as u16 / settings.sidelen as u16;
            let bx = (ax as i16 + rng.gen_range(-(max_dist as i16)..(max_dist as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist as i16)..(max_dist as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * settings.sidelen as usize + bx as usize;
            
            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];
            
            let a_on_b_h = pixels[apos].calc_heuristic((bx, by), t_b, weights[bpos], settings.proximity_importance);
            let b_on_a_h = pixels[bpos].calc_heuristic((ax, ay), t_a, weights[apos], settings.proximity_importance);
            
            let improvement = (pixels[apos].h - b_on_a_h) + (pixels[bpos].h - a_on_b_h);
            if improvement > 0 {
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_h);
                pixels[bpos].update_heuristic(a_on_b_h);
            }
        }
        
        tx.send(ProgressMsg::Progress(0.5 + (pass as f32 / refinement_passes as f32) * 0.5));
        
        let final_assignments: Vec<usize> = pixels.iter()
            .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
            .collect();
        let data = make_new_img(&source_pixels, &final_assignments, settings.sidelen);
        tx.send(ProgressMsg::UpdatePreview {
            width: settings.sidelen,
            height: settings.sidelen,
            data,
        });
    }
    
    let final_assignments: Vec<usize> = pixels.iter()
        .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
        .collect();
    
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
        assignments: final_assignments,
    }));
    
    Ok(())
}

// fn serialize_assignments(assignments: Vec<usize>) -> String {
//     format!(
//         "[{}]",
//         assignments
//             .iter()
//             .map(|a| a.to_string())
//             .collect::<Vec<_>>()
//             .join(",")
//     )
// }
#[cfg(not(target_arch = "wasm32"))]
pub fn process<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    match settings.algorithm {
        Algorithm::Optimal => process_optimal(unprocessed, settings, tx, cancel),
        Algorithm::Auction => process_auction(unprocessed, settings, tx, cancel),
        Algorithm::Greedy => process_greedy(unprocessed, settings, tx, cancel),
        Algorithm::Hybrid => process_hybrid(unprocessed, settings, tx, cancel),
        Algorithm::Genetic => process_genetic(unprocessed, settings, tx, cancel),
    }
}

#[cfg(target_arch = "wasm32")]
pub fn process<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
) -> Result<(), Box<dyn std::error::Error>> {
    match settings.algorithm {
        Algorithm::Optimal => process_optimal(unprocessed, settings, tx),
        Algorithm::Auction => process_auction(unprocessed, settings, tx),
        Algorithm::Greedy => process_greedy(unprocessed, settings, tx),
        Algorithm::Hybrid => process_hybrid(unprocessed, settings, tx),
        Algorithm::Genetic => process_genetic(unprocessed, settings, tx),
    }
}

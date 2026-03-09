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
    Done {
        preset: Preset,
        elapsed_seconds: f32,
    },
    Error(String),
    Cancelled,
}

impl ProgressMsg {
    pub fn typ(&self) -> &'static str {
        match self {
            ProgressMsg::Progress(_) => "progress",
            ProgressMsg::UpdatePreview { .. } => "update_preview",
            ProgressMsg::UpdateAssignments(_) => "update_assignments",
            ProgressMsg::Done { .. } => "done",
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

    tx.send(ProgressMsg::Done {
        preset: Preset {
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
        },
        elapsed_seconds: 0.0,
    });

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

const SWAPS_PER_GENERATION_PER_PIXEL: usize = 8;

use std::time::Instant;

pub fn process_genetic<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
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
    let mut assignments = vec![0; pixels.len()];
    let positions: Vec<(u16, u16)> = (0..pixels.len())
    .map(|i| ((i as u32 % settings.sidelen) as u16, (i as u32 / settings.sidelen) as u16))
    .collect();

    let mut max_dist = settings.sidelen;
    loop {
        let mut swaps_made = 0;
        for _ in 0..swaps_per_generation {
            let apos = rng.gen_range(0..pixels.len() as u32) as usize;
            let (ax, ay) = positions[apos];
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
                let elapsed_time = now.elapsed();
                println!("Running morph() took {:.3} seconds.", elapsed_time.as_secs_f32());
                return Ok(());
            }
        }

        for (i, p) in pixels.iter().enumerate() {
            assignments[i] = p.src_y as usize * settings.sidelen as usize + p.src_x as usize;
        }
        //debug_print(format!("max_dist = {max_dist}, swaps made = {swaps_made}"));
        if max_dist < 4 && swaps_made < 10 {
            //let dir_name = util::save_result(target, base_name, source, assignments, img)?;
            let elapsed_time = now.elapsed();
            tx.send(ProgressMsg::Done {
                preset: Preset {
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
                },
                elapsed_seconds: elapsed_time.as_secs_f32(),
            });
            //println!("Running morph() took {:.3} seconds.", elapsed_time.as_secs_f32());
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
        Algorithm::Genetic => process_genetic(unprocessed, settings, tx),
    }
}
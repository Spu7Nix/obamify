use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

pub mod drawing_process;
pub mod util;

use egui::ahash::AHasher;
use image::GenericImageView;
use pathfinding::prelude::Weights;
use rand::Rng;
use std::sync::mpsc;

#[derive(Clone, Copy)]
pub enum Algorithm {
    Optimal,
    Genetic,
}

#[derive(Clone)]
pub struct GenerationSettings {
    pub proximity_importance: i64,
    pub algorithm: Algorithm,
    pub rescale: Option<u32>,
    pub custom_target_image: Option<PathBuf>,
    pub auto_resize: bool,
}

impl GenerationSettings {
    pub fn default() -> Self {
        Self {
            proximity_importance: 10, // 20
            algorithm: Algorithm::Genetic,
            rescale: None,
            custom_target_image: None,
            auto_resize: true,
        }
    }

    // pub fn quick_process() -> Self {
    //     Self {
    //         proximity_importance: 10,
    //         rescale: Some(64),
    //     }
    // }
}

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

struct ImgDiffWeights {
    source: Vec<(u8, u8, u8)>,
    target: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    sidelen: usize,
    settings: GenerationSettings,
}

const TARGET_IMAGE_PATH: &str = "./target.png";
const TARGET_WEIGHTS_PATH: &str = "./weights.png";

impl Weights<i64> for ImgDiffWeights {
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

pub enum ProgressMsg {
    Progress(f32),
    UpdatePreview(image::ImageBuffer<image::Rgb<u8>, Vec<u8>>),
    UpdateAssignments(Vec<usize>),
    Done(PathBuf), // result directory
    Error(String),
    Cancelled,
}

type FxIndexSet<K> = indexmap::IndexSet<K, std::hash::BuildHasherDefault<AHasher>>;

pub fn process_optimal<P: AsRef<Path>>(
    source_path: P,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let start_time = std::time::Instant::now();
    let (target, base_name, source, source_pixels, target_pixels, weights) =
        util::get_images(source_path, &settings)?;

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target_pixels,
        weights,
        sidelen: target.width() as usize,
        settings,
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
            if root.is_multiple_of(100) {
                // send progress
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    tx.send(ProgressMsg::Cancelled).unwrap();
                    return Ok(());
                }
                tx.send(ProgressMsg::Progress(root as f32 / nx as f32))?;

                let img = make_new_img(
                    &source_pixels,
                    &xy.clone()
                        .into_iter()
                        .map(|a| a.unwrap_or(0))
                        .collect::<Vec<_>>(),
                    target.width(),
                );

                tx.send(ProgressMsg::UpdatePreview(img))?;
            }
        }
        (
            lx.into_iter().sum::<i64>() + ly.into_iter().sum::<i64>(),
            xy.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
        )
    };

    let img = make_new_img(&source_pixels, &assignments, target.width());

    let dir_name = util::save_result(target, base_name, source, assignments, img)?;

    tx.send(ProgressMsg::Done(PathBuf::from(format!(
        "./presets/{}",
        dir_name
    ))))?;

    println!(
        "finished in {:.2?} seconds",
        std::time::Instant::now().duration_since(start_time)
    );

    Ok(())
}

fn make_new_img(
    source_pixels: &[(u8, u8, u8)],
    assignments: &[usize],
    sidelen: u32,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let mut img = image::ImageBuffer::new(sidelen, sidelen);

    for (target_idx, source_idx) in assignments.iter().enumerate() {
        let (x, y) = (
            (target_idx % sidelen as usize) as u32,
            (target_idx / sidelen as usize) as u32,
        );
        let (r, g, b) = source_pixels[*source_idx];
        img.put_pixel(x, y, image::Rgb([r, g, b]));
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

const SWAPS_PER_GENERATION: usize = 200000;

pub fn process_genetic<P: AsRef<Path>>(
    source_path: P,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let (target, base_name, source, source_pixels, target_pixels, weights) =
        util::get_images(source_path, &settings)?;

    let mut pixels = source_pixels
        .iter()
        .enumerate()
        .map(|(i, &(r, g, b))| {
            let x = (i as u32 % source.width()) as u16;
            let y = (i as u32 / source.width()) as u16;
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

    let mut rng = rand::thread_rng();
    let mut max_dist = target.width();
    loop {
        let mut swaps_made = 0;
        for _ in 0..SWAPS_PER_GENERATION {
            let apos = rng.gen_range(0..pixels.len());
            let ax = apos as u16 % target.width() as u16;
            let ay = apos as u16 / target.width() as u16;
            let bx = (ax as i16 + rng.gen_range(-(max_dist as i16)..=(max_dist as i16)))
                .clamp(0, target.width() as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist as i16)..=(max_dist as i16)))
                .clamp(0, target.width() as i16 - 1) as u16;
            let bpos = by as usize * target.width() as usize + bx as usize;

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
        let assignments = pixels
            .iter()
            .map(|p| p.src_y as usize * source.width() as usize + p.src_x as usize)
            .collect::<Vec<_>>();
        let img = make_new_img(&source_pixels, &assignments, target.width());
        if swaps_made < 10 || cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            let dir_name = util::save_result(target, base_name, source, assignments, img)?;
            tx.send(ProgressMsg::Done(PathBuf::from(format!(
                "./presets/{}",
                dir_name
            ))))?;
            return Ok(());
        }
        tx.send(ProgressMsg::UpdatePreview(img))?;
        tx.send(ProgressMsg::Progress(
            1.0 - max_dist as f32 / target.width() as f32,
        ))?;

        max_dist = (max_dist as f32 * 0.99).max(2.0) as u32;
    }
}

fn load_weights(source: &image::DynamicImage) -> Vec<i64> {
    let (width, height) = source.dimensions();
    let mut weights = vec![0; (width * height) as usize];
    for (x, y, pixel) in source.pixels() {
        let weight = pixel[0] as i64;
        weights[(y * width + x) as usize] = weight;
    }
    weights
}

fn serialize_assignments(assignments: Vec<usize>) -> String {
    format!(
        "[{}]",
        assignments
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub fn process(
    source_path: PathBuf,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    match settings.algorithm {
        Algorithm::Optimal => process_optimal(source_path, settings, tx, cancelled),
        Algorithm::Genetic => process_genetic(source_path, settings, tx, cancelled),
    }
}

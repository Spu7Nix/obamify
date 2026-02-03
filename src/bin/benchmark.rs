//! CLI benchmark tool for testing obamify algorithms
//! 
//! Usage: cargo run --release --bin benchmark -- [OPTIONS]
//! 
//! Options:
//!   --resolution <N>      Resolution to test (default: 64)
//!   --algorithm <NAME>    Algorithm to test: greedy, genetic, auction, hybrid, optimal, spatial (default: all)
//!   --preset <NAME>       Preset to use: blackhole, wisetree, cat, colorful (default: blackhole)
//!   --all                 Run all combinations

use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use obamify::app::calculate::{self, util::{Algorithm, GenerationSettings}, ProgressMsg};
use obamify::app::preset::UnprocessedPreset;

/// A simple progress sink that does nothing (for benchmarking)
struct NullSink;

impl calculate::util::ProgressSink for NullSink {
    fn send(&mut self, _msg: ProgressMsg) {}
}

fn load_preset(name: &str) -> Option<UnprocessedPreset> {
    let path = format!("presets/{}/source.png", name);
    let img = image::open(&path).ok()?.to_rgb8();
    Some(UnprocessedPreset {
        name: name.to_string(),
        width: img.width(),
        height: img.height(),
        source_img: img.into_raw(),
    })
}

fn run_algorithm(
    algorithm: Algorithm,
    preset: &UnprocessedPreset,
    resolution: u32,
) -> std::time::Duration {
    let mut settings = GenerationSettings::default(uuid::Uuid::new_v4(), preset.name.clone());
    settings.sidelen = resolution;
    settings.algorithm = algorithm;
    
    let mut sink = NullSink;
    let cancel = Arc::new(AtomicBool::new(false));
    
    let start = Instant::now();
    
    let result = match algorithm {
        Algorithm::Optimal => calculate::process_optimal(preset.clone(), settings, &mut sink, cancel),
        Algorithm::Auction => calculate::process_auction(preset.clone(), settings, &mut sink, cancel),
        Algorithm::Greedy => calculate::process_greedy(preset.clone(), settings, &mut sink, cancel),
        Algorithm::Hybrid => calculate::process_hybrid(preset.clone(), settings, &mut sink, cancel),
        Algorithm::Genetic => calculate::process_genetic(preset.clone(), settings, &mut sink, cancel),
        Algorithm::Spatial => calculate::process_spatial(preset.clone(), settings, &mut sink, cancel),
    };
    
    let elapsed = start.elapsed();
    
    if let Err(e) = result {
        eprintln!("Algorithm {:?} failed: {}", algorithm, e);
    }
    
    elapsed
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse arguments
    let mut resolution: u32 = 64;
    let mut algorithm_filter: Option<String> = None;
    let mut preset_name = "blackhole".to_string();
    let mut run_all = false;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--resolution" | "-r" => {
                if i + 1 < args.len() {
                    resolution = args[i + 1].parse().unwrap_or(64);
                    i += 1;
                }
            }
            "--algorithm" | "-a" => {
                if i + 1 < args.len() {
                    algorithm_filter = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--preset" | "-p" => {
                if i + 1 < args.len() {
                    preset_name = args[i + 1].clone();
                    i += 1;
                }
            }
            "--all" => {
                run_all = true;
            }
            "--help" | "-h" => {
                println!("Obamify Algorithm Benchmark Tool");
                println!();
                println!("Usage: benchmark [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -r, --resolution <N>   Resolution to test (default: 64)");
                println!("  -a, --algorithm <NAME> Algorithm: greedy, genetic, auction, hybrid, optimal");
                println!("  -p, --preset <NAME>    Preset: blackhole, wisetree, cat, colorful");
                println!("  --all                  Run all algorithm/resolution combinations");
                println!("  -h, --help             Show this help");
                return;
            }
            _ => {}
        }
        i += 1;
    }
    
    // Load preset
    let preset = match load_preset(&preset_name) {
        Some(p) => p,
        None => {
            eprintln!("Failed to load preset: {}", preset_name);
            eprintln!("Available presets: blackhole, wisetree, cat, cat2, colorful");
            return;
        }
    };
    
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             OBAMIFY ALGORITHM BENCHMARK                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Preset: {} ({}x{})", preset_name, preset.width, preset.height);
    println!();
    
    // Algorithms to test
    let algorithms = if let Some(ref name) = algorithm_filter {
        match name.to_lowercase().as_str() {
            "greedy" => vec![Algorithm::Greedy],
            "genetic" => vec![Algorithm::Genetic],
            "auction" => vec![Algorithm::Auction],
            "hybrid" => vec![Algorithm::Hybrid],
            "spatial" => vec![Algorithm::Spatial],
            "optimal" => vec![Algorithm::Optimal],
            _ => {
                eprintln!("Unknown algorithm: {}", name);
                return;
            }
        }
    } else {
        vec![
            Algorithm::Greedy,
            Algorithm::Spatial,
            Algorithm::Genetic,
            Algorithm::Auction,
            Algorithm::Hybrid,
            // Skip optimal by default (too slow)
        ]
    };
    
    // Resolutions to test
    let resolutions = if run_all {
        vec![64, 128, 256, 512]
    } else {
        vec![resolution]
    };
    
    // Print header
    println!("┌─────────────────┬──────────┬──────────────┬─────────────┐");
    println!("│ Algorithm       │ Res      │ Time         │ Pixels/sec  │");
    println!("├─────────────────┼──────────┼──────────────┼─────────────┤");
    
    for res in &resolutions {
        for algo in &algorithms {
            let pixels = res * res;
            
            print!("│ {:15} │ {:4}x{:<4}│ ", format!("{:?}", algo), res, res);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            
            let elapsed = run_algorithm(*algo, &preset, *res);
            
            let secs = elapsed.as_secs_f64();
            let pixels_per_sec = if secs > 0.0 { pixels as f64 / secs } else { 0.0 };
            
            let time_str = if secs < 0.001 {
                format!("{:.2}µs", secs * 1_000_000.0)
            } else if secs < 1.0 {
                format!("{:.2}ms", secs * 1000.0)
            } else if secs < 60.0 {
                format!("{:.2}s", secs)
            } else {
                format!("{:.1}min", secs / 60.0)
            };
            
            println!("{:12} │ {:11.0} │", time_str, pixels_per_sec);
        }
        
        if resolutions.len() > 1 {
            println!("├─────────────────┼──────────┼──────────────┼─────────────┤");
        }
    }
    
    println!("└─────────────────┴──────────┴──────────────┴─────────────┘");
    println!();
    println!("✓ Benchmark complete!");
}

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub inner: UnprocessedPreset,
    pub assignments: Vec<usize>,
    /// How much colors should shift toward target (0.0 = none, 1.0 = full)
    #[serde(default)]
    pub color_shift: f32,
    /// Target colors for each pixel position (for color morphing)
    /// If not empty, colors will interpolate: source + t * color_shift * (target - source)
    #[serde(default)]
    pub target_colors: Vec<u8>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UnprocessedPreset {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub source_img: Vec<u8>,
}

use crate::morph_sim::preset_path_to_name;

use super::load_weights;

use super::TARGET_WEIGHTS_PATH;

use super::TARGET_IMAGE_PATH;

use super::GenerationSettings;

use super::serialize_assignments;

use std::path::Path;

use std::error::Error;

pub(crate) fn save_result(
    target: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    base_name: String,
    source: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    assignments: Vec<usize>,
    img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Result<String, Box<dyn Error>> {
    let mut dir_name = base_name.clone();
    let mut counter = 1;
    while std::path::Path::new(&format!("./presets/{}", dir_name)).exists() {
        dir_name = format!("{}_{}", base_name, counter);
        counter += 1;
    }
    std::fs::create_dir_all(format!("./presets/{}", dir_name))?;
    img.save(format!("./presets/{}/output.png", dir_name))?;
    source.save(format!("./presets/{}/source.png", dir_name))?;
    target.save(format!("./presets/{}/target.png", dir_name))?;
    std::fs::write(
        format!("./presets/{}/assignments.json", dir_name),
        serialize_assignments(assignments),
    )?;
    Ok(dir_name)
}

pub(crate) fn get_images<P: AsRef<Path>>(
    source_path: P,
    settings: &GenerationSettings,
) -> Result<
    (
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        String,
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        Vec<(u8, u8, u8)>,
        Vec<(u8, u8, u8)>,
        Vec<i64>,
    ),
    Box<dyn Error>,
> {
    let (target, target_pixels, weights) = load_target(settings)?;

    let base_name = preset_path_to_name(source_path.as_ref());
    let mut source = image::open(source_path)?.to_rgb8();
    
    // Apply auto-resize if enabled
    if settings.auto_resize {
        source = resize_image_if_needed(source, 128);
    }
    
    let source = image::imageops::resize(
        &source,
        target.width(),
        target.height(),
        image::imageops::FilterType::Lanczos3,
    );
    let source_pixels = source
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    Ok((
        target,
        base_name,
        source,
        source_pixels,
        target_pixels,
        weights,
    ))
}

pub(crate) fn load_target(
    settings: &GenerationSettings,
) -> Result<
    (
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        Vec<(u8, u8, u8)>,
        Vec<i64>,
    ),
    Box<dyn Error>,
> {
    let mut target = if let Some(ref custom_target) = settings.custom_target_image {
        image::open(custom_target)?.to_rgb8()
    } else {
        image::open(TARGET_IMAGE_PATH)?.to_rgb8()
    };
    
    // For custom targets, we'll generate a default weights image (all white)
    let mut target_weights = if settings.custom_target_image.is_some() {
        // Create a white image with the same dimensions as target for default weights
        image::ImageBuffer::from_pixel(target.width(), target.height(), image::Rgb([255, 255, 255]))
    } else {
        image::open(TARGET_WEIGHTS_PATH)?.to_rgb8()
    };
    if target.dimensions().0 != target.dimensions().1 {
        return Err("Target image must be square".into());
    }
    if target.dimensions() != target_weights.dimensions() {
        return Err("Target and weights images must have the same dimensions".into());
    }
    
    // Apply auto-resize if enabled
    if settings.auto_resize {
        target = resize_image_if_needed(target, 128);
        target_weights = resize_image_if_needed(target_weights, 128);
    }
    
    if let Some(rescale) = settings.rescale {
        target = image::imageops::resize(
            &target,
            rescale,
            rescale,
            image::imageops::FilterType::Lanczos3,
        );
        target_weights = image::imageops::resize(
            &target_weights,
            rescale,
            rescale,
            image::imageops::FilterType::Lanczos3,
        );
    }
    let target_pixels = target.pixels().map(|p| (p[0], p[1], p[2])).collect();
    let weights = load_weights(&target_weights.into());
    Ok((target, target_pixels, weights))
}

pub fn resize_image_if_needed(img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>, max_size: u32) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let max_dimension = width.max(height);
    
    if max_dimension <= max_size {
        return img; // No resize needed
    }
    
    // Calculate new dimensions maintaining aspect ratio
    let scale = max_size as f32 / max_dimension as f32;
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;
    
    image::imageops::resize(
        &img,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    )
}

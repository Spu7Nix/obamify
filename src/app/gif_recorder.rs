#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

use color_quant::NeuQuant;

use crate::{ObamifyApp, app::SeedColor};

pub const GIF_FRAMERATE: u32 = 8;
pub const GIF_RESOLUTION: u32 = 400;
pub const GIF_MAX_FRAMES: u32 = 140;
pub const GIF_MIN_FRAMES: u32 = 100;
pub const GIF_MAX_SIZE: usize = 10 * 1024 * 1024; // 10 MB
pub const GIF_SPEED: f32 = 1.5;
pub const GIF_PALETTE_SAMPLEFAC: i32 = 1;

#[derive(Clone, Debug)]
pub enum GifStatus {
    None,
    Recording,
    #[cfg(not(target_arch = "wasm32"))]
    Complete(PathBuf),
    #[cfg(target_arch = "wasm32")]
    Complete,
    Error(String),
}
impl GifStatus {
    fn is_recording(&self) -> bool {
        matches!(self, GifStatus::Recording)
    }

    fn not_recording(&self) -> bool {
        matches!(self, GifStatus::None)
    }
}

struct InFlight {
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
}

pub struct GifRecorder {
    pub id: u32,
    pub status: GifStatus,
    pub encoder: Option<gif::Encoder<Vec<u8>>>,
    pub palette: Option<NeuQuant>,
    pub frame_count: u32,
    inflight: Option<InFlight>,
    should_stop: bool,
}

impl GifRecorder {
    pub fn new() -> Self {
        Self {
            id: 0,
            status: GifStatus::None,
            encoder: None,
            palette: None,
            frame_count: 0,
            inflight: None,
            should_stop: false,
        }
    }

    pub fn is_recording(&self) -> bool {
        self.status.is_recording()
    }

    pub fn not_recording(&self) -> bool {
        self.status.not_recording()
    }

    fn poll_inflight(&mut self) -> Option<Vec<u8>> {
        if let Some(inflight) = &self.inflight {
            if inflight.ready.load(std::sync::atomic::Ordering::Acquire) {
                let slice = inflight.buffer.slice(..);
                let mapped = slice.get_mapped_range();
                // Remove row padding
                let width = GIF_RESOLUTION;
                let height = GIF_RESOLUTION;
                let bpp = 4u32; // RGBA8
                let unpadded_bytes_per_row = width * bpp;
                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
                let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

                let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                for y in 0..height as usize {
                    let start = y * padded_bytes_per_row as usize;
                    let end = start + unpadded_bytes_per_row as usize;
                    rgba.extend_from_slice(&mapped[start..end]);
                }
                drop(mapped);
                inflight.buffer.unmap();
                self.inflight = None;
                Some(rgba)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn try_write_frame(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(rgba) = self.poll_inflight() {
            if let Some(encoder) = &mut self.encoder {
                let nq = self.palette.as_ref().unwrap();
                let pixels: Vec<u8> = rgba
                    .chunks_exact(4)
                    .map(|pix| nq.index_of(pix) as u8)
                    .collect();
                let mut frame = gif::Frame::from_indexed_pixels(
                    GIF_RESOLUTION as u16,
                    GIF_RESOLUTION as u16,
                    pixels,
                    None,
                );
                let frame_size = encoder.get_ref().len() + frame.buffer.len() + 32; // idk if this is exact but its a conservative estimate
                if frame_size > GIF_MAX_SIZE {
                    self.should_stop = true;
                    return Ok(true);
                }

                frame.delay = ((100.0 / GIF_FRAMERATE as f32) / GIF_SPEED) as u16; // delay in 1/100 sec
                encoder.write_frame(&frame)?;

                Ok(true)
            } else {
                // shouldn't happen
                Err("No encoder".into())
            }
        } else {
            Ok(false)
        }
    }

    pub fn init_encoder(
        &mut self,
        active_colors: &[SeedColor],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let colors = active_colors
            .iter()
            .flat_map(|s| {
                s.rgba
                    .map(|f| (if f == 1.0 { 255.0 } else { f * 256.0 }) as u8)
            })
            .collect::<Vec<u8>>();
        let gif_palette = NeuQuant::new(GIF_PALETTE_SAMPLEFAC, 256, &colors);
        let mut encoder = gif::Encoder::new(
            vec![],
            GIF_RESOLUTION as u16,
            GIF_RESOLUTION as u16,
            &gif_palette.color_map_rgb(),
        )?;
        self.palette = Some(gif_palette);
        encoder.set_repeat(gif::Repeat::Infinite)?;
        self.encoder = Some(encoder);
        self.frame_count = 0;
        self.status = GifStatus::Recording;
        Ok(())
    }

    pub fn finish(&mut self, name: String) -> bool {
        match (
            self.status.clone(),
            self.encoder.take().unwrap().into_inner(),
        ) {
            (GifStatus::Recording, Ok(data)) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let file = rfd::FileDialog::new()
                        .set_title("save gif")
                        .add_filter("gif", &["gif"])
                        .set_file_name(format!("{}.gif", name))
                        .save_file();
                    if let Some(path) = file {
                        std::fs::write(&path, data).unwrap();
                        self.status = GifStatus::Complete(path);
                    } else {
                        return false;
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.status = GifStatus::None;
                    use wasm_bindgen_futures::spawn_local;
                    let status_ptr: *mut GifStatus = &mut self.status;

                    spawn_local(async move {
                        if let Some(handle) = rfd::AsyncFileDialog::new()
                            .set_title("Recording complete!")
                            .set_file_name(format!("{}.gif", name))
                            .save_file()
                            .await
                        {
                            handle.write(&data).await.ok();
                            // SAFETY: We ensure the app outlives the async task (eframe app is long-lived).
                            unsafe {
                                *status_ptr = GifStatus::Complete;
                            }
                        }
                    });
                }
            }
            (a, b) => {
                self.status = GifStatus::Error(format!("Something weird happened: {:?}", (a, b)));
            }
        }
        true
    }

    pub fn no_inflight(&self) -> bool {
        self.inflight.is_none()
    }

    pub fn stop(&mut self) {
        self.status = GifStatus::None;
        self.encoder = None;
        self.palette = None;
        self.frame_count = 0;
        self.inflight = None;
        self.id += 1;
    }

    pub fn should_stop(&self) -> bool {
        if self.frame_count < GIF_MIN_FRAMES {
            false
        } else if self.frame_count >= GIF_MAX_FRAMES {
            true
        } else {
            self.should_stop
        }
    }

    pub(crate) fn get_name(&self, name: String, reverse: bool) -> String {
        if reverse {
            format!("unobamify_{}", name)
        } else {
            format!("obamify_{}", name)
        }
    }
}

impl ObamifyApp {
    pub fn get_color_image_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let width = self.size.0;
        let height = self.size.1;
        let bpp = 4u32; // RGBA8
        let unpadded_bytes_per_row = width * bpp;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row as u64 * height as u64;

        // Staging buffer to receive the texture
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("color readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encode copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy color_tex -> buffer"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        let ready = Arc::new(AtomicBool::new(false));
        let slice = readback.slice(..);
        let ready_in_cb = Arc::clone(&ready);

        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                ready_in_cb.store(true, std::sync::atomic::Ordering::Release);
            }
        });

        self.gif_recorder.inflight = Some(InFlight {
            buffer: readback,
            ready,
        });

        Ok(())

        // let slice = readback.slice(..);
        // let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        // slice.map_async(wgpu::MapMode::Read, move |res| {
        //     // res: Result<(), wgpu::BufferAsyncError>
        //     let _ = tx.send(res);
        // });

        // // Ensure the callback runs
        // device.poll(wgpu::PollType::Wait)?;

        // // Wait for the result and propagate any map error
        // pollster::block_on(rx.receive()).expect("map_async sender dropped")?;
        // let mapped = slice.get_mapped_range();
        // // Remove row padding
        // let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        // for y in 0..height as usize {
        //     let start = y * padded_bytes_per_row as usize;
        //     let end = start + unpadded_bytes_per_row as usize;
        //     rgba.extend_from_slice(&mapped[start..end]);
        // }
        // drop(mapped);
        // readback.unmap();
        // Ok(rgba)
    }
}

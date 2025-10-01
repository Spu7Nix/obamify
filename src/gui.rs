use super::get_presets;
use super::GifStatus;
use super::GuiMode;
use super::VoronoiApp;
use super::DRAWING_ALPHA;
use super::GIF_FRAMERATE;
use super::GIF_NUM_FRAMES;
use super::GIF_RESOLUTION;
use crate::calculate;
use crate::calculate::GenerationSettings;
use crate::calculate::ProgressMsg;
use crate::morph_sim::preset_path_to_name;
use crate::DEFAULT_RESOLUTION;
use eframe::App;
use eframe::Frame;
use egui::Color32;
use egui::Modal;
use egui::Window;
use image::buffer::ConvertBuffer;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum ExportFormat {
    Gif,
    WebM,
    Mp4,
}

impl Default for ExportFormat {
    fn default() -> Self {
        ExportFormat::Gif
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Gif => write!(f, "GIF"),
            ExportFormat::WebM => write!(f, "WebM"),
            ExportFormat::Mp4 => write!(f, "MP4"),
        }
    }
}

pub(crate) struct GuiState {
    pub last_mouse_pos: Option<(f32, f32)>,
    pub drawing_color: [f32; 4],
    pub mode: GuiMode,
    pub animate: bool,
    pub fps_text: String,
    pub show_progress_modal: bool,
    pub last_progress: f32,
    pub process_cancelled: Arc<AtomicBool>,
    pub currently_processing: Option<PathBuf>,
    pub presets: Vec<PathBuf>,
    pub current_settings: GenerationSettings,
    pub configuring_generation: Option<PathBuf>,
    pub custom_start_image: Option<PathBuf>,
    pub custom_end_image: Option<PathBuf>,
    pub using_custom_images: bool,
    pub auto_resize_images: bool,
    pub export_format: ExportFormat,
}

impl GuiState {
    pub fn default(presets: Vec<PathBuf>) -> GuiState {
        GuiState {
            animate: true,
            fps_text: String::new(),
            presets,
            mode: GuiMode::Transform,
            show_progress_modal: false,
            last_progress: 0.0,
            process_cancelled: Arc::new(AtomicBool::new(false)),
            last_mouse_pos: None,
            drawing_color: [0.0, 0.0, 0.0, DRAWING_ALPHA],
            currently_processing: None,
            current_settings: GenerationSettings::default(),
            configuring_generation: None,
            custom_start_image: None,
            custom_end_image: None,
            using_custom_images: false,
            auto_resize_images: true,
            export_format: ExportFormat::default(),
        }
    }
}

impl App for VoronoiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };

        let device = &rs.device;
        // Resize handling (match the egui "central panel" size)
        //let available = ctx.available_rect();
        // let target_size = (
        //     available.width().max(1.0) as u32,
        //     available.height().max(1.0) as u32,
        // );
        // if target_size != self.size {
        //     self.resize(rs, target_size);
        // }

        // Ensure texture is registered exactly once per allocation
        self.ensure_registered_texture(rs);

        // Run GPU pipeline
        if let Some(img) = &self.preview_image {
            // show image
            let img = image::imageops::resize(
                img,
                self.size.0,
                self.size.1,
                image::imageops::FilterType::Nearest,
            );
            let rgba: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = img.convert();
            let rgba = rgba.into_raw();
            rs.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.color_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.size.0),
                    rows_per_image: Some(self.size.1),
                },
                wgpu::Extent3d {
                    width: self.size.0,
                    height: self.size.1,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            self.run_gpu(rs);

            if self.gui.animate {
                if self.gif_status.is_recording() {
                    for _ in 0..(60 / GIF_FRAMERATE) {
                        self.sim.update(&mut self.seeds, self.size.0);
                    }

                    if let Err(e) = self.write_frame_to_gif(device, &rs.queue) {
                        self.gif_status = GifStatus::Error(e.to_string());
                        self.gui.animate = false;
                    } else {
                        self.gif_frame_count += 1;

                        if self.gif_frame_count >= GIF_NUM_FRAMES {
                            if let Some(_enc) = self.gif_encoder.take() {
                                // dropping encoder flushes file
                            }
                            if let GifStatus::Recording(Some(path)) = self.gif_status.clone() {
                                self.gif_status = GifStatus::Complete(path);
                            } else {
                                self.gif_status = GifStatus::Error(format!(
                                    "Something weird happened: {:?}",
                                    self.gif_status
                                ));
                            }
                            self.gui.animate = false;
                        }
                    }
                } else {
                    self.sim.update(&mut self.seeds, self.size.0);
                }
                rs.queue
                    .write_buffer(&self.seed_buf, 0, bytemuck::cast_slice(&self.seeds));
            }
        }

        let dt = self.prev_frame_time.elapsed();
        self.prev_frame_time = std::time::Instant::now();
        self.gui.fps_text = format!(
            "{:5.2} ms/frame (~{:06.0} FPS)",
            dt.as_secs_f64() * 1000.0,
            1.0 / dt.as_secs_f64()
        );

        // ===== UI =====
        ctx.set_zoom_factor(1.4);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                match self.gui.mode {
                    GuiMode::Draw => {
                        if ui.button("reset").clicked() {
                            self.init_canvas(device);
                        }

                        if let Ok(msg) = self.progress_rx.try_recv() {
                            match msg {
                                ProgressMsg::UpdatePreview(image) => {
                                    self.preview_image = Some(image);
                                }
                                ProgressMsg::Cancelled => {
                                    self.gui
                                        .process_cancelled
                                        .store(false, std::sync::atomic::Ordering::Relaxed);
                                    self.preview_image = None;

                                    ui.close();
                                }
                                ProgressMsg::UpdateAssignments(assignments) => {
                                    self.sim.set_assignments(assignments, self.size.0)
                                }
                                ProgressMsg::Progress(_) => todo!(),
                                ProgressMsg::Done(path_buf) => todo!(),
                                ProgressMsg::Error(_) => todo!(),
                            }
                        }

                        if ui
                            .add(egui::Button::new(egui::RichText::new("🏠")))
                            .on_hover_text("transform mode")
                            .clicked()
                        {
                            self.gui.mode = GuiMode::Transform;
                            self.change_sim(device, self.gui.presets[0].clone());
                        }
                    }
                    GuiMode::Transform => {
                        if ui
                            .add_enabled(
                                !self.gui.animate,
                                egui::Button::new("play transformation"), //.fill(egui::Color32::from_rgb(47, 92, 34)),
                            )
                            .clicked()
                        {
                            self.gui.animate = true;
                        }
                        if ui
                            .add_enabled(self.gui.animate, egui::Button::new("switch target"))
                            .clicked()
                        {
                            self.sim.switch();
                        }
                        if ui.button("reload").clicked() {
                            self.change_sim(device, self.sim.source_path());
                            self.gui.animate = false;
                        }
                        ui.separator();

                        if ui.button("save gif").clicked() {
                            self.gif_status = GifStatus::Recording(None);
                            self.resize_textures(device, (GIF_RESOLUTION, GIF_RESOLUTION), false);
                            self.change_sim(device, self.sim.source_path());
                            self.gui.animate = true;
                            self.gif_frame_count = 0; // reset counter
                            for _ in 0..20 {
                                self.sim.update(&mut self.seeds, self.size.0);
                            }
                        }

                        // GIF recording progress / status
                        match &self.gif_status {
                            GifStatus::Recording(Some(_)) => {
                                let progress = self.gif_frame_count as f32 / GIF_NUM_FRAMES as f32;
                                ui.add(
                                    egui::ProgressBar::new(progress)
                                        .desired_width(120.0)
                                        .text(format!("GIF {:3.0}%", progress * 100.0)),
                                );
                                if ui.button("cancel gif").clicked() {
                                    // cancel: drop encoder
                                    self.gif_encoder = None;
                                    self.gif_status = GifStatus::None;
                                    self.gif_frame_count = 0;
                                    self.resize_textures(device, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION), false);
                                    self.change_sim(device, self.sim.source_path());
                                }
                            }
                            GifStatus::Complete(path) => {
                                ui.colored_label(egui::Color32::LIGHT_GREEN, "GIF saved");
                                if ui.button("open file").clicked() {
                                    let _ = opener::open(path);
                                }
                                if ui.button("reset size").clicked() {
                                    self.resize_textures(device, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION), false);
                                }
                            }
                            GifStatus::Error(err) => {
                                ui.colored_label(egui::Color32::RED, format!("gif error: {err}"));
                            }
                            _ => {}
                        }

                        ui.separator();
                        
                        ui.horizontal(|ui| {
                            ui.label("Mode:");
                            if ui.radio(!self.gui.using_custom_images, "📂 Use Presets").clicked() {
                                self.gui.using_custom_images = false;
                                // Reset custom image selections when switching to presets
                                self.gui.custom_start_image = None;
                                self.gui.custom_end_image = None;
                                if !self.gui.presets.is_empty() {
                                    self.change_sim(device, self.gui.presets[0].clone());
                                    self.gui.animate = false;
                                }
                            }
                            if ui.radio(self.gui.using_custom_images, "🖼️ Custom Images").clicked() {
                                self.gui.using_custom_images = true;
                            }
                        });

                        if !self.gui.using_custom_images {
                            // Preset mode
                            ui.label("Choose preset:");
                            egui::ComboBox::from_label("")
                                .selected_text({
                                    let name = self.sim.name();
                                    if name.chars().count() > 13 {
                                        let truncated: String = name.chars().take(10).collect();
                                        format!("{truncated}…")
                                    } else {
                                        name.to_string()
                                    }
                                })
                                .show_ui(ui, |ui| {
                                    for preset in self.gui.presets.clone().into_iter() {
                                        if ui.button(preset_path_to_name(&preset)).clicked() {
                                            // Call change_sim when a new preset is selected
                                            self.change_sim(device, preset);
                                            self.gui.animate = false;
                                        }
                                    }
                                });
                            if ui.button("obamify new image").clicked() {
                                // open file select
                                let file = rfd::FileDialog::new()
                                    .set_title("choose image (square aspect ratio recommended)")
                                    .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
                                    .pick_file();

                                if let Some(path) = file {
                                    self.gui.configuring_generation = Some(path);
                                }
                            }
                        } else {
                            // Custom images mode
                            ui.horizontal(|ui| {
                                if ui.button("Select Start Image").clicked() {
                                    let file = rfd::FileDialog::new()
                                        .set_title("Choose start image (square aspect ratio recommended)")
                                        .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
                                        .pick_file();

                                    if let Some(path) = file {
                                        self.gui.custom_start_image = Some(path);
                                    }
                                }
                                
                                if let Some(ref path) = self.gui.custom_start_image {
                                    ui.label(format!("✓ {}", path.file_name().unwrap().to_string_lossy()));
                                } else {
                                    ui.label("No start image selected");
                                }
                            });

                            ui.horizontal(|ui| {
                                if ui.button("Select End Image").clicked() {
                                    let file = rfd::FileDialog::new()
                                        .set_title("Choose end image (square aspect ratio recommended)")
                                        .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
                                        .pick_file();

                                    if let Some(path) = file {
                                        self.gui.custom_end_image = Some(path);
                                    }
                                }
                                
                                if let Some(ref path) = self.gui.custom_end_image {
                                    ui.label(format!("✓ {}", path.file_name().unwrap().to_string_lossy()));
                                } else {
                                    ui.label("No end image selected");
                                }
                            });

                            if self.gui.custom_start_image.is_some() && self.gui.custom_end_image.is_some() {
                                if ui.button("Generate Transformation").clicked() {
                                    // Set up custom transformation with proper settings
                                    self.gui.configuring_generation = self.gui.custom_start_image.clone();
                                    // Update current settings for custom transformation
                                    self.gui.current_settings.custom_target_image = self.gui.custom_end_image.clone();
                                }
                            } else {
                                ui.add_enabled(false, egui::Button::new("Generate Transformation"));
                                ui.label("Select both start and end images to generate transformation");
                            }
                        }

                        ui.separator();

                        // Auto-resize setting
                        ui.horizontal(|ui| {
                            let tooltip = "When enabled, input & target images are downscaled so the longest side is 128px (keeps aspect ratio). Speeds generation & GIF saving.";
                            if ui.checkbox(&mut self.gui.auto_resize_images, "Auto-resize (≤128px longest side)").on_hover_text(tooltip).changed() {
                                // No immediate action needed; applied on next generation run.
                            }
                        });

                        if ui
                            .add(egui::Button::new(egui::RichText::new("✏")))
                            .on_hover_text("drawing mode")
                            .clicked()
                        {
                            self.gui.mode = GuiMode::Draw;
                            self.init_canvas(device);
                        }

                        if self.gui.configuring_generation.is_some() {
                            Modal::new("configuring_generation".into()).show(ui.ctx(), |ui| {
                                ui.vertical(|ui| {
                                    ui.label(
                                        egui::RichText::new(if self.gui.using_custom_images {
                                            "Generating Custom Transformation".to_string()
                                        } else {
                                            format!(
                                                "Obamifying: {}",
                                                preset_path_to_name(
                                                    self.gui.configuring_generation.as_ref().unwrap()
                                                )
                                            )
                                        })
                                        .heading()
                                        .strong(),
                                    );
                                    ui.separator();

                                    ui.horizontal(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label("proximity importance:");
                                            ui.add(egui::Slider::new(
                                                &mut self.gui.current_settings.proximity_importance,
                                                0..=50,
                                            ));
                                        });

                                        let mut algorithm =
                                            match self.gui.current_settings.algorithm {
                                                calculate::Algorithm::Optimal => {
                                                    "optimal algorithm"
                                                }
                                                calculate::Algorithm::Genetic => "fast algorithm",
                                            };
                                        egui::ComboBox::from_label("")
                                            .width(200.0)
                                            .selected_text(algorithm)
                                            .show_ui(ui, |ui| {
                                                if ui.button("optimal algorithm").clicked() {
                                                    algorithm = "optimal algorithm";
                                                    self.gui.current_settings.algorithm =
                                                        calculate::Algorithm::Optimal;
                                                }
                                                if ui.button("fast algorithm").clicked() {
                                                    algorithm = "fast algorithm";
                                                    self.gui.current_settings.algorithm =
                                                        calculate::Algorithm::Genetic;
                                                }
                                            });
                                    });

                                    ui.separator();
                                    ui.horizontal(|ui| {
                                        if ui
                                            .add(egui::Button::new(
                                                egui::RichText::new("start!").strong(),
                                            ))
                                            .clicked()
                                        {
                                            if let Some(path) =
                                                self.gui.configuring_generation.take()
                                            {
                                                self.gui.show_progress_modal = true;

                                                let mut settings = self.gui.current_settings.clone();
                                                // If using custom images, set the target image
                                                if self.gui.using_custom_images {
                                                    settings.custom_target_image = self.gui.custom_end_image.clone();
                                                }
                                                // Apply auto-resize setting
                                                settings.auto_resize = self.gui.auto_resize_images;
                                                self.gui.currently_processing = Some(path.clone());
                                                //self.change_sim(device, path.clone(), false);

                                                self.gui.process_cancelled.store(
                                                    false,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                );

                                                std::thread::spawn({
                                                    let tx = self.progress_tx.clone();
                                                    let cancelled =
                                                        self.gui.process_cancelled.clone();
                                                    move || {
                                                        let result = calculate::process(
                                                            path,
                                                            settings,
                                                            tx.clone(),
                                                            cancelled,
                                                        );
                                                        match result {
                                                            Ok(()) => {}
                                                            Err(err) => {
                                                                tx.send(ProgressMsg::Error(
                                                                    err.to_string(),
                                                                ))
                                                                .ok();
                                                            }
                                                        }
                                                    }
                                                });
                                            }
                                        }
                                        if ui.button("cancel").clicked() {
                                            self.gui.configuring_generation = None;
                                        }
                                    });
                                });
                            });
                        }

                        if self.gui.show_progress_modal {
                            Window::new("progress")
                                .title_bar(false)
                                .collapsible(false)
                                .resizable(false)
                                .movable(false)
                                .anchor(egui::Align2::CENTER_BOTTOM, (0.0, 0.0))
                                .show(ui.ctx(), |ui| {
                                    let processing_label_message = if self.gui.using_custom_images {
                                        "generating custom transformation..."
                                    } else {
                                        "processing..."
                                    };
                                    ui.vertical(|ui| {
                                        ui.set_min_width(ui.available_width().min(400.0));
                                        if let Ok(msg) = self.progress_rx.try_recv() {
                                            match msg {
                                                ProgressMsg::Done(path) => {
                                                    self.preview_image = None;
                                                    self.gui.presets = get_presets();
                                                    self.change_sim(device, path);
                                                    self.gui.animate = true;
                                                    self.gui.show_progress_modal = false;
                                                    ui.close();
                                                }
                                                ProgressMsg::Progress(p) => {
                                                    self.gui.last_progress = p;
                                                }
                                                ProgressMsg::Error(err) => {
                                                    ui.label(format!("error: {}", err));
                                                    if ui.button("close").clicked() {
                                                        ui.close();
                                                    }
                                                }
                                                ProgressMsg::UpdatePreview(image) => {
                                                    self.preview_image = Some(image);
                                                }
                                                ProgressMsg::Cancelled => {
                                                    self.gui.process_cancelled.store(
                                                        false,
                                                        std::sync::atomic::Ordering::Relaxed,
                                                    );
                                                    self.preview_image = None;
                                                    self.gui.show_progress_modal = false;
                                                    ui.close();
                                                }
                                                ProgressMsg::UpdateAssignments(assignments) => self
                                                    .sim
                                                    .set_assignments(assignments, self.size.0),
                                            }
                                        }

                                        if self
                                            .gui
                                            .process_cancelled
                                            .load(std::sync::atomic::Ordering::Relaxed)
                                        {
                                            ui.label("cancelling...");
                                        } else {
                                            ui.label(processing_label_message);
                                        }
                                        ui.add(
                                            egui::ProgressBar::new(self.gui.last_progress)
                                                .show_percentage(),
                                        );

                                        ui.horizontal(|ui| {
                                            if ui.button("cancel").clicked() {
                                                self.gui.process_cancelled.store(
                                                    true,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                );
                                                self.gui.last_progress = 0.0;
                                            }
                                        })
                                    });
                                });

                            // if modal.should_close() {
                            //     self.gui.show_progress_modal = false;
                            // }
                        } else if !self.gif_status.not_recording() {
                            Modal::new("recording_progress".into()).show(ui.ctx(), |ui| match self
                                .gif_status
                                .clone()
                            {
                                GifStatus::Recording(_) => {
                                    ui.label("Recording GIF...");
                                }

                                GifStatus::Error(err) => {
                                    ui.label(format!("Error: {}", err));
                                    ui.horizontal(|ui| {
                                        if ui.button("close").clicked() {
                                            self.stop_recording_gif(device);
                                        }
                                    });
                                }
                                GifStatus::Complete(path) => {
                                    ui.label("gif saved!");
                                    ui.horizontal(|ui| {
                                        if ui.button("open file").clicked() {
                                            opener::reveal(path).ok();
                                        }
                                        if ui.button("close").clicked() {
                                            self.stop_recording_gif(device);
                                        }
                                    });
                                }
                                GifStatus::None => unreachable!(),
                            });
                        }

                        // ui.separator();
                        // ui.label(&self.gui.fps_text);
                    }
                }
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ctx, |ui| {
                ui.vertical_centered_justified(|ui| {
                    if let Some(id) = self.egui_tex_id {
                        let full = ui.available_size();
                        let aspect = self.size.0 as f32 / self.size.1 as f32;
                        let desired = full.x.min(full.y) * egui::vec2(1.0, aspect);
                        ui.add(egui::Image::new((id, desired)).maintain_aspect_ratio(true));

                        if matches!(self.gui.mode, GuiMode::Draw) {
                            self.handle_drawing(ctx, device, ui, aspect);
                        }
                    } else {
                        ui.colored_label(Color32::LIGHT_RED, "Texture not ready");
                    }
                });
            });
        if matches!(self.gui.mode, GuiMode::Draw) {
            let number_keys = [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
                egui::Key::Num5,
            ];

            // DBECEE,383232, 6B5E57,D49976

            let colors = [
                ("black", 0x000000),
                ("a", 0x86d9e3),
                ("b", 0x383232),
                ("c", 0xD49976),
                ("d", 0x793025),
            ];

            for (idx, (name, color)) in colors.iter().enumerate() {
                if ctx.input(|i| i.key_pressed(number_keys[idx])) {
                    let hex = *color;
                    let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
                    let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
                    let b = (hex & 0xFF) as f32 / 255.0;
                    let a = 0.5;

                    self.gui.drawing_color = [r, g, b, a];
                }
            }
            // show selected drawing color
            egui::Area::new("drawing_color".into())
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 30.0))
                .show(ctx, |ui| {
                    let rect_size = 30.0;
                    let (rect, resp) = ui.allocate_exact_size(
                        egui::vec2(rect_size, rect_size),
                        egui::Sense::hover(),
                    );
                    let color = egui::Color32::from_rgba_unmultiplied(
                        (self.gui.drawing_color[0] * 255.0) as u8,
                        (self.gui.drawing_color[1] * 255.0) as u8,
                        (self.gui.drawing_color[2] * 255.0) as u8,
                        255,
                    );
                    ui.painter().rect_filled(rect, 15.0, color);
                    if ui.is_rect_visible(rect) {
                        ui.painter().rect_stroke(
                            rect,
                            15.0,
                            (2.0, egui::Color32::WHITE),
                            egui::StrokeKind::Inside,
                        );
                    }

                    // Keep the picker visible while hovering either the main swatch or the picker area.
                    let spacing = 10.0;
                    let btn_size = rect_size / 2.0;
                    let gap = 4.0;

                    // Layout picker row next to the swatch, vertically centered.
                    let n_buttons = colors.len() as f32;
                    let picker_width = n_buttons * btn_size + (n_buttons - 1.0).max(0.0) * gap;
                    let picker_min =
                        rect.min + egui::vec2(rect_size + spacing, (rect_size - btn_size) * 0.5);
                    let picker_rect = egui::Rect::from_min_size(
                        rect.min,
                        egui::vec2(picker_width + rect_size + spacing, rect_size),
                    );

                    // Decide visibility purely from pointer position to avoid z-order flicker.
                    let pointer_pos = ctx.input(|i| i.pointer.hover_pos());
                    let show_picker = ui.is_rect_visible(rect)
                        && pointer_pos.is_some_and(|p| rect.contains(p) || picker_rect.contains(p));

                    // Global visibility animation driver
                    let base_t = ui
                        .ctx()
                        .animate_bool(egui::Id::new("color_picker_visible"), show_picker);

                    // Helpers
                    let saturate = |x: f32| x.clamp(0.0, 1.0);
                    let smoothstep = |x: f32| {
                        let x = saturate(x);
                        x * x * (3.0 - 2.0 * x)
                    };

                    // Start position = centered under the main swatch so buttons "emerge" from it
                    let start_pos = egui::pos2(
                        rect.min.x + (rect_size - btn_size) * 0.5,
                        rect.min.y + (rect_size - btn_size) * 0.5,
                    );

                    // Per-button stagger
                    let per_btn_delay = 0.08_f32;
                    // Ensure the last button still reaches t=1 when base_t=1
                    let total_stagger = (n_buttons - 1.0).max(0.0) * per_btn_delay;
                    let denom = (1.0 - total_stagger).max(1e-6);

                    for (idx, (_name, hex)) in colors.iter().enumerate() {
                        let rgba = {
                            let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
                            let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
                            let b = (hex & 0xFF) as f32 / 255.0;
                            let a = DRAWING_ALPHA;
                            [r, g, b, a]
                        };
                        let i = idx as f32;

                        // Staggered progress for each button; normalized so the last also reaches 1.0
                        let raw = (base_t - per_btn_delay * i) / denom;
                        let t_i = smoothstep(raw);

                        // Only draw while animating or visible to avoid early reveal
                        if t_i <= 0.001 {
                            continue;
                        }

                        // Target position to the right of the swatch
                        let end_pos = egui::pos2(picker_min.x + i * (btn_size + gap), picker_min.y);

                        // Interpolate from under the swatch to the target
                        let pos = egui::pos2(
                            egui::lerp(start_pos.x..=end_pos.x, t_i),
                            egui::lerp(start_pos.y..=end_pos.y, t_i),
                        );

                        egui::Area::new(egui::Id::new(format!("color_picker_btn_{idx}")))
                            .fixed_pos(pos)
                            .show(ctx, |ui| {
                                let (btn_rect, btn_resp) = ui.allocate_exact_size(
                                    egui::vec2(btn_size, btn_size),
                                    egui::Sense::click(),
                                );

                                // Fade with the slide
                                let a = (255.0 * t_i) as u8;
                                let color32 = egui::Color32::from_rgba_unmultiplied(
                                    (rgba[0] * 255.0) as u8,
                                    (rgba[1] * 255.0) as u8,
                                    (rgba[2] * 255.0) as u8,
                                    a,
                                );

                                ui.painter().rect_filled(btn_rect, 15.0 / 2.0, color32);
                                if ui.is_rect_visible(btn_rect) {
                                    ui.painter().rect_stroke(
                                        btn_rect,
                                        15.0 / 2.0,
                                        (
                                            2.0,
                                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, a),
                                        ),
                                        egui::StrokeKind::Inside,
                                    );
                                }

                                if btn_resp.clicked() {
                                    self.gui.drawing_color = rgba;
                                }
                            });
                    }
                });
        }

        // continuous repaint for animation
        ctx.request_repaint();
        self.frame_count += 1;
    }
}

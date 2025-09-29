#![windows_subsystem = "windows"]

mod morph_sim;
use std::{
    fs::File,
    num::NonZeroU64,
    path::PathBuf,
    sync::{atomic::AtomicBool, mpsc, Arc},
};

use bytemuck::{Pod, Zeroable};
use color_quant::NeuQuant;
use eframe::{egui, App, CreationContext, Frame, NativeOptions};
use egui::{Color32, Modal, ViewportBuilder};
use egui_wgpu::{self, wgpu};
use wgpu::util::DeviceExt;

use crate::{
    calculate::{GenerationSettings, ProgressMsg},
    morph_sim::{preset_path_to_name, Sim},
};

const WG_SIZE_XY: u32 = 8;
const WG_SIZE_SEEDS: u32 = 256;
//const INVALID_ID: u32 = 0xFFFF_FFFF;

mod calculate;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SeedPos {
    xy: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SeedColor {
    rgba: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsCommon {
    width: u32,
    height: u32,
    n_seeds: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsJfa {
    width: u32,
    height: u32,
    step: u32,
    _pad: u32,
}

const DEFAULT_RESOLUTION: u32 = 2048;
const GIF_FRAMERATE: u32 = 8;
const GIF_RESOLUTION: u32 = 460;
const GIF_NUM_FRAMES: u32 = 140;
const GIF_SPEED: f32 = 1.5;
const GIF_PALETTE_SAMPLEFAC: i32 = 1;

#[derive(Clone)]
enum GifStatus {
    None,
    Recording(Option<PathBuf>),
    Complete(PathBuf),
    Error(String),
}
impl GifStatus {
    fn is_recording(&self) -> bool {
        match self {
            GifStatus::Recording(_) => true,
            _ => false,
        }
    }

    fn not_recording(&self) -> bool {
        match self {
            GifStatus::None => true,
            _ => false,
        }
    }
}

pub struct VoronoiApp {
    prev_frame_time: std::time::Instant,
    // UI state
    size: (u32, u32),
    seed_count: u32,
    animate: bool,
    speed: f32,
    fps_text: String,
    show_progress_modal: bool,
    progress_tx: mpsc::SyncSender<ProgressMsg>,
    progress_rx: mpsc::Receiver<ProgressMsg>,
    last_progress: f32,
    process_cancelled: Arc<AtomicBool>,
    quick_process: bool,
    currently_processing: Option<PathBuf>,

    presets: Vec<PathBuf>,

    gif_status: GifStatus,
    gif_encoder: Option<gif::Encoder<File>>,
    gif_palette: Option<NeuQuant>,
    gif_frame_count: u32,
    frame_counter: u32,
    sim: Sim,

    // Seeds CPU copy
    seeds: Vec<SeedPos>,
    colors: Vec<SeedColor>,

    // EGUI texture id for presenting the shaded RGBA texture
    egui_tex_id: Option<egui::TextureId>,

    // GPU resources (lifetime tied to eframe's RenderState device)
    // Buffers
    seed_buf: wgpu::Buffer,
    color_buf: wgpu::Buffer,
    params_common_buf: wgpu::Buffer,
    params_jfa_buf: wgpu::Buffer,

    // Textures & views
    ids_a: wgpu::Texture,
    ids_b: wgpu::Texture,
    ids_a_view: wgpu::TextureView,
    ids_b_view: wgpu::TextureView,

    // Color (linear storage + srgb view for egui)
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,

    // Pipelines
    clear_pipeline: wgpu::ComputePipeline,
    seed_splat_pipeline: wgpu::ComputePipeline,
    jfa_pipeline: wgpu::ComputePipeline,
    shade_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    clear_bgl: wgpu::BindGroupLayout,
    seed_bgl: wgpu::BindGroupLayout,
    jfa_bgl: wgpu::BindGroupLayout,
    shade_bgl: wgpu::BindGroupLayout,

    // Bind groups that are re-created when textures change
    clear_bg_a: wgpu::BindGroup,
    clear_bg_b: wgpu::BindGroup,
    seed_bg: wgpu::BindGroup,
    jfa_bg_a_to_b: wgpu::BindGroup,
    jfa_bg_b_to_a: wgpu::BindGroup,
    shade_bg: wgpu::BindGroup,
}

impl VoronoiApp {
    pub fn change_sim(&mut self, device: &wgpu::Device, source_dir: PathBuf) {
        let (seed_count, seeds, colors, sim) = morph_sim::init_image(self.size.0, source_dir);
        self.seed_count = seed_count;
        self.seeds = seeds;

        self.sim = sim;

        // Update GPU buffers
        self.seed_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seeds"),
            contents: bytemuck::cast_slice(&self.seeds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params_common = ParamsCommon {
            width: self.size.0,
            height: self.size.1,
            n_seeds: self.seed_count,
            _pad: 0,
        };
        self.params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        self.color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        self.colors = colors;

        self.rebuild_bind_groups(device);
    }

    pub fn new(cc: &CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be built with the 'wgpu' feature and Renderer::Wgpu")
            .clone();
        let device = &rs.device;
        let size = (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION);
        // let seed_count = 4096;
        // let mut seeds = Vec::with_capacity(seed_count as usize);
        // let mut colors = Vec::with_capacity(seed_count as usize);
        // for x in 0..64 {
        //     for y in 0..64 {
        //         seeds.push(SeedPos {
        //             xy: [
        //                 (x as f32 + 0.5 + rng.gen::<f32>()) * (size.0 as f32 / 64.0),
        //                 (y as f32 + 0.5 + rng.gen::<f32>()) * (size.1 as f32 / 64.0),
        //             ],
        //             // rgb: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][rng.gen_range(0..3)],
        //             // _pad: [0.0, 0.0, 0.0],
        //         });

        //         colors.push(SeedColor {
        //             rgba: [
        //                 [1.0, 0.0, 0.0, 1.0],
        //                 [0.0, 1.0, 0.0, 1.0],
        //                 [0.0, 0.0, 1.0, 1.0],
        //             ][rng.gen_range(0..3)],
        //         });
        //     }
        // }
        // seeds.push(SeedPos {
        //     xy: [size.1 as f32 * 0.55, size.1 as f32 * 0.55],
        // });
        // seeds.push(SeedPos {
        //     xy: [size.1 as f32 * 0.45, size.1 as f32 * 0.45],
        // });

        // get all folders in ../presets
        let presets: Vec<PathBuf> = get_presets();

        let (seed_count, seeds, colors, sim) = morph_sim::init_image(
            size.0,
            presets[rand::random::<usize>() % presets.len()].clone(),
        );

        // === Buffers ===
        let seed_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seeds"),
            contents: bytemuck::cast_slice(&seeds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params_common = ParamsCommon {
            width: size.0,
            height: size.1,
            n_seeds: seed_count,
            _pad: 0,
        };
        let params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_jfa = ParamsJfa {
            width: size.0,
            height: size.1,
            step: 1,
            _pad: 0,
        };
        let params_jfa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_jfa"),
            contents: bytemuck::bytes_of(&params_jfa),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Textures ===
        let (ids_a, ids_a_view) = Self::make_ids_texture(device, size, Some("ids_a"));
        let (ids_b, ids_b_view) = Self::make_ids_texture(device, size, Some("ids_b"));
        let (color_tex, color_view) = Self::make_color_texture(device, size, Some("color"));

        // === Pipelines ===
        let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_clear"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let seed_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_seed_splat"),
            entries: &[
                // seeds
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params common
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ParamsCommon>() as u64
                        ),
                    },
                    count: None,
                },
                // dst ids
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let jfa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_jfa"),
            entries: &[
                // seeds
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // src ids (read-only sampled as integer texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // dst ids (write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // params_jfa
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<ParamsJfa>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let shade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_shade"),
            entries: &[
                // ids texture (u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // out color (rgba8unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // seeds (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // colors
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Shader modules
        let clear_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clear.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear.wgsl").into()),
        });
        let seed_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seed_splat.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/seed.wgsl").into()),
        });
        let jfa_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jfa.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/jfa.wgsl").into()),
        });
        let shade_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shade.wgsl").into()),
        });

        // Pipelines
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_clear"),
                    bind_group_layouts: &[&clear_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &clear_sm,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let seed_splat_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("seed_splat_pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("pl_seed"),
                        bind_group_layouts: &[&seed_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &seed_sm,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let jfa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jfa_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_jfa"),
                    bind_group_layouts: &[&jfa_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &jfa_sm,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let shade_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shade_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_shade"),
                    bind_group_layouts: &[&shade_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shade_sm,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Bind groups
        let clear_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_a"),
            layout: &clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ids_a_view),
            }],
        });
        let clear_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_b"),
            layout: &clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ids_b_view),
            }],
        });

        let seed_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_seed_splat"),
            layout: &seed_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_common_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
            ],
        });

        let jfa_bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_a_to_b"),
            layout: &jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_jfa_buf.as_entire_binding(),
                },
            ],
        });

        let jfa_bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_b_to_a"),
            layout: &jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_jfa_buf.as_entire_binding(),
                },
            ],
        });

        let shade_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade"),
            layout: &shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view), // will point to the final ids
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: color_buf.as_entire_binding(),
                },
            ],
        });

        let (progress_tx, progress_rx) = mpsc::sync_channel::<ProgressMsg>(1);

        Self {
            size,
            seed_count,
            animate: true,
            speed: 1.0,
            fps_text: String::new(),
            seeds,
            colors,
            egui_tex_id: None,
            seed_buf,
            color_buf,
            sim,
            params_common_buf,
            params_jfa_buf,
            ids_a,
            ids_b,
            ids_a_view,
            ids_b_view,
            color_tex,
            color_view,
            clear_pipeline,
            seed_splat_pipeline,
            jfa_pipeline,
            shade_pipeline,
            clear_bgl,
            seed_bgl,
            jfa_bgl,
            shade_bgl,
            clear_bg_a,
            clear_bg_b,
            seed_bg,
            jfa_bg_a_to_b,
            jfa_bg_b_to_a,
            shade_bg,
            prev_frame_time: std::time::Instant::now(),
            presets,

            progress_tx,
            progress_rx,
            show_progress_modal: false,
            last_progress: 0.0,
            process_cancelled: Arc::new(AtomicBool::new(false)),
            quick_process: false,
            currently_processing: None,

            gif_encoder: None,
            gif_status: GifStatus::None,
            gif_frame_count: 0,
            gif_palette: None,
            frame_counter: 0,
        }
    }

    fn make_ids_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        label: Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ids_view"),
            format: Some(wgpu::TextureFormat::R32Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            ..Default::default()
        });
        (tex, view)
    }

    fn make_color_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        label: Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn ensure_registered_texture(&mut self, rs: &egui_wgpu::RenderState) {
        if self.egui_tex_id.is_none() {
            let id = rs.renderer.write().register_native_texture(
                &rs.device,
                &self.color_view,
                wgpu::FilterMode::Linear,
            );
            self.egui_tex_id = Some(id);
        }
    }

    fn rebuild_bind_groups(&mut self, device: &wgpu::Device) {
        // Rebuild any BGs that reference texture views
        self.clear_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_a"),
            layout: &self.clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
            }],
        });
        self.clear_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_b"),
            layout: &self.clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
            }],
        });
        self.seed_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_seed_splat"),
            layout: &self.seed_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_common_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
            ],
        });
        self.jfa_bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_a_to_b"),
            layout: &self.jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_jfa_buf.as_entire_binding(),
                },
            ],
        });
        self.jfa_bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_b_to_a"),
            layout: &self.jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_jfa_buf.as_entire_binding(),
                },
            ],
        });
        self.shade_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade"),
            layout: &self.shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.color_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn resize_textures(&mut self, device: &wgpu::Device, new_size: (u32, u32), rebuild_bg: bool) {
        self.size = new_size;
        // Recreate textures
        let (ids_a, ids_a_view) = Self::make_ids_texture(device, self.size, Some("ids_a"));
        let (ids_b, ids_b_view) = Self::make_ids_texture(device, self.size, Some("ids_b"));
        let (color_tex, color_view) = Self::make_color_texture(device, self.size, Some("color"));
        self.ids_a = ids_a;
        self.ids_a_view = ids_a_view;
        self.ids_b = ids_b;
        self.ids_b_view = ids_b_view;
        self.color_tex = color_tex;
        self.color_view = color_view;

        // Update params_common
        let params_common = ParamsCommon {
            width: self.size.0,
            height: self.size.1,
            n_seeds: self.seed_count,
            _pad: 0,
        };
        self.params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_jfa = ParamsJfa {
            width: self.size.0,
            height: self.size.1,
            step: 1,
            _pad: 0,
        };

        self.params_jfa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_jfa"),
            contents: bytemuck::bytes_of(&params_jfa),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        if rebuild_bg {
            self.rebuild_bind_groups(device);
        }

        // Force re-registering the egui texture
        self.egui_tex_id = None;
    }

    fn run_gpu(&mut self, rs: &egui_wgpu::RenderState) {
        let device = &rs.device;

        // Prepare commands
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voronoi_jfa_encoder"),
        });

        // 1) Clear both ID textures
        for bg in [&self.clear_bg_a, &self.clear_bg_b] {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_ids_a"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            cpass.dispatch_workgroups(
                self.size.0.div_ceil(WG_SIZE_XY),
                self.size.1.div_ceil(WG_SIZE_XY),
                1,
            );
        }

        // 2) Seed splat into A
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("seed_splat"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.seed_splat_pipeline);
            cpass.set_bind_group(0, &self.seed_bg, &[]);
            cpass.dispatch_workgroups(self.seed_count.div_ceil(WG_SIZE_SEEDS), 1, 1);
            //
        }

        // 3) JFA passes, ping-pong A<->B

        let max_dim = self.size.0.max(self.size.1);
        let mut step = 1u32;
        while step < max_dim {
            step <<= 1;
        }
        step >>= 1;

        let groups_x = self.size.0.div_ceil(WG_SIZE_XY);
        let groups_y = self.size.1.div_ceil(WG_SIZE_XY);

        let mut flip = false;
        while step >= 1 {
            let pj = ParamsJfa {
                width: self.size.0,
                height: self.size.1,
                step,
                _pad: 0,
            };
            let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_jfa_staging"),
                contents: bytemuck::bytes_of(&pj),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(
                &staging,
                0,
                &self.params_jfa_buf,
                0,
                std::mem::size_of::<ParamsJfa>() as u64,
            );
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jfa_step"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.jfa_pipeline);
                cpass.set_bind_group(
                    0,
                    if !flip {
                        &self.jfa_bg_a_to_b
                    } else {
                        &self.jfa_bg_b_to_a
                    },
                    &[],
                );
                cpass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            flip = !flip;
            step >>= 1;
        }

        // if self.refined {
        //     for _ in 0..2 {
        //         let pj = ParamsJfa {
        //             width: self.size.0,
        //             height: self.size.1,
        //             step: 1,
        //             _pad: 0,
        //         };
        //         let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: Some("params_jfa_staging"),
        //             contents: bytemuck::bytes_of(&pj),
        //             usage: wgpu::BufferUsages::COPY_SRC,
        //         });
        //         encoder.copy_buffer_to_buffer(
        //             &staging,
        //             0,
        //             &self.params_jfa_buf,
        //             0,
        //             std::mem::size_of::<ParamsJfa>() as u64,
        //         );
        //         {
        //             let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //                 label: Some("jfa_step"),
        //                 timestamp_writes: None,
        //             });
        //             cpass.set_pipeline(&self.jfa_pipeline);
        //             cpass.set_bind_group(
        //                 0,
        //                 if !flip {
        //                     &self.jfa_bg_a_to_b
        //                 } else {
        //                     &self.jfa_bg_b_to_a
        //                 },
        //                 &[],
        //             );
        //             cpass.dispatch_workgroups(groups_x, groups_y, 1);
        //         }
        //         flip = !flip;
        //     }
        // }

        // 4) Shade to color (the final IDs are in A if flip is true, else in B).
        // Our shade BG was built with ids_a_view at binding 0. If the last write ended in B,
        // we temporarily rebind with B for this dispatch.
        let shade_with_b = flip; // if true, IDs live in B
        if shade_with_b {
            let tmp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_shade_tmp_b"),
                layout: &self.shade_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.seed_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.color_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_pipeline);
            cpass.set_bind_group(0, &tmp_bg, &[]);
            cpass.dispatch_workgroups(groups_x, groups_y, 1);
        } else {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_pipeline);
            cpass.set_bind_group(0, &self.shade_bg, &[]);
            cpass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // Submit
        rs.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn get_color_image_data(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let width = self.size.0;
        let height = self.size.1;
        let bpp = 4u32; // RGBA8
        let unpadded_bytes_per_row = width * bpp;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
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

        let slice = readback.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        slice.map_async(wgpu::MapMode::Read, move |res| {
            // res: Result<(), wgpu::BufferAsyncError>
            let _ = tx.send(res);
        });

        // Ensure the callback runs
        device.poll(wgpu::PollType::Wait)?;

        // Wait for the result and propagate any map error
        pollster::block_on(rx.receive()).expect("map_async sender dropped")?;
        let mapped = slice.get_mapped_range();
        // Remove row padding
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height as usize {
            let start = y * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            rgba.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        readback.unmap();
        Ok(rgba)
    }

    fn write_frame_to_gif(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rgba = self.get_color_image_data(device, queue)?;
        // let image =
        //     image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(self.size.0, self.size.1, rgba)
        //         .unwrap();
        // // resize to GIF_RESOLUTION
        // let resized = image::imageops::resize(
        //     &image,
        //     GIF_RESOLUTION,
        //     GIF_RESOLUTION,
        //     image::imageops::FilterType::Lanczos3,
        // );

        // if self.gif_status.is_recording() && self.gif_encoder.is_none() {
        //     resized.save("testframe.png")?;
        // }

        // let rgba = resized.into_raw();

        // if first frame, create encoder
        if self.gif_status.is_recording() && self.gif_encoder.is_none() {
            let file = rfd::FileDialog::new()
                .set_title("save gif")
                .add_filter("gif", &["gif"])
                .set_file_name(format!("{}.gif", self.sim.name()))
                .save_file();
            if let Some(path) = file {
                let colors = self
                    .colors
                    .iter()
                    .flat_map(|s| s.rgba.map(|f| (f * 256.0) as u8))
                    .collect::<Vec<u8>>();
                let gif_palette = NeuQuant::new(GIF_PALETTE_SAMPLEFAC, 256, &colors);

                let file = std::fs::File::create(&path)?;
                // save a test image of the palette
                // {
                //     let mut img = image::ImageBuffer::new(16, 16);
                //     for (i, pixel) in gif_palette.color_map_rgb().chunks_exact(3).enumerate() {
                //         img.put_pixel(
                //             i as u32 % 16,
                //             i as u32 / 16,
                //             image::Rgba([pixel[0], pixel[1], pixel[2], 255]),
                //         );
                //     }
                //     img.save("palette.png")?;
                // }
                // clear file
                file.set_len(0)?;
                let mut encoder = gif::Encoder::new(
                    file,
                    GIF_RESOLUTION as u16,
                    GIF_RESOLUTION as u16,
                    &gif_palette.color_map_rgb(),
                )?;
                self.gif_palette = Some(gif_palette);
                encoder.set_repeat(gif::Repeat::Infinite)?;
                self.gif_encoder = Some(encoder);
                self.gif_frame_count = 0;
                self.gif_status = GifStatus::Recording(Some(path));
            } else {
                // cancelled
                self.stop_recording_gif(device);
                return Ok(());
            }
        }

        if let Some(encoder) = &mut self.gif_encoder {
            let nq = self.gif_palette.as_ref().unwrap();
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

            frame.delay = ((100.0 / GIF_FRAMERATE as f32) / GIF_SPEED) as u16; // delay in 1/100 sec
            encoder.write_frame(&frame)?;
        }

        Ok(())
    }

    fn stop_recording_gif(&mut self, device: &wgpu::Device) {
        self.gif_status = GifStatus::None;
        self.gif_encoder = None;
        self.animate = false;
        self.resize_textures(device, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION), false);
        self.change_sim(device, self.sim.source_path());
    }
}

fn get_presets() -> Vec<PathBuf> {
    if let Ok(dir) = std::fs::read_dir("./presets") {
        dir.filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.is_dir() {
                Some(path)
            } else {
                None
            }
        })
        .collect()
    } else {
        Vec::new()
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

        self.run_gpu(rs);

        // Optionally animate seeds a bit
        if self.animate {
            if self.gif_status.is_recording() {
                for _ in 0..(60 / GIF_FRAMERATE) {
                    self.sim.update(&mut self.seeds, self.size.0);
                }

                if let Err(e) = self.write_frame_to_gif(device, &rs.queue) {
                    self.gif_status = GifStatus::Error(e.to_string());
                    self.animate = false;
                } else {
                    self.gif_frame_count += 1;

                    if self.gif_frame_count >= GIF_NUM_FRAMES {
                        // finish recording
                        // self.stop_recording_gif(device);
                        if let GifStatus::Recording(Some(path)) = self.gif_status.clone() {
                            self.gif_status = GifStatus::Complete(path);
                        } else {
                            self.gif_status = GifStatus::Error("Something weird happened".into());
                        }

                        self.animate = false;
                    }
                }
            } else {
                if self.speed == 2.0 {
                    self.sim.update(&mut self.seeds, self.size.0);
                    self.sim.update(&mut self.seeds, self.size.0);
                } else if self.speed == 1.0 {
                    self.sim.update(&mut self.seeds, self.size.0);
                } else if self.speed == 0.5 {
                    if self.frame_counter % 2 == 0 {
                        self.sim.update(&mut self.seeds, self.size.0);
                    }
                }
            }
            self.frame_counter += 1;
            rs.queue
                .write_buffer(&self.seed_buf, 0, bytemuck::cast_slice(&self.seeds));
        }

        let dt = self.prev_frame_time.elapsed();
        self.prev_frame_time = std::time::Instant::now();
        self.fps_text = format!(
            "{:5.2} ms/frame (~{:06.0} FPS)",
            dt.as_secs_f64() * 1000.0,
            1.0 / dt.as_secs_f64()
        );

        // ===== UI =====
        ctx.set_zoom_factor(1.4);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !self.animate,
                        egui::Button::new("play transformation"), //.fill(egui::Color32::from_rgb(47, 92, 34)),
                    )
                    .clicked()
                {
                    self.animate = true;
                }
                if ui
                    .add_enabled(self.animate, egui::Button::new("switch target"))
                    .clicked()
                {
                    self.sim.switch();
                }
                if ui.button("reload").clicked() {
                    self.change_sim(device, self.sim.source_path());
                    self.animate = false;
                }
                ui.separator();

                ui.label("speed:");
                if ui.button("0.5x").clicked() {
                    self.speed = 0.5;
                }
                if ui.button("1x").clicked() {
                    self.speed = 1.0;
                }
                if ui.button("2x").clicked() {
                    self.speed = 2.0;
                }

                ui.separator();

                if ui.button("save gif").clicked() {
                    self.gif_status = GifStatus::Recording(None);
                    self.resize_textures(device, (GIF_RESOLUTION, GIF_RESOLUTION), false);
                    self.change_sim(device, self.sim.source_path());
                    self.animate = true;
                    for _ in 0..20 {
                        self.sim.update(&mut self.seeds, self.size.0);
                    }
                }

                ui.separator();
                // choose preset
                // for (i, preset) in self.presets.clone().into_iter().enumerate() {
                //     if ui.button(i.to_string()).clicked() {
                //         self.change_sim(device, preset);
                //         self.animate = false;
                //     }
                // }
                ui.label("choose preset:");
                egui::ComboBox::from_label("")
                    .selected_text(self.sim.name())
                    .show_ui(ui, |ui| {
                        for preset in self.presets.clone().into_iter() {
                            if ui.button(preset_path_to_name(&preset)).clicked() {
                                // Call change_sim when a new preset is selected
                                self.change_sim(device, preset);
                                self.animate = false;
                            }
                        }
                    });
                ui.separator();
                if ui.button("obamify new image").clicked() {
                    // open file select
                    let file = rfd::FileDialog::new()
                        .set_title("choose image (square aspect ratio recommended)")
                        .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
                        .pick_file();
                    if let Some(path) = file {
                        self.show_progress_modal = true;
                        self.quick_process = false;

                        let settings = GenerationSettings::default();
                        self.currently_processing = Some(path.clone());

                        std::thread::spawn({
                            let tx = self.progress_tx.clone();
                            let cancelled = self.process_cancelled.clone();
                            move || match calculate::process(path, settings, tx.clone(), cancelled)
                            {
                                Ok(()) => {}
                                Err(err) => {
                                    tx.send(ProgressMsg::Error(err.to_string())).ok();
                                }
                            }
                        });
                    }
                }

                if self.show_progress_modal {
                    Modal::new("progress_modal".into()).show(ui.ctx(), |ui| {
                        let processing_label_message = if self.quick_process {
                            "processing..."
                        } else {
                            "processing... (could take a while)"
                        };
                        ui.vertical(|ui| {
                            ui.set_min_width(ui.available_width().min(400.0));
                            if let Ok(msg) = self.progress_rx.try_recv() {
                                match msg {
                                    ProgressMsg::Done(path) => {
                                        self.presets = get_presets();
                                        self.change_sim(device, path);
                                        self.animate = true;
                                        self.show_progress_modal = false;
                                        ui.close();
                                    }
                                    ProgressMsg::Progress(p) => {
                                        ui.label(processing_label_message);
                                        self.last_progress = p;
                                        ui.add(egui::ProgressBar::new(p).show_percentage());
                                    }
                                    ProgressMsg::Error(err) => {
                                        ui.label(format!("error: {}", err));
                                        if ui.button("close").clicked() {
                                            ui.close();
                                        }
                                    }
                                    ProgressMsg::Cancelled => {
                                        self.process_cancelled
                                            .store(false, std::sync::atomic::Ordering::Relaxed);
                                        if self.quick_process {
                                            let settings = GenerationSettings::quick_process();

                                            std::thread::spawn({
                                                let tx = self.progress_tx.clone();
                                                let cancelled = self.process_cancelled.clone();
                                                let path =
                                                    self.currently_processing.clone().unwrap();
                                                move || match calculate::process(
                                                    path,
                                                    settings,
                                                    tx.clone(),
                                                    cancelled,
                                                ) {
                                                    Ok(()) => {}
                                                    Err(err) => {
                                                        tx.send(ProgressMsg::Error(
                                                            err.to_string(),
                                                        ))
                                                        .ok();
                                                    }
                                                }
                                            });
                                        } else {
                                            self.show_progress_modal = false;
                                            ui.close();
                                        }
                                    }
                                }
                            } else {
                                if self
                                    .process_cancelled
                                    .load(std::sync::atomic::Ordering::Relaxed)
                                {
                                    ui.label("cancelling...");
                                } else {
                                    ui.label(processing_label_message);
                                }
                                ui.add(
                                    egui::ProgressBar::new(self.last_progress).show_percentage(),
                                );
                            }
                            ui.horizontal(|ui| {
                                if ui.button("cancel").clicked() {
                                    self.quick_process = false;
                                    self.process_cancelled
                                        .store(true, std::sync::atomic::Ordering::Relaxed);
                                    self.last_progress = 0.0;
                                }

                                if !self.quick_process
                                    && ui
                                        .button("make faster, lower quality result instead")
                                        .clicked()
                                {
                                    self.process_cancelled
                                        .store(true, std::sync::atomic::Ordering::Relaxed);
                                    self.last_progress = 0.0;
                                    self.quick_process = true;
                                }
                            })
                        });
                    });

                    // if modal.should_close() {
                    //     self.show_progress_modal = false;
                    // }
                } else if !self.gif_status.not_recording() {
                    Modal::new("recording_progress".into()).show(ui.ctx(), |ui| {
                        match self.gif_status.clone() {
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
                                    if ui.button("open folder").clicked() {
                                        if let Some(parent) = path.parent() {
                                            opener::open(parent).ok();
                                        }
                                    }
                                    if ui.button("close").clicked() {
                                        self.stop_recording_gif(device);
                                    }
                                });
                            }
                            GifStatus::None => unreachable!(),
                        }
                    });
                }

                // ui.separator();
                // ui.label(&self.fps_text);
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
                    } else {
                        ui.colored_label(Color32::LIGHT_RED, "Texture not ready");
                    }
                });
            });

        // continuous repaint for animation
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result<()> {
    // get screen dimensions
    let native_options = NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size((1024.0, 1024.0))
            .with_clamp_size_to_monitor_size(true)
            .with_resizable(true),
        ..Default::default()
    }; // wgpu backend is selected via the `wgpu` feature
    eframe::run_native(
        "obamify",
        native_options,
        Box::new(|cc| Ok(Box::new(VoronoiApp::new(cc)))),
    )
}

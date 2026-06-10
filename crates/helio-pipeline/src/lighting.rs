//! Fullscreen deferred PBR lighting sampling the tiled 2D shadow atlas.
//!
//! Mirrors the classic `DeferredLightPass` output (Cook-Torrance BRDF,
//! normal-offset shadow bias, CSM blend zones, ACES tonemap) so switching to
//! the GPU-driven pipeline causes no visual shift, but reads shadows from
//! [`PipelineShared`]'s single tiled atlas through the per-face matrices that
//! `CullPass` uploads each frame. Publishes `frame.pre_aa` with the same
//! semantics as the classic pass so the TAA / overlay tail keeps working.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

use crate::PipelineShared;

/// WGSL mirror: `Globals` in shaders/lighting.wgsl. Uniform-compatible layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    /// rgb = ambient colour, w = ambient intensity.
    ambient_color: [f32; 4],
    csm_splits: [f32; 4],
    /// Reserved (always zero) — keeps the layout stable for future extension.
    camera_unused: [f32; 4],
    light_count: u32,
    frame: u32,
    has_sky: u32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<Globals>() == 64);

pub struct LightingPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    /// (albedo view ptr, depth view ptr, lights buffer ptr, lights buffer size).
    /// The size term catches GrowableBuffer reallocation: the `&Buffer` address
    /// can stay identical across a realloc while the underlying buffer changed.
    bind_group_key: Option<(usize, usize, usize, u64)>,
    globals_buf: wgpu::Buffer,
    /// Kept alive alongside its view; never read directly.
    #[allow(dead_code)]
    pre_aa_tex: wgpu::Texture,
    pub pre_aa_view: wgpu::TextureView,
    shared: Arc<Mutex<PipelineShared>>,
    /// Clear colour for the no-sky path, refreshed from main_scene in prepare().
    clear_color: [f64; 4],
}

impl LightingPass {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        pre_aa_format: wgpu::TextureFormat,
        shared: Arc<Mutex<PipelineShared>>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PipelineLighting Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/lighting.wgsl").into()),
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PipelineLighting Globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One bind group for everything — unlike the classic pass there is no
        // env/RC/caustics plumbing here, so a single group keeps rebuilds cheap.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PipelineLighting BGL"),
            entries: &[
                uniform_entry(0),    // camera
                uniform_entry(1),    // globals
                storage_ro_entry(2), // lights
                storage_ro_entry(3), // face view-proj matrices
                texture_entry(4),    // gbuffer albedo
                texture_entry(5),    // gbuffer normal
                texture_entry(6),    // gbuffer orm
                texture_entry(7),    // gbuffer emissive
                depth_entry(8),      // scene depth
                depth_entry(9),      // tiled shadow atlas
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLighting PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PipelineLighting Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: pre_aa_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // HDR output consumed by TAA; same usage as the classic pre-AA target.
        let pre_aa_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PipelineLighting PreAA"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: pre_aa_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pre_aa_view = pre_aa_tex.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            globals_buf,
            pre_aa_tex,
            pre_aa_view,
            shared,
            clear_color: [0.02, 0.02, 0.03, 1.0],
        }
    }
}

impl RenderPass for LightingPass {
    fn name(&self) -> &'static str {
        "PipelineLighting"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Ambient + clear colour come from the high-level scene when present;
        // the fallbacks render a dim editor void rather than pure black.
        let (ambient_color, ambient_intensity, clear_color) =
            match ctx.frame_resources.main_scene.as_ref() {
                Some(main) => (main.ambient_color, main.ambient_intensity, main.clear_color),
                None => ([0.1, 0.1, 0.15], 0.1, [0.02, 0.02, 0.03, 1.0]),
            };
        self.clear_color = [
            clear_color[0] as f64,
            clear_color[1] as f64,
            clear_color[2] as f64,
            clear_color[3] as f64,
        ];

        let globals = Globals {
            ambient_color: [
                ambient_color[0],
                ambient_color[1],
                ambient_color[2],
                ambient_intensity,
            ],
            // Must match the splits views.rs built the cascade matrices for, or
            // cascade selection samples outside the matrices' valid range.
            csm_splits: libhelio::CSM_SPLITS,
            camera_unused: [0.0; 4],
            // Only movable lights exist at runtime (static/stationary are baked).
            light_count: ctx.scene.movable_light_count,
            frame: ctx.frame_num as u32,
            has_sky: ctx.frame_resources.sky_lut.is_some() as u32,
            _pad: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gbuffer = ctx.resources.gbuffer.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig("PipelineLighting requires frame.gbuffer".into())
        })?;

        // Atlas view, face matrices and comparison sampler live in the shared
        // pipeline state; hold the guard for the rest of execute() so nothing
        // can swap them while we encode against them.
        let shared = self.shared.lock().unwrap();

        // Lazy bind group: the shared atlas resources are created once and never
        // reallocated, so the key only tracks the per-frame-variable bindings.
        let key = (
            gbuffer.albedo as *const _ as usize,
            ctx.depth as *const _ as usize,
            ctx.scene.lights as *const _ as usize,
            ctx.scene.lights.size(),
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PipelineLighting BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.globals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shared.face_mats_buf.as_entire_binding(),
                    },
                    texture_view_entry(4, gbuffer.albedo),
                    texture_view_entry(5, gbuffer.normal),
                    texture_view_entry(6, gbuffer.orm),
                    texture_view_entry(7, gbuffer.emissive),
                    texture_view_entry(8, ctx.depth),
                    texture_view_entry(9, &shared.atlas_view),
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&shared.compare_sampler),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        // Same target contract as the classic DeferredLightPass: render into an
        // upstream-published pre_aa when present, else our own. When a sky pass
        // ran, load its output (the shader discards sky pixels); otherwise clear
        // to the scene's background colour.
        let target = ctx.resources.pre_aa.unwrap_or(&self.pre_aa_view);
        let load = if ctx.resources.sky_lut.is_some() {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: self.clear_color[0],
                g: self.clear_color[1],
                b: self.clear_color[2],
                a: self.clear_color[3],
            })
        };

        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("PipelineLighting"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        if frame.pre_aa.is_none() {
            frame.pre_aa = Some(&self.pre_aa_view);
        }
    }
}

// ── Bind group layout helpers ──────────────────────────────────────────────────

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // G-buffer reads use textureLoad, so non-filterable Float suffices
            // (Rgba16Float targets are not filterable without extra features).
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

//! Pipeline G-buffer geometry pass.
//!
//! Contract-identical sibling of the classic `GBufferPass` (helio-pass-gbuffer):
//! same 5-MRT layout, same bindless material model, same `frame.gbuffer`
//! publish contract — so every tail pass (VG, TAA, billboards, water, perf
//! overlay) keeps working. The only behavioural difference is the draw source:
//! geometry is drawn from the camera's (view 0) GPU-culled indirect list in
//! [`crate::PipelineShared`] produced by `CullPass`, instead of the scene's
//! full `ctx.scene.indirect` list.
//!
//! # Render Targets (owned by this pass)
//!
//! | Slot | Name     | Format        | Contents                          |
//! |------|----------|---------------|-----------------------------------|
//! | 0    | albedo   | Rgba8Unorm    | albedo.rgb + alpha                |
//! | 1    | normal   | Rgba16Float   | world normal.xyz + F0.r           |
//! | 2    | orm      | Rgba8Unorm    | AO, roughness, metallic, F0.g     |
//! | 3    | emissive | Rgba16Float   | emissive.rgb + F0.b               |
//!
//! # Material Bind Group
//!
//! Group 1 provides bindless texture access:
//!  - binding 0: materials storage buffer
//!  - binding 1: material_textures storage buffer (MaterialTextureData array)
//!  - binding 2: scene_textures (256-slot texture array)
//!  - binding 3: scene_samplers (256-slot sampler array)
//!
//! # Vertex / Index Buffers
//!
//! This pass owns no mesh data.  The caller must bind the shared mesh vertex
//! buffer (slot 0) and index buffer before this pass executes.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

/// Bindless texture array size per shader stage.
/// Capped at 16 on wasm32 and Apple native Metal; 256 on other desktop backends.
#[cfg(not(any(target_arch = "wasm32", target_os = "macos", target_os = "ios")))]
const MAX_TEXTURES: usize = 256;
#[cfg(any(target_arch = "wasm32", target_os = "macos", target_os = "ios"))]
const MAX_TEXTURES: usize = 16;

// ── Uniform types ─────────────────────────────────────────────────────────────

/// Per-frame globals uploaded to the GPU each frame (matches `Globals` in geometry.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GBufferGlobals {
    pub frame: u32,
    pub delta_time: f32,
    pub light_count: u32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 4],
    pub rc_world_min: [f32; 4],
    pub rc_world_max: [f32; 4],
    pub csm_splits: [f32; 4],
    pub debug_mode: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct GeometryPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    /// Group 0: camera + globals + instance_data. Rebuilt when buffer identity changes.
    bind_group_0: Option<wgpu::BindGroup>,
    /// (camera ptr, camera size, instances ptr, instances size). Size is part of
    /// the key because a GrowableBuffer reallocation replaces the inner
    /// `wgpu::Buffer` in place — the `&Buffer` address may stay identical while
    /// the underlying GPU resource changed.
    bind_group_0_key: Option<(usize, u64, usize, u64)>,
    /// Group 1: materials + material_textures + bindless texture arrays.
    bind_group_1: Option<wgpu::BindGroup>,
    /// (material textures version, materials ptr, materials size) — sizes for
    /// the same GrowableBuffer-reallocation reason as `bind_group_0_key`.
    bind_group_1_key: Option<(u64, usize, u64)>,
    /// Per-frame globals uploaded in `prepare()`.
    globals_buf: wgpu::Buffer,
    /// Cross-pass pipeline state: per-view culled indirect lists + count buffer
    /// written by `CullPass`. This pass only reads view 0 (the camera).
    shared: Arc<Mutex<crate::PipelineShared>>,
    /// CSM cascade split distances. Must match the values used in shadow_matrices.wgsl
    /// so that cascade selection in any shader that reads `globals.csm_splits` is
    /// consistent with the shadow maps that were actually generated.
    pub csm_splits: [f32; 4],
    /// Debug visualisation mode forwarded to the GBuffer shader (0 = off).
    pub debug_mode: u32,
    // ── GBuffer textures (owned; exposed for downstream passes) ───────────────
    pub albedo_tex: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
    pub normal_tex: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub orm_tex: wgpu::Texture,
    pub orm_view: wgpu::TextureView,
    pub emissive_tex: wgpu::Texture,
    pub emissive_view: wgpu::TextureView,
    pub lightmap_uv_tex: wgpu::Texture,
    pub lightmap_uv_view: wgpu::TextureView,
    /// Lightmap atlas regions buffer (empty until bake data is loaded)
    lightmap_atlas_regions_buf: wgpu::Buffer,
}

impl GeometryPass {
    /// Create the geometry pass.
    ///
    /// * `width/height` – initial render resolution
    /// * `shared`       – pipeline shared state (culled draw lists from CullPass)
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        shared: Arc<Mutex<crate::PipelineShared>>,
    ) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader_src = include_str!("../shaders/geometry.wgsl")
            .replace(
                "binding_array<texture_2d<f32>, 256>",
                &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
            )
            .replace(
                "binding_array<sampler, 256>",
                &format!("binding_array<sampler, {MAX_TEXTURES}>"),
            );
        // WebGPU does not support binding arrays; collapse to single texture/sampler.
        #[cfg(target_arch = "wasm32")]
        let shader_src = shader_src
            .replace(
                &format!("binding_array<sampler, {MAX_TEXTURES}>"),
                "sampler",
            )
            .replace("scene_samplers[slot.texture_index]", "scene_samplers")
            .replace(
                &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
                "texture_2d<f32>",
            )
            .replace("scene_textures[slot.texture_index]", "scene_textures")
            // textureSample requires uniform control flow on WebGPU; use textureSampleLevel (LOD 0) instead.
            .replace(
                "return textureSample(scene_textures, scene_samplers, uv);",
                "return textureSampleLevel(scene_textures, scene_samplers, uv, 0.0);",
            );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PipelineGeometry Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Globals buffer ────────────────────────────────────────────────────
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PipelineGeometry/Globals"),
            size: std::mem::size_of::<GBufferGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind Group Layout 0 ───────────────────────────────────────────────
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PipelineGeometry BGL 0"),
                entries: &[
                    // binding 0: camera (uniform, VERTEX | FRAGMENT)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: globals (uniform, FRAGMENT)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: instance_data (storage read, VERTEX)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: lightmap_atlas_regions (storage read, VERTEX)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // ── Bind Group Layout 1: material + textures ──────────────────────────
        let bind_group_layout_1 = create_geometry_material_bgl(device);

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineGeometry PL"),
            bind_group_layouts: &[Some(&bind_group_layout_0), Some(&bind_group_layout_1)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PipelineGeometry Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Full vertex layout (stride = 40 bytes, matching shared mesh buffer).
                //   offset  0 — position       Float32x3  location 0
                //   offset 12 — bitangent_sign Float32    location 1
                //   offset 16 — tex_coords0   Float32x2  location 2  (UV0: material/albedo)
                //   offset 24 — tex_coords1   Float32x2  location 5  (UV1: lightmap)
                //   offset 32 — normal        Uint32     location 3
                //   offset 36 — tangent       Uint32     location 4
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2,
                        },
                        // UV1 — dedicated lightmap UV channel (non-overlapping, [0,1] per mesh).
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 32,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 36,
                            shader_location: 4,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Same depth contract as the classic GBufferPass: this pass owns
                // the depth write (no depth prepass runs ahead of it); LessEqual
                // keeps early-Z robust against floating-point precision issues.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── GBuffer textures ──────────────────────────────────────────────────
        let (albedo_tex, albedo_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            "GBuffer/Albedo",
        );
        let (normal_tex, normal_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "GBuffer/Normal",
        );
        let (orm_tex, orm_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            "GBuffer/ORM",
        );
        let (emissive_tex, emissive_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "GBuffer/Emissive",
        );
        let (lightmap_uv_tex, lightmap_uv_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rg16Float,
            "GBuffer/LightmapUV",
        );

        // Create empty lightmap atlas regions buffer (populated when bake data is loaded).
        // Each region is 32 bytes: uv_offset(8) + uv_scale(8) + uv_clamp_min(8) + uv_clamp_max(8).
        let lightmap_atlas_regions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lightmap Atlas Regions"),
            size: 32, // One empty region sentinel; recreated when real data loads.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_0: None,
            bind_group_0_key: None,
            bind_group_1: None,
            bind_group_1_key: None,
            globals_buf,
            shared,
            // Default CSM splits — single source of truth is libhelio::CSM_SPLITS.
            csm_splits: libhelio::CSM_SPLITS,
            debug_mode: 0,
            albedo_tex,
            albedo_view,
            normal_tex,
            normal_view,
            orm_tex,
            orm_view,
            emissive_tex,
            emissive_view,
            lightmap_uv_tex,
            lightmap_uv_view,
            lightmap_atlas_regions_buf,
        }
    }
}

impl RenderPass for GeometryPass {
    fn name(&self) -> &'static str {
        "PipelineGeometry"
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.resize(device, width, height);
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.gbuffer = Some(libhelio::GBufferViews {
            albedo: &self.albedo_view,
            normal: &self.normal_view,
            orm: &self.orm_view,
            emissive: &self.emissive_view,
        });
        frame.gbuffer_lightmap_uv = Some(&self.lightmap_uv_view);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Read per-scene values from frame_resources so the GBuffer globals match
        // what the renderer configured (ambient light, GI bounds, etc.).
        let (ambient_color, ambient_intensity, rc_world_min, rc_world_max) =
            if let Some(ref ms) = ctx.frame_resources.main_scene {
                (
                    [ms.ambient_color[0], ms.ambient_color[1], ms.ambient_color[2], 1.0],
                    ms.ambient_intensity,
                    [ms.rc_world_min[0], ms.rc_world_min[1], ms.rc_world_min[2], 0.0],
                    [ms.rc_world_max[0], ms.rc_world_max[1], ms.rc_world_max[2], 0.0],
                )
            } else {
                // Fallback for headless / test usage without a full renderer.
                ([0.1, 0.1, 0.15, 1.0], 0.1, [-100.0_f32; 4], [100.0_f32; 4])
            };

        // Upload per-frame globals (O(1) — fixed-size struct).
        let globals = GBufferGlobals {
            frame: ctx.frame_num as u32,
            delta_time: ctx.delta_time,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity,
            ambient_color,
            rc_world_min,
            rc_world_max,
            csm_splits: self.csm_splits,
            debug_mode: self.debug_mode,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Draw source is view 0 (camera) of the culled lists written by CullPass.
        // shared.draw_count was captured by CullPass::prepare, so every pipeline
        // pass sees the same stable-slot list length this frame. The guard stays
        // alive for the whole function: the render pass below borrows
        // shared.culled_indirect / shared.view_counts.
        let shared = self.shared.lock().unwrap();
        let draw_count = shared.draw_count.min(crate::MAX_DRAWS_PER_VIEW);
        if draw_count == 0 {
            return Ok(());
        }
        let main_scene = ctx.resources.main_scene.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig(
                "PipelineGeometry requires main_scene mesh buffers".to_string(),
            )
        })?;

        // Rebuild bind group 0 when camera or instances buffer identity changes
        // (pointer OR size — GrowableBuffer realloc can keep the same &Buffer address).
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let instances_ptr = ctx.scene.instances as *const _ as usize;
        let key = (
            camera_ptr,
            ctx.scene.camera.size(),
            instances_ptr,
            ctx.scene.instances.size(),
        );
        if self.bind_group_0_key != Some(key) {
            log::debug!("PipelineGeometry: rebuilding bind group 0 (buffer identity changed)");
            self.bind_group_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PipelineGeometry BG 0"),
                layout: &self.bind_group_layout_0,
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
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lightmap_atlas_regions_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_0_key = Some(key);
        }

        // Rebuild bind group 1 when the material textures version or the
        // materials buffer identity changes.
        let materials_ptr = ctx.scene.materials as *const _ as usize;
        let key1 = (
            main_scene.material_textures.version,
            materials_ptr,
            ctx.scene.materials.size(),
        );
        let needs_rebuild = self.bind_group_1_key != Some(key1) || self.bind_group_1.is_none();
        if needs_rebuild {
            log::debug!("PipelineGeometry: rebuilding bind group 1 (material textures changed)");
            self.bind_group_1 = Some(
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PipelineGeometry BG 1"),
                    layout: &self.bind_group_layout_1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ctx.scene.materials.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: main_scene
                                .material_textures
                                .material_textures
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            #[cfg(not(target_arch = "wasm32"))]
                            resource: wgpu::BindingResource::TextureViewArray(
                                main_scene.material_textures.texture_views,
                            ),
                            #[cfg(target_arch = "wasm32")]
                            resource: wgpu::BindingResource::TextureView(
                                main_scene
                                    .material_textures
                                    .texture_views
                                    .first()
                                    .copied()
                                    .expect("scene must have at least one texture view"),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            #[cfg(not(target_arch = "wasm32"))]
                            resource: wgpu::BindingResource::SamplerArray(
                                main_scene.material_textures.samplers,
                            ),
                            #[cfg(target_arch = "wasm32")]
                            resource: wgpu::BindingResource::Sampler(
                                main_scene
                                    .material_textures
                                    .samplers
                                    .first()
                                    .copied()
                                    .expect("scene must have at least one sampler"),
                            ),
                        },
                    ],
                }),
            );
            self.bind_group_1_key = Some(key1);
        }

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PipelineGeometry"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.albedo_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.normal_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.orm_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.emissive_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.lightmap_uv_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0), // Geometry owns depth; clear to far
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
        pass.set_index_buffer(
            main_scene.mesh_buffers.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );

        // With MULTI_DRAW_INDIRECT_COUNT the cull shader compacted view 0's list
        // and the GPU count buffer bounds the actual draw count. Without it the
        // shader wrote stable slots (culled slots keep instance_count = 0, which
        // costs nothing to issue), so we draw the full slot range.
        #[cfg(not(target_arch = "wasm32"))]
        {
            if shared.supports_multi_draw_count {
                pass.multi_draw_indexed_indirect_count(
                    &shared.culled_indirect,
                    crate::PipelineShared::indirect_offset(0),
                    &shared.view_counts,
                    crate::PipelineShared::count_offset(0),
                    draw_count,
                );
            } else {
                pass.multi_draw_indexed_indirect(
                    &shared.culled_indirect,
                    crate::PipelineShared::indirect_offset(0),
                    draw_count,
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        for i in 0..draw_count {
            pass.draw_indexed_indirect(
                &shared.culled_indirect,
                crate::PipelineShared::indirect_offset(0) + i as u64 * 20,
            );
        }
        Ok(())
    }
}

impl GeometryPass {
    /// Recreates GBuffer textures at a new resolution (call on window resize).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (albedo_tex, albedo_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            "GBuffer/Albedo",
        );
        let (normal_tex, normal_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "GBuffer/Normal",
        );
        let (orm_tex, orm_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            "GBuffer/ORM",
        );
        let (emissive_tex, emissive_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "GBuffer/Emissive",
        );
        let (lightmap_uv_tex, lightmap_uv_view) = gbuffer_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rg16Float,
            "GBuffer/LightmapUV",
        );
        self.albedo_tex = albedo_tex;
        self.albedo_view = albedo_view;
        self.normal_tex = normal_tex;
        self.normal_view = normal_view;
        self.orm_tex = orm_tex;
        self.orm_view = orm_view;
        self.emissive_tex = emissive_tex;
        self.emissive_view = emissive_view;
        self.lightmap_uv_tex = lightmap_uv_tex;
        self.lightmap_uv_view = lightmap_uv_view;
    }

    /// Populate the lightmap atlas regions buffer from baked data.
    ///
    /// Called by the renderer after a successful bake to upload per-mesh UV atlas regions.
    /// Each region maps a mesh_id to its atlas location with precomputed clamp bounds that
    /// prevent bilinear filtering from bleeding across adjacent atlas region boundaries.
    ///
    /// # Arguments
    /// * `queue`   - GPU queue for buffer uploads
    /// * `regions` - `[uv_offset.x, uv_offset.y, uv_scale.x, uv_scale.y,
    ///                uv_clamp_min.x, uv_clamp_min.y, uv_clamp_max.x, uv_clamp_max.y]`
    ///              per mesh (32 bytes each, matching `LightmapAtlasRegion` in geometry.wgsl)
    pub fn upload_lightmap_atlas_regions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        regions: &[[f32; 8]],
    ) {
        if regions.is_empty() {
            return;
        }

        // Each region is 32 bytes: uv_offset(8) + uv_scale(8) + uv_clamp_min(8) + uv_clamp_max(8).
        let buf_size = (regions.len() * 32) as u64;

        // Recreate buffer if size changed
        if self.lightmap_atlas_regions_buf.size() < buf_size {
            self.lightmap_atlas_regions_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lightmap Atlas Regions"),
                size: buf_size.max(32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Invalidate bind group since buffer changed
            self.bind_group_0_key = None;
        }

        queue.write_buffer(&self.lightmap_atlas_regions_buf, 0, bytemuck::cast_slice(regions));

        log::info!(
            "[GeometryPass] Uploaded {} lightmap atlas regions ({} bytes)",
            regions.len(),
            buf_size
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Create a screen-sized GBuffer texture and its default view.
fn gbuffer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Build the BGL for group 1 (bindless materials + textures).
fn create_geometry_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_array_count =
        NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero texture table size");

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("PipelineGeometry BGL 1"),
        entries: &[
            // binding 0: materials storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: material_textures storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: scene_textures (texture array; collapsed to 1 on wasm32)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(texture_array_count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
            // binding 3: scene_samplers (256-slot sampler array)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(texture_array_count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
        ],
    })
}

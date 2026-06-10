//! Single-pass tiled shadow atlas.
//!
//! Renders every dirty shadow face depth-only into ONE 2D `Depth32Float`
//! atlas inside ONE render pass, using per-face viewports/scissors instead of
//! per-face array layers. This is the structural fix for the classic design's
//! light-move cost: one attachment transition per frame (vs 12 render passes
//! against two 1 GB texture arrays), and each face draws a pre-culled list
//! produced by `CullPass` (face `f` consumes view `1 + f`).
//!
//! Tile lifecycle:
//! * First run — `LoadOp::Clear(1.0)` initialises the whole attachment so the
//!   lighting pass never samples undefined depth, and all active faces render.
//! * Steady state — `LoadOp::Load` preserves clean tiles; each dirty tile is
//!   cleared in-pass by a scissored fullscreen triangle at z = 1.0
//!   (`DepthCompare::Always`), then re-renders its culled geometry.
//! * No dirty faces — the pass returns without touching the encoder.
//!
//! Depth conventions match the classic `ShadowPass` exactly (front-face
//! culling, slope-scaled bias 2.0) so the lighting math is unchanged.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

use crate::{PipelineShared, MAX_DRAWS_PER_VIEW, MAX_SHADOW_FACES};

/// Byte stride between consecutive entries in `face_view_idx_buf`.
///
/// Must satisfy `device.limits().min_uniform_buffer_offset_alignment`, which is
/// guaranteed to be ≤ 256 on every wgpu backend (Metal, Vulkan, DX12, WebGPU).
const FACE_BUF_STRIDE: u64 = 256;

pub struct ShadowAtlasPass {
    /// Depth-only geometry pipeline (front-face culled, slope-scaled bias).
    pipeline: wgpu::RenderPipeline,

    /// Tile-clear pipeline — scissored fullscreen triangle at z = 1.0 with
    /// `DepthCompare::Always`, so a stale tile can be reset without a
    /// `LoadOp::Clear` on the whole 8192² attachment.
    clear_pipeline: wgpu::RenderPipeline,

    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    /// (instances buffer ptr, instances buffer size). GrowableBuffer realloc
    /// can preserve the `&Buffer` address while the size changes, so the size
    /// participates in the key.
    bind_group_key: Option<(usize, u64)>,

    /// Per-face VIEW index — entry `f` holds `1 + f` (view 0 is the camera).
    /// One u32 per `FACE_BUF_STRIDE` slot, selected via dynamic offset.
    /// Written once at construction; the CPU never touches it again.
    face_view_idx_buf: wgpu::Buffer,

    shared: Arc<Mutex<PipelineShared>>,
}

impl ShadowAtlasPass {
    /// Allocate all GPU resources. Called once; zero allocations after this
    /// (besides the dirty-face list, which is empty in steady state).
    pub fn new(device: &wgpu::Device, shared: Arc<Mutex<PipelineShared>>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PipelineShadowAtlas"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_atlas.wgsl").into()),
        });

        // ── Bind group layout ─────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PipelineShadowAtlas BGL"),
            entries: &[
                // binding 0: unified view array (camera + shadow faces)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: per-instance world transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: face → view index, dynamic offset selects the face
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        });

        // ── Geometry pipeline ─────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineShadowAtlas PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PipelineShadowAtlas Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Shared mesh vertex buffer layout (stride = 40 bytes, matches
                // the G-buffer pass). Only position is needed for depth.
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            // Depth-only: no colour outputs, no fragment shader.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Front-face culling: writing the light-facing surfaces' depth
                // directly causes self-shadow acne; culling them is the same
                // convention as the classic ShadowPass (and UE4/Unity).
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                // slope_scale compensates FP depth precision on surfaces at
                // grazing angles to the light; a constant bias caused a visible
                // offset in the classic pass, so it stays 0.
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Tile-clear pipeline ───────────────────────────────────────────────
        // No bindings, no vertex buffers: vs_clear synthesises a fullscreen
        // triangle at z = 1.0 (far plane) and DepthCompare::Always overwrites
        // whatever the stale tile held. The per-face scissor confines it.
        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PipelineShadowAtlas/Clear PL"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PipelineShadowAtlas/Clear Pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_clear"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Face → view-index uniform ─────────────────────────────────────────
        // Entry f holds (1 + f): the culling view that owns face f's draw list.
        let face_view_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PipelineShadowAtlas/FaceViewIdx"),
            size: MAX_SHADOW_FACES as u64 * FACE_BUF_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut map = face_view_idx_buf.slice(..).get_mapped_range_mut();
            for f in 0..MAX_SHADOW_FACES as usize {
                let offset = f * FACE_BUF_STRIDE as usize;
                // Write the view index as a native-endian u32; the rest of the
                // 256-byte slot is zero-initialised by wgpu (mapped buffers are
                // zeroed).
                map[offset..offset + 4].copy_from_slice(&(1 + f as u32).to_ne_bytes());
            }
        }
        face_view_idx_buf.unmap();

        Self {
            pipeline,
            clear_pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            face_view_idx_buf,
            shared,
        }
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowAtlasPass {
    fn name(&self) -> &'static str {
        "PipelineShadowAtlas"
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        // Nothing to upload: face_view_idx_buf is static, and all per-frame
        // inputs (view matrices, dirty flags, draw counts) are produced by
        // CullPass::prepare into PipelineShared.
        Ok(())
    }

    // Note: no publish() override. `frame.shadow_atlas` is a D2Array contract
    // in FrameResources; this atlas is a plain D2 texture and the lighting
    // pass reaches it through PipelineShared instead.

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let mut shared = self.shared.lock().unwrap();

        let face_count = (shared.face_count as usize).min(MAX_SHADOW_FACES as usize);
        // The first run must clear the whole attachment so the lighting pass
        // never samples undefined depth — even if no face is active yet.
        let needs_init = !shared.atlas_initialized;
        let dirty_faces: Vec<usize> = (0..face_count)
            .filter(|&f| shared.face_active[f] && (shared.face_dirty[f] || needs_init))
            .collect();
        if dirty_faces.is_empty() && !needs_init {
            return Ok(());
        }

        let main_scene = ctx.resources.main_scene.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig("PipelineShadowAtlas requires main_scene".into())
        })?;
        let vertices = main_scene.mesh_buffers.vertices;
        let indices = main_scene.mesh_buffers.indices;

        // ── Lazy bind group ───────────────────────────────────────────────────
        // views_buf and face_view_idx_buf never reallocate; only the instances
        // GrowableBuffer can, so it alone keys the rebuild (O(1) amortised).
        let key = (
            ctx.scene.instances as *const _ as usize,
            ctx.scene.instances.size(),
        );
        if self.bind_group_key != Some(key) {
            log::debug!("PipelineShadowAtlas: rebuilding bind group (instances buffer changed)");
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PipelineShadowAtlas BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared.views_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.face_view_idx_buf,
                            offset: 0,
                            size: NonZeroU64::new(16),
                        }),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }
        let bg = self.bind_group.as_ref().unwrap();

        // ── ONE render pass for every dirty face ──────────────────────────────
        // Scoped so the pass's borrows of shared.atlas_view / culled_indirect /
        // view_counts end before the dirty flags are mutated below.
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PipelineShadowAtlas"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shared.atlas_view,
                    depth_ops: Some(wgpu::Operations {
                        load: if needs_init {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            // Preserve every clean tile; dirty tiles are reset
                            // in-pass by the scissored clear triangle.
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            for &f in &dirty_faces {
                let (vx, vy, vw, vh) = crate::atlas::face_viewport(f as u32);
                pass.set_viewport(vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0);
                pass.set_scissor_rect(vx, vy, vw, vh);

                // (a) Reset the stale tile to the far plane. Skipped on the
                //     first run: LoadOp::Clear already cleared the attachment.
                if !needs_init {
                    pass.set_pipeline(&self.clear_pipeline);
                    pass.draw(0..3, 0..1);
                }

                // (b) Draw this face's pre-culled geometry (view 1 + f).
                if shared.draw_count > 0 {
                    let dyn_offset = (f as u64 * FACE_BUF_STRIDE) as u32;
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, bg, &[dyn_offset]);
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);

                    let view_idx = (1 + f) as u32;
                    let max = shared.draw_count.min(MAX_DRAWS_PER_VIEW);
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if shared.supports_multi_draw_count {
                            // Compacted list: the GPU count buffer holds the
                            // exact number of surviving draws for this view.
                            pass.multi_draw_indexed_indirect_count(
                                &shared.culled_indirect,
                                PipelineShared::indirect_offset(view_idx),
                                &shared.view_counts,
                                PipelineShared::count_offset(view_idx),
                                max,
                            );
                        } else {
                            // Stable-slot mode: culled slots were written with
                            // instance_count = 0 and cost nothing to consume.
                            pass.multi_draw_indexed_indirect(
                                &shared.culled_indirect,
                                PipelineShared::indirect_offset(view_idx),
                                max,
                            );
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    for i in 0..max {
                        pass.draw_indexed_indirect(
                            &shared.culled_indirect,
                            PipelineShared::indirect_offset(view_idx) + i as u64 * 20,
                        );
                    }
                }
            }
        }

        // Pass dropped — safe to mutate the guard. Clearing per-face (rather
        // than wholesale) keeps flags CullPass set after our prepare() intact.
        for &f in &dirty_faces {
            shared.face_dirty[f] = false;
            shared.face_rendered_once[f] = true;
        }
        shared.atlas_initialized = true;

        Ok(())
    }
}

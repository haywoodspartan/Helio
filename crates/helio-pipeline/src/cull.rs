//! Unified multi-view culling pass.
//!
//! ONE compute dispatch frustum-culls every draw template against every view —
//! camera (view 0) plus all shadow faces (views 1..N) — writing per-view
//! `DrawIndexedIndirect` lists into `PipelineShared::culled_indirect` and
//! per-view counts into `PipelineShared::view_counts`. Per-face shadow culling
//! falls out for free: shadow faces are just more views (DESIGN.md D2).
//!
//! `prepare()` also carries this pipeline's per-frame CPU work:
//! * rebuild the view set on CPU ([`crate::views::ViewBuilder`] — DESIGN.md D3)
//!   and upload it (views + face matrices),
//! * fold the builder's per-face change reports + the movable-objects
//!   generation into `PipelineShared::face_dirty` so ShadowAtlasPass knows
//!   which tiles to re-render (DESIGN.md D4),
//! * zero the per-view count buffer so the shader's `atomicAdd` slot
//!   allocation starts fresh.
//!
//! The shader emits in one of two modes (`Params.compact`, both paths in one
//! shader): compact append when `MULTI_DRAW_INDIRECT_COUNT` is available,
//! stable slots with `instance_count = 0` for culled draws otherwise.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

use crate::{views::ViewBuilder, PipelineShared, MAX_DRAWS_PER_VIEW, MAX_SHADOW_FACES};

/// Threads per workgroup along the draw axis; must match `@workgroup_size` in
/// cull.wgsl.
const WORKGROUP_SIZE: u32 = 64;

// ── Uniforms ──────────────────────────────────────────────────────────────────

/// Mirrors `Params` in cull.wgsl (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    draw_count: u32,
    view_count: u32,
    /// 1 = compact append mode (multi-draw-count), 0 = stable-slot mode.
    compact: u32,
    max_draws_per_view: u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct CullPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    /// (ptr, size) per scene buffer bound. GrowableBuffer reallocation can hand
    /// the new `wgpu::Buffer` the old address, but never the old address AND
    /// the old size, so the pair detects reallocation reliably.
    bind_group_key: Option<(usize, u64, usize, u64, usize, u64)>,
    params_buf: wgpu::Buffer,
    builder: ViewBuilder,
    shared: Arc<Mutex<PipelineShared>>,
    /// `movable_objects_generation` seen last prepare. Starts at `u64::MAX` so
    /// the first frame counts as "objects moved" and all faces begin dirty.
    prev_objects_gen: u64,
    warned_overflow: bool,
}

impl CullPass {
    pub fn new(device: &wgpu::Device, shared: Arc<Mutex<PipelineShared>>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PipelineCull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cull.wgsl").into()),
        });

        // Buffer-only layout; bindings mirror cull.wgsl group(0) exactly.
        let buffer_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PipelineCull BGL"),
            entries: &[
                buffer_entry(0, ro),                              // draw_calls
                buffer_entry(1, ro),                              // instances
                buffer_entry(2, ro),                              // views
                buffer_entry(3, rw),                              // out_draws
                buffer_entry(4, rw),                              // out_counts
                buffer_entry(5, wgpu::BufferBindingType::Uniform), // params
                buffer_entry(6, ro),                              // visibility
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineCull PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PipelineCull Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PipelineCull Params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            params_buf,
            builder: ViewBuilder::new(),
            shared,
            prev_objects_gen: u64::MAX,
            warned_overflow: false,
        }
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for CullPass {
    fn name(&self) -> &'static str {
        "PipelineCull"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let scene = ctx.scene;
        let camera = scene.camera.data();
        let lights = scene.lights.as_slice();
        let face_count = scene
            .shadow_matrices
            .len()
            .min(MAX_SHADOW_FACES as usize) as u32;
        // Same source SceneResources.draw_count uses (GpuScene::resources()),
        // so prepare- and execute-side counts agree within a frame.
        let draw_count = scene.draw_calls.len() as u32;

        // Generation-counter dirty tracking (DESIGN.md D4): any movable object
        // motion re-dirties every active face — the per-face culled single-pass
        // atlas makes that cheap, so no per-caster hashing is needed.
        let objects_moved = scene.movable_objects_generation != self.prev_objects_gen;
        self.prev_objects_gen = scene.movable_objects_generation;

        let view_count = self.builder.build(camera, lights, face_count);

        let shared = &mut *self.shared.lock().unwrap();
        for f in 0..face_count as usize {
            let active = self.builder.face_active[f];
            shared.face_active[f] = active;
            if active {
                // Accumulate — ShadowAtlasPass clears per-face once rendered.
                if self.builder.face_changed[f] || objects_moved || !shared.face_rendered_once[f]
                {
                    shared.face_dirty[f] = true;
                }
            } else {
                // Deactivated face: drop pending work and force a re-render on
                // re-activation (rendered_once gates the dirty fold above).
                shared.face_rendered_once[f] = false;
                shared.face_dirty[f] = false;
            }
        }
        shared.view_count = view_count;
        shared.face_count = face_count;
        shared.draw_count = draw_count;

        // Uploads (prepare() is the only place this pass touches the queue).
        ctx.write_buffer(
            &shared.views_buf,
            0,
            bytemuck::cast_slice(&self.builder.views()[..view_count as usize]),
        );
        if face_count > 0 {
            ctx.write_buffer(
                &shared.face_mats_buf,
                0,
                bytemuck::cast_slice(&self.builder.face_mats()[..face_count as usize]),
            );
        }
        let params = Params {
            draw_count,
            view_count,
            compact: shared.supports_multi_draw_count as u32,
            max_draws_per_view: MAX_DRAWS_PER_VIEW,
        };
        ctx.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        // Zero per-view counts so the shader's atomicAdd allocates dense slots
        // from 0 each frame (also keeps counts at 0 for views with no draws).
        ctx.write_buffer(
            &shared.view_counts,
            0,
            &vec![0u8; (view_count * 4) as usize],
        );

        if draw_count > MAX_DRAWS_PER_VIEW && !self.warned_overflow {
            log::warn!(
                "PipelineCull: scene has {draw_count} draw calls but per-view list \
                 capacity is {MAX_DRAWS_PER_VIEW}; draws beyond capacity are dropped. \
                 Raise helio_pipeline::MAX_DRAWS_PER_VIEW."
            );
            self.warned_overflow = true;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Counts captured at prepare time — uniform across passes this frame.
        let (draw_count, view_count) = {
            let shared = self.shared.lock().unwrap();
            (shared.draw_count, shared.view_count)
        };
        if draw_count == 0 || view_count == 0 {
            return Ok(());
        }

        // Lazy bind group, rebuilt on GrowableBuffer reallocation. Shared
        // pipeline buffers (views/culled_indirect/view_counts) are created once
        // and never reallocated, so only scene buffers key the rebuild.
        let key = (
            ctx.scene.draw_calls as *const _ as usize,
            ctx.scene.draw_calls.size(),
            ctx.scene.instances as *const _ as usize,
            ctx.scene.instances.size(),
            ctx.scene.visibility as *const _ as usize,
            ctx.scene.visibility.size(),
        );
        if self.bind_group_key != Some(key) {
            let shared = self.shared.lock().unwrap();
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PipelineCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.draw_calls.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: shared.views_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shared.culled_indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: shared.view_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: ctx.scene.visibility.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        // One dispatch covers every (draw, view) pair: x walks draws in
        // 64-thread groups, y is the view index (≤ MAX_VIEWS = 257).
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PipelineCull"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(draw_count.div_ceil(WORKGROUP_SIZE), view_count, 1);
        Ok(())
    }
}

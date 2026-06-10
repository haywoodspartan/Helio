//! GPU-driven render pipeline for Helio.
//!
//! An opt-in alternative to the default render graph, built around four passes:
//!
//! 1. [`CullPass`] — ONE compute dispatch frustum-culls every draw against every
//!    view (camera + all shadow faces), producing per-view indirect draw lists.
//! 2. [`ShadowAtlasPass`] — ONE render pass re-renders all dirty shadow faces
//!    into a single 2D tiled `Depth32Float` atlas via per-face viewports and
//!    per-face culled draws. Replaces the 12-render-pass / 2 GB dual-atlas
//!    design that made light movement cost ~60 ms.
//! 3. [`GeometryPass`] — G-buffer fill from the camera's culled draw list.
//!    Publishes the same `frame.gbuffer` contract as the classic GBufferPass so
//!    every tail pass (VG, TAA, billboards, water, perf overlay) keeps working.
//! 4. [`LightingPass`] — fullscreen deferred PBR sampling the tiled atlas.
//!    Publishes `frame.pre_aa` with the same semantics as DeferredLightPass.
//!
//! Cross-pass state lives in [`PipelineShared`] (`Arc<Mutex<…>>`, same pattern
//! as `PerfOverlayShared`). View/cascade matrices are computed on CPU in
//! [`views`] — an exact transcription of `shadow_matrices.wgsl` so shadow
//! placement is bit-compatible with the classic pipeline.

use std::sync::{Arc, Mutex};

pub mod atlas;
pub mod views;

mod cull;
mod geometry;
mod lighting;
mod shadow;

pub use cull::CullPass;
pub use geometry::GeometryPass;
pub use lighting::LightingPass;
pub use shadow::ShadowAtlasPass;

// ── Pipeline constants ─────────────────────────────────────────────────────────
// These are compile-time constants mirrored in the WGSL shaders; if you change
// one here, change the matching `const` in the shader (asserted by tests).

/// Shadow tile resolution (one cube face / cascade per tile).
pub const TILE_RES: u32 = 512;
/// Tiles per atlas row/column. 16×16 = 256 tiles.
pub const ATLAS_TILES_PER_ROW: u32 = 16;
/// Full atlas resolution: 8192² × Depth32Float = 256 MB (vs 2 GB classic).
pub const ATLAS_RES: u32 = TILE_RES * ATLAS_TILES_PER_ROW;
/// Maximum shadow faces (tiles). Matches the classic 42-caster × 6-face budget.
pub const MAX_SHADOW_FACES: u32 = ATLAS_TILES_PER_ROW * ATLAS_TILES_PER_ROW;
/// View 0 is the camera; views 1..=MAX_SHADOW_FACES are shadow faces.
pub const MAX_VIEWS: u32 = 1 + MAX_SHADOW_FACES;
/// Per-view indirect draw list capacity. Draws beyond this are dropped with a
/// log warning (cull.rs); raise if scenes exceed it.
pub const MAX_DRAWS_PER_VIEW: u32 = 4096;
/// Cube faces / matrix slots per shadow caster (parity with the classic pipeline).
pub const FACES_PER_CASTER: u32 = 6;

/// `GpuView.flags` bit: this view is a shadow face — cull only instances whose
/// `flags` bit 0 (`casts_shadow`) is set.
pub const VIEW_FLAG_SHADOW: u32 = 1 << 0;
/// `GpuView.flags` bit: view slot is inactive (spot faces 1-5, directional 4-5,
/// gaps). The cull shader emits zero draws for inactive views.
pub const VIEW_FLAG_INACTIVE: u32 = 1 << 1;

// ── GPU structs ────────────────────────────────────────────────────────────────

/// One culling/rendering view: the camera or a single shadow face.
///
/// WGSL mirror (cull.wgsl / shadow_atlas.wgsl):
/// ```wgsl
/// struct GpuView {
///     view_proj: mat4x4f,          // 64 B
///     planes:    array<vec4f, 6>,  // 96 B — normalized frustum planes, inward
///     flags:     u32,
///     _pad0: u32, _pad1: u32, _pad2: u32,
/// }                                 // 176 B total
/// ```
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuView {
    pub view_proj: [f32; 16],
    /// Normalized frustum planes (nx, ny, nz, d), pointing inward:
    /// `dot(n, p) + d >= -radius` ⇒ sphere not culled.
    pub planes: [[f32; 4]; 6],
    pub flags: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<GpuView>() == 176);

// ── Shared pipeline state ──────────────────────────────────────────────────────

/// GPU resources + per-frame CPU state shared by the four pipeline passes.
///
/// Created once by the graph builder and handed to each pass as
/// `Arc<Mutex<PipelineShared>>`. CullPass (first in the graph) refreshes the
/// per-frame fields in `prepare()`; later passes only read them.
pub struct PipelineShared {
    // ── GPU resources (created once, never reallocated) ───────────────────
    /// `array<GpuView, MAX_VIEWS>` storage buffer. Written by CullPass.prepare.
    pub views_buf: wgpu::Buffer,
    /// `array<mat4x4f, MAX_SHADOW_FACES>` face view-proj matrices for the
    /// lighting shader (shadow UV projection). Written by CullPass.prepare.
    pub face_mats_buf: wgpu::Buffer,
    /// Per-view culled `DrawIndexedIndirect` lists, written by the cull shader.
    /// Layout: `[view][slot]`, stride `MAX_DRAWS_PER_VIEW * 20` bytes per view.
    pub culled_indirect: wgpu::Buffer,
    /// Per-view emitted-draw counts (`array<atomic<u32>, MAX_VIEWS>`), used as
    /// the count buffer for `multi_draw_indexed_indirect_count`.
    pub view_counts: wgpu::Buffer,
    /// Single 2D tiled shadow atlas (Depth32Float, ATLAS_RES²).
    pub atlas_tex: wgpu::Texture,
    pub atlas_view: wgpu::TextureView,
    /// PCF comparison sampler (LessEqual), exposed to the lighting pass and
    /// published as `frame.shadow_sampler`.
    pub compare_sampler: wgpu::Sampler,

    // ── Device capabilities ────────────────────────────────────────────────
    /// MULTI_DRAW_INDIRECT_COUNT available → cull compacts draw lists and
    /// passes use the GPU count buffer. Otherwise the cull shader writes
    /// stable slots with `instance_count = 0` for culled draws and passes
    /// issue `draw_count` indirect draws.
    pub supports_multi_draw_count: bool,

    // ── Per-frame state (refreshed by CullPass::prepare) ───────────────────
    /// Total views uploaded this frame: 1 (camera) + active shadow face range.
    pub view_count: u32,
    /// Shadow face slots in use this frame (= scene shadow_matrices length,
    /// clamped to MAX_SHADOW_FACES).
    pub face_count: u32,
    /// Per-face: belongs to a real caster face this frame (vs gap/identity).
    pub face_active: [bool; MAX_SHADOW_FACES as usize],
    /// Per-face: must re-render this frame (matrix changed / never rendered /
    /// movable objects moved). Set by CullPass::prepare (accumulating), cleared
    /// per-face by ShadowAtlasPass once the face is actually re-rendered.
    pub face_dirty: [bool; MAX_SHADOW_FACES as usize],
    /// Per-face: rendered at least once with its current activation. Cleared
    /// when a face goes inactive so re-activation forces a re-render.
    pub face_rendered_once: [bool; MAX_SHADOW_FACES as usize],
    /// Draw count captured at prepare time (uniform across passes this frame).
    pub draw_count: u32,
    /// False until the atlas has been cleared once (first ShadowAtlasPass run).
    pub atlas_initialized: bool,
}

impl PipelineShared {
    pub fn new(device: &wgpu::Device) -> Arc<Mutex<Self>> {
        let views_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline/Views"),
            size: MAX_VIEWS as u64 * std::mem::size_of::<GpuView>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let face_mats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline/FaceMatrices"),
            size: MAX_SHADOW_FACES as u64 * 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let culled_indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline/CulledIndirect"),
            size: MAX_VIEWS as u64 * MAX_DRAWS_PER_VIEW as u64 * 20,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let view_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline/ViewCounts"),
            size: MAX_VIEWS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pipeline/ShadowAtlas"),
            size: wgpu::Extent3d {
                width: ATLAS_RES,
                height: ATLAS_RES,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Pipeline/ShadowCompare"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let supports_multi_draw_count = device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

        Arc::new(Mutex::new(Self {
            views_buf,
            face_mats_buf,
            culled_indirect,
            view_counts,
            atlas_tex,
            atlas_view,
            compare_sampler,
            supports_multi_draw_count,
            view_count: 1,
            face_count: 0,
            face_active: [false; MAX_SHADOW_FACES as usize],
            face_dirty: [false; MAX_SHADOW_FACES as usize],
            face_rendered_once: [false; MAX_SHADOW_FACES as usize],
            draw_count: 0,
            atlas_initialized: false,
        }))
    }

    /// Byte offset of `view`'s slice in `culled_indirect`.
    pub fn indirect_offset(view: u32) -> u64 {
        view as u64 * MAX_DRAWS_PER_VIEW as u64 * 20
    }

    /// Byte offset of `view`'s entry in `view_counts`.
    pub fn count_offset(view: u32) -> u64 {
        view as u64 * 4
    }
}

// Single-pass tiled shadow atlas — depth-only, GPU-driven.
//
// Two vertex entry points, no fragment stage (the rasteriser writes depth):
//
//   vs_main  — projects scene geometry with this face's pre-computed view-proj
//              matrix from the unified view array (view 0 = camera, view 1 + f
//              = shadow face f).  The face's VIEW index arrives via a dynamic-
//              offset uniform so every face shares one bind group.
//   vs_clear — fullscreen "giant triangle" at z = 1.0; with DepthCompare::Always
//              and the per-face scissor rect it clears exactly one stale tile,
//              letting the attachment use LoadOp::Load to preserve clean tiles.

// ── Types ─────────────────────────────────────────────────────────────────────

// Must match `helio_pipeline::GpuView` (176-byte stride).  Only view_proj is
// read here; the frustum planes are consumed by cull.wgsl.
struct GpuView {
    view_proj: mat4x4f,         // offset   0
    planes:    array<vec4f, 6>, // offset  64 (unused in this pass)
    flags:     u32,             // offset 160
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

// Per-instance world transform.  Must match GpuInstanceData in libhelio (144 bytes).
struct GpuInstance {
    transform:      mat4x4f, // model matrix, 64 bytes
    normal_mat_0:   vec4f,   // 16 bytes (unused in shadow pass)
    normal_mat_1:   vec4f,   // 16 bytes
    normal_mat_2:   vec4f,   // 16 bytes
    bounds:         vec4f,   // xyz = bounding sphere center, w = radius
    mesh_id:        u32,
    material_id:    u32,
    flags:          u32,
    lightmap_index: u32,
}

// VIEW index (1 + face) of the tile being rendered.  Addressed via dynamic
// uniform buffer offset — one 16-byte slot per face at 256-byte stride.
struct FaceIdx {
    value: u32,
    _p0:   u32,
    _p1:   u32,
    _p2:   u32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

// Unified view array written by CullPass (camera + all shadow faces).
@group(0) @binding(0) var<storage, read> views:     array<GpuView>;
// Per-instance world transforms for the entire scene.
@group(0) @binding(1) var<storage, read> instances: array<GpuInstance>;
// Current face's view index, selected per face via dynamic offset.
@group(0) @binding(2) var<uniform>       face_idx:  FaceIdx;

// ── Geometry: depth projection ────────────────────────────────────────────────

@vertex
fn vs_main(
    @location(0)             position: vec3<f32>,
    @builtin(instance_index) slot:     u32,
) -> @builtin(position) vec4<f32> {
    let world = instances[slot].transform * vec4<f32>(position, 1.0);
    return views[face_idx.value].view_proj * world;
}

// ── Tile clear: fullscreen triangle at the far plane ──────────────────────────

@vertex
fn vs_clear(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Three clip-space vertices whose convex hull covers all of NDC, z = 1.0.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

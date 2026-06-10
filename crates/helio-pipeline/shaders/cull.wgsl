// Unified multi-view culling — ONE dispatch for camera + every shadow face.
//
// Thread grid: x = draw template index, y = view index. Each thread sphere-tests
// one draw's bounding sphere against one view's 6 frustum planes and emits (or
// suppresses) one DrawIndexedIndirect command into that view's slice of
// `out_draws` (stride = params.max_draws_per_view commands per view).
//
// Two emission modes (selected by `params.compact`, set from the device's
// MULTI_DRAW_INDIRECT_COUNT support so both paths live in one shader):
//   compact == 1: visible draws are appended at `atomicAdd(&out_counts[view])`
//                 — consumers use multi_draw_indexed_indirect_count.
//   compact == 0: every draw writes its STABLE slot `[view][draw]`, culled
//                 draws with instance_count = 0 — consumers issue draw_count
//                 indirect draws (zero-instance commands cost nothing).
//
// Frustum planes are precomputed on CPU (views.rs::extract_frustum_planes),
// normalized and inward-facing: dot(n, p) + d < -radius ⇒ fully outside.
// Extension point: Hi-Z occlusion test would slot in after the sphere test.

// ── Constants (mirror src/lib.rs — asserted by tests) ──────────────────────────

/// GpuView.flags bit 0: shadow view — only shadow-casting instances draw.
const VIEW_FLAG_SHADOW: u32 = 1u;
/// GpuView.flags bit 1: inactive slot (spot faces 1-5, directional 4-5, gaps).
const VIEW_FLAG_INACTIVE: u32 = 2u;
/// GpuInstanceData.flags bit 0: casts_shadow.
const INSTANCE_FLAG_CASTS_SHADOW: u32 = 1u;

// ── Structs ────────────────────────────────────────────────────────────────────

/// Draw template — must match libhelio::GpuDrawCall (NOT indirect-args order:
/// instance_count is LAST here, second in DrawIndexedIndirect).
struct GpuDrawCall {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,
    instance_count: u32,
}

/// wgpu indirect command layout (libhelio::DrawIndexedIndirectArgs, 20 bytes).
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// Must match GpuInstanceData in libhelio/src/instance.rs (144 bytes).
/// Only `bounds` and `flags` are read here.
struct GpuInstance {
    transform:      mat4x4f,  // model matrix, 64 bytes
    normal_mat_0:   vec4f,    // 16 bytes
    normal_mat_1:   vec4f,    // 16 bytes
    normal_mat_2:   vec4f,    // 16 bytes
    bounds:         vec4f,    // xyz = world-space bounding sphere center, w = radius
    mesh_id:        u32,
    material_id:    u32,
    flags:          u32,
    lightmap_index: u32,
}

/// Must match helio_pipeline::GpuView (176 bytes). Planes are normalized,
/// inward-facing (nx, ny, nz, d).
struct GpuView {
    view_proj: mat4x4f,
    planes:    array<vec4f, 6>,
    flags:     u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

/// Mirrors `Params` in src/cull.rs (16 bytes).
struct Params {
    draw_count:         u32,
    view_count:         u32,
    compact:            u32,
    max_draws_per_view: u32,
}

// ── Bindings ───────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       draw_calls: array<GpuDrawCall>;
@group(0) @binding(1) var<storage, read>       instances:  array<GpuInstance>;
@group(0) @binding(2) var<storage, read>       views:      array<GpuView>;
@group(0) @binding(3) var<storage, read_write> out_draws:  array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write> out_counts: array<atomic<u32>>;
@group(0) @binding(5) var<uniform>             params:     Params;
/// Per-instance visibility (1 = visible, 0 = hidden group), indexed like `instances`.
@group(0) @binding(6) var<storage, read>       visibility: array<u32>;

// ── Main ───────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let draw = gid.x;
    let view = gid.y;
    if draw >= params.draw_count || view >= params.view_count {
        return;
    }

    let v = views[view];
    let tpl = draw_calls[draw];
    let inst = instances[tpl.first_instance];

    // Inactive view slots (spot faces 1-5, directional 4-5, gaps) emit nothing.
    var visible = (v.flags & VIEW_FLAG_INACTIVE) == 0u;

    // Hidden groups: editor-toggled visibility lives per-instance.
    if visible && visibility[tpl.first_instance] == 0u {
        visible = false;
    }

    // Shadow views only draw shadow-casting instances.
    if visible && (v.flags & VIEW_FLAG_SHADOW) != 0u
        && (inst.flags & INSTANCE_FLAG_CASTS_SHADOW) == 0u {
        visible = false;
    }

    // Sphere-vs-frustum on single-instance draws only. Merged instanced batches
    // share one template but span many instances, so a single bounding sphere
    // does not cover them — keep those conservatively (always drawn).
    if visible && tpl.instance_count <= 1u {
        let c = inst.bounds.xyz;
        let r = inst.bounds.w;
        var inside = true;
        for (var i = 0u; i < 6u; i++) {
            let p = v.planes[i];
            if dot(p.xyz, c) + p.w < -r {
                inside = false;
            }
        }
        if !inside {
            visible = false;
        }
    }

    // Note the field reorder: template is (count, first, offset, inst, n);
    // the indirect command wants (count, n, first, offset, inst).
    let cmd = DrawIndexedIndirect(
        tpl.index_count,
        select(0u, max(tpl.instance_count, 1u), visible),
        tpl.first_index,
        tpl.vertex_offset,
        tpl.first_instance,
    );

    if params.compact == 1u {
        // Append mode: counts were zeroed in prepare(), so atomicAdd hands out
        // dense slots. Slots past capacity are dropped (warned once on CPU).
        if visible {
            let slot = atomicAdd(&out_counts[view], 1u);
            if slot < params.max_draws_per_view {
                out_draws[view * params.max_draws_per_view + slot] = cmd;
            }
        }
    } else {
        // Stable-slot mode: culled draws still write (with instance_count = 0)
        // so every slot consumers replay holds a valid command for this frame.
        if draw < params.max_draws_per_view {
            out_draws[view * params.max_draws_per_view + draw] = cmd;
        }
    }
}

# helio-pipeline — GPU-driven render pipeline (ground-up rework)

Status: v1 design. Opt-in alternative to `build_default_graph`; the default
pipeline is untouched.

## Why this exists

Profiling the editor showed that moving one light cost ~60 ms on a 24 GB GPU in
a ~50k-triangle scene. The cost is structural, not geometric:

1. **12 render passes per light move** — the moved caster's 6 cube faces
   re-render in BOTH a static and a dynamic shadow atlas (6 × 2), each its own
   `begin_render_pass` with attachment transitions against giant resources.
2. **2 GB of shadow atlas** — two pre-allocated `Depth32Float` 1024² × 256-layer
   arrays (1 GB each). Per-subresource state tracking across 256 layers × 2
   atlases is heavy, and every pass touches one layer of a huge resource.
3. **No per-face culling** — every face draws the entire static or movable draw
   list, regardless of what its 90° frustum can see.
4. **CPU dirty-tracking spaghetti** — per-caster FNV hashes, camera-snap
   quantisation, static/movable generation counters, all to decide what the GPU
   should already know.

## Architecture

```
            ┌────────────────────────────────────────────────┐
 CPU        │ ViewSet: view 0 = camera, views 1..N = shadow  │
 (prepare)  │ faces of active casters. Frustum planes + VP    │
            │ matrices computed on CPU (N ≤ 49, trivial).     │
            └────────────────────────────────────────────────┘
                 │ one storage buffer upload (dirty-gated)
                 ▼
 GPU  [1] UnifiedCullPass (compute, ONE dispatch)
          threads = draw_count × view_count
          sphere-vs-frustum per (draw, view) →
          per-view culled DrawIndexedIndirect lists + count buffer
                 │
                 ▼
      [2] ShadowAtlasPass (ONE render pass, depth-only)
          Single 2D tiled atlas (Depth32Float, 8192² = 16×16 tiles of 512²,
          256 MB total — vs 2 GB before). All dirty faces render in ONE pass
          via set_viewport per face + per-face culled multi-draw.
          No static/dynamic split. Skipped entirely when no face is dirty.
                 │
                 ▼
      [3] GeometryPass (ONE render pass)
          Same G-buffer MRT layout + bindless material model as the existing
          GBufferPass (publishes frame.gbuffer so VG/overlay tails work),
          but consumes view-0 culled draws from [1].
                 │
                 ▼
      [4] LightingPass (fullscreen)
          Deferred PBR. Samples the tiled 2D atlas (face → tile UV transform).
          All-lights loop v1 (editor-scale); tiled light culling is a marked
          extension point. Writes pre_aa (Rgba16Float), publishes it for TAA.
                 │
                 ▼
      [tail] existing passes, unchanged: VirtualGeometry (optional), Billboard,
             WaterSim (optional), TAA, PerfOverlay, DebugDraw.
```

### Key decisions

**D1 — single 2D tiled shadow atlas, one render pass.**
A render pass cannot switch array layers mid-pass, so per-face array layers
force one pass per face. A 2D atlas with per-face *viewports* renders every
dirty face inside ONE render pass: one attachment transition total, one
resource for the tracker, ~8× less VRAM. Face *i* occupies tile
`(i % tiles_per_row, i / tiles_per_row)`; the lighting shader maps face UV →
atlas UV with a scale+offset. Tile resolution and atlas dimensions are
constants in `atlas.rs` with pure-math helpers (unit-tested).

**D2 — one culling dispatch for all views.**
The cull shader is view-count-agnostic: `(draw, view)` thread grid, sphere vs
6 precomputed planes, append to that view's list (or write
`instance_count = 0` at a stable slot when `MULTI_DRAW_INDIRECT_COUNT` is
unavailable — both modes in one shader, selected by a uniform). Per-face
culling falls out for free: shadow faces are just views 1..N.

**D3 — CPU computes view matrices, GPU consumes.**
≤ 49 mat4s per frame is nanoseconds on CPU and removes the ShadowMatrixPass +
ShadowDirtyPass compute prelude and their cross-pass buffer plumbing. The
cube-face orientation and CSM cascade math is ported from
`shadow_matrices.wgsl` verbatim and unit-tested against known geometry.

**D4 — dirty tracking by generation counters only.**
A view set re-uploads when `(camera_generation, movable_lights_generation,
movable_objects_generation, draw_count)` changes. Shadow faces re-render when
their light moved or objects moved (per-face culled, single pass, so even
"re-render all active faces" is cheap). No hashes, no quantisation, no
per-caster CPU state.

**D5 — keep the executor and the tail.**
`helio_v3::RenderGraph` (profiler, transients, external-device mode) is sound;
the problem was the passes, not the executor. Tail passes keep working because
GeometryPass publishes the same `frame.gbuffer` / `frame.pre_aa` contracts.

### v1 limitations (explicit)

- Frustum culling only (no Hi-Z occlusion in the cull dispatch yet — extension
  point in `cull.wgsl`).
- All-lights loop in lighting (no tile lists yet); fine below ~64 lights.
- Sky: if the scene has a sky, the existing Sky passes are kept in the graph
  ahead of lighting (lighting clears to ambient where no sky output exists).
- Shadow caster budget: `MAX_ATLAS_FACES` (default 64 tiles → 10 casters × 6
  faces + 4 cascades) instead of 42 casters; configurable.

## Files

- `src/lib.rs` — pub exports, `PipelineConfig`, shared constants.
- `src/atlas.rs` — tile math (face → viewport rect, face → UV scale/offset).
- `src/views.rs` — ViewSet builder: camera + cube faces + spot + CSM cascades,
  frustum plane extraction. Pure math, unit-tested.
- `src/cull.rs` + `shaders/cull.wgsl` — UnifiedCullPass.
- `src/shadow.rs` + `shaders/shadow_atlas.wgsl` — ShadowAtlasPass.
- `src/geometry.rs` + `shaders/geometry.wgsl` — GeometryPass (G-buffer).
- `src/lighting.rs` + `shaders/lighting.wgsl` — LightingPass.
- `tests/` — atlas math, frustum extraction, WGSL↔Rust layout assertions,
  naga parse validation of all four shaders.
- Integration: `build_gpu_driven_graph()` in `helio/src/renderer/graph.rs`,
  selected via `RendererConfig` (default remains the classic graph).

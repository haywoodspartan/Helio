//! Rust ↔ WGSL layout and constant-sync assertions.
//!
//! The pipeline constants in `helio_pipeline` (lib.rs) are mirrored as plain
//! `const`s inside the WGSL shaders; nothing enforces that at compile time,
//! so these tests fail loudly if either side drifts.

use helio_pipeline::{
    GpuView, ATLAS_RES, ATLAS_TILES_PER_ROW, MAX_SHADOW_FACES, MAX_VIEWS, TILE_RES,
};

#[test]
fn gpu_view_matches_wgsl_layout() {
    // mat4x4f (64) + array<vec4f, 6> (96) + flags + 3×pad (16) = 176 bytes.
    assert_eq!(std::mem::size_of::<GpuView>(), 176);
}

#[test]
fn pipeline_constants_are_consistent() {
    assert_eq!(ATLAS_RES, TILE_RES * ATLAS_TILES_PER_ROW);
    assert_eq!(MAX_VIEWS, 1 + MAX_SHADOW_FACES);
    assert_eq!(MAX_SHADOW_FACES, ATLAS_TILES_PER_ROW * ATLAS_TILES_PER_ROW);
}

#[test]
fn lighting_wgsl_atlas_constants_in_sync() {
    // lighting.wgsl maps face UV → atlas UV with its own copies of the tile
    // constants; a silent mismatch would shift every shadow lookup.
    let src = include_str!("../shaders/lighting.wgsl");
    assert!(
        src.contains("TILE_RES: f32 = 512"),
        "lighting.wgsl TILE_RES out of sync with helio_pipeline::TILE_RES ({TILE_RES})"
    );
    assert!(
        src.contains("ATLAS_TILES_PER_ROW: f32 = 16"),
        "lighting.wgsl ATLAS_TILES_PER_ROW out of sync with helio_pipeline::ATLAS_TILES_PER_ROW ({ATLAS_TILES_PER_ROW})"
    );
}

#[test]
fn cull_wgsl_declares_view_flags() {
    // The cull shader must honor both view flags (shadow-only filtering and
    // inactive-slot skipping) declared in lib.rs.
    let src = include_str!("../shaders/cull.wgsl");
    assert!(
        src.contains("VIEW_FLAG_SHADOW"),
        "cull.wgsl missing VIEW_FLAG_SHADOW"
    );
    assert!(
        src.contains("VIEW_FLAG_INACTIVE"),
        "cull.wgsl missing VIEW_FLAG_INACTIVE"
    );
}

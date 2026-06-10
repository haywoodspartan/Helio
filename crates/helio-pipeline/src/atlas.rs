//! Shadow atlas tile math.
//!
//! The atlas is a single 2D `Depth32Float` texture of `ATLAS_RES²`, divided into
//! `ATLAS_TILES_PER_ROW × ATLAS_TILES_PER_ROW` tiles of `TILE_RES²`. Face `f`
//! occupies tile `(f % ATLAS_TILES_PER_ROW, f / ATLAS_TILES_PER_ROW)`.
//!
//! The same mapping is implemented in `shaders/lighting.wgsl` (`face_uv_to_atlas`);
//! tests assert both stay in sync via the constants.

use crate::{ATLAS_TILES_PER_ROW, MAX_SHADOW_FACES, TILE_RES};
#[cfg(test)]
use crate::ATLAS_RES;

/// Pixel-space viewport `(x, y, w, h)` of a face's tile, for
/// `RenderPass::set_viewport` / `set_scissor_rect`.
pub fn face_viewport(face: u32) -> (u32, u32, u32, u32) {
    debug_assert!(face < MAX_SHADOW_FACES);
    let tx = face % ATLAS_TILES_PER_ROW;
    let ty = face / ATLAS_TILES_PER_ROW;
    (tx * TILE_RES, ty * TILE_RES, TILE_RES, TILE_RES)
}

/// UV transform mapping in-face UV `[0,1]²` to atlas UV:
/// `atlas_uv = face_uv * [s.x, s.y] + [o.x, o.y]`, returned as `[sx, sy, ox, oy]`.
pub fn face_uv_transform(face: u32) -> [f32; 4] {
    debug_assert!(face < MAX_SHADOW_FACES);
    let tx = (face % ATLAS_TILES_PER_ROW) as f32;
    let ty = (face / ATLAS_TILES_PER_ROW) as f32;
    let s = 1.0 / ATLAS_TILES_PER_ROW as f32;
    [s, s, tx * s, ty * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_dimensions_consistent() {
        assert_eq!(ATLAS_RES, TILE_RES * ATLAS_TILES_PER_ROW);
        assert_eq!(MAX_SHADOW_FACES, ATLAS_TILES_PER_ROW * ATLAS_TILES_PER_ROW);
    }

    #[test]
    fn face_zero_is_top_left() {
        assert_eq!(face_viewport(0), (0, 0, TILE_RES, TILE_RES));
        assert_eq!(face_uv_transform(0), [
            1.0 / ATLAS_TILES_PER_ROW as f32,
            1.0 / ATLAS_TILES_PER_ROW as f32,
            0.0,
            0.0
        ]);
    }

    #[test]
    fn last_face_is_bottom_right() {
        let last = MAX_SHADOW_FACES - 1;
        let (x, y, w, h) = face_viewport(last);
        assert_eq!((x + w, y + h), (ATLAS_RES, ATLAS_RES));
    }

    #[test]
    fn viewports_never_overlap_or_escape() {
        let mut seen = std::collections::HashSet::new();
        for f in 0..MAX_SHADOW_FACES {
            let (x, y, w, h) = face_viewport(f);
            assert!(x + w <= ATLAS_RES && y + h <= ATLAS_RES, "face {f} escapes");
            assert!(seen.insert((x, y)), "face {f} overlaps another tile");
        }
    }

    #[test]
    fn uv_transform_maps_into_tile() {
        for f in [0u32, 1, 15, 16, 17, 255] {
            let [sx, sy, ox, oy] = face_uv_transform(f);
            let (px, py, _, _) = face_viewport(f);
            // face uv (0,0) → tile origin in normalized coords
            assert!((ox - px as f32 / ATLAS_RES as f32).abs() < 1e-6);
            assert!((oy - py as f32 / ATLAS_RES as f32).abs() < 1e-6);
            // face uv (1,1) stays inside the atlas
            assert!(sx + ox <= 1.0 + 1e-6 && sy + oy <= 1.0 + 1e-6);
        }
    }
}

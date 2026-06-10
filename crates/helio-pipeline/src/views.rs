//! CPU view-set construction: camera + shadow-face matrices + frustum planes.
//!
//! The shadow matrix math is an **exact transcription** of
//! `helio-pass-shadow-matrix/shaders/shadow_matrices.wgsl` — including its
//! nonstandard orthographic convention and the `ATLAS_TEXELS = 2048` snap
//! constant — so shadow placement is bit-compatible with the classic pipeline.
//! Do NOT replace the `*_wgsl` helpers with `glam`'s built-in projections; the
//! conventions differ and shadows would silently shift.
//!
//! ≤ 257 matrices per frame on CPU is nanoseconds; computing them here removes
//! the ShadowMatrixPass + ShadowDirtyPass GPU preludes and their cross-pass
//! buffer plumbing entirely.

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use libhelio::{GpuCameraUniforms, GpuLight};

use crate::{
    GpuView, FACES_PER_CASTER, MAX_SHADOW_FACES, VIEW_FLAG_INACTIVE, VIEW_FLAG_SHADOW,
};

// ── Constants (parity with shadow_matrices.wgsl:13-16) ─────────────────────────

/// CSM split distances; must match `libhelio::CSM_SPLITS` and the WGSL const.
const CSM_SPLITS: [f32; 4] = libhelio::CSM_SPLITS;
/// Directional light pull-back distance along the light direction.
const SCENE_DEPTH: f32 = 4000.0;
/// Texel-snap granularity for cascades. NOTE: the classic shader snaps against
/// 2048 regardless of the actual atlas face resolution — preserved for parity.
const ATLAS_TEXELS: f32 = 2048.0;

const LIGHT_TYPE_DIRECTIONAL: u32 = 0;
const LIGHT_TYPE_POINT: u32 = 1;
const LIGHT_TYPE_SPOT: u32 = 2;

// ── WGSL matrix helper transcriptions (shadow_matrices.wgsl:76-111) ────────────
// Column-major construction, identical formulas. mat4x4f(c0,c1,c2,c3) ⇒
// Mat4::from_cols.

/// `mat4_perspective_rh` — RH, depth [0,1].
pub fn perspective_rh_wgsl(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy * 0.5).tan();
    let nf = 1.0 / (near - far);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far * nf, -1.0),
        Vec4::new(0.0, 0.0, near * far * nf, 0.0),
    )
}

/// `mat4_orthographic_rh` — transcribed verbatim (nonstandard z mapping kept).
pub fn orthographic_rh_wgsl(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let rml = 1.0 / (right - left);
    let tmb = 1.0 / (top - bottom);
    let fmn = 1.0 / (far - near);
    Mat4::from_cols(
        Vec4::new(2.0 * rml, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 * tmb, 0.0, 0.0),
        Vec4::new(0.0, 0.0, fmn, 0.0),
        Vec4::new(-(right + left) * rml, -(top + bottom) * tmb, -near * fmn, 1.0),
    )
}

/// `mat4_look_at_rh`.
pub fn look_at_rh_wgsl(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

// ── Per-light matrix computation (shadow_matrices.wgsl:115-244) ────────────────

/// Six cube-face view-proj matrices for a point light (wgsl:115-134).
pub fn point_light_face_matrices(position: Vec3, range: f32) -> [Mat4; 6] {
    let far_plane = range.max(0.1) * 2.5;
    let proj = perspective_rh_wgsl(std::f32::consts::FRAC_PI_2, 1.0, 0.05, far_plane);
    let dirs_ups: [(Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::NEG_Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y, Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z, Vec3::NEG_Y),
        (Vec3::NEG_Z, Vec3::NEG_Y),
    ];
    dirs_ups.map(|(dir, up)| proj * look_at_rh_wgsl(position, position + dir, up))
}

/// Spot light view-proj matrix (wgsl:138-151).
pub fn spot_light_matrix(position: Vec3, direction: Vec3, range: f32, cos_outer: f32) -> Mat4 {
    let dir = direction.normalize();
    let outer_angle = cos_outer.clamp(-1.0, 1.0).acos();
    let fov = (outer_angle * 2.0).clamp(
        std::f32::consts::PI * 0.25,
        std::f32::consts::PI - 0.01,
    );
    // WGSL select(false_val, true_val, cond):
    // up = Y when |dot(dir, Y)| < 0.99, else Z.
    let up = if dir.dot(Vec3::Y).abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let view = look_at_rh_wgsl(position, position + dir, up);
    let proj = perspective_rh_wgsl(fov, 1.0, 0.05, range.max(0.1));
    proj * view
}

/// Four CSM cascade matrices for a directional light (wgsl:155-233).
pub fn directional_cascade_matrices(camera: &GpuCameraUniforms, direction: Vec3) -> [Mat4; 4] {
    let dir = direction.normalize();
    // up = Z when |dot(dir, Y)| > 0.99, else Y (wgsl:158).
    let up = if dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };

    let inv_view_proj = Mat4::from_cols_array(&camera.inv_view_proj);
    let cam_pos = Vec3::new(
        camera.position_near[0],
        camera.position_near[1],
        camera.position_near[2],
    );

    // Unproject the 8 NDC frustum corners (z=0 near, z=1 far).
    let ndc: [Vec4; 8] = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new(1.0, -1.0, 0.0, 1.0),
        Vec4::new(-1.0, 1.0, 0.0, 1.0),
        Vec4::new(1.0, 1.0, 0.0, 1.0),
        Vec4::new(-1.0, -1.0, 1.0, 1.0),
        Vec4::new(1.0, -1.0, 1.0, 1.0),
        Vec4::new(-1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
    ];
    let mut world = [Vec3::ZERO; 8];
    for i in 0..8 {
        let v = inv_view_proj * ndc[i];
        world[i] = v.xyz() / v.w;
    }

    let mut near_dist = 0.0;
    let mut far_dist = 0.0;
    for i in 0..4 {
        near_dist += (world[i] - cam_pos).length();
        far_dist += (world[i + 4] - cam_pos).length();
    }
    near_dist /= 4.0;
    far_dist /= 4.0;
    let depth = (far_dist - near_dist).max(1.0);

    let prev_d = [0.0, CSM_SPLITS[0], CSM_SPLITS[1], CSM_SPLITS[2]];

    std::array::from_fn(|cascade_idx| {
        let t0 = ((prev_d[cascade_idx] - near_dist) / depth).clamp(0.0, 1.0);
        let t1 = ((CSM_SPLITS[cascade_idx] - near_dist) / depth).clamp(0.0, 1.0);

        // 8 world-space corners of this frustum slice.
        let mut cc = [Vec3::ZERO; 8];
        for j in 0..4 {
            cc[j * 2] = world[j].lerp(world[j + 4], t0);
            cc[j * 2 + 1] = world[j].lerp(world[j + 4], t1);
        }

        // Sphere fit.
        let mut centroid = Vec3::ZERO;
        for c in &cc {
            centroid += *c;
        }
        centroid /= 8.0;
        let mut radius: f32 = 0.0;
        for c in &cc {
            radius = radius.max((*c - centroid).length());
        }

        // Texel snap (against ATLAS_TEXELS=2048, parity with the WGSL).
        let texel_size = (2.0 * radius) / ATLAS_TEXELS;
        let radius_snap = (radius / texel_size).ceil() * texel_size;

        let light_view_raw = look_at_rh_wgsl(centroid - dir * SCENE_DEPTH, centroid, up);
        let centroid_ls_v4 = light_view_raw * centroid.extend(1.0);
        let centroid_ls = centroid_ls_v4.xyz() / centroid_ls_v4.w;
        let snap = texel_size;
        let snapped_x = (centroid_ls.x / snap).round() * snap;
        let snapped_y = (centroid_ls.y / snap).round() * snap;

        // WGSL m[col][row]: right_ws = rows 0 of cols 0..2 = view-space X axis.
        let cols = light_view_raw.to_cols_array_2d();
        let right_ws = Vec3::new(cols[0][0], cols[1][0], cols[2][0]).normalize();
        let up_ws = Vec3::new(cols[0][1], cols[1][1], cols[2][1]).normalize();
        let snap_offset =
            right_ws * (snapped_x - centroid_ls.x) + up_ws * (snapped_y - centroid_ls.y);
        let stable_centroid = centroid + snap_offset;

        let light_view = look_at_rh_wgsl(stable_centroid - dir * SCENE_DEPTH, stable_centroid, up);
        let proj = orthographic_rh_wgsl(
            -radius_snap,
            radius_snap,
            -radius_snap,
            radius_snap,
            0.1,
            SCENE_DEPTH * 2.0,
        );
        proj * light_view
    })
}

// ── Frustum plane extraction (Gribb-Hartmann, normalized) ──────────────────────

/// Extract 6 normalized inward-facing frustum planes from a view-proj matrix
/// (depth [0,1] convention; same row combinations as shadow_dirty.wgsl, but
/// normalized so sphere distance tests are metrically correct).
pub fn extract_frustum_planes(vp: &Mat4) -> [[f32; 4]; 6] {
    let m = vp.to_cols_array_2d(); // m[col][row]
    let row = |r: usize| Vec4::new(m[0][r], m[1][r], m[2][r], m[3][r]);
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let raw = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near (z >= 0)
        r3 - r2, // far
    ];
    raw.map(|p| {
        let n = Vec3::new(p.x, p.y, p.z);
        let len = n.length().max(1e-12);
        [p.x / len, p.y / len, p.z / len, p.w / len]
    })
}

// ── View set builder ───────────────────────────────────────────────────────────

/// Builds the per-frame view array (camera + shadow faces) and reports which
/// face matrices changed since the previous build.
///
/// Render-state (has this face ever been rendered, is it pending re-render) is
/// owned by `PipelineShared` — CullPass folds `face_changed` + objects-moved +
/// never-rendered into `shared.face_dirty`, and ShadowAtlasPass clears it.
pub struct ViewBuilder {
    views: Vec<GpuView>,
    face_mats: Vec<[f32; 16]>,
    /// Per-face: slot belongs to an active caster face this frame.
    pub face_active: [bool; MAX_SHADOW_FACES as usize],
    /// Per-face: view-proj matrix differs from the previous `build()`.
    pub face_changed: [bool; MAX_SHADOW_FACES as usize],
    prev_face_mats: Vec<[f32; 16]>,
}

impl Default for ViewBuilder {
    fn default() -> Self {
        Self::new()
    }
}

const IDENTITY: [f32; 16] = {
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
};

impl ViewBuilder {
    pub fn new() -> Self {
        let n = MAX_SHADOW_FACES as usize;
        Self {
            views: vec![GpuView::zeroed_inactive(); n + 1],
            face_mats: vec![IDENTITY; n],
            face_active: [false; MAX_SHADOW_FACES as usize],
            face_changed: [false; MAX_SHADOW_FACES as usize],
            prev_face_mats: vec![IDENTITY; n],
        }
    }

    /// Rebuild the view set for this frame.
    ///
    /// * `camera` — current (jittered) camera uniforms; sub-pixel jitter is
    ///   irrelevant for culling.
    /// * `lights` — the movable lights buffer (CPU mirror). `shadow_index`
    ///   carries the face base slot assigned by `Scene::flush()`.
    /// * `shadow_face_count` — active face range (scene shadow-matrix count).
    ///
    /// Returns `view_count` = 1 + active face range.
    pub fn build(
        &mut self,
        camera: &GpuCameraUniforms,
        lights: &[GpuLight],
        shadow_face_count: u32,
    ) -> u32 {
        let face_count = shadow_face_count.min(MAX_SHADOW_FACES) as usize;

        // View 0: camera.
        let cam_vp = Mat4::from_cols_array(&camera.view_proj);
        self.views[0] = GpuView {
            view_proj: camera.view_proj,
            planes: extract_frustum_planes(&cam_vp),
            flags: 0,
            _pad: [0; 3],
        };

        // Reset all face slots to inactive identity; lights re-activate theirs.
        for f in 0..face_count {
            self.face_active[f] = false;
            self.face_changed[f] = false;
            self.face_mats[f] = IDENTITY;
        }

        for light in lights {
            if light.shadow_index == u32::MAX {
                continue;
            }
            let base = light.shadow_index as usize;
            if base + FACES_PER_CASTER as usize > face_count {
                continue;
            }
            let pos = Vec3::new(
                light.position_range[0],
                light.position_range[1],
                light.position_range[2],
            );
            let dir = Vec3::new(
                light.direction_outer[0],
                light.direction_outer[1],
                light.direction_outer[2],
            );
            let range = light.position_range[3];

            match light.light_type {
                LIGHT_TYPE_POINT => {
                    let mats = point_light_face_matrices(pos, range);
                    for (i, m) in mats.iter().enumerate() {
                        self.set_face(base + i, m);
                    }
                }
                LIGHT_TYPE_SPOT => {
                    let m = spot_light_matrix(pos, dir, range, light.direction_outer[3]);
                    self.set_face(base, &m);
                }
                LIGHT_TYPE_DIRECTIONAL => {
                    let mats = directional_cascade_matrices(camera, dir);
                    for (i, m) in mats.iter().enumerate() {
                        self.set_face(base + i, m);
                    }
                }
                _ => {} // Area / unknown: no shadow faces (parity with classic shader)
            }
        }

        // Change detection + view emission for every face slot.
        for f in 0..face_count {
            if self.face_active[f] {
                self.face_changed[f] = self.face_mats[f] != self.prev_face_mats[f];
                self.prev_face_mats[f] = self.face_mats[f];
                let vp = Mat4::from_cols_array(&self.face_mats[f]);
                self.views[1 + f] = GpuView {
                    view_proj: self.face_mats[f],
                    planes: extract_frustum_planes(&vp),
                    flags: VIEW_FLAG_SHADOW,
                    _pad: [0; 3],
                };
            } else {
                self.views[1 + f] = GpuView::zeroed_inactive();
            }
        }

        1 + face_count as u32
    }

    fn set_face(&mut self, face: usize, mat: &Mat4) {
        if face < self.face_mats.len() {
            self.face_mats[face] = mat.to_cols_array();
            self.face_active[face] = true;
        }
    }

    /// The view array for upload — `views()[0..view_count]`.
    pub fn views(&self) -> &[GpuView] {
        &self.views
    }

    /// Face view-proj matrices for the lighting shader's shadow projection —
    /// upload `face_mats()[0..face_count]` to `PipelineShared::face_mats_buf`.
    pub fn face_mats(&self) -> &[[f32; 16]] {
        &self.face_mats
    }
}

impl GpuView {
    fn zeroed_inactive() -> Self {
        Self {
            view_proj: IDENTITY,
            planes: [[0.0; 4]; 6],
            flags: VIEW_FLAG_SHADOW | VIEW_FLAG_INACTIVE,
            _pad: [0; 3],
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn camera_at_origin() -> GpuCameraUniforms {
        // Simple camera at origin looking down -Z with a 90° perspective.
        let view = look_at_rh_wgsl(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = perspective_rh_wgsl(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0);
        let vp = proj * view;
        GpuCameraUniforms {
            view: view.to_cols_array(),
            proj: proj.to_cols_array(),
            view_proj: vp.to_cols_array(),
            inv_view_proj: vp.inverse().to_cols_array(),
            position_near: [0.0, 0.0, 0.0, 0.1],
            forward_far: [0.0, 0.0, -1.0, 1000.0],
            jitter_frame: [0.0; 4],
            prev_view_proj: vp.to_cols_array(),
        }
    }

    #[test]
    fn perspective_matches_wgsl_formula() {
        let m = perspective_rh_wgsl(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 100.0);
        let c = m.to_cols_array_2d();
        let f = 1.0 / (std::f32::consts::FRAC_PI_2 * 0.5).tan();
        let nf = 1.0 / (0.05 - 100.0);
        assert!((c[0][0] - f).abs() < 1e-6);
        assert!((c[1][1] - f).abs() < 1e-6);
        assert!((c[2][2] - 100.0 * nf).abs() < 1e-6);
        assert!((c[2][3] - -1.0).abs() < 1e-6);
        assert!((c[3][2] - 0.05 * 100.0 * nf).abs() < 1e-6);
    }

    #[test]
    fn look_at_matches_wgsl_formula() {
        let eye = Vec3::new(1.0, 2.0, 3.0);
        let m = look_at_rh_wgsl(eye, Vec3::new(4.0, 2.0, 3.0), Vec3::Y);
        // Looking down +X: s = cross(f=X, up=Y) = -Z … verify orthonormality +
        // the translation row matches the WGSL last column.
        let c = m.to_cols_array_2d();
        let s = Vec3::new(c[0][0], c[1][0], c[2][0]);
        let u = Vec3::new(c[0][1], c[1][1], c[2][1]);
        let f = -Vec3::new(c[0][2], c[1][2], c[2][2]);
        assert!((s.length() - 1.0).abs() < 1e-5);
        assert!((u.length() - 1.0).abs() < 1e-5);
        assert!(s.dot(u).abs() < 1e-5);
        assert!((c[3][0] - -s.dot(eye)).abs() < 1e-5);
        assert!((c[3][1] - -u.dot(eye)).abs() < 1e-5);
        assert!((c[3][2] - f.dot(eye)).abs() < 1e-5);
    }

    #[test]
    fn point_in_front_is_inside_point_face_frustum() {
        let mats = point_light_face_matrices(Vec3::new(5.0, 1.0, 5.0), 10.0);
        // A point 3 m along +X from the light must be inside face 0 (+X).
        let planes = extract_frustum_planes(&mats[0]);
        let p = Vec3::new(8.0, 1.0, 5.0);
        for pl in &planes {
            let d = pl[0] * p.x + pl[1] * p.y + pl[2] * p.z + pl[3];
            assert!(d > 0.0, "point should be inside all planes, got {d}");
        }
        // …and outside face 1 (-X).
        let planes_neg = extract_frustum_planes(&mats[1]);
        let mut outside_any = false;
        for pl in &planes_neg {
            if pl[0] * p.x + pl[1] * p.y + pl[2] * p.z + pl[3] < 0.0 {
                outside_any = true;
            }
        }
        assert!(outside_any, "+X point must be outside the -X face frustum");
    }

    #[test]
    fn sphere_culling_is_conservative() {
        let mats = point_light_face_matrices(Vec3::ZERO, 10.0);
        let planes = extract_frustum_planes(&mats[0]); // +X face
        // Sphere centered slightly outside the frustum but with radius reaching in.
        let center = Vec3::new(3.0, 3.5, 0.0); // above the 90° cone at x=3
        let radius = 2.0;
        let mut culled = false;
        for pl in &planes {
            let d = pl[0] * center.x + pl[1] * center.y + pl[2] * center.z + pl[3];
            if d < -radius {
                culled = true;
            }
        }
        assert!(!culled, "sphere overlapping the frustum must not be culled");
    }

    #[test]
    fn view_builder_reports_moved_light_faces_changed() {
        let cam = camera_at_origin();
        let mut light = GpuLight {
            position_range: [0.0, 5.0, 0.0, 20.0],
            direction_outer: [0.0, -1.0, 0.0, 0.5],
            color_intensity: [1.0, 1.0, 1.0, 100.0],
            shadow_index: 0,
            light_type: LIGHT_TYPE_POINT,
            inner_angle: 0.6,
            _pad: 0,
        };
        let mut b = ViewBuilder::new();
        let count = b.build(&cam, std::slice::from_ref(&light), 6);
        assert_eq!(count, 7);
        assert!(b.face_active[..6].iter().all(|&a| a));
        assert!(
            b.face_changed[..6].iter().all(|&d| d),
            "first build: all faces differ from identity"
        );

        // Same inputs → no change.
        let _ = b.build(&cam, std::slice::from_ref(&light), 6);
        assert!(b.face_changed[..6].iter().all(|&d| !d), "unchanged: clean");

        // Move the light → all 6 faces changed again.
        light.position_range[0] = 3.0;
        let _ = b.build(&cam, std::slice::from_ref(&light), 6);
        assert!(b.face_changed[..6].iter().all(|&d| d), "moved: changed");
    }

    #[test]
    fn view_builder_spot_activates_only_face_zero() {
        let cam = camera_at_origin();
        let light = GpuLight {
            position_range: [0.0, 5.0, 0.0, 20.0],
            direction_outer: [0.0, -1.0, 0.0, 0.5],
            color_intensity: [1.0, 1.0, 1.0, 100.0],
            shadow_index: 0,
            light_type: LIGHT_TYPE_SPOT,
            inner_angle: 0.6,
            _pad: 0,
        };
        let mut b = ViewBuilder::new();
        let _ = b.build(&cam, std::slice::from_ref(&light), 6);
        assert!(b.face_active[0]);
        assert!(b.face_active[1..6].iter().all(|&a| !a));
        // Inactive views must carry the INACTIVE flag for the cull shader.
        assert_ne!(b.views()[2].flags & crate::VIEW_FLAG_INACTIVE, 0);
        assert_eq!(b.views()[1].flags, crate::VIEW_FLAG_SHADOW);
    }

    #[test]
    fn directional_cascades_shrink_with_cascade_index() {
        let cam = camera_at_origin();
        let mats = directional_cascade_matrices(&cam, Vec3::new(-0.3, -1.0, -0.2));
        // Earlier cascades cover smaller world areas → larger ortho scale.
        let scale = |m: &Mat4| m.to_cols_array_2d()[0][0].abs();
        assert!(
            scale(&mats[0]) >= scale(&mats[3]),
            "cascade 0 must be tighter (larger scale) than cascade 3"
        );
    }
}

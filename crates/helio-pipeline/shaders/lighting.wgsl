//! Fullscreen deferred PBR lighting for the GPU-driven pipeline.
//!
//! Runs over the G-buffer written by geometry.wgsl. The BRDF (Cook-Torrance /
//! GGX), light attenuation, spot cone falloff, normal-offset shadow bias, CSM
//! cascade blend zones and ACES tonemap are copied from the classic
//! deferred_lighting.wgsl so switching pipelines causes no visual shift.
//!
//! The structural difference: shadows live in ONE tiled 2D depth atlas instead
//! of two texture_depth_2d_array atlases. Face `layer` occupies tile
//! `(layer % 16, layer / 16)`; in-face UV is remapped to the tile before the
//! comparison sample (same mapping as src/atlas.rs::face_uv_transform).

// ── Constants (mirrored in src/lib.rs; keep in sync) ──────────────────────────

const TILE_RES: f32 = 512.0;
const ATLAS_TILES_PER_ROW: f32 = 16.0;
// Normal-offset bias constant (world-space units). Shifts the shadow query
// point along the surface normal before projecting into light space — same
// technique as UE4 "Normal Shadow Bias" / Unity HDRP, matching the classic pass.
const NORMAL_OFFSET_SCALE: f32 = 0.05;
const PCF_TAPS: u32 = 8u;
// 10% smoothstep blend zone around each CSM split boundary.
const BLEND_ZONE: f32 = 0.1;
const PI: f32 = 3.14159265359;

// ── Uniforms / storage ─────────────────────────────────────────────────────────

// Mirrors libhelio::GpuCameraUniforms (same layout as taa.wgsl).
struct CameraUniforms {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

// Mirrors lighting.rs::Globals (64 bytes).
struct Globals {
    ambient_color: vec4<f32>,  // rgb = ambient colour, w = ambient intensity
    csm_splits:    vec4<f32>,
    camera_unused: vec4<f32>,  // reserved, zero
    light_count:   u32,
    frame:         u32,
    has_sky:       u32,
    _pad:          u32,
}

// GpuLight (64 bytes, matches libhelio::GpuLight)
struct GpuLight {
    position_range:  vec4<f32>,  // xyz = position, w = range
    direction_outer: vec4<f32>,  // xyz = direction, w = spot outer cos angle
    color_intensity: vec4<f32>,  // xyz = color, w = intensity
    shadow_index:    u32,        // -1u32 if no shadow
    light_type:      u32,        // LightType enum (0=directional, 1=point, 2=spot)
    inner_angle:     f32,        // spot inner cos angle
    _pad:            u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var<storage, read> lights: array<GpuLight>;
// Per-face view-proj matrices, slot-aligned with atlas tiles (CullPass uploads).
@group(0) @binding(3) var<storage, read> face_mats: array<mat4x4<f32>>;
@group(0) @binding(4) var gbuf_albedo:   texture_2d<f32>;   // Rgba8Unorm  albedo.rgb + alpha
@group(0) @binding(5) var gbuf_normal:   texture_2d<f32>;   // Rgba16Float world normal + F0.r
@group(0) @binding(6) var gbuf_orm:      texture_2d<f32>;   // Rgba8Unorm  AO/rough/metal + F0.g
@group(0) @binding(7) var gbuf_emissive: texture_2d<f32>;   // Rgba16Float emissive + F0.b
@group(0) @binding(8) var depth_tex:     texture_depth_2d;  // scene depth
@group(0) @binding(9) var shadow_atlas:  texture_depth_2d;  // tiled 2D shadow atlas
@group(0) @binding(10) var shadow_sampler: sampler_comparison;

// ── Fullscreen-triangle vertex shader ──────────────────────────────────────────

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var out: VSOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// ── Shadow sampling (tiled atlas) ──────────────────────────────────────────────

// Vogel disk sampling - blue-noise-like spiral pattern for high-quality PCF
fn vogel_disk_sample(sample_idx: u32, sample_count: u32, theta: f32) -> vec2<f32> {
    let GOLDEN_ANGLE = 2.39996323;  // 2π / φ² (golden angle in radians)
    let r = sqrt(f32(sample_idx) + 0.5) / sqrt(f32(sample_count));
    let angle = f32(sample_idx) * GOLDEN_ANGLE + theta;
    return vec2<f32>(cos(angle), sin(angle)) * r;
}

// Per-pixel hash for PCF rotation (reduces banding artifacts)
fn hash22(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    let d = dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * d);
}

fn point_light_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(0u, 1u, dir.x < 0.0);
    } else if a.y >= a.x && a.y >= a.z {
        return select(2u, 3u, dir.y < 0.0);
    } else {
        return select(4u, 5u, dir.z < 0.0);
    }
}

// Project `biased_pos` through face `layer`'s view-proj matrix, remap in-face UV
// to the face's atlas tile, then 8-tap Vogel-disk PCF. `cascade_scale` widens
// the kernel for distant CSM cascades so penumbrae stay visually consistent.
fn sample_face_shadow(
    layer: u32,
    cascade_scale: f32,
    biased_pos: vec3<f32>,
    frag_coord: vec2<f32>,
) -> f32 {
    let clip = face_mats[layer] * vec4<f32>(biased_pos, 1.0);
    if clip.w <= 0.0 { return 1.0; }

    let ndc       = clip.xyz / clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
        || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    // Pull the sample centre 1.5 texels inside the tile so the bilinear
    // comparison footprint cannot read a neighbouring face's depth.
    let edge = 1.5 / TILE_RES;
    let uvc  = clamp(shadow_uv, vec2<f32>(edge), vec2<f32>(1.0 - edge));

    // Face → tile mapping (must match src/atlas.rs::face_uv_transform).
    let tile = vec2<f32>(
        f32(layer % u32(ATLAS_TILES_PER_ROW)),
        f32(layer / u32(ATLAS_TILES_PER_ROW)),
    );
    let atlas_uv = (tile + uvc) / ATLAS_TILES_PER_ROW;

    // 2-texel base radius in face UV, converted to atlas-UV units.
    let filter_radius = (2.0 / TILE_RES) * cascade_scale / ATLAS_TILES_PER_ROW;

    // Per-pixel rotation to break up banding (stable with frame counter).
    let theta = hash22(frag_coord + vec2<f32>(f32(globals.frame))) * 6.28318530718;

    var lit_sum = 0.0;
    for (var i = 0u; i < PCF_TAPS; i++) {
        let offset = vogel_disk_sample(i, PCF_TAPS, theta) * filter_radius;
        lit_sum += textureSampleCompareLevel(
            shadow_atlas, shadow_sampler,
            atlas_uv + offset,
            ndc.z,
        );
    }
    return lit_sum / f32(PCF_TAPS);
}

fn shadow_factor(light: GpuLight, world_pos: vec3<f32>, N: vec3<f32>, frag_coord: vec2<f32>) -> f32 {
    if light.shadow_index == 0xffffffffu { return 1.0; }

    // Normal-offset: scale by (1 - NdotL) so face-on surfaces get near-zero
    // offset while grazing surfaces get the full amount (classic-pass parity).
    var light_dir: vec3<f32>;
    if light.light_type == 0u {
        light_dir = normalize(-light.direction_outer.xyz);
    } else {
        light_dir = normalize(light.position_range.xyz - world_pos);
    }
    let ndl        = max(dot(N, light_dir), 0.0);
    let biased_pos = world_pos + N * NORMAL_OFFSET_SCALE * (1.0 - ndl);

    if light.light_type == 1u {
        // Point light: pick the cube face the fragment falls in (one tile each).
        let face = point_light_face(biased_pos - light.position_range.xyz);
        return sample_face_shadow(light.shadow_index + face, 1.0, biased_pos, frag_coord);
    }

    if light.light_type == 0u {
        // Directional: CSM cascade selection by view distance, with smoothstep
        // blend zones around each split so transitions are invisible.
        let dist   = length(world_pos - camera.position_near.xyz);
        let splits = globals.csm_splits;

        var cascade_a = 3u;
        var cascade_b = 3u;
        var blend     = 0.0;

        if dist < splits.x * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 0u;
        } else if dist < splits.x * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 0u;
            cascade_b = 1u;
            blend = smoothstep(
                splits.x * (1.0 - BLEND_ZONE / 2.0),
                splits.x * (1.0 + BLEND_ZONE / 2.0),
                dist,
            );
        } else if dist < splits.y * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 1u;
        } else if dist < splits.y * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 1u;
            cascade_b = 2u;
            blend = smoothstep(
                splits.y * (1.0 - BLEND_ZONE / 2.0),
                splits.y * (1.0 + BLEND_ZONE / 2.0),
                dist,
            );
        } else if dist < splits.z * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 2u;
        } else if dist < splits.z * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 2u;
            cascade_b = 3u;
            blend = smoothstep(
                splits.z * (1.0 - BLEND_ZONE / 2.0),
                splits.z * (1.0 + BLEND_ZONE / 2.0),
                dist,
            );
        } else {
            cascade_a = 3u;
        }

        // Distant cascades cover more world per texel → widen the PCF kernel.
        let scale_a  = 1.0 + f32(cascade_a) * 1.5;
        let shadow_a = sample_face_shadow(light.shadow_index + cascade_a, scale_a, biased_pos, frag_coord);

        if blend <= 0.001 { return shadow_a; }

        if cascade_b != cascade_a {
            let scale_b  = 1.0 + f32(cascade_b) * 1.5;
            let shadow_b = sample_face_shadow(light.shadow_index + cascade_b, scale_b, biased_pos, frag_coord);
            return mix(shadow_a, shadow_b, blend);
        }
        return shadow_a;
    }

    // Spot light: single face at the base slot.
    return sample_face_shadow(light.shadow_index, 1.0, biased_pos, frag_coord);
}

// ── Cook-Torrance BRDF (copied from deferred_lighting.wgsl) ────────────────────

fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a    = roughness * roughness;
    let a2   = a * a;
    let NdH  = max(dot(N, H), 0.0);
    let denom = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdV = max(dot(N, V), 0.0);
    let NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

// Evaluate one direct light with the full Cook-Torrance BRDF.
// `sf` is the shadow factor (0=shadowed, 1=lit), computed at the call site.
fn pbr_direct_light(
    light:     GpuLight,
    world_pos: vec3<f32>,
    N:         vec3<f32>,
    V:         vec3<f32>,
    F0:        vec3<f32>,
    albedo:    vec3<f32>,
    roughness: f32,
    metallic:  f32,
    sf:        f32,
) -> vec3<f32> {
    var L:        vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {  // Directional light
        L        = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {  // Point or spot light
        let to_light = light.position_range.xyz - world_pos;
        let dist     = length(to_light);
        if dist > light.position_range.w { return vec3<f32>(0.0); }
        L = to_light / dist;
        let ratio   = dist / light.position_range.w;
        let falloff = max(0.0, 1.0 - ratio * ratio);
        var atten   = falloff * falloff;
        if light.light_type == 2u {  // Spot light
            let cos_a = dot(-L, light.direction_outer.xyz);
            atten    *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
    }

    let NdL = max(dot(N, L), 0.0);
    if NdL == 0.0 { return vec3<f32>(0.0); }

    if all(radiance < vec3<f32>(0.002)) { return vec3<f32>(0.0); }

    let H        = normalize(V + L);
    let D        = distribution_ggx(N, H, roughness);
    let G        = geometry_smith(N, V, L, roughness);
    let F        = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let kD       = (1.0 - F) * (1.0 - metallic);
    let specular = D * G * F / (4.0 * max(dot(N, V), 0.0) * NdL + 0.0001);

    return (kD * albedo / PI + specular) * radiance * NdL * sf;
}

// ── Tonemapping ────────────────────────────────────────────────────────────────

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ── Fragment entry ─────────────────────────────────────────────────────────────

@fragment
fn fs_main(
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(frag_coord.xy);

    // Sky pixels (depth = far plane) already hold the sky/clear colour → keep.
    let depth = textureLoad(depth_tex, pix, 0);
    if depth >= 1.0 { discard; }

    // Reconstruct world position from depth + inverse view-proj.
    let ndc       = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h   = camera.inv_view_proj * ndc;
    let world_pos = world_h.xyz / world_h.w;

    // G-buffer decode — same packing as the classic pass (F0 split across the
    // three alpha channels).
    let albedo    = textureLoad(gbuf_albedo, pix, 0).rgb;
    let nrm4      = textureLoad(gbuf_normal, pix, 0);
    let N         = normalize(nrm4.xyz);
    let orm4      = textureLoad(gbuf_orm, pix, 0);
    let ao        = orm4.x;
    let roughness = orm4.y;
    let metallic  = orm4.z;
    let emis4     = textureLoad(gbuf_emissive, pix, 0);
    let emissive  = emis4.rgb;
    let F0        = clamp(vec3<f32>(nrm4.w, orm4.w, emis4.w), vec3<f32>(0.0), vec3<f32>(0.999));
    let V         = normalize(camera.position_near.xyz - world_pos);

    // Direct lighting: all-lights loop (v1 — tiled light culling is the marked
    // extension point in DESIGN.md; fine below ~64 lights).
    var direct = vec3<f32>(0.0);
    let n_lights = min(globals.light_count, arrayLength(&lights));
    for (var i = 0u; i < n_lights; i++) {
        let light = lights[i];
        if light.light_type != 0u {
            // Cheap radius reject before any shadow sampling.
            if length(light.position_range.xyz - world_pos) > light.position_range.w {
                continue;
            }
        }
        let sf = shadow_factor(light, world_pos, N, frag_coord.xy);
        direct += pbr_direct_light(light, world_pos, N, V, F0, albedo, roughness, metallic, sf);
    }

    // Ambient is shadow-independent (AO occludes indirect light, shadow maps do
    // not) and emissive arrives pre-multiplied from the G-buffer — both exactly
    // as in the classic pass.
    var color = direct
        + albedo * globals.ambient_color.rgb * globals.ambient_color.w * ao
        + emissive;
    color = aces_tonemap(color);
    return vec4<f32>(color, 1.0);
}

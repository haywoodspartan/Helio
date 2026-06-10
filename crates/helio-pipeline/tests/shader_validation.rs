//! naga parse + validation of the four pipeline shaders.
//!
//! Catches WGSL syntax and type errors at `cargo test` time instead of at
//! first device launch. `Capabilities::all()` mirrors what the engine
//! requests from the adapter — in particular `geometry.wgsl` uses
//! `binding_array<…, 256>` (bindless materials), which is beyond the
//! baseline WebGPU capability set.

fn parse_and_validate(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}: WGSL parse error:\n{e}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name}: WGSL validation error:\n{e}"));
}

#[test]
fn cull_wgsl_is_valid() {
    parse_and_validate("cull.wgsl", include_str!("../shaders/cull.wgsl"));
}

#[test]
fn shadow_atlas_wgsl_is_valid() {
    parse_and_validate(
        "shadow_atlas.wgsl",
        include_str!("../shaders/shadow_atlas.wgsl"),
    );
}

#[test]
fn geometry_wgsl_is_valid() {
    parse_and_validate("geometry.wgsl", include_str!("../shaders/geometry.wgsl"));
}

#[test]
fn lighting_wgsl_is_valid() {
    parse_and_validate("lighting.wgsl", include_str!("../shaders/lighting.wgsl"));
}

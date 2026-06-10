use std::sync::Arc;

use helio_pass_billboard::BillboardPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky_lut::SkyLutPass;
use helio_pass_sky::SkyPass;
use helio_pass_taa::TaaPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_perf_overlay::{PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass, PerfOverlayShared};
use helio_pass_water_sim::WaterSimPass;
use helio_pipeline::{CullPass, GeometryPass, LightingPass, PipelineShared, ShadowAtlasPass};
use helio_v3::RenderGraph;

use crate::scene::Scene;
use crate::renderer::debug::DebugDrawPass;
use crate::renderer::config::RendererConfig;

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../../spotlight.png");

pub fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
) -> RenderGraph {
    build_default_graph_internal(device, queue, scene, config, debug_state, debug_camera_buf, true)
}

/// Same as `build_default_graph` but marks the graph as operating against an
/// externally-owned device.  Blocking `device.poll` calls are replaced with a
/// single non-blocking tick so Helio never races with the device owner's event
/// loop (e.g. GPUI's winit event loop).
pub fn build_default_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
) -> RenderGraph {
    build_default_graph_internal(device, queue, scene, config, debug_state, debug_camera_buf, false)
}

fn build_default_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    owns_device: bool,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let mut graph = if owns_device {
        RenderGraph::new(device, queue)
    } else {
        RenderGraph::new_with_external_device(device, queue)
    };

    let camera_buf = gpu_scene.camera.buffer();

    let hiz_pass = HiZBuildPass::new(device, config.internal_width(), config.internal_height());
    let hiz_view = Arc::clone(&hiz_pass.hiz_view);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        gpu_scene.lights.buffer(),
        gpu_scene.shadow_matrices.buffer(),
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
    )));

    let shadow_dirty_pass = ShadowDirtyPass::new(device);
    let face_dirty_buf = Arc::clone(&shadow_dirty_pass.face_dirty_buf);
    let face_geom_count_buf = Arc::clone(&shadow_dirty_pass.face_geom_count_buf);
    graph.add_pass(Box::new(shadow_dirty_pass));

    graph.add_pass(Box::new(ShadowPass::new(device, face_dirty_buf, face_geom_count_buf)));

    let has_sky = scene.sky_context().has_sky;
    if has_sky {
        let sky_lut_pass = SkyLutPass::new(device, camera_buf);
        let sky_lut_view = sky_lut_pass.sky_lut_view.clone();
        graph.add_pass(Box::new(sky_lut_pass));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            &sky_lut_view,
            config.internal_width(),
            config.internal_height(),
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        true,
    )));

    graph.add_pass(Box::new(IndirectDispatchPass::new(device)));
    graph.add_pass(Box::new(OcclusionCullPass::new(
        device,
        hiz_view,
        hiz_sampler,
        config.internal_width(),
        config.internal_height(),
    )));

    let perf_overlay_shared = PerfOverlayShared::new(device, config.internal_width(), config.internal_height());
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(hiz_pass));

    graph.add_pass(Box::new(LightCullPass::new(device, config.internal_width(), config.internal_height())));

    graph.add_pass(Box::new(GBufferPass::new(device, config.internal_width(), config.internal_height())));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    let mut deferred_light_pass = DeferredLightPass::new(
        device,
        queue,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    );
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(deferred_light_pass));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(WaterSimPass::new(
        device,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.width,
        config.height,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    let mut perf_overlay_pass = PerfOverlayPass::new(
        device,
        Arc::clone(&perf_overlay_shared),
        config.surface_format,
    );
    perf_overlay_pass.set_mode(config.perf_overlay_mode);
    graph.add_pass(Box::new(perf_overlay_pass));

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        false,
    )));

    // Use init_transients instead of set_render_size so that passes keep
    // their constructor-provided sizes. Default graph transients are internal-
    // resolution, so initialize with render-scaled dimensions.
    graph.init_transients(config.internal_width(), config.internal_height());

    graph
}

pub fn build_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let mut graph = RenderGraph::new(device, queue);

    let camera_buf = gpu_scene.camera.buffer();

    let hiz_pass = HiZBuildPass::new(device, config.internal_width(), config.internal_height());
    let hiz_view = Arc::clone(&hiz_pass.hiz_view);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        gpu_scene.lights.buffer(),
        gpu_scene.shadow_matrices.buffer(),
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
    )));

    let shadow_dirty_pass = ShadowDirtyPass::new(device);
    let face_dirty_buf = Arc::clone(&shadow_dirty_pass.face_dirty_buf);
    let face_geom_count_buf = Arc::clone(&shadow_dirty_pass.face_geom_count_buf);
    graph.add_pass(Box::new(shadow_dirty_pass));

    graph.add_pass(Box::new(ShadowPass::new(device, face_dirty_buf, face_geom_count_buf)));

    let has_sky = scene.sky_context().has_sky;
    if has_sky {
        let sky_lut_pass = SkyLutPass::new(device, camera_buf);
        let sky_lut_view = sky_lut_pass.sky_lut_view.clone();
        graph.add_pass(Box::new(sky_lut_pass));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            &sky_lut_view,
            config.internal_width(),
            config.internal_height(),
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        true,
    )));

    graph.add_pass(Box::new(IndirectDispatchPass::new(device)));
    graph.add_pass(Box::new(OcclusionCullPass::new(
        device,
        hiz_view,
        hiz_sampler,
        config.internal_width(),
        config.internal_height(),
    )));

    let perf_overlay_shared = PerfOverlayShared::new(device, config.internal_width(), config.internal_height());
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(hiz_pass));

    graph.add_pass(Box::new(GBufferPass::new(device, config.internal_width(), config.internal_height())));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));

    // ── HLFS Pass: Replaces traditional deferred lighting ──
    let mut hlfs_pass = HlfsPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    );
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(WaterSimPass::new(
        device,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.width,
        config.height,
        config.surface_format,
    )));

    let mut perf_overlay_pass = PerfOverlayPass::new(
        device,
        Arc::clone(&perf_overlay_shared),
        config.surface_format,
    );
    perf_overlay_pass.set_mode(config.perf_overlay_mode);
    graph.add_pass(Box::new(perf_overlay_pass));

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        false,
    )));

    // HLFS graph also runs its main transient resources at internal resolution.
    graph.init_transients(config.internal_width(), config.internal_height());

    graph
}

/// Opt-in GPU-driven pipeline graph (see `helio-pipeline/DESIGN.md`).
///
/// Replaces the classic shadow/cull prelude (ShadowMatrix → ShadowDirty →
/// Shadow → IndirectDispatch → OcclusionCull → HiZ → LightCull → GBuffer →
/// DeferredLight) with four GPU-driven passes: one unified multi-view cull
/// dispatch, one single-pass tiled shadow atlas, a culled G-buffer fill, and
/// deferred lighting against the tiled atlas. The tail (sky, VG, billboards,
/// water, TAA, perf overlay, debug draw) is wired exactly like the default
/// graph since GeometryPass/LightingPass publish the same frame contracts.
pub fn build_gpu_driven_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let camera_buf = gpu_scene.camera.buffer();
    let mut graph = RenderGraph::new(device, queue);

    // Cross-pass pipeline state (view set, culled indirect lists, atlas).
    let shared = PipelineShared::new(device);

    // [1] Unified cull: ONE compute dispatch produces per-view indirect lists
    // for the camera and every shadow face. [2] then renders all dirty faces
    // in ONE render pass via per-face viewports on the tiled 2D atlas.
    graph.add_pass(Box::new(CullPass::new(device, shared.clone())));
    graph.add_pass(Box::new(ShadowAtlasPass::new(device, shared.clone())));

    let has_sky = scene.sky_context().has_sky;
    if has_sky {
        let sky_lut_pass = SkyLutPass::new(device, camera_buf);
        let sky_lut_view = sky_lut_pass.sky_lut_view.clone();
        graph.add_pass(Box::new(sky_lut_pass));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            &sky_lut_view,
            config.internal_width(),
            config.internal_height(),
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        true,
    )));

    let perf_overlay_shared = PerfOverlayShared::new(device, config.internal_width(), config.internal_height());
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    // [3] G-buffer fill from the camera's culled draw list (view 0). Publishes
    // the same frame.gbuffer contract as GBufferPass so the tail keeps working.
    let mut geometry = GeometryPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        shared.clone(),
    );
    geometry.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(geometry));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    // [4] Deferred lighting sampling the tiled shadow atlas; publishes
    // frame.pre_aa with DeferredLightPass semantics.
    graph.add_pass(Box::new(LightingPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
        shared.clone(),
    )));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(WaterSimPass::new(
        device,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.width,
        config.height,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    let mut perf_overlay_pass = PerfOverlayPass::new(
        device,
        Arc::clone(&perf_overlay_shared),
        config.surface_format,
    );
    perf_overlay_pass.set_mode(config.perf_overlay_mode);
    graph.add_pass(Box::new(perf_overlay_pass));

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        false,
    )));

    // Same as the default graph: transients run at internal resolution.
    graph.init_transients(config.internal_width(), config.internal_height());

    graph
}

pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
    graph
}

pub fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Helio Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

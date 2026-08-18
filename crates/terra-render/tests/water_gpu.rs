//! The water pass, on a real GPU.
//!
//! Two things are checked that nothing else can:
//!
//! * the shader compiles as `Water::new` composes it -- this is the first blended
//!   pipeline in the renderer and the first module to pull in both `atmosphere` and
//!   `cdlod`, so a name collision between those two would only show up here
//! * the whole pipeline builds, which validates the bind group layouts against what
//!   the shader declares. A wrong binding index is a validation error, not a warning.
//!
//! No window is opened.

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let limits = adapter.limits();
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("water-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

#[test]
fn the_water_shader_compiles() {
    let Some((device, _queue)) = gpu() else { return };
    let src = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        include_str!("../../../assets/shaders/common/noise.wgsl"),
        include_str!("../../../assets/shaders/common/camera.wgsl"),
        include_str!("../../../assets/shaders/common/lighting.wgsl"),
        include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
        include_str!("../../../assets/shaders/common/cdlod.wgsl"),
        include_str!("../../../assets/shaders/render/water.wgsl"),
    );
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("water"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("water failed to compile:\n{err}");
    }
}

/// Building the real pipeline, which is what validates the bind group layouts against
/// the bindings the shader declares. A wrong group or binding index is a hard error
/// here and invisible to a shader-only compile.
///
/// Goes through `Water::new`, so the composition, the layouts, the blend state and the
/// reversed-Z depth state are all the shipped ones.
#[test]
fn the_water_pipeline_builds_against_its_bind_groups() {
    let Some((device, queue)) = gpu() else { return };
    // Lighting wants the volumetric fog volume; a 1x1x1 stand-in is enough to build the
    // bind group, since nothing here samples it.
    let fog = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test-fog"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let lighting = terra_render::lighting::Lighting::new(&device, Default::default(), &fog);
    let env = terra_render::environment::EnvironmentGpu::new(&device);

    // A flat heightfield, as the terrain's own storage buffer would hold.
    const RES: u32 = 64;
    let heights = wgpu::util::DeviceExt::create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("test-heights"),
            contents: bytemuck::cast_slice(&vec![100.0f32; (RES * RES) as usize]),
            usage: wgpu::BufferUsages::STORAGE,
        },
    );

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut water =
        terra_render::water::Water::new(&device, &lighting, &env, &heights, 4096.0, RES);
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("the water pipeline failed validation:\n{err}");
    }

    // And a prepare with water on has to select patches rather than nothing, or the
    // draw would be a no-op that looks like the feature is missing.
    let settings = terra_render::water::WaterSettings { enabled: true, ..Default::default() };
    let camera = terra_render::camera::Camera::default();
    water.prepare(&queue, &settings, &camera, 1.0, 0.0);
    assert!(water.triangle_count() > 0, "no patches were selected for a world with water");

    // Off means nothing submitted, not a hidden surface still costing patches.
    let off = terra_render::water::WaterSettings { enabled: false, ..Default::default() };
    water.prepare(&queue, &off, &camera, 1.0, 0.0);
    assert_eq!(water.triangle_count(), 0, "disabled water still submitted geometry");
}

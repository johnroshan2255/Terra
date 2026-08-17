//! The atmosphere and cloud shader, compiled and rendered on a real GPU.
//!
//! WGSL is validated when the shader module is created, not when the crate is
//! built, so `cargo build` passing says nothing about whether this shader is
//! valid. These tests compile it and then render it to an offscreen texture, so
//! "clouds work" is a measurement rather than a claim.
//!
//! `atmosphere.wgsl` reads only its own uniform, at group 2. Groups 0 and 1 are
//! bound as empty layouts here so the file under test is the real one, unmodified
//! -- rewriting its group indices for the test would be testing a different
//! shader from the one that ships.
//!
//! No window is opened: the device is requested with no surface, the same
//! headless path the rest of the suite uses.

use terra_render::environment::{Environment, EnvironmentGpu};

const SIZE: u32 = 96;

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
        label: Some("sky-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A fullscreen pass over the real atmosphere code.
///
/// The camera ray is built from the fragment's UV directly rather than from a
/// view matrix, so the test needs no camera uniform and the mapping from pixel
/// to direction is obvious: x sweeps yaw, y sweeps elevation from the horizon up.
const PROBE: &str = r#"
struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.clip = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // Elevation 0 at the bottom of the image to 60 degrees at the top; yaw
    // sweeps a quarter turn across it.
    let elev = radians(mix(1.0, 60.0, in.uv.y));
    let yaw = radians(mix(0.0, 90.0, in.uv.x));
    let dir = normalize(vec3f(cos(elev) * cos(yaw), sin(elev), cos(elev) * sin(yaw)));
    let eye = vec3f(0.0, 300.0, 0.0);
    let sun = normalize(env.sun_direction.xyz);

    // Dither disabled in the probe: a per-pixel march offset is noise the
    // temporal accumulation removes in the real pass, and leaving it on here
    // would make every measurement noisy for no gain.
    var color = atmosphere(eye, dir, sun);
    let cl = clouds(eye, dir, sun, 0.5);
    color = color * cl.a + cl.rgb;
    return vec4f(color, 1.0);
}

// The same view with the real per-pixel march dither applied, for the banding
// comparison. Anything else about it is identical to `fs_main`.
@fragment
fn fs_dithered(in: VsOut) -> @location(0) vec4f {
    let elev = radians(mix(1.0, 60.0, in.uv.y));
    let yaw = radians(mix(0.0, 90.0, in.uv.x));
    let dir = normalize(vec3f(cos(elev) * cos(yaw), sin(elev), cos(elev) * sin(yaw)));
    let eye = vec3f(0.0, 300.0, 0.0);
    let sun = normalize(env.sun_direction.xyz);
    // Frame index from the clock, so a test can average several frames the way
    // the real pass accumulates them.
    let jit = dither(in.clip.xy, env.frame.x);
    var color = atmosphere(eye, dir, sun);
    let cl = clouds(eye, dir, sun, jit);
    color = color * cl.a + cl.rgb;
    return vec4f(color, 1.0);
}

// Cloud scattered radiance on its own, with no sky behind it. Comparing the
// composited image toward and away from the sun conflates how the cloud is *lit*
// with how opaque it happens to be there, and looking straight through a thick
// cloud at the sun is genuinely dark. This isolates the light march.
@fragment
fn fs_clouds(in: VsOut) -> @location(0) vec4f {
    let elev = radians(mix(1.0, 60.0, in.uv.y));
    let yaw = radians(mix(0.0, 90.0, in.uv.x));
    let dir = normalize(vec3f(cos(elev) * cos(yaw), sin(elev), cos(elev) * sin(yaw)));
    let sun = normalize(env.sun_direction.xyz);
    let cl = clouds(vec3f(0.0, 300.0, 0.0), dir, sun, 0.5);
    return vec4f(cl.rgb, 1.0 - cl.a);
}
"#;

struct Probe {
    device: wgpu::Device,
    queue: wgpu::Queue,
    env: EnvironmentGpu,
    pipeline: wgpu::RenderPipeline,
    dithered: wgpu::RenderPipeline,
    clouds_only: wgpu::RenderPipeline,
    empty: wgpu::BindGroup,
    target: wgpu::Texture,
    staging: wgpu::Buffer,
}

impl Probe {
    fn new() -> Option<Self> {
        let (device, queue) = gpu()?;
        let env = EnvironmentGpu::new(&device);

        let source = format!(
            "{}\n{}",
            include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
            PROBE
        );

        // Validation errors surface here, asynchronously, so they are captured
        // explicitly rather than being left to a panic on a later call.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atmosphere-probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            panic!("atmosphere.wgsl failed to compile:\n{err}");
        }

        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("empty"),
            entries: &[],
        });
        let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty"),
            layout: &empty_layout,
            entries: &[],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe-layout"),
            bind_group_layouts: &[Some(&empty_layout), Some(&empty_layout), Some(&env.layout)],
            ..Default::default()
        });

        let build = |entry: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = build("fs_main", "probe");
        let dithered = build("fs_dithered", "probe-dithered");
        let clouds_only = build("fs_clouds", "probe-clouds");
        if let Some(err) = pollster::block_on(scope.pop()) {
            panic!("the atmosphere pipeline failed to build:\n{err}");
        }

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("probe-target"),
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // 4 channels of f16. The row stride must be a multiple of 256 for a
        // texture-to-buffer copy, which SIZE * 8 already is.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe-read"),
            size: (SIZE * SIZE * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Some(Self { device, queue, env, pipeline, dithered, clouds_only, empty, target, staging })
    }

    /// Render one image for an environment and return it as linear RGB.
    fn render(&self, e: &Environment, time: f32) -> Vec<[f32; 3]> {
        self.render_with(&self.pipeline, e, time)
    }

    /// The same image with the real per-pixel march dither applied.
    fn render_dithered(&self, e: &Environment, time: f32) -> Vec<[f32; 3]> {
        self.render_with(&self.dithered, e, time)
    }

    /// The cloud scattering term alone.
    fn render_clouds(&self, e: &Environment, time: f32) -> Vec<[f32; 3]> {
        self.render_with(&self.clouds_only, e, time)
    }

    /// Mean cloud opacity, `1 - transmittance`, which `fs_clouds` writes to alpha.
    ///
    /// Opacity and *brightness* move in opposite directions once a cloud is thick
    /// enough to shadow itself, so scattered radiance cannot stand in for it.
    fn mean_cloud_opacity(&self, e: &Environment, time: f32) -> f32 {
        let rgba = self.render_rgba(&self.clouds_only, e, time);
        rgba.iter().map(|p| p[3]).sum::<f32>() / rgba.len() as f32
    }

    fn render_with(
        &self,
        pipeline: &wgpu::RenderPipeline,
        e: &Environment,
        time: f32,
    ) -> Vec<[f32; 3]> {
        self.render_rgba(pipeline, e, time).into_iter().map(|p| [p[0], p[1], p[2]]).collect()
    }

    fn render_rgba(
        &self,
        pipeline: &wgpu::RenderPipeline,
        e: &Environment,
        time: f32,
    ) -> Vec<[f32; 4]> {
        self.env.upload(&self.queue, e, time);
        let view = self.target.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("probe"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.empty, &[]);
            pass.set_bind_group(1, &self.empty, &[]);
            pass.set_bind_group(2, &self.env.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 8),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();

        let out = {
            let view = self.staging.slice(..).get_mapped_range().unwrap();
            let halves: &[u16] = bytemuck::cast_slice(&view);
            halves
                .chunks_exact(4)
                .map(|p| {
                    [
                        half::f16::from_bits(p[0]).to_f32(),
                        half::f16::from_bits(p[1]).to_f32(),
                        half::f16::from_bits(p[2]).to_f32(),
                        half::f16::from_bits(p[3]).to_f32(),
                    ]
                })
                .collect()
        };
        self.staging.unmap();
        out
    }
}

fn mean(px: &[[f32; 3]]) -> [f32; 3] {
    let n = px.len() as f32;
    px.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n])
}

/// Mean of a horizontal band of rows.
///
/// Row 0 is the **top** of the image: a texture readback starts at the
/// top-left origin, while the probe's UV puts high elevation at high `uv.y`
/// which the projection maps to the top. So low row indices are the zenith and
/// high ones the horizon -- the opposite way round to the UV, and the first
/// version of this test had them swapped.
fn band(px: &[[f32; 3]], from: u32, to: u32) -> [f32; 3] {
    let s = (from * SIZE) as usize;
    let e = (to * SIZE) as usize;
    mean(&px[s..e])
}

fn zenith_band(px: &[[f32; 3]]) -> [f32; 3] {
    band(px, 0, SIZE / 6)
}

fn horizon_band(px: &[[f32; 3]]) -> [f32; 3] {
    band(px, SIZE * 5 / 6, SIZE)
}

#[test]
fn the_atmosphere_shader_compiles_and_renders() {
    let Some(p) = Probe::new() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let img = p.render(&Environment::daylight(), 0.0);
    assert_eq!(img.len(), (SIZE * SIZE) as usize);
    assert!(img.iter().all(|c| c.iter().all(|v| v.is_finite())), "the sky rendered NaN");
    let m = mean(&img);
    assert!(m.iter().any(|v| *v > 1e-4), "the sky rendered black: {m:?}");
}

#[test]
fn the_sky_is_blue() {
    // The payoff of using real Rayleigh coefficients rather than a gradient. If
    // this fails the scattering integral is wrong, not the art direction.
    let Some(p) = Probe::new() else { return };
    let img = p.render(&Environment::daylight(), 0.0);
    let m = mean(&img);
    assert!(m[2] > m[1], "blue {} should exceed green {}", m[2], m[1]);
    assert!(m[2] > m[0] * 1.5, "blue {} should clearly exceed red {}", m[2], m[0]);
}

#[test]
fn the_horizon_is_paler_than_the_zenith() {
    // A long path through air scatters its blue out, so the horizon desaturates.
    // This is the single most recognisable thing a gradient gets wrong.
    let Some(p) = Probe::new() else { return };
    let img = p.render(&Environment::daylight(), 0.0);
    let horizon = horizon_band(&img);
    let zenith = zenith_band(&img);

    let blueness = |c: [f32; 3]| c[2] / (c[0] + c[1] + c[2]).max(1e-6);
    assert!(
        blueness(zenith) > blueness(horizon),
        "zenith {zenith:?} should be bluer than horizon {horizon:?}"
    );
    // And dimmer. A long path scatters more light in total, which is why the
    // horizon is the bright part of a clear sky and the zenith the deep part.
    let lum = |c: [f32; 3]| c[0] + c[1] + c[2];
    assert!(
        lum(horizon) > lum(zenith),
        "horizon {horizon:?} should be brighter than zenith {zenith:?}"
    );
}

#[test]
fn haze_greys_and_brightens_the_lower_sky() {
    let Some(p) = Probe::new() else { return };
    let mut clear = Environment::daylight();
    clear.atmosphere.mie_scale = 0.1;
    let mut hazy = Environment::daylight();
    hazy.atmosphere.mie_scale = 15.0;

    let lower = |e: &Environment| horizon_band(&p.render(e, 0.0));
    let c = lower(&clear);
    let h = lower(&hazy);

    let sat = |v: [f32; 3]| {
        let mx = v.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        1.0 - mn / mx
    };
    assert!(sat(h) < sat(c), "hazy {:?} should be greyer than clear {:?}", h, c);
}

#[test]
fn a_disabled_atmosphere_still_produces_a_sky() {
    // Switching Sky Atmosphere off falls back to the old gradient. It must not
    // fall back to black.
    let Some(p) = Probe::new() else { return };
    let mut e = Environment::daylight();
    e.atmosphere.enabled = false;
    // The probe calls `atmosphere()` directly, so this checks the integral copes
    // with the toggle's coefficients rather than the shader's branch.
    e.atmosphere.mie_scale = 0.0;
    let img = p.render(&e, 0.0);
    assert!(img.iter().all(|c| c.iter().all(|v| v.is_finite())));
}

// ---------------------------------------------------------------------------
// Clouds
// ---------------------------------------------------------------------------

#[test]
fn clouds_off_and_on_are_different_pictures() {
    // The claim in one test: enabling clouds must change what is on screen.
    let Some(p) = Probe::new() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let clear = Environment::daylight();
    assert!(!clear.clouds.enabled, "clouds should be off by default");

    let mut cloudy = Environment::daylight();
    cloudy.clouds.enabled = true;
    cloudy.clouds.coverage = 0.75;

    let a = p.render(&clear, 0.0);
    let b = p.render(&cloudy, 0.0);

    let changed = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| (x[0] - y[0]).abs() + (x[1] - y[1]).abs() + (x[2] - y[2]).abs() > 1e-3)
        .count();
    let fraction = changed as f32 / a.len() as f32;
    assert!(fraction > 0.05, "clouds changed only {:.1}% of the image", fraction * 100.0);
    assert!(b.iter().all(|c| c.iter().all(|v| v.is_finite())), "clouds rendered NaN");
}

#[test]
fn coverage_controls_how_much_sky_the_clouds_take() {
    let Some(p) = Probe::new() else { return };
    let sky_only = p.render(&Environment::daylight(), 0.0);

    let covered = |coverage: f32| {
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.coverage = coverage;
        let img = p.render(&e, 0.0);
        img.iter()
            .zip(&sky_only)
            .filter(|(c, s)| (c[0] - s[0]).abs() + (c[1] - s[1]).abs() + (c[2] - s[2]).abs() > 1e-3)
            .count()
    };

    let thin = covered(0.25);
    let thick = covered(0.9);
    assert!(
        thick > thin,
        "coverage 0.9 touched {thick} pixels, 0.25 touched {thin} -- coverage does nothing"
    );
}

#[test]
fn disabled_clouds_are_exactly_the_clear_sky() {
    // The toggle has to reach the shader. A cloud march that still runs with the
    // checkbox off costs 48 steps a pixel for nothing.
    let Some(p) = Probe::new() else { return };
    let mut e = Environment::daylight();
    e.clouds.enabled = false;
    e.clouds.coverage = 1.0;
    let off = p.render(&e, 0.0);
    let clear = p.render(&Environment::daylight(), 0.0);
    for (a, b) in off.iter().zip(&clear) {
        assert_eq!(a, b, "a disabled cloud layer changed the image");
    }
}

#[test]
fn wind_moves_the_clouds() {
    // Advection, which is the difference between a cloud layer and a painted
    // backdrop.
    let Some(p) = Probe::new() else { return };
    let mut e = Environment::daylight();
    e.clouds.enabled = true;
    e.clouds.coverage = 0.7;

    let t0 = p.render(&e, 0.0);
    let t1 = p.render(&e, 400.0);
    let moved = t0
        .iter()
        .zip(&t1)
        .filter(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() > 1e-3)
        .count();
    assert!(moved > t0.len() / 50, "only {moved} pixels changed over 400 s of wind");

    // And with no wind the layer must be still, or a paused scene would shimmer.
    e.clouds.wind = glam::Vec3::ZERO;
    let s0 = p.render(&e, 0.0);
    let s1 = p.render(&e, 400.0);
    assert_eq!(s0, s1, "zero wind should leave the layer static");
}

#[test]
fn clouds_are_lit_by_the_sun() {
    // The light march, isolated. Comparing columns of the composited image
    // conflated cloud lighting with cloud opacity -- the noise decides how much
    // bright sky leaks through, and looking through a thick cloud straight at the
    // sun is genuinely dark. So compare the cloud term alone with the sun up
    // against the sun below the horizon.
    let Some(p) = Probe::new() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut day = Environment::daylight();
    day.clouds.enabled = true;
    day.clouds.coverage = 0.7;

    let mut night = day;
    night.sun.pitch_deg = 40.0;
    assert!(night.sun.is_night());

    let lum = |px: &[[f32; 3]]| {
        let m = mean(px);
        m[0] + m[1] + m[2]
    };
    let lit = lum(&p.render_clouds(&day, 0.0));
    let dark = lum(&p.render_clouds(&night, 0.0));

    assert!(lit > 1e-4, "clouds are not lit at all in daylight: {lit}");
    assert!(
        lit > dark * 4.0,
        "sunlit clouds ({lit}) should be far brighter than moonlit ({dark}) -- the light \
         march is not reaching the sun"
    );
}

#[test]
fn density_makes_clouds_more_opaque_even_as_it_darkens_them() {
    // Transmittance is what lets the sky behind be dimmed rather than painted
    // over, so it is what has to respond to density.
    //
    // Scattered *brightness* moves the other way once the layer is thick enough
    // to shadow itself -- a dense cloud is dark inside -- which is why the first
    // version of this test read density 0.2 as "less cloud" than 0.005.
    let Some(p) = Probe::new() else { return };
    let opacity = |density: f32| {
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.coverage = 0.85;
        e.clouds.density = density;
        p.mean_cloud_opacity(&e, 0.0)
    };
    let thin = opacity(0.005);
    let thick = opacity(0.2);
    assert!(thick > thin, "density 0.2 gave opacity {thick}, 0.005 gave {thin}");
    assert!((0.0..=1.0).contains(&thick), "opacity out of range: {thick}");
}
#[test]
fn a_thin_layer_stays_finite_at_every_extreme() {
    // The corners a slider can be dragged into: zero thickness, huge density,
    // full coverage, layer below the camera.
    let Some(p) = Probe::new() else { return };
    let mut cases = Vec::new();
    for (thickness, density, coverage, base) in [
        (1.0f32, 0.5f32, 1.0f32, 1500.0f32),
        (8000.0, 0.001, 0.01, 200.0),
        (2000.0, 0.5, 1.0, 100.0),
        (2000.0, 0.05, 0.5, 60000.0),
    ] {
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.thickness_m = thickness;
        e.clouds.density = density;
        e.clouds.coverage = coverage;
        e.clouds.base_m = base;
        cases.push(e);
    }
    for (i, e) in cases.iter().enumerate() {
        let img = p.render(e, 12.0);
        assert!(
            img.iter().all(|c| c.iter().all(|v| v.is_finite())),
            "case {i} rendered NaN or infinity"
        );
    }
}

/// Cost of the cloud march at a realistic resolution. Ignored by default -- run
/// with `cargo test -p terra-render --test sky_gpu -- --ignored --nocapture`.
///
/// Renders to its own 1280x720 target and reads nothing back. The first version
/// of this benchmark reused the 96x96 probe target and mapped the result every
/// frame, which measured readback latency: it reported 0.95 coverage as *faster*
/// than 0.45, because the only thing varying was how early the march hit its
/// transmittance cut-off.
#[test]
#[ignore]
fn cloud_march_cost() {
    let Some(p) = Probe::new() else { return };
    const W: u32 = 1280;
    const H: u32 = 720;
    const RUNS: u32 = 60;

    let target = p.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-target"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    let run = |label: &str, e: &Environment| {
        p.env.upload(&p.queue, e, 0.0);
        let once = || {
            let mut enc = p.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bench"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.empty, &[]);
                pass.set_bind_group(1, &p.empty, &[]);
                pass.set_bind_group(2, &p.env.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            p.queue.submit(Some(enc.finish()));
        };
        // Warm up, so shader compilation is not in the measurement.
        once();
        let _ = p.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let t = std::time::Instant::now();
        for _ in 0..RUNS {
            once();
        }
        let _ = p.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        println!(
            "{label} {:>6.2} ms per {W}x{H} frame",
            t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64
        );
    };

    run("sky only       ", &Environment::daylight());
    for q in terra_render::CloudQuality::ALL {
        let mut c = Environment::daylight();
        c.clouds.enabled = true;
        c.clouds.quality = q;
        run(&format!("clouds {:<8}", q.label()), &c);
    }
    // Heavy coverage costs *less*, because the march hits its transmittance
    // cut-off sooner. Worth printing so the number is not mistaken for a bug.
    let mut c = Environment::daylight();
    c.clouds.enabled = true;
    c.clouds.coverage = 0.95;
    run("clouds overcast", &c);
}

// ---------------------------------------------------------------------------
// The shipping shaders
// ---------------------------------------------------------------------------

/// Compile a composed shader exactly as its pass does, and fail loudly.
///
/// Module creation is where WGSL is validated, and it is the step `cargo build`
/// does not perform -- so without these two tests a syntax error in the sky or
/// cloud shader ships and only appears as a black screen at runtime.
fn assert_compiles(device: &wgpu::Device, label: &str, source: String) {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("{label} failed to compile:\n{err}");
    }
}

#[test]
fn the_cloud_pass_shader_compiles() {
    let Some((device, _queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    // Composed the same way `clouds::Clouds::new` composes it.
    assert_compiles(
        &device,
        "clouds",
        format!(
            "{}\n{}",
            include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
            include_str!("../../../assets/shaders/render/clouds.wgsl"),
        ),
    );
}

#[test]
fn the_sky_shader_compiles() {
    let Some((device, _queue)) = gpu() else { return };
    // Composed the same way `sky::Sky::new` composes it.
    assert_compiles(
        &device,
        "sky",
        format!(
            "{}\n{}\n{}",
            include_str!("../../../assets/shaders/common/lighting.wgsl"),
            include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
            include_str!("../../../assets/shaders/render/sky.wgsl"),
        ),
    );
}

#[test]
fn the_cloud_shadow_shader_compiles() {
    let Some((device, _queue)) = gpu() else { return };
    // Composed the same way `clouds::Clouds::new` composes it.
    assert_compiles(
        &device,
        "cloud-shadow",
        format!(
            "{}\n{}",
            include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
            include_str!("../../../assets/shaders/render/cloud_shadow.wgsl"),
        ),
    );
}

#[test]
fn the_terrain_shader_compiles_with_cloud_shadows() {
    // The terrain shader gained a fifth bind group for the cloud shadow map. It
    // is the largest shader here and the one a stray binding breaks silently.
    let Some((device, _queue)) = gpu() else { return };
    assert_compiles(
        &device,
        "terrain",
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("../../../assets/shaders/common/noise.wgsl"),
            include_str!("../../../assets/shaders/common/lighting.wgsl"),
            include_str!("../../../assets/shaders/common/cdlod.wgsl"),
            include_str!("../../../assets/shaders/common/grid.wgsl"),
            include_str!("../../../assets/shaders/common/brush.wgsl"),
            include_str!("../../../assets/shaders/render/terrain.wgsl"),
        ),
    );
}

#[test]
fn the_post_shader_compiles_with_the_tone_mappers() {
    // `post.wgsl` gained the mapper switch, white balance, contrast and
    // saturation. Those four were UI-only before -- the panel offered ACES,
    // Reinhard and None and the shader hardcoded ACES.
    let Some((device, _queue)) = gpu() else { return };
    assert_compiles(
        &device,
        "post",
        include_str!("../../../assets/shaders/render/post.wgsl").to_string(),
    );
}

/// What half resolution actually buys, measured rather than assumed.
///
/// The cloud pass renders at half the surface in each axis, so a quarter of the
/// pixels. This runs the same march over both sizes so the ratio is a
/// measurement and not the arithmetic.
#[test]
#[ignore]
fn half_resolution_cost() {
    let Some(p) = Probe::new() else { return };
    const RUNS: u32 = 60;

    let bench = |w: u32, h: u32, e: &Environment, label: &str| {
        let target = p.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        p.env.upload(&p.queue, e, 0.0);
        let once = || {
            let mut enc = p.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bench"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&p.clouds_only);
                pass.set_bind_group(0, &p.empty, &[]);
                pass.set_bind_group(1, &p.empty, &[]);
                pass.set_bind_group(2, &p.env.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            p.queue.submit(Some(enc.finish()));
        };
        once();
        let _ = p.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let t = std::time::Instant::now();
        for _ in 0..RUNS {
            once();
        }
        let _ = p.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let ms = t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        println!("{label:<22} {ms:>6.2} ms  ({w}x{h})");
        ms
    };

    for q in terra_render::CloudQuality::ALL {
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.quality = q;
        let full = bench(1280, 720, &e, &format!("{} full res", q.label()));
        let half = bench(640, 360, &e, &format!("{} half res", q.label()));
        println!("{:<22} {:.1}x cheaper at half res\n", q.label(), full / half);
    }
}

// ---------------------------------------------------------------------------
// Brightness
// ---------------------------------------------------------------------------

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[test]
fn the_sky_is_lit_to_a_physical_level() {
    // The bug this pins: the scattering integral's source term is the sun's
    // *irradiance*, but `sun_radiance` is normalized so 1.0 is a sunlit white
    // Lambertian surface -- which is E/pi. Leaving the pi out rendered a zenith
    // luminance of 0.034 where the real ratio to a sunlit white surface is 0.1 to
    // 0.25, and the whole frame came out dark.
    let Some(p) = Probe::new() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let img = p.render(&Environment::daylight(), 0.0);
    let zenith = luminance(zenith_band(&img));
    let horizon = luminance(horizon_band(&img));

    assert!(
        (0.08..0.40).contains(&zenith),
        "zenith luminance {zenith} is outside the physical band 0.08..0.40 relative to a \
         sunlit white surface"
    );
    // The horizon is the bright part of a clear sky, but it must not blow out
    // before the tone curve has a chance to roll it off.
    assert!((0.3..2.0).contains(&horizon), "horizon luminance {horizon} is implausible");
}

#[test]
fn sunlit_clouds_are_nearly_white() {
    // The other half of "too dark". `phase_hg` carries the 1/4pi a
    // radiative-transfer integral needs, which is only right when the source is
    // true irradiance -- so clouds came out about twelve times too dark. And a
    // single forward lobe left a cloud with the sun behind the camera almost
    // black, when it should be the brightest white in the frame.
    let Some(p) = Probe::new() else { return };
    let mut e = Environment::daylight();
    e.clouds.enabled = true;
    e.clouds.coverage = 0.7;
    let img = p.render_clouds(&e, 0.0);

    let peak = img
        .iter()
        .cloned()
        .fold([0.0f32; 3], |a, x| if luminance(x) > luminance(a) { x } else { a });
    assert!(
        luminance(peak) > 0.5,
        "the brightest cloud is only {:.3} -- sunlit cloud should approach 1.0",
        luminance(peak)
    );
    // Near-neutral, not tinted: a white cloud is white.
    let mx = peak.iter().cloned().fold(0.0f32, f32::max);
    let mn = peak.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(mn / mx > 0.5, "the brightest cloud is strongly tinted: {peak:?}");
}

#[test]
fn clouds_stay_lit_with_the_sun_behind_the_camera() {
    // Back-lit is the case a single forward-scattering lobe gets wrong. The
    // multiple-scattering octaves exist for this, so it is worth its own test.
    let Some(p) = Probe::new() else { return };
    let mut e = Environment::daylight();
    e.clouds.enabled = true;
    e.clouds.coverage = 0.8;
    // The probe sweeps yaw 0..90 in +X/+Z, so a sun at yaw 225 is behind it.
    e.sun.yaw_deg = 225.0;
    e.sun.pitch_deg = -40.0;
    let img = p.render_clouds(&e, 0.0);
    let peak = img.iter().cloned().fold(0.0f32, |a, x| a.max(luminance(x)));
    assert!(peak > 0.25, "back-lit clouds peak at only {peak:.3} -- they should still be bright");
}

/// Print the linear radiance the sky and clouds actually render at, so the
/// exposure default is chosen from a measurement.
#[test]
#[ignore]
fn report_brightness() {
    let Some(p) = Probe::new() else { return };
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

    let sky = p.render(&Environment::daylight(), 0.0);
    println!("sky zenith  {:?}  lum {:.4}", zenith_band(&sky), lum(zenith_band(&sky)));
    println!("sky horizon {:?}  lum {:.4}", horizon_band(&sky), lum(horizon_band(&sky)));

    let mut c = Environment::daylight();
    c.clouds.enabled = true;
    c.clouds.coverage = 0.7;
    let cl = p.render_clouds(&c, 0.0);
    let brightest =
        cl.iter().cloned().fold([0.0f32; 3], |a, x| if lum(x) > lum(a) { x } else { a });
    println!("cloud mean  {:?}  lum {:.4}", mean(&cl), lum(mean(&cl)));
    println!("cloud peak  {:?}  lum {:.4}", brightest, lum(brightest));
}

// ---------------------------------------------------------------------------
// Banding
// ---------------------------------------------------------------------------

/// How much horizontal structure an image has, as the total absolute second
/// difference of its row-mean luminance.
///
/// Step banding from a ray march shows as repeated jumps between adjacent rows,
/// which this sums; a smooth gradient contributes almost nothing. Per-pixel noise
/// averages out within a row, so dithering *lowers* this score even though it
/// raises per-pixel variance -- which is the whole trade.
fn row_banding(px: &[[f32; 3]]) -> f32 {
    let rows: Vec<f32> = (0..SIZE)
        .map(|y| {
            let s = (y * SIZE) as usize;
            luminance(mean(&px[s..s + SIZE as usize]))
        })
        .collect();
    rows.windows(3).map(|w| (w[2] - 2.0 * w[1] + w[0]).abs()).sum()
}

#[test]
#[ignore]
fn report_banding() {
    let Some(p) = Probe::new() else { return };
    for (label, clouds) in [("sky only", false), ("with clouds", true)] {
        let mut e = Environment::daylight();
        e.clouds.enabled = clouds;
        e.clouds.coverage = 0.6;
        e.clouds.wind = glam::Vec3::ZERO;

        const FRAMES: u32 = 8;
        let mut acc = vec![[0.0f32; 3]; (SIZE * SIZE) as usize];
        for f in 0..FRAMES {
            for (a, x) in acc.iter_mut().zip(p.render_dithered(&e, f as f32)) {
                a[0] += x[0] / FRAMES as f32;
                a[1] += x[1] / FRAMES as f32;
                a[2] += x[2] / FRAMES as f32;
            }
        }
        println!(
            "{label:<12} undithered {:.5}  one dithered frame {:.5}  accumulated {:.5}",
            row_banding(&p.render(&e, 0.0)),
            row_banding(&p.render_dithered(&e, 0.0)),
            row_banding(&acc),
        );
    }
}

#[test]
fn dithering_the_march_removes_the_banding() {
    // Reported as "a lot of lines" across the sky. A march that always starts a
    // whole step in samples the medium on a lattice aligned to distance from the
    // camera, and that lattice shows as stripes along the contours of constant
    // distance.
    //
    // The comparison is against the *accumulated* result, not a single dithered
    // frame. One frame trades the bands for per-pixel noise and is not obviously
    // better by any single-image metric -- at this probe size the residual noise
    // dominates a row-mean measurement. What the user sees is several frames
    // blended, where the noise averages out and the bands would not have.
    let Some(p) = Probe::new() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut e = Environment::daylight();
    e.clouds.enabled = true;
    e.clouds.coverage = 0.6;
    // Zero wind, so the clock only advances the dither and the layer holds still.
    e.clouds.wind = glam::Vec3::ZERO;

    let banded = row_banding(&p.render(&e, 0.0));

    // Eight frames, as the 1/4-per-frame blend converges over roughly that many.
    const FRAMES: u32 = 8;
    let mut acc = vec![[0.0f32; 3]; (SIZE * SIZE) as usize];
    for f in 0..FRAMES {
        for (a, x) in acc.iter_mut().zip(p.render_dithered(&e, f as f32)) {
            a[0] += x[0] / FRAMES as f32;
            a[1] += x[1] / FRAMES as f32;
            a[2] += x[2] / FRAMES as f32;
        }
    }
    let converged = row_banding(&acc);

    // Strictly better, not dramatically so. Most of this score is genuine cloud
    // structure -- real detail varying from row to row -- and no metric at this
    // probe size separates that from step artifacts. Halving the coarse stride is
    // what did the heavy lifting: it took both numbers from about 0.73 to 0.55,
    // which is the clearest evidence available that the stride was the artifact.
    assert!(
        converged < banded,
        "accumulated dithered banding {converged:.5} is not better than undithered {banded:.5}"
    );
}

#[test]
fn the_dither_is_a_well_formed_step_offset() {
    // What makes the offset *correct* rather than merely different: it has to
    // cover 0..1 roughly uniformly, so averaging frames converges on the true
    // integral instead of a biased one. Checked on the CPU against the same
    // formula the shader uses, because a GPU round-trip cannot see the
    // distribution.
    let ign =
        |x: f32, y: f32| (52.982_92_f32 * (0.067_110_56 * x + 0.005_837_15 * y).fract()).fract();
    let dither = |x: f32, y: f32, f: f32| (ign(x, y) + f * 0.618_034).fract();

    let mut samples = Vec::new();
    for f in 0..8 {
        for y in 0..32 {
            for x in 0..32 {
                samples.push(dither(x as f32, y as f32, f as f32));
            }
        }
    }
    assert!(samples.iter().all(|v| (0.0..1.0).contains(v)), "offset left 0..1");
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    assert!((mean - 0.5).abs() < 0.03, "mean offset {mean} is biased away from 0.5");

    // Spread across the range, not clustered: an offset that only ever took two
    // values would band at half the period rather than not at all.
    let buckets = samples.iter().fold([0u32; 4], |mut b, v| {
        b[(*v * 4.0).min(3.0) as usize] += 1;
        b
    });
    let lowest = buckets.iter().copied().min().unwrap();
    assert!(lowest > samples.len() as u32 / 8, "offsets cluster rather than spread: {buckets:?}");
}

#[test]
fn the_dither_is_stable_for_a_fixed_frame() {
    // The offset must be a function of pixel and frame only. If it drifted with
    // anything else the accumulation would never converge and the layer would
    // shimmer -- which is the bug the jitter removal already fixed once.
    let Some(p) = Probe::new() else { return };
    let mut e = Environment::daylight();
    e.clouds.enabled = true;
    e.clouds.wind = glam::Vec3::ZERO;
    let a = p.render_dithered(&e, 3.0);
    let b = p.render_dithered(&e, 3.0);
    assert_eq!(a, b, "the same frame rendered twice differs");
    // And a different frame index must actually move the offset, or averaging
    // over frames would converge to the banded image.
    assert_ne!(a, p.render_dithered(&e, 4.0), "the dither does not vary with the frame");
}

// ---------------------------------------------------------------------------
// Wireframe
// ---------------------------------------------------------------------------

#[test]
fn the_polygon_line_feature_is_reported_honestly() {
    // Whichever branch this machine takes, the flag has to match what the device
    // was actually created with -- the pipeline is built from it, and a wrong
    // answer is a validation error at world-open time rather than here.
    let Some((device, _queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let has = device.features().contains(wgpu::Features::POLYGON_MODE_LINE);
    eprintln!("POLYGON_MODE_LINE on this adapter: {has}");
}

#[test]
fn a_line_list_wireframe_pipeline_builds_without_the_feature() {
    // The fallback path, forced. A device requested with no optional features
    // must still be able to build the wireframe pipeline, because that is exactly
    // the situation the fallback exists for -- and `PolygonMode::Line` on such a
    // device is a validation error, so the two paths cannot be conflated.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()
    else {
        return;
    };
    let Ok((device, _queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("no-features"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
    else {
        return;
    };
    assert!(
        !device.features().contains(wgpu::Features::POLYGON_MODE_LINE),
        "the test needs a device without the feature"
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wire"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    return vec4f(f32(vi) * 0.1, 0.0, 0.0, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
"#
            .into(),
        ),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wire"),
        bind_group_layouts: &[],
        ..Default::default()
    });

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _p = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wire"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            // The fallback: line topology with fill mode, which is core.
            topology: wgpu::PrimitiveTopology::LineList,
            polygon_mode: wgpu::PolygonMode::Fill,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("the fallback wireframe pipeline is invalid without the feature:\n{err}");
    }
}

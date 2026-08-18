//! Parallax occlusion mapping, measured on a real GPU.
//!
//! The march replaced a single-step offset, and the properties worth pinning are
//! the ones a screenshot cannot show:
//!
//! * a flat height channel displaces **nothing**, exactly. This is what makes
//!   parallax safe to leave on by default -- a set with no displacement map
//!   decodes to mid-grey, and mid-grey has to be a no-op rather than a uniform
//!   slide of the whole texture.
//! * the depth returned is bounded by the relief amplitude, so a grazing view of
//!   a deep material cannot walk the lookup off into another tile.
//! * nothing returns a NaN. `slope` divides by the view-normal dot product, which
//!   is why it is clamped, and a NaN here would spread through the uv into every
//!   channel of the material.
//!
//! `relief_at` uses `textureSampleLevel` rather than `textureSample`, so the POM
//! functions are callable from a compute entry point -- which is what lets them be
//! probed directly instead of inferred from rendered pixels.
//!
//! No window is opened.

use wgpu::util::DeviceExt;

/// Matches the harness struct below: uv, slope, then (amp, steps, axis, mode).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe {
    uv: [f32; 4],
    slope: [f32; 4],
    args: [f32; 4],
}

impl Probe {
    fn depth(uv: [f32; 3], slope: [f32; 3], amp: f32, steps: i32) -> Self {
        Self {
            uv: [uv[0], uv[1], uv[2], 0.0],
            slope: [slope[0], slope[1], slope[2], 0.0],
            // axis 1 = the XZ projection, which is what flat ground uses.
            args: [amp, steps as f32, 1.0, 0.0],
        }
    }
    fn shadow(uv: [f32; 3], slope: [f32; 3], amp: f32, steps: i32) -> Self {
        let mut p = Self::depth(uv, slope, amp, steps);
        p.args[3] = 1.0;
        p
    }
}

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
        label: Some("pom-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A compute entry appended to the real terrain shader, so the functions under
/// test are the shipped ones rather than a copy that can drift.
const HARNESS: &str = r#"
struct Probe { uv: vec4f, slope: vec4f, args: vec4f };
@group(0) @binding(1) var<storage, read> probes: array<Probe>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(1)
fn cs_pom(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let p = probes[i];
    let amp = p.args.x;
    let steps = i32(p.args.y);
    let axis = u32(p.args.z);
    let depth = pom_depth(0u, p.uv.xyz, p.slope.xyz, axis, amp, steps);
    if (p.args.w < 0.5) {
        results[i] = depth;
    } else {
        results[i] = pom_shadow(0u, p.uv.xyz - p.slope.xyz * depth, depth, p.slope.xyz, axis, amp);
    }
}

// One march per invocation over a screen's worth of invocations, for the
// benchmark. `uv` comes from the invocation id rather than a buffer so the cost
// measured is the march and not two million buffer reads.
struct Bench { args: vec4f, slope: vec4f };
@group(0) @binding(3) var<uniform> bench: Bench;
@group(0) @binding(4) var<storage, read_write> bench_out: array<f32>;

@compute @workgroup_size(64)
fn cs_pom_bench(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let uv = vec3f(f32(i % 1920u) * 0.013, 0.0, f32(i / 1920u) * 0.013);
    let amp = bench.args.x;
    let steps = i32(bench.args.y);
    let d = pom_depth(0u, uv, bench.slope.xyz, 1u, amp, steps);
    let s = pom_shadow(0u, uv - bench.slope.xyz * d, d, bench.slope.xyz, 1u, amp);
    bench_out[i] = d + s;
}
"#;

/// The terrain shader, composed exactly as `Terrain::new` composes it, plus the
/// probe entry point.
fn module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let src = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        include_str!("../../../assets/shaders/common/noise.wgsl"),
        include_str!("../../../assets/shaders/common/lighting.wgsl"),
        include_str!("../../../assets/shaders/common/cdlod.wgsl"),
        include_str!("../../../assets/shaders/common/grid.wgsl"),
        include_str!("../../../assets/shaders/common/brush.wgsl"),
        include_str!("../../../assets/shaders/render/terrain.wgsl"),
        HARNESS,
    );
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pom-harness"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

/// A one-layer height field in the alpha channel, `n` by `n`.
///
/// `Rgba8UnormSrgb` to match the real palette exactly -- alpha is linear even in
/// an sRGB format, which is why the height channel lives there. The 1/255
/// quantization is part of what is being tested: a decoded mid-grey map is 128,
/// not an exact 0.5.
fn height_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: u32,
    alpha: &[u8],
) -> wgpu::Texture {
    assert_eq!(alpha.len() as u32, n * n);
    let mut data = vec![0u8; (n * n * 4) as usize];
    for (i, a) in alpha.iter().enumerate() {
        data[i * 4] = 128;
        data[i * 4 + 1] = 128;
        data[i * 4 + 2] = 128;
        data[i * 4 + 3] = *a;
    }
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("pom-height"),
            size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &data,
    )
}

/// Run every probe against one height field and read the results back.
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    alpha: &[u8],
    n: u32,
    probes: &[Probe],
) -> Vec<f32> {
    let module = module(device);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pom"),
        layout: None,
        module: &module,
        entry_point: Some("cs_pom"),
        compilation_options: Default::default(),
        cache: None,
    });

    let tex = height_texture(device, queue, n, alpha);
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("pom-samp"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let probe_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probes"),
        contents: bytemuck::cast_slice(probes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (probes.len() * 4) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: probe_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
        ],
    });
    let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g2"),
        layout: &pipeline.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &g0, &[]);
        pass.set_bind_group(2, &g2, &[]);
        pass.dispatch_workgroups(probes.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    queue.submit([enc.finish()]);

    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let data = read.slice(..).get_mapped_range().expect("mapped range");
    bytemuck::cast_slice::<u8, f32>(&data).to_vec()
}

/// Mid-grey everywhere: what a set with no displacement map decodes to.
fn flat(n: u32) -> Vec<u8> {
    vec![128u8; (n * n) as usize]
}

/// A trench down the middle: the left half at the reference plane, the right half
/// at the bottom of the relief.
fn trench(n: u32) -> Vec<u8> {
    let mut v = vec![128u8; (n * n) as usize];
    for z in 0..n {
        for x in 0..n {
            if x >= n / 2 {
                v[(z * n + x) as usize] = 0;
            }
        }
    }
    v
}

#[test]
fn a_flat_height_map_displaces_nothing() {
    let Some((device, queue)) = gpu() else { return };
    let n = 32;
    // A spread of view slopes, including steeply oblique ones where a
    // reference-at-the-top implementation would slide the texture furthest.
    let probes: Vec<Probe> = [0.0f32, 0.25, 1.0, 4.0]
        .iter()
        .map(|s| Probe::depth([0.3, 0.0, 0.7], [*s, 0.0, *s * 0.5], 0.02, 16))
        .collect();
    let got = run(&device, &queue, &flat(n), n, &probes);
    for (i, d) in got.iter().enumerate() {
        assert_eq!(*d, 0.0, "probe {i}: flat relief displaced by {d}");
    }
}

#[test]
fn a_trench_is_found_and_the_depth_is_bounded() {
    let Some((device, queue)) = gpu() else { return };
    let n = 32;
    let amp = 0.05f32;
    // Start inside the trench, looking obliquely across it.
    let probes = vec![
        Probe::depth([0.8, 0.0, 0.5], [1.0, 0.0, 0.0], amp, 24),
        Probe::depth([0.9, 0.0, 0.5], [2.0, 0.0, 0.0], amp, 24),
    ];
    let got = run(&device, &queue, &trench(n), n, &probes);
    for (i, d) in got.iter().enumerate() {
        assert!(*d > 0.0, "probe {i}: a trench should have depth, got {d}");
        // The reference plane sits at mid-grey, so the relief only reaches half
        // the amplitude below it. Anything deeper means the march ran past its
        // bound and the lookup could land in a neighbouring tile.
        assert!(
            *d <= amp * 0.5 + 1e-6,
            "probe {i}: depth {d} exceeds the half-amplitude bound {}",
            amp * 0.5
        );
    }
}

#[test]
fn no_view_angle_produces_a_nan_or_a_negative_depth() {
    let Some((device, queue)) = gpu() else { return };
    let n = 32;
    let mut probes = Vec::new();
    // Slope is the view direction divided by its normal component, so it grows
    // without bound as the view goes edge-on. These are past what the shader's
    // own clamp admits, deliberately.
    for s in [0.0f32, 0.5, 2.0, 20.0, 200.0] {
        for amp in [0.0f32, 0.001, 0.05, 0.25] {
            probes.push(Probe::depth([0.4, 0.0, 0.6], [s, 0.0, s], amp, 16));
        }
    }
    let got = run(&device, &queue, &trench(n), n, &probes);
    for (i, d) in got.iter().enumerate() {
        assert!(d.is_finite(), "probe {i} returned {d}");
        assert!(*d >= 0.0, "probe {i} returned a negative depth {d}");
    }
}

#[test]
fn zero_amplitude_is_a_no_op() {
    // The slider's own zero. It has to cost nothing and change nothing, since
    // that is the documented way to turn the effect off.
    let Some((device, queue)) = gpu() else { return };
    let n = 32;
    let probes = vec![Probe::depth([0.5, 0.0, 0.5], [1.0, 0.0, 1.0], 0.0, 16)];
    let got = run(&device, &queue, &trench(n), n, &probes);
    assert_eq!(got[0], 0.0);
}

#[test]
fn flat_relief_casts_no_self_shadow() {
    let Some((device, queue)) = gpu() else { return };
    let n = 32;
    let probes = vec![Probe::shadow([0.3, 0.0, 0.7], [1.0, 0.0, 0.5], 0.02, 16)];
    let got = run(&device, &queue, &flat(n), n, &probes);
    assert_eq!(got[0], 1.0, "a flat surface cannot shadow itself");
}

#[test]
fn self_shadowing_stays_within_range() {
    // It multiplies the sun term, so a value above 1 would brighten the surface
    // and a negative one would produce black.
    let Some((device, queue)) = gpu() else { return };
    let n = 32;
    let mut probes = Vec::new();
    for s in [0.0f32, 0.5, 2.0, 20.0] {
        probes.push(Probe::shadow([0.85, 0.0, 0.5], [s, 0.0, s * 0.25], 0.05, 24));
    }
    let got = run(&device, &queue, &trench(n), n, &probes);
    for (i, v) in got.iter().enumerate() {
        assert!(v.is_finite(), "probe {i} returned {v}");
        assert!((0.0..=1.0).contains(v), "probe {i} returned {v}, outside 0..1");
    }
}

/// Marginal cost of the march, printed rather than asserted.
///
/// `cargo test -p terra-render --test pom_gpu -- --ignored --nocapture`
///
/// This is a **compute proxy, not a frame time**: one march per invocation over
/// 1920x1080 invocations, which isolates the march but does not reproduce the
/// fragment shader's texture cache behaviour, the two layers a real pixel blends,
/// or anything else in the frame. What it is good for is the comparison that
/// decided the default -- a flat mid-grey height channel against real relief.
/// The flat row is the one that matters: it is what a set with no displacement
/// map costs, and the `prev_gap <= 0` early exit is meant to make it ~free.
#[test]
#[ignore]
fn march_cost_against_a_flat_height_channel() {
    let Some((device, queue)) = gpu() else { return };
    let n = 64u32;
    const PIXELS: u32 = 1920 * 1080;
    const RUNS: u32 = 20;

    let module = module(&device);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pom-bench"),
        layout: None,
        module: &module,
        entry_point: Some("cs_pom_bench"),
        compilation_options: Default::default(),
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("bench-samp"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench-out"),
        size: (PIXELS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    println!("marching {PIXELS} invocations, mean of {RUNS} dispatches (compute proxy)");
    for (label, alpha) in [("flat mid-grey", flat(n)), ("real relief", trench(n))] {
        let tex = height_texture(&device, &queue, n, &alpha);
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g2"),
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        for (amp, steps) in [(0.0f32, 16), (0.03, 16), (0.03, 32), (0.10, 32)] {
            let args: [f32; 8] = [amp, steps as f32, 0.0, 0.0, 1.5, 0.0, 0.6, 0.0];
            let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bench-args"),
                contents: bytemuck::cast_slice(&args),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("g0"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 3, resource: ubo.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                ],
            });

            let dispatch = |device: &wgpu::Device, queue: &wgpu::Queue| {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &g0, &[]);
                    pass.set_bind_group(2, &g2, &[]);
                    pass.dispatch_workgroups(PIXELS / 64, 1, 1);
                }
                queue.submit([enc.finish()]);
            };

            // Warm-up, so pipeline compilation and first-use allocation are not
            // counted as march time.
            dispatch(&device, &queue);
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

            let t0 = std::time::Instant::now();
            for _ in 0..RUNS {
                dispatch(&device, &queue);
            }
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let per = t0.elapsed() / RUNS;
            println!(
                "  {label:<14} amp {amp:>4.2} m, {steps:>2} steps: {:>7.3} ms",
                per.as_secs_f64() * 1000.0
            );
        }
    }
}

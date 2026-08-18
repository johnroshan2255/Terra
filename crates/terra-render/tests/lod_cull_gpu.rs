//! LOD binning in the scatter cull pass, on a real GPU.
//!
//! The pass writes its per-level counts into the indirect draw arguments and
//! nothing on the CPU ever reads them in normal operation, so the only way to
//! know the banding is right is to run the shipped shader and read them back.
//! What is asserted here is the set of properties a wrong band selection breaks:
//!
//! * the three counters **sum to** the number of instances inside the draw
//!   distance and the frustum -- never more, which would mean an instance was
//!   emitted twice and drawn twice, and never fewer, which would mean a gap
//! * each instance lands in the band its distance says, so the switch happens
//!   where the setting says it does
//! * `overflow` stays zero, because each output buffer is sized to the whole
//!   species and every instance goes to exactly one of them
//!
//! No window is opened.

use terra_render::mesh::Instance;
use terra_render::scatter::LOD_COUNT;
use wgpu::util::DeviceExt;

const INSTANCE_BYTES: usize = 32;
const DRAW_ARGS_WORDS: usize = 5;

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
        label: Some("lod-cull-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// Mirrors `CullParams` in `scatter.rs` and the `Cull` block in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Cull {
    planes: [[f32; 4]; 6],
    prev_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    cull_distance: f32,
    radius: f32,
    count: u32,
    capacity: u32,
    /// xy switch distances, z Hi-Z levels, w occlusion enabled.
    lod_bands: [f32; 4],
    hiz_size: [f32; 4],
}

/// What one cull run produced.
struct Outcome {
    /// Survivors per band.
    counts: [u32; LOD_COUNT],
    /// The x coordinate of every instance written into each band, so an instance
    /// can be traced to the band it landed in.
    xs: [Vec<f32>; LOD_COUNT],
    overflow: u32,
}

/// Run the real cull entry point over `instances`.
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    instances: &[Instance],
    cull_distance: f32,
    bands_m: [f32; 2],
) -> Outcome {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scatter-cull"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../assets/shaders/render/scatter_cull.wgsl").into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cull"),
        layout: None,
        module: &module,
        entry_point: Some("cull"),
        compilation_options: Default::default(),
        cache: None,
    });

    let n = instances.len();
    let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("src"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dst: Vec<wgpu::Buffer> = (0..LOD_COUNT)
        .map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dst{i}")),
                size: (n * INSTANCE_BYTES) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();

    // Frustum planes that accept everything, so this isolates the banding from the
    // frustum test that already has its own coverage.
    let planes = [[0.0f32, 1.0, 0.0, 1.0e9]; 6];
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&Cull {
            planes,
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            eye: [0.0, 0.0, 0.0, 0.0],
            cull_distance,
            radius: 1.0,
            count: n as u32,
            capacity: n as u32,
            // Occlusion off: these tests are about the distance banding, and the Hi-Z
            // test has its own coverage. `w = 0` is also what the first frame passes,
            // before any pyramid exists.
            lod_bands: [bands_m[0] * bands_m[0], bands_m[1] * bands_m[1], 1.0, 0.0],
            hiz_size: [256.0, 256.0, 0.0, 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Zeroed args, as the per-frame fill writes them.
    let args_words = vec![0u32; LOD_COUNT * DRAW_ARGS_WORDS];
    let args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("args"),
        contents: bytemuck::cast_slice(&args_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let overflow = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("overflow"),
        contents: &0u32.to_le_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let mut entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
    ];
    for (i, d) in dst.iter().enumerate() {
        entries
            .push(wgpu::BindGroupEntry { binding: 2 + i as u32, resource: d.as_entire_binding() });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: 2 + LOD_COUNT as u32,
        resource: args.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3 + LOD_COUNT as u32,
        resource: overflow.as_entire_binding(),
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cull"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    // A 1x1 pyramid, bound so the pipeline is complete. Never sampled, because occlusion
    // is switched off above.
    let hiz_tex = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test-hiz"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    // Texture only: the shader reads the pyramid with `textureLoad`, so it needs no
    // sampler and the derived layout has none.
    let hiz_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hiz"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&hiz_tex),
        }],
    });

    // Staging: args, then overflow, then each band's instance buffer.
    let args_bytes = (LOD_COUNT * DRAW_ARGS_WORDS * 4) as u64;
    let inst_bytes = (n * INSTANCE_BYTES) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: args_bytes + 4 + inst_bytes * LOD_COUNT as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_bind_group(1, &hiz_bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&args, 0, &staging, 0, args_bytes);
    enc.copy_buffer_to_buffer(&overflow, 0, &staging, args_bytes, 4);
    for (i, d) in dst.iter().enumerate() {
        enc.copy_buffer_to_buffer(
            d,
            0,
            &staging,
            args_bytes + 4 + inst_bytes * i as u64,
            inst_bytes,
        );
    }
    queue.submit([enc.finish()]);

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let view = staging.slice(..).get_mapped_range().expect("mapped range");

    let words: &[u32] = bytemuck::cast_slice(&view[..args_bytes as usize]);
    let mut counts = [0u32; LOD_COUNT];
    for (i, c) in counts.iter_mut().enumerate() {
        *c = words[i * DRAW_ARGS_WORDS + 1];
    }
    let overflow_n =
        u32::from_le_bytes(view[args_bytes as usize..args_bytes as usize + 4].try_into().unwrap());

    // Read each band's written records back as instances and pull out their x.
    let mut xs: [Vec<f32>; LOD_COUNT] = Default::default();
    for (i, slot) in xs.iter_mut().enumerate() {
        let base = (args_bytes + 4 + inst_bytes * i as u64) as usize;
        let bytes = &view[base..base + inst_bytes as usize];
        let recs: &[Instance] = bytemuck::cast_slice(bytes);
        *slot = recs[..counts[i] as usize].iter().map(|r| r.pos[0]).collect();
    }
    drop(view);
    staging.unmap();
    Outcome { counts, xs, overflow: overflow_n }
}

/// Instances laid out along +X at the given distances from the origin.
fn along_x(distances: &[f32]) -> Vec<Instance> {
    distances
        .iter()
        .map(|d| {
            Instance::from_parts(
                glam::Vec3::new(*d, 0.0, 0.0),
                glam::Quat::IDENTITY,
                1.0,
                glam::Vec3::ONE,
                0,
            )
        })
        .collect()
}

#[test]
fn each_instance_lands_in_the_band_its_distance_says() {
    let Some((device, queue)) = gpu() else { return };
    // Bands at 100 m and 400 m, drawn to 900 m.
    let near = [10.0f32, 50.0, 99.0];
    let mid = [101.0f32, 200.0, 399.0];
    let far = [401.0f32, 600.0, 899.0];
    let all: Vec<f32> = near.iter().chain(&mid).chain(&far).copied().collect();

    let out = run(&device, &queue, &along_x(&all), 900.0, [100.0, 400.0]);

    assert_eq!(out.counts, [3, 3, 3], "banding is wrong: {:?}", out.counts);
    for (band, want) in [(0usize, &near[..]), (1, &mid[..]), (2, &far[..])] {
        let mut got = out.xs[band].clone();
        got.sort_by(f32::total_cmp);
        let mut expect = want.to_vec();
        expect.sort_by(f32::total_cmp);
        assert_eq!(got, expect, "band {band} holds the wrong instances");
    }
    assert_eq!(out.overflow, 0);
}

#[test]
fn the_counters_sum_to_what_is_inside_the_draw_distance() {
    // The acceptance property. Summing to more means an instance was emitted into
    // two bands and is drawn twice; summing to less means a hole in the scatter.
    let Some((device, queue)) = gpu() else { return };
    let inside = [5.0f32, 80.0, 150.0, 300.0, 500.0, 880.0];
    let outside = [901.0f32, 1500.0, 5000.0];
    let all: Vec<f32> = inside.iter().chain(&outside).copied().collect();

    let out = run(&device, &queue, &along_x(&all), 900.0, [100.0, 400.0]);
    let drawn: u32 = out.counts.iter().sum();
    assert_eq!(
        drawn,
        inside.len() as u32,
        "counters {:?} sum to {drawn}, expected {}",
        out.counts,
        inside.len()
    );
    assert_eq!(out.overflow, 0);
}

#[test]
fn no_instance_is_written_into_two_bands() {
    // Traced by position rather than by count, because two errors could cancel in
    // a sum: an instance duplicated into one band and dropped from another.
    let Some((device, queue)) = gpu() else { return };
    let distances: Vec<f32> = (0..200).map(|i| 4.0 + i as f32 * 4.0).collect();
    let out = run(&device, &queue, &along_x(&distances), 900.0, [100.0, 400.0]);

    let mut seen: Vec<f32> = out.xs.iter().flatten().copied().collect();
    let before = seen.len();
    seen.sort_by(f32::total_cmp);
    seen.dedup();
    assert_eq!(before, seen.len(), "an instance was emitted into more than one band");

    let expected = distances.iter().filter(|d| **d <= 900.0).count();
    assert_eq!(seen.len(), expected, "every in-range instance must appear exactly once");
}

#[test]
fn the_switch_distance_is_where_the_setting_says() {
    // Walking the threshold, because an off-by-one in the comparison chain would
    // still pass a coarse banding test.
    let Some((device, queue)) = gpu() else { return };
    let out = run(&device, &queue, &along_x(&[99.9, 100.1]), 900.0, [100.0, 400.0]);
    assert_eq!(out.counts[0], 1, "the instance inside the band should be LOD 0");
    assert_eq!(out.counts[1], 1, "the instance past it should be LOD 1");
}

#[test]
fn collapsing_the_bands_puts_everything_in_the_coarsest_level() {
    // What a user dragging both sliders to the minimum should get: all far, none
    // near, and still no instance lost.
    let Some((device, queue)) = gpu() else { return };
    let distances: Vec<f32> = (1..50).map(|i| i as f32 * 10.0).collect();
    let out = run(&device, &queue, &along_x(&distances), 900.0, [0.0, 0.0]);
    assert_eq!(out.counts[0], 0);
    assert_eq!(out.counts[1], 0);
    assert_eq!(out.counts[2] as usize, distances.len());
    assert_eq!(out.overflow, 0);
}

#[test]
fn pushing_the_bands_past_the_draw_distance_puts_everything_in_lod0() {
    let Some((device, queue)) = gpu() else { return };
    let distances: Vec<f32> = (1..50).map(|i| i as f32 * 10.0).collect();
    let out = run(&device, &queue, &along_x(&distances), 900.0, [900.0, 900.0]);
    assert_eq!(out.counts[0] as usize, distances.len());
    assert_eq!(out.counts[1], 0);
    assert_eq!(out.counts[2], 0);
}

#[test]
fn height_above_the_camera_does_not_change_the_band() {
    // The cull uses horizontal distance -- a tree directly below a mountaintop
    // camera is not smaller on screen for being lower. The band has to agree with
    // the cull, or an instance is culled by one measure and shaded by another.
    let Some((device, queue)) = gpu() else { return };
    let instances: Vec<Instance> = [-800.0f32, 0.0, 800.0]
        .iter()
        .map(|y| {
            Instance::from_parts(
                glam::Vec3::new(50.0, *y, 0.0),
                glam::Quat::IDENTITY,
                1.0,
                glam::Vec3::ONE,
                0,
            )
        })
        .collect();
    let out = run(&device, &queue, &instances, 900.0, [100.0, 400.0]);
    assert_eq!(out.counts, [3, 0, 0], "vertical offset moved the band: {:?}", out.counts);
}

/// Counts at three camera positions, printed rather than asserted.
///
/// `cargo test -p terra-render --test lod_cull_gpu -- --ignored --nocapture`
///
/// The distribution is what decides whether the default switch distances are
/// sensible, and it is a property of the scene rather than of the code -- so this
/// reports rather than judges.
#[test]
#[ignore]
fn the_distribution_at_three_camera_positions() {
    let Some((device, queue)) = gpu() else { return };
    // A uniform grid, which is what scatter placement approximates: instance count
    // grows with the square of the radius, so most of them are far away.
    let mut instances = Vec::new();
    let step = 12.0f32;
    let half = 80;
    for gz in -half..=half {
        for gx in -half..=half {
            instances.push(Instance::from_parts(
                glam::Vec3::new(gx as f32 * step, 0.0, gz as f32 * step),
                glam::Quat::IDENTITY,
                1.0,
                glam::Vec3::ONE,
                0,
            ));
        }
    }
    println!("{} instances on a {step} m grid, draw distance 900 m", instances.len());
    println!("LOD triangles assumed 6000 / 1500 / 400");
    for (label, bands) in [
        ("default 120/350", [120.0f32, 350.0]),
        ("tight 60/200", [60.0, 200.0]),
        ("loose 300/600", [300.0, 600.0]),
    ] {
        let out = run(&device, &queue, &instances, 900.0, bands);
        let drawn: u32 = out.counts.iter().sum();
        let tris: u64 =
            out.counts.iter().zip([6000u64, 1500, 400]).map(|(n, t)| *n as u64 * t).sum();
        let flat = drawn as u64 * 6000;
        println!(
            "  {label:<16} counts {:?}  drawn {drawn:>6}  tris {:>10} vs {:>10} flat  ({:.0}% saved)  overflow {}",
            out.counts,
            tris,
            flat,
            100.0 - (tris as f64 / flat as f64) * 100.0,
            out.overflow
        );
    }
}

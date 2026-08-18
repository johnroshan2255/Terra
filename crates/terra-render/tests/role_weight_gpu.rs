//! Automatic role weights, evaluated by the shipped terrain shader.
//!
//! These exist because the bands were tuned for terrain the generator cannot produce,
//! and nothing caught it. `slope` in the shader is `1 - cos(theta)`, so the numbers do
//! not read as angles and a band starting at 0.26 looks reasonable until you work out
//! that it means 42 degrees -- and then measure a generated world and find its steepest
//! sample is 39.6.
//!
//! The consequence was silent and total: a palette whose only material held the rock
//! role rendered at **zero weight everywhere**, the no-claimant fallback forced slot 0
//! across the whole map, and it looked exactly like a texture deliberately applied to
//! everything. Three rounds of reading the code did not find it; measuring the
//! heightfield did.
//!
//! So this measures instead of reading. The bands are probed through the real shader at
//! slopes the pipeline actually produces, and the reachable maximum is pinned as a
//! constant so a future retune has to argue with a number.
//!
//! No window is opened.

use wgpu::util::DeviceExt;

/// Steepest slope a generated 4 km world contains, as `1 - cos(theta)`.
///
/// Measured on `global_height.r16` from a real project: median 11.5 degrees, p90 29.3,
/// p99 35.1, max **39.6** -- which is 0.229. Two things cap it and neither is going to
/// change: the heightfield samples every 4 m, so a 45 degree face is a 4 m step between
/// neighbours, and erosion exists to relax steep ground.
///
/// A little margin below the measured figure, because a different seed will differ.
const REACHABLE_MAX_SLOPE: f32 = 0.20;

/// Typical ground, for the "does grass actually cover the map" end.
const MEDIAN_SLOPE: f32 = 0.020;

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
        label: Some("role-weight-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A compute entry appended to the real terrain shader, evaluating the same band
/// expressions the fragment stage does.
///
/// The bands are read from the shader's own constants, so this cannot drift from what
/// ships -- which is the whole point. It is a transcription of the weight block rather
/// than a call into it, because that block reads storage buffers and interpolated inputs
/// a compute pass has none of; the constants are what matter and they are shared.
const PROBE: &str = r#"
struct Probe { slope: f32, dep: f32, wet: f32, altitude: f32 };
@group(0) @binding(0) var<storage, read> probes: array<Probe>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn cs_roles(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let p = probes[i];
    let base = i * 6u;

    out[base + 0u] = 0.45 + 0.35 * smoothstep(SOIL_FROM, SOIL_TO, p.slope);
    let gentle = 1.0 - smoothstep(GRASS_FROM, GRASS_TO, p.slope);
    out[base + 1u] = gentle * (0.75 + 0.45 * clamp(p.dep, 0.0, 1.0));
    out[base + 2u] =
        smoothstep(ROCK_FROM, ROCK_TO, p.slope) * 1.3 + clamp(-p.dep, 0.0, 1.0) * 0.8;
    out[base + 3u] =
        clamp(p.dep, 0.0, 1.0) * 0.55 + smoothstep(0.30, 0.72, p.wet) * 1.5;
    let cold = smoothstep(SNOW_ALTITUDE_FROM, SNOW_ALTITUDE_TO, p.altitude);
    out[base + 4u] = cold * (1.0 - smoothstep(SNOW_FROM, SNOW_TO, p.slope)) * 2.0;
    out[base + 5u] = 0.0;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe {
    slope: f32,
    dep: f32,
    wet: f32,
    altitude: f32,
}

const SOIL: usize = 0;
const GRASS: usize = 1;
const ROCK: usize = 2;
const SNOW: usize = 4;

/// Role weights for each probe, six per probe.
fn weights(device: &wgpu::Device, queue: &wgpu::Queue, probes: &[Probe]) -> Vec<[f32; 6]> {
    let src = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        include_str!("../../../assets/shaders/common/noise.wgsl"),
        include_str!("../../../assets/shaders/common/lighting.wgsl"),
        include_str!("../../../assets/shaders/common/cdlod.wgsl"),
        include_str!("../../../assets/shaders/common/grid.wgsl"),
        include_str!("../../../assets/shaders/common/brush.wgsl"),
        include_str!("../../../assets/shaders/render/terrain.wgsl"),
        PROBE,
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("role-probe"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("roles"),
        layout: None,
        module: &module,
        entry_point: Some("cs_roles"),
        compilation_options: Default::default(),
        cache: None,
    });

    let inp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probes"),
        contents: bytemuck::cast_slice(probes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (probes.len() * 6 * 4) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("weights"),
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
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("roles"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(probes.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    queue.submit([enc.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let view = read.slice(..).get_mapped_range().expect("mapped range");
    let flat: &[f32] = bytemuck::cast_slice(&view);
    flat.chunks_exact(6).map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect()
}

/// Neutral erosion and low ground, so only the slope varies.
fn at_slope(slope: f32) -> Probe {
    Probe { slope, dep: 0.0, wet: 0.0, altitude: 300.0 }
}

#[test]
fn rock_actually_appears_on_the_steepest_ground_a_world_contains() {
    // The bug, pinned. Rock's band began at 0.26 -- 42 degrees -- and a generated world
    // has nothing steeper than 0.229, so rock's weight was zero across the entire map and
    // a rock material could never be placed by the automatic system at all.
    let Some((device, queue)) = gpu() else { return };
    let w = weights(&device, &queue, &[at_slope(REACHABLE_MAX_SLOPE)]);
    assert!(
        w[0][ROCK] > 0.5,
        "at the steepest slope a world contains, rock weighs {:.3} -- it has to be the \
         dominant material there, not absent",
        w[0][ROCK]
    );
}

#[test]
fn rock_is_absent_on_gentle_ground() {
    // The other half: rock must not creep onto the flats, or every world is a quarry.
    let Some((device, queue)) = gpu() else { return };
    let w = weights(&device, &queue, &[at_slope(0.0), at_slope(MEDIAN_SLOPE)]);
    assert_eq!(w[0][ROCK], 0.0, "rock on dead flat ground");
    assert!(w[1][ROCK] < 0.05, "rock at the median slope: {:.3}", w[1][ROCK]);
}

#[test]
fn grass_gives_up_before_the_steepest_ground() {
    // Grass's band ended at 0.34 -- 49 degrees -- so it never fully faded and held most of
    // its weight even on the steepest face, leaving nothing for rock to break through.
    let Some((device, queue)) = gpu() else { return };
    let w = weights(&device, &queue, &[at_slope(0.0), at_slope(REACHABLE_MAX_SLOPE)]);
    assert!(w[0][GRASS] > 0.7, "grass should own flat ground, got {:.3}", w[0][GRASS]);
    assert!(
        w[1][GRASS] < 0.05,
        "grass still weighs {:.3} on the steepest ground a world has",
        w[1][GRASS]
    );
}

#[test]
fn rock_overtakes_grass_somewhere_inside_the_reachable_range() {
    // What makes a visible transition rather than one material everywhere: the two have to
    // actually cross, and the crossing has to happen on ground that exists.
    let Some((device, queue)) = gpu() else { return };
    let probes: Vec<Probe> =
        (0..=40).map(|i| at_slope(REACHABLE_MAX_SLOPE * i as f32 / 40.0)).collect();
    let w = weights(&device, &queue, &probes);

    let cross = w.iter().position(|r| r[ROCK] > r[GRASS]);
    let Some(i) = cross else {
        panic!("rock never overtakes grass at any slope a world contains");
    };
    let at = REACHABLE_MAX_SLOPE * i as f32 / 40.0;
    // Not right at the top, or only a handful of texels ever see rock.
    assert!(
        at < REACHABLE_MAX_SLOPE * 0.95,
        "rock only overtakes grass at {at:.3}, the very top of the range"
    );
}

#[test]
fn snow_can_reach_the_top_of_a_default_world() {
    // Snow's altitude band was 900-1350 m on a world whose default amplitude is 900 and
    // which measured an 864 m peak, so snow never appeared either.
    let Some((device, queue)) = gpu() else { return };
    let peak = Probe { slope: 0.02, dep: 0.0, wet: 0.0, altitude: 860.0 };
    let w = weights(&device, &queue, &[peak]);
    assert!(w[0][SNOW] > 0.5, "snow weighs {:.3} on an 860 m peak", w[0][SNOW]);
}

#[test]
fn snow_does_not_sit_on_a_steep_face() {
    let Some((device, queue)) = gpu() else { return };
    let steep = Probe { slope: REACHABLE_MAX_SLOPE, dep: 0.0, wet: 0.0, altitude: 860.0 };
    let w = weights(&device, &queue, &[steep]);
    assert!(w[0][SNOW] < 0.2, "snow clinging to a steep face at {:.3}", w[0][SNOW]);
}

#[test]
fn soil_never_vanishes_and_is_never_alone_in_owning_the_map() {
    // Soil is the base coat, so it must not drop to zero -- there has to be something for
    // the others to break through. But it must not dominate everywhere either, or every
    // palette looks like dirt.
    let Some((device, queue)) = gpu() else { return };
    let probes: Vec<Probe> =
        (0..=20).map(|i| at_slope(REACHABLE_MAX_SLOPE * i as f32 / 20.0)).collect();
    let w = weights(&device, &queue, &probes);
    for (i, r) in w.iter().enumerate() {
        assert!(r[SOIL] > 0.3, "soil fell to {:.3} at probe {i}", r[SOIL]);
    }
    // On flat ground grass leads; on the steepest, rock does. Soil is the filler.
    assert!(w[0][GRASS] > w[0][SOIL], "soil out-weighs grass on flat ground");
    assert!(w[20][ROCK] > w[20][SOIL], "soil out-weighs rock on the steepest ground");
}

#[test]
fn every_band_both_starts_and_finishes_inside_the_reachable_range() {
    // The class of bug, not just the instance. A band whose onset is past what the
    // pipeline produces is inert, and one whose end is past it never completes -- both
    // are invisible in the shader source and obvious the moment they are measured.
    let Some((device, queue)) = gpu() else { return };
    let lo = weights(&device, &queue, &[at_slope(0.0)])[0];
    let hi = weights(&device, &queue, &[at_slope(REACHABLE_MAX_SLOPE)])[0];
    for (name, i) in [("soil", SOIL), ("grass", GRASS), ("rock", ROCK)] {
        assert!(
            (hi[i] - lo[i]).abs() > 0.2,
            "{name} barely moves across the whole reachable slope range \
             ({:.3} -> {:.3}), so its band is tuned outside it",
            lo[i],
            hi[i]
        );
    }
}

/// The weight curve across the reachable range, printed rather than asserted.
///
/// `cargo test -p terra-render --test role_weight_gpu -- --ignored --nocapture`
#[test]
#[ignore]
fn the_weight_curve() {
    let Some((device, queue)) = gpu() else { return };
    println!(" deg   slope    soil   grass    rock    snow   winner");
    for i in 0..=20 {
        let s = REACHABLE_MAX_SLOPE * i as f32 / 20.0;
        let deg = (1.0 - s).clamp(-1.0, 1.0).acos().to_degrees();
        let p = Probe { slope: s, dep: 0.0, wet: 0.0, altitude: 860.0 };
        let w = weights(&device, &queue, &[p])[0];
        let names = ["soil", "grass", "rock", "gravel", "snow", "road"];
        let best = (0..5).max_by(|a, b| w[*a].total_cmp(&w[*b])).unwrap();
        println!(
            "{deg:5.1}  {s:.3}   {:.3}   {:.3}   {:.3}   {:.3}   {}",
            w[SOIL], w[GRASS], w[ROCK], w[SNOW], names[best]
        );
    }
}

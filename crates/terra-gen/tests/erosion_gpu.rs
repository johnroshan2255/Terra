//! End-to-end check of the generation pipeline against a real GPU.
//!
//! The solver is numerically delicate -- an oversized `dt` violates the CFL
//! condition and the flux field diverges into NaN within a few hundred
//! iterations. That failure is invisible in the editor until the terrain turns
//! into noise, so it is worth catching here.

use terra_project::params::{ErosionParams, RmfParams, ThermalParams};

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
        label: Some("erosion-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// Mean absolute gradient, in meters per meter.
fn mean_slope(h: &[f32], res: u32, cell: f32) -> f32 {
    let n = res as i32;
    let at = |x: i32, y: i32| h[(y.clamp(0, n - 1) * n + x.clamp(0, n - 1)) as usize];
    let mut sum = 0.0;
    for y in 0..n {
        for x in 0..n {
            let dx = (at(x + 1, y) - at(x - 1, y)) / (2.0 * cell);
            let dy = (at(x, y + 1) - at(x, y - 1)) / (2.0 * cell);
            sum += (dx * dx + dy * dy).sqrt();
        }
    }
    sum / (n * n) as f32
}

#[test]
fn full_pipeline_is_stable_and_carves_terrain() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    const RES: u32 = 256;
    const EXTENT: f32 = 4096.0;
    let cell = EXTENT / (RES - 1) as f32;

    let rmf = RmfParams::default();
    let base: Vec<f32> =
        terra_gen::heightfield::generate(RES, EXTENT, &rmf).iter().map(|h| h + 256.0).collect();

    let thermal = ThermalParams::default();
    let relaxed = terra_gen::thermal::run(&base, RES, cell, &thermal, 10);

    let params = ErosionParams::default();
    let sim = terra_gen::erosion::Erosion::new(&device, &queue, RES, cell, &params);
    let run = sim.run(&device, &queue, &relaxed, 400, |_| {});
    let eroded = run.height;

    assert_eq!(eroded.len(), relaxed.len(), "readback returned the wrong length");

    // Divergence check: this is what a CFL violation looks like.
    assert!(eroded.iter().all(|h| h.is_finite()), "solver produced NaN or infinity");
    let lo = eroded.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = eroded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (-500.0..20_000.0).contains(&lo) && (-500.0..20_000.0).contains(&hi),
        "heights left a plausible range: {lo}..{hi}"
    );

    // It must actually do something.
    let changed: f32 =
        eroded.iter().zip(&relaxed).map(|(a, b)| (a - b).abs()).sum::<f32>() / eroded.len() as f32;
    assert!(changed > 0.05, "erosion barely moved anything: {changed} m mean change");

    // And it must smooth on average: water carries material downhill, so mean
    // slope falls. A rise would mean the solver is amplifying noise instead.
    let before = mean_slope(&relaxed, RES, cell);
    let after = mean_slope(&eroded, RES, cell);
    assert!(after < before, "mean slope rose: {before} -> {after}");

    // The drainage network is a by-product of the same solve. If it comes back
    // flat, the material masks downstream have nothing to work with.
    let flow = terra_gen::erosion::Erosion::normalize_flow(&run.flow);
    assert_eq!(flow.len(), eroded.len());
    assert!(flow.iter().all(|f| (0.0..=1.0).contains(f)), "flow must normalize into 0..1");
    let channels = flow.iter().filter(|f| **f > 0.5).count();
    assert!(channels > 0, "no cell accumulated significant discharge");
    // Channels are a minority of any landscape. If most of the map lights up,
    // the metric is measuring rainfall rather than concentration.
    assert!(
        channels < flow.len() / 4,
        "too much of the map reads as channel: {channels} of {}",
        flow.len()
    );

    let dep = terra_gen::erosion::Erosion::deposition_map(&relaxed, &eroded);
    assert!(dep.iter().any(|d| *d > 0.55), "nothing was deposited");
    assert!(dep.iter().any(|d| *d < 0.45), "nothing was scoured");
}

#[test]
fn zero_iterations_is_a_no_op() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    const RES: u32 = 64;
    let input: Vec<f32> = (0..RES * RES).map(|i| 100.0 + (i % 17) as f32).collect();

    let sim =
        terra_gen::erosion::Erosion::new(&device, &queue, RES, 4.0, &ErosionParams::default());
    let out = sim.run(&device, &queue, &input, 0, |_| {}).height;

    assert_eq!(out, input, "running no iterations must return the input unchanged");
}

/// Timing at the resolutions the editor actually uses. Ignored by default --
/// run with `cargo test -p terra-gen --test erosion_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_at_production_resolution() {
    let Some((device, queue)) = gpu() else { return };

    for (res, extent) in [(1024u32, 4096.0f32), (2048, 8192.0)] {
        let cell = extent / (res - 1) as f32;
        let rmf = RmfParams::default();

        let t0 = std::time::Instant::now();
        let base = terra_gen::heightfield::generate(res, extent, &rmf);
        let t_rmf = t0.elapsed();

        let t1 = std::time::Instant::now();
        let relaxed = terra_gen::thermal::run(&base, res, cell, &ThermalParams::default(), 50);
        let t_thermal = t1.elapsed();

        let params = ErosionParams::default();
        let sim = terra_gen::erosion::Erosion::new(&device, &queue, res, cell, &params);
        let t2 = std::time::Instant::now();
        let out = sim.run(&device, &queue, &relaxed, params.iterations, |_| {}).height;
        let t_erode = t2.elapsed();

        assert!(out.iter().all(|h| h.is_finite()), "{res}: diverged");
        println!(
            "{res}x{res}: rmf {:?}, thermal(50) {:?}, erosion({} iters) {:?}",
            t_rmf, t_thermal, params.iterations, t_erode
        );
    }
}

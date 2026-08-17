//! The compute extraction, checked against the CPU reference on a real GPU.
//!
//! The two implementations are the same algorithm written twice, and they are
//! allowed to differ in exactly one way: the GPU assigns vertex indices with
//! an atomic counter, so its buffers come back in nondeterministic order.
//! Every comparison here is therefore on *geometry* -- which triangles exist,
//! where their corners are -- and never on buffer contents.
//!
//! Nothing in this file opens a window. `request_adapter` is called with no
//! surface, which is the same headless path `terra-gen`'s erosion tests use.

use glam::Vec3;
use std::collections::HashMap;
use terra_voxel::gpu::{Capacity, Extractor};
use terra_voxel::sdf;
use terra_voxel::surface_nets::{self, Mesh, SampleGrid};

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
        label: Some("voxel-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// Quantize a position so two independently-computed copies of the same vertex
/// hash together. A tenth of a millimetre is far below any geometric feature
/// and far above float reassociation noise.
fn key(p: Vec3) -> (i64, i64, i64) {
    let q = |v: f32| (v as f64 * 10_000.0).round() as i64;
    (q(p.x), q(p.y), q(p.z))
}

/// A triangle as its three corner positions, rotated so the smallest corner
/// comes first. Rotation preserves winding; sorting would destroy it, and
/// winding is exactly what decides which side of a cave wall you can see.
fn triangles(m: &Mesh) -> Vec<[(i64, i64, i64); 3]> {
    let mut out = Vec::with_capacity(m.triangle_count());
    for t in m.indices.chunks_exact(3) {
        let mut c = [
            key(m.positions[t[0] as usize]),
            key(m.positions[t[1] as usize]),
            key(m.positions[t[2] as usize]),
        ];
        let lowest = (0..3).min_by_key(|i| c[*i]).unwrap();
        c.rotate_left(lowest);
        out.push(c);
    }
    out.sort();
    out
}

type Key = (i64, i64, i64);

fn edge_use(m: &Mesh) -> HashMap<(Key, Key), u32> {
    let mut map = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (ka, kb) = (key(m.positions[a as usize]), key(m.positions[b as usize]));
            *map.entry((ka.min(kb), ka.max(kb))).or_insert(0) += 1;
        }
    }
    map
}

fn run_both(
    dim: u32,
    voxel: f32,
    origin: Vec3,
    f: impl Fn(Vec3) -> f32 + Sync + Copy,
) -> Option<(Mesh, Mesh)> {
    let (device, queue) = gpu()?;
    let grid = SampleGrid::sample(dim, voxel, origin, f);
    let cpu = surface_nets::extract(&grid);

    let extractor = Extractor::new(&device);
    let buffers = extractor.buffers(&device, dim, Capacity::worst_case(dim));
    let (gpu_mesh, stats) = extractor.readback(&device, &queue, &buffers, &grid);
    assert!(!stats.overflowed, "worst-case capacity should never overflow");
    Some((cpu, gpu_mesh))
}

#[test]
fn gpu_matches_cpu_reference_on_a_sphere() {
    let Some((cpu, gpu)) = run_both(24, 4.0 / 24.0, Vec3::splat(-2.0), |p| p.length() - 1.0) else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    assert!(!cpu.is_empty(), "the reference produced nothing; the test proves nothing");
    assert_eq!(
        gpu.vertex_count(),
        cpu.vertex_count(),
        "vertex counts differ: gpu {} vs cpu {}",
        gpu.vertex_count(),
        cpu.vertex_count()
    );
    assert_eq!(gpu.triangle_count(), cpu.triangle_count(), "triangle counts differ");
    assert_eq!(triangles(&gpu), triangles(&cpu), "the two meshes are not the same surface");
}

#[test]
fn gpu_normals_match_the_reference() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let dim = 20;
    let grid = SampleGrid::sample(dim, 0.2, Vec3::splat(-2.0), |p| p.length() - 1.0);
    let cpu = surface_nets::extract(&grid);

    let extractor = Extractor::new(&device);
    let buffers = extractor.buffers(&device, dim, Capacity::worst_case(dim));
    let (gpu_mesh, _) = extractor.readback(&device, &queue, &buffers, &grid);

    // Match vertices by position, then compare the normal at each. A sign flip
    // here would light the whole cave system inside out.
    let by_pos: HashMap<_, _> =
        cpu.positions.iter().zip(&cpu.normals).map(|(p, n)| (key(*p), *n)).collect();
    let mut compared = 0;
    for (p, n) in gpu_mesh.positions.iter().zip(&gpu_mesh.normals) {
        let Some(want) = by_pos.get(&key(*p)) else { continue };
        assert!(n.dot(*want) > 0.999, "normal mismatch at {p}: gpu {n} vs cpu {want}");
        compared += 1;
    }
    assert!(compared > 100, "only {compared} normals compared; test is too weak");
}

#[test]
fn gpu_mesh_of_a_carved_cave_is_watertight() {
    // The load-bearing case: a lump of rock with a tunnel bored through it, so
    // the outer surface, the bore wall and the two mouths all have to be
    // triangulated consistently by the shader.
    let Some((cpu, gpu)) = run_both(32, 0.125, Vec3::splat(-2.0), |p| {
        let rock = p.length() - 1.5;
        let bore = sdf::capsule(p, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), 0.4);
        sdf::subtract(rock, bore)
    }) else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    assert!(!gpu.is_empty());
    assert_eq!(triangles(&gpu), triangles(&cpu));
    for (edge, n) in edge_use(&gpu) {
        assert_eq!(n, 2, "GPU mesh leaks at edge {edge:?}: used by {n} triangles");
    }
}

#[test]
fn gpu_triangles_wind_outward() {
    let Some((_, gpu)) = run_both(20, 0.2, Vec3::splat(-2.0), |p| p.length() - 1.0) else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut checked = 0;
    for t in gpu.indices.chunks_exact(3) {
        let (a, b, c) = (
            gpu.positions[t[0] as usize],
            gpu.positions[t[1] as usize],
            gpu.positions[t[2] as usize],
        );
        let n = (b - a).cross(c - a);
        if n.length_squared() < 1e-12 {
            continue;
        }
        assert!(n.dot((a + b + c) / 3.0) > 0.0, "inward-facing GPU triangle at {a}");
        checked += 1;
    }
    assert!(checked > 100, "only {checked} triangles checked");
}

#[test]
fn a_uniform_block_produces_an_empty_draw() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let dim = 8;
    let grid = SampleGrid::sample(dim, 1.0, Vec3::ZERO, |_| -5.0);
    let extractor = Extractor::new(&device);
    let buffers = extractor.buffers(&device, dim, Capacity::worst_case(dim));
    let (mesh, stats) = extractor.readback(&device, &queue, &buffers, &grid);

    assert!(mesh.is_empty(), "solid rock must extract to nothing");
    assert_eq!(stats.index_count, 0);
    assert!(!stats.overflowed);
}

#[test]
fn overflow_is_reported_rather_than_silently_truncated() {
    // A chunk given less room than its surface needs must say so. Losing
    // geometry quietly reads as a hole in the world with no way to trace it.
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let dim = 24;
    let grid = SampleGrid::sample(dim, 4.0 / 24.0, Vec3::splat(-2.0), |p| p.length() - 1.0);

    let extractor = Extractor::new(&device);
    let starved = Capacity { max_vertices: 16, max_indices: 24 };
    let buffers = extractor.buffers(&device, dim, starved);
    let (mesh, stats) = extractor.readback(&device, &queue, &buffers, &grid);

    assert!(stats.overflowed, "a starved chunk must set the overflow flag");
    // Whatever did fit must still be within the allocation, not past it.
    assert!(mesh.vertex_count() <= starved.max_vertices as usize);
    assert!(mesh.indices.len() <= starved.max_indices as usize);
}

#[test]
fn typical_capacity_is_enough_for_a_real_surface() {
    // `Capacity::typical` is a heuristic, and a heuristic that turns out to be
    // too small in ordinary use is a bug. A sphere filling the chunk is about
    // as much surface as a chunk ever holds.
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let dim = terra_voxel::CHUNK_DIM;
    let grid = SampleGrid::sample(dim, 1.0, Vec3::splat(-(dim as f32) / 2.0), |p| {
        p.length() - dim as f32 * 0.45
    });
    let extractor = Extractor::new(&device);
    let buffers = extractor.buffers(&device, dim, Capacity::typical(dim));
    let (mesh, stats) = extractor.readback(&device, &queue, &buffers, &grid);

    assert!(!stats.overflowed, "typical capacity overflowed on an ordinary sphere");
    assert!(!mesh.is_empty());
}

#[test]
fn indirect_args_describe_the_mesh_that_was_written() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let dim = 16;
    let grid = SampleGrid::sample(dim, 0.25, Vec3::splat(-2.0), |p| p.length() - 1.0);
    let extractor = Extractor::new(&device);
    let buffers = extractor.buffers(&device, dim, Capacity::worst_case(dim));
    let (mesh, stats) = extractor.readback(&device, &queue, &buffers, &grid);
    assert!(!mesh.is_empty());

    // Read the indirect buffer the shader filled in. This is the block the GPU
    // would consume in `draw_indexed_indirect`, so a wrong index_count here
    // draws a partial cave with no error anywhere.
    let size = std::mem::size_of::<terra_voxel::DrawIndexedIndirect>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("args-read"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&buffers.args, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().unwrap().unwrap();

    let args: terra_voxel::DrawIndexedIndirect = {
        let view = staging.slice(..).get_mapped_range().expect("mapped range");
        *bytemuck::from_bytes(&view)
    };
    staging.unmap();

    assert_eq!(args.index_count, stats.index_count, "args disagree with the counter");
    assert_eq!(args.index_count as usize, mesh.indices.len());
    assert_eq!(args.instance_count, 1, "a non-empty chunk must draw one instance");
    assert_eq!(args.first_index, 0);
    assert_eq!(args.base_vertex, 0);
}

/// Extraction cost at the chunk size the renderer actually uses. Ignored by
/// default -- run with
/// `cargo test -p terra-voxel --test extract_gpu -- --ignored --nocapture`.
///
/// The number that matters is per-chunk cost against the re-extraction budget:
/// a sculpt stroke dirties a handful of chunks per frame, and at 60 Hz the
/// whole frame is 16.7 ms with terrain, shadows and post already in it.
#[test]
#[ignore]
fn extraction_cost_at_chunk_resolution() {
    let Some((device, queue)) = gpu() else { return };
    let extractor = Extractor::new(&device);

    for dim in [16u32, 32, 48] {
        let voxel = 1.0f32;
        let origin = Vec3::splat(-(dim as f32) / 2.0);
        // A sphere filling the chunk: about as much surface as a chunk holds.
        let field = |p: Vec3| p.length() - dim as f32 * 0.4;

        let t0 = std::time::Instant::now();
        let grid = SampleGrid::sample(dim, voxel, origin, field);
        let t_sample = t0.elapsed();

        let t1 = std::time::Instant::now();
        let cpu = surface_nets::extract(&grid);
        let t_cpu = t1.elapsed();

        let buffers = extractor.buffers(&device, dim, Capacity::worst_case(dim));
        // One warm-up so shader compilation and first-use allocation are not
        // counted as extraction time.
        extractor.dispatch(&device, &queue, &buffers, &grid);
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        const RUNS: u32 = 50;
        let t2 = std::time::Instant::now();
        for _ in 0..RUNS {
            extractor.dispatch(&device, &queue, &buffers, &grid);
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let t_gpu = t2.elapsed() / RUNS;

        println!(
            "dim {dim:>2}: sample {:>8.3} ms | cpu extract {:>8.3} ms | gpu extract {:>8.3} ms \
             | {} verts, {} tris",
            t_sample.as_secs_f64() * 1000.0,
            t_cpu.as_secs_f64() * 1000.0,
            t_gpu.as_secs_f64() * 1000.0,
            cpu.vertex_count(),
            cpu.triangle_count(),
        );
    }
}

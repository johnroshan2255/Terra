//! The brush cursor ring, measured on the GPU across every brush size the editor
//! offers and every camera distance it allows.
//!
//! This exists because the ring silently vanished. Its width was set in world
//! metres, `radius * 0.02 + 0.5`, so at the editor's default camera height an 8 m
//! brush -- the smallest `[` gives you -- drew a ring a third of a pixel wide.
//! Nothing rendered, and nothing in the codebase noticed, because the shader was
//! perfectly correct and simply drew something too small to see.
//!
//! So the invariant under test is not "the maths is right", it is **the ring is
//! actually visible**, at every combination of the two things that made it
//! disappear: brush radius and metres per pixel.
//!
//! `brush_ring_weights` takes the screen footprint as a parameter rather than
//! calling `fwidth` itself, which is what lets this be a compute pass over a grid of
//! cases instead of one render per case.
//!
//! No window is opened.

/// Brush radii the editor can produce. `[` and `]` scale by 0.85 and 1.18 between
/// hard limits of 8 m and 800 m, so these are the ends and a spread between.
const RADII: [f32; 7] = [8.0, 14.0, 30.0, 60.0, 120.0, 350.0, 800.0];

/// Metres per pixel, from a camera almost on the ground to one 40 km out.
const FOOTPRINTS: [f32; 6] = [0.05, 0.5, 1.94, 6.0, 20.0, 90.0];

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
        label: Some("brush-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// Walks a ray of pixels straight outward through the ring, one thread a pixel, and
/// reports the core and halo weight at each.
const HARNESS: &str = r#"
struct Args {
    // x = brush radius in metres, y = metres per pixel, z = sample count.
    params: vec4f,
};
@group(0) @binding(0) var<uniform> args: Args;
@group(0) @binding(1) var<storage, read_write> out: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let n = u32(args.params.z);
    if (gid.x >= n) {
        return;
    }
    let radius = args.params.x;
    let mpp = args.params.y;
    // Centre the walk on the ring and step one pixel at a time, so the index maps
    // directly to "pixels from the ring edge".
    let dist = radius + (f32(gid.x) - f32(n) * 0.5) * mpp;
    let w = brush_ring_weights(dist, radius, mpp);
    out[gid.x] = vec4f(w, dist);
}
"#;

/// `(core, halo, fill)` for a pixel-by-pixel walk across the ring.
fn walk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    radius_m: f32,
    metres_per_pixel: f32,
) -> Vec<[f32; 4]> {
    use wgpu::util::DeviceExt;

    const N: u32 = 256;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("brush-harness"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{}", include_str!("../../../assets/shaders/common/brush.wgsl"), HARNESS)
                .into(),
        ),
    });

    let args = [radius_m, metres_per_pixel, N as f32, 0.0];
    let args_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("args"),
        contents: bytemuck::cast_slice(&args),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bytes = (N as usize * 16) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
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

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("brush-harness"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: args_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(N.div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read, 0, bytes);
    queue.submit([enc.finish()]);

    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let out = {
        let data = read.slice(..).get_mapped_range().expect("mapped range");
        bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
    };
    read.unmap();
    out
}

/// Pixels whose core weight is at least half -- the ring's visible thickness.
fn solid_pixels(w: &[[f32; 4]]) -> usize {
    w.iter().filter(|p| p[0] >= 0.5).count()
}

#[test]
fn the_ring_is_visible_at_every_brush_size_and_distance() {
    // The regression itself. Under the old metre-based width, the 8 m entry here
    // produced a third of a pixel and this test would have failed on it.
    let Some((device, queue)) = gpu() else { return };
    for r in RADII {
        for mpp in FOOTPRINTS {
            let w = walk(&device, &queue, r, mpp);
            let solid = solid_pixels(&w);
            assert!(
                solid >= 2,
                "radius {r} m at {mpp} m/px draws only {solid} solid pixels -- invisible"
            );
            let peak = w.iter().map(|p| p[0]).fold(0.0f32, f32::max);
            assert!(peak > 0.99, "radius {r} m at {mpp} m/px peaks at {peak}, never fully opaque");
        }
    }
}

#[test]
fn the_ring_is_the_same_thickness_however_far_away_it_is() {
    // The property that makes it visible everywhere: constant weight on screen. A
    // ring that thins with distance is the old bug returning, and one that thickens
    // swallows the terrain under a large brush.
    let Some((device, queue)) = gpu() else { return };
    let mut widths = Vec::new();
    for r in RADII {
        for mpp in FOOTPRINTS {
            widths.push(solid_pixels(&walk(&device, &queue, r, mpp)));
        }
    }
    let lo = *widths.iter().min().unwrap();
    let hi = *widths.iter().max().unwrap();
    // Integer pixel counting on a smoothstep ramp gives a pixel of slack either way.
    assert!(hi - lo <= 1, "ring thickness ranges from {lo} to {hi} pixels across cases");
    assert!((3..=5).contains(&hi), "a {hi}-pixel ring is not the intended weight");
}

#[test]
fn the_dark_halo_frames_the_ring_without_dimming_it() {
    // The halo is what makes the ring readable over pale ground and over the default
    // world grid's bright lines. It has to sit *outside* the core, not on top of it.
    let Some((device, queue)) = gpu() else { return };
    let w = walk(&device, &queue, 60.0, 1.94);
    for p in &w {
        let (core, halo) = (p[0], p[1]);
        assert!(
            !(core > 0.5 && halo > 0.1),
            "halo {halo} overlaps a core of {core} and would dim the ring"
        );
    }
    let halo_px = w.iter().filter(|p| p[1] >= 0.4).count();
    assert!(halo_px >= 2, "the halo is only {halo_px} pixels wide, so it frames nothing");
}

#[test]
fn the_ring_sits_on_the_brush_radius() {
    // A ring drawn at the wrong radius is worse than none: the stroke would land
    // somewhere other than where the cursor says. Checks the peak is at the radius,
    // not merely that a peak exists.
    let Some((device, queue)) = gpu() else { return };
    for r in [8.0f32, 120.0, 800.0] {
        let mpp = 1.94;
        let w = walk(&device, &queue, r, mpp);
        let peak = w.iter().max_by(|a, b| a[0].partial_cmp(&b[0]).unwrap()).expect("a peak");
        // `peak[3]` is the world distance that sample was taken at.
        assert!(
            (peak[3] - r).abs() <= mpp * 1.5,
            "radius {r}: the brightest pixel is at {} m",
            peak[3]
        );
    }
}

#[test]
fn the_interior_fill_marks_the_disc_and_scales_with_the_brush() {
    // Unlike the ring, the wash *should* be in metres: it shows the area the brush
    // covers, so it has to grow with the brush.
    let Some((device, queue)) = gpu() else { return };
    let w = walk(&device, &queue, 120.0, 1.94);
    let inside = w.iter().filter(|p| p[3] < 60.0).collect::<Vec<_>>();
    let outside = w.iter().filter(|p| p[3] > 180.0).collect::<Vec<_>>();
    assert!(!inside.is_empty() && !outside.is_empty(), "the walk did not span the disc");
    assert!(inside.iter().all(|p| p[2] > 0.4), "the disc interior has no fill");
    assert!(outside.iter().all(|p| p[2] == 0.0), "fill leaks outside the brush");
}

#[test]
fn a_degenerate_footprint_cannot_flood_the_screen() {
    // A perfectly flat, axis-aligned surface can hand `fwidth` a zero, and dividing
    // by it would make the ring cover everything -- a full-screen yellow wash rather
    // than a missing ring. Reachable: the default terrain is exactly flat.
    let Some((device, queue)) = gpu() else { return };
    let w = walk(&device, &queue, 120.0, 0.0);
    assert!(
        w.iter().all(|p| p.iter().all(|v| v.is_finite())),
        "a zero footprint produced a non-finite weight"
    );
    // With a zero step every sample sits exactly on the radius, so a solid core
    // there is correct -- what must not happen is a NaN, which the guard prevents.
    assert!(w.iter().all(|p| p[0] >= 0.0 && p[0] <= 1.0), "core weight left 0..1");
}

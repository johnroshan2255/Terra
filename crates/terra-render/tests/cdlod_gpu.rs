//! CDLOD vertex placement, run on a real GPU and compared against the CPU.
//!
//! `cdlod.rs` is thoroughly tested on the CPU, but the CPU is not what draws the
//! terrain. The vertex shader is, and between the two sits a Rust struct that has to
//! match a WGSL struct byte for byte and arithmetic that has to match line for line.
//! Neither is checked by `cargo build`, and both fail silently: a field read at the
//! wrong offset gives patches at nonsense positions, and a morph that disagrees with
//! the CPU's leaves hairline cracks that flicker sky through the ground as the
//! camera moves.
//!
//! So the CPU is used as the oracle, the way `terra-voxel`'s Surface Nets tests do:
//! the shared `common/cdlod.wgsl` chunk is compiled into a compute shader, fed the
//! real `Patch` buffer that `Cdlod::select` produced, and every vertex it computes is
//! required to equal `Patch::vertex_xz` exactly.
//!
//! No window is opened -- the device is requested with no surface.

use glam::{Vec2, Vec3};
use terra_render::cdlod::{Cdlod, PATCH_QUADS, PATCH_VERTS, Patch, vertical_gap};

const EXTENT: f32 = 4000.0;
const TARGET_SPACING: f32 = 0.5;

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
        label: Some("cdlod-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A compute pass over the shared placement function.
///
/// The chunk under test is prepended unmodified, exactly as `Terrain::new` prepends
/// it -- rewriting it for the test would be testing a different shader from the one
/// that ships.
const HARNESS: &str = r#"
struct Args {
    // xz = eye world XZ, z = vertical gap to the height slab, w = patch count.
    eye: vec4f,
};
@group(0) @binding(0) var<uniform> args: Args;
@group(0) @binding(1) var<storage, read> patches: array<CdlodPatch>;
@group(0) @binding(2) var<storage, read_write> out_xz: array<vec2f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let verts = PATCH_VERTS * PATCH_VERTS;
    let total = u32(args.eye.w) * verts;
    if (gid.x >= total) {
        return;
    }
    // Not `patch`: that is a reserved WGSL keyword and the module fails to parse.
    let p = patches[gid.x / verts];
    let vi = gid.x % verts;
    out_xz[gid.x] = cdlod_vertex_xz(p, vi, PATCH_QUADS, args.eye.xyz);
}
"#;

/// Every vertex of every selected patch, as the GPU places it.
fn gpu_vertices(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    patches: &[Patch],
    eye_xz: Vec2,
    gap: f32,
) -> Vec<Vec2> {
    use wgpu::util::DeviceExt;

    // `PATCH_QUADS` is a Rust constant, so it is substituted rather than passed:
    // if the shader read a different patch size from the CPU every vertex would be
    // wrong, and a mismatch should be impossible to introduce rather than caught.
    let source = format!(
        "{}\n{}",
        include_str!("../../../assets/shaders/common/cdlod.wgsl"),
        HARNESS
            .replace("PATCH_VERTS", &format!("{}u", PATCH_VERTS))
            .replace("PATCH_QUADS", &format!("{}u", PATCH_QUADS)),
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cdlod-harness"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let verts_per_patch = (PATCH_VERTS * PATCH_VERTS) as usize;
    let total = patches.len() * verts_per_patch;
    let out_bytes = (total * std::mem::size_of::<[f32; 2]>()) as u64;

    let args = [eye_xz.x, eye_xz.y, gap, patches.len() as f32];
    let args_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("args"),
        contents: bytemuck::bytes_of(&args),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // The real `Patch` slice, cast straight to bytes. If the WGSL struct disagreed
    // about a single offset this is where it would show.
    let patch_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("patches"),
        contents: bytemuck::cast_slice(patches),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cdlod-harness"),
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
            wgpu::BindGroupEntry { binding: 1, resource: patch_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
    queue.submit([enc.finish()]);

    read_buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let out: Vec<Vec2> = {
        let data = read_buf.slice(..).get_mapped_range().expect("mapped range");
        bytemuck::cast_slice::<u8, [f32; 2]>(&data).iter().map(|v| Vec2::from(*v)).collect()
    };
    read_buf.unmap();
    out
}

#[test]
fn the_shader_places_every_vertex_exactly_where_the_cpu_does() {
    let Some((device, queue)) = gpu() else { return };

    // Several cameras, because the morph is distance-driven: one position exercises
    // one slice of the factor's range, and the disagreements worth finding are at
    // the ends (fully morphed, not morphed at all) and in the middle.
    //
    // The flag is whether that camera should see any morphing at all. The last one
    // should not: 6 km up, every patch is the root level, whose band starts at
    // 7776 m, so nothing has begun morphing. It stays in the list because exact
    // agreement still has to hold there -- an unmorphed patch placed wrongly is just
    // as broken.
    for (eye, expect_morph) in [
        (Vec3::new(0.0, 30.0, 0.0), true),
        (Vec3::new(-1750.0, 12.0, 1400.0), true),
        (Vec3::new(620.0, 900.0, -80.0), true),
        (Vec3::new(0.0, 6000.0, 0.0), false),
    ] {
        let range = (0.0, 120.0);
        let mut lod = Cdlod::new(EXTENT, TARGET_SPACING);
        let patches = lod.select(eye, range, EXTENT).to_vec();
        assert!(!patches.is_empty());

        let eye_xz = Vec2::new(eye.x, eye.z);
        let gap = vertical_gap(eye.y, range);
        let got = gpu_vertices(&device, &queue, &patches, eye_xz, gap);

        let verts_per_patch = (PATCH_VERTS * PATCH_VERTS) as usize;
        assert_eq!(got.len(), patches.len() * verts_per_patch);

        let mut worst = 0.0f32;
        let mut morphed = 0usize;
        for (pi, p) in patches.iter().enumerate() {
            for gz in 0..PATCH_VERTS {
                for gx in 0..PATCH_VERTS {
                    let want = p.vertex_xz(gx, gz, eye_xz, gap);
                    let idx = pi * verts_per_patch + (gz * PATCH_VERTS + gx) as usize;
                    let diff = (got[idx] - want).abs().max_element();
                    worst = worst.max(diff);
                    if p.grid_xz(gx, gz) != want {
                        morphed += 1;
                    }
                }
            }
        }
        // Not approximate equality with a generous tolerance: both sides run the
        // same operations on the same f32 inputs, so they should agree to the last
        // bit. A tolerance here would hide exactly the kind of reordered arithmetic
        // that opens a crack.
        assert!(
            worst <= f32::EPSILON * 4096.0,
            "eye {eye}: worst disagreement {worst} m between shader and CPU"
        );
        // And confirm the comparison had teeth: if nothing morphed, this test just
        // checked that two grids agree about a grid.
        if expect_morph {
            assert!(
                morphed > 1000,
                "eye {eye}: only {morphed} vertices morphed, so the morph went untested"
            );
        } else {
            assert_eq!(morphed, 0, "eye {eye}: nothing should be morphing this far out");
        }
    }
}

#[test]
fn the_shader_reads_the_patch_struct_at_the_right_offsets() {
    // Isolates the layout from the arithmetic. A single field read at the wrong
    // offset -- the mistake that once rendered the whole terrain red through
    // `LayerParams` -- puts patches at nonsense positions, and the failure above
    // would not say which of the two causes it was.
    //
    let Some((device, queue)) = gpu() else { return };
    // Unreachably distant morph band, so every vertex has a factor of zero and the
    // output is `origin + g * size / quads` and nothing else.
    let p = Patch::new(Vec2::new(-1234.5, 678.25), 512.0, 3, 1.0e9, 2.0e9);
    let got = gpu_vertices(&device, &queue, &[p], Vec2::ZERO, 0.0);

    let step = 512.0 / PATCH_QUADS as f32;
    assert_eq!(got[0], Vec2::new(-1234.5, 678.25), "origin misread");
    assert_eq!(got[1], Vec2::new(-1234.5 + step, 678.25), "size or origin.x misread");
    assert_eq!(
        got[PATCH_VERTS as usize],
        Vec2::new(-1234.5, 678.25 + step),
        "origin.z or the row stride is wrong"
    );
    let last = got[(PATCH_VERTS * PATCH_VERTS - 1) as usize];
    assert_eq!(last, Vec2::new(-1234.5 + 512.0, 678.25 + 512.0), "the patch is the wrong size");
}

#[test]
fn the_shader_fully_morphs_a_patch_that_is_far_away() {
    // The other end of the factor's range, and the one the crack-free property
    // depends on: past `morph_end` every odd vertex must sit exactly on its even
    // neighbour, which is a parent-level vertex.
    let Some((device, queue)) = gpu() else { return };
    // A 1 m morph band, and the eye 100 km away below, so every vertex is past it.
    let p = Patch::new(Vec2::ZERO, 320.0, 0, 0.0, 1.0);
    let got = gpu_vertices(&device, &queue, &[p], Vec2::new(-100_000.0, -100_000.0), 0.0);

    let step = 320.0 / PATCH_QUADS as f32;
    let mut collapsed = 0;
    for gz in 0..PATCH_VERTS {
        for gx in 0..PATCH_VERTS {
            let v = got[(gz * PATCH_VERTS + gx) as usize];
            let even = Vec2::new((gx - gx % 2) as f32, (gz - gz % 2) as f32) * step;
            assert_eq!(v, even, "vertex ({gx}, {gz}) did not collapse onto the parent grid");
            if gx % 2 == 1 || gz % 2 == 1 {
                collapsed += 1;
            }
        }
    }
    assert!(collapsed > 0);
    // Fully morphed, the patch has only the parent level's vertices left.
    let distinct: std::collections::HashSet<(i64, i64)> =
        got.iter().map(|v| ((v.x * 256.0).round() as i64, (v.y * 256.0).round() as i64)).collect();
    let want = ((PATCH_QUADS / 2 + 1) * (PATCH_QUADS / 2 + 1)) as usize;
    assert_eq!(distinct.len(), want, "expected {want} distinct positions after the morph");
}

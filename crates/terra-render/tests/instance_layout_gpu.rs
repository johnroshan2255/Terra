//! The 32-byte instance record, checked against the shaders that read it.
//!
//! This is the layout class of bug the material `tint` offset already cost this
//! project once: the Rust side and the WGSL side each looked correct, the sizes
//! agreed, and the shader read from the wrong byte anyway. A size assertion did
//! not catch it and neither would one here, so this reads the fields back out
//! *through the shader* and compares them to what Rust wrote.
//!
//! Two shaders touch the record and neither had a compile test before:
//!
//! * `mesh.wgsl` reads it as instance-step vertex attributes, by byte offset
//! * `scatter_cull.wgsl` reads it as two `vec4u` and bitcasts, by byte offset
//!
//! No window is opened.

use glam::{Quat, Vec3};
use terra_render::mesh::Instance;
use wgpu::util::DeviceExt;

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
        label: Some("instance-layout-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

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
fn the_mesh_shader_compiles() {
    // Composed as `MeshRenderer::new` composes it. It reconstructs the model
    // matrix from a quaternion now, so a typo here is a silent wrong transform.
    let Some((device, _queue)) = gpu() else { return };
    assert_compiles(
        &device,
        "mesh",
        format!(
            "{}\n{}",
            include_str!("../../../assets/shaders/common/lighting.wgsl"),
            include_str!("../../../assets/shaders/render/mesh.wgsl"),
        ),
    );
}

#[test]
fn the_scatter_cull_shader_compiles() {
    let Some((device, _queue)) = gpu() else { return };
    assert_compiles(
        &device,
        "scatter_cull",
        include_str!("../../../assets/shaders/render/scatter_cull.wgsl").to_string(),
    );
}

/// A probe entry appended to the real cull shader, so the decode under test is
/// the shipped one rather than a copy that can drift.
const PROBE: &str = r#"
@group(0) @binding(5) var<storage, read_write> probe_out: array<f32>;

@compute @workgroup_size(1)
fn cs_probe(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let inst = src[i];
    let p = inst_pos(inst);
    probe_out[i * 4u + 0u] = p.x;
    probe_out[i * 4u + 1u] = p.y;
    probe_out[i * 4u + 2u] = p.z;
    probe_out[i * 4u + 3u] = inst_scale(inst);
}
"#;

/// Read `pos` and `scale` back out of each record, through the cull shader.
fn decode_on_gpu(device: &wgpu::Device, queue: &wgpu::Queue, instances: &[Instance]) -> Vec<f32> {
    let src =
        format!("{}\n{}", include_str!("../../../assets/shaders/render/scatter_cull.wgsl"), PROBE);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cull-probe"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cull-probe"),
        layout: None,
        module: &module,
        entry_point: Some("cs_probe"),
        compilation_options: Default::default(),
        cache: None,
    });

    let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instances"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (instances.len() * 4 * 4) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
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

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(instances.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    queue.submit([enc.finish()]);

    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let data = read.slice(..).get_mapped_range().expect("mapped range");
    bytemuck::cast_slice::<u8, f32>(&data).to_vec()
}

#[test]
fn the_cull_shader_reads_the_position_rust_wrote() {
    let Some((device, queue)) = gpu() else { return };
    // Positions spanning a 16 km world, because the whole point of keeping the
    // position at full f32 is that a distant instance does not quantize.
    let cases = [
        (Vec3::new(0.0, 0.0, 0.0), 1.0f32),
        (Vec3::new(1234.5, 67.25, -890.75), 3.75),
        (Vec3::new(-8000.0, 1500.5, 8000.0), 0.125),
        (Vec3::new(7999.5, -200.25, -1.5), 24.0),
    ];
    let instances: Vec<Instance> = cases
        .iter()
        .map(|(p, s)| Instance::from_parts(*p, Quat::from_rotation_y(0.7), *s, Vec3::ONE, 0))
        .collect();

    let got = decode_on_gpu(&device, &queue, &instances);
    for (i, (p, s)) in cases.iter().enumerate() {
        let (gx, gy, gz, gs) = (got[i * 4], got[i * 4 + 1], got[i * 4 + 2], got[i * 4 + 3]);
        assert_eq!(gx, p.x, "instance {i}: x read as {gx}, wrote {}", p.x);
        assert_eq!(gy, p.y, "instance {i}: y read as {gy}, wrote {}", p.y);
        assert_eq!(gz, p.z, "instance {i}: z read as {gz}, wrote {}", p.z);
        // Scale is f16, so relative rather than exact -- but it has to be the
        // right *field*, which is what a byte-offset mistake would break.
        assert!(
            (gs - s).abs() / s < 1e-2,
            "instance {i}: scale read as {gs}, wrote {s} -- wrong field or offset"
        );
    }
}

#[test]
fn the_cull_shader_does_not_confuse_scale_with_the_padding() {
    // `scale` is read as the low half of a `Float16x2` spanning bytes 20..24,
    // where the high half is padding. Reading `.y` instead of `.x` would give
    // zero for every instance, which culls everything and looks like an empty
    // world rather than like a layout bug.
    let Some((device, queue)) = gpu() else { return };
    let instances: Vec<Instance> = [0.25f32, 1.0, 7.5, 60.0]
        .iter()
        .map(|s| Instance::from_parts(Vec3::ZERO, Quat::IDENTITY, *s, Vec3::ONE, 0))
        .collect();

    let got = decode_on_gpu(&device, &queue, &instances);
    for (i, s) in [0.25f32, 1.0, 7.5, 60.0].iter().enumerate() {
        let gs = got[i * 4 + 3];
        assert!(gs > 0.0, "instance {i}: scale came back {gs} -- reading the padding");
        assert!((gs - s).abs() / s < 1e-2, "instance {i}: scale {gs}, wrote {s}");
    }
}

#[test]
fn a_seed_in_the_last_word_does_not_corrupt_the_scale() {
    // `seed` sits at bytes 28..32, immediately after the colour. A record whose
    // trailing words are non-zero must decode exactly as one whose are zero --
    // this is what catches an off-by-four in either direction.
    let Some((device, queue)) = gpu() else { return };
    let plain =
        Instance::from_parts(Vec3::new(10.0, 20.0, 30.0), Quat::IDENTITY, 2.0, Vec3::ONE, 0);
    let seeded = Instance::from_parts(
        Vec3::new(10.0, 20.0, 30.0),
        Quat::IDENTITY,
        2.0,
        Vec3::new(0.1, 0.2, 0.3),
        0xDEAD_BEEF,
    );

    let got = decode_on_gpu(&device, &queue, &[plain, seeded]);
    assert_eq!(&got[0..4], &got[4..8], "the seed or colour leaked into position or scale");
}

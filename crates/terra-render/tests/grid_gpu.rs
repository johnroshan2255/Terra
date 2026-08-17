//! The default world grid, rendered and measured.
//!
//! The grid replaces the flat grey an untextured terrain used to be, and it is only
//! an improvement if it antialiases properly. A grid drawn naively is *worse* than
//! flat grey: past a couple of pixels per cell the lines merge into a solid wash,
//! and under TAA jitter that wash crawls. `grid.wgsl` is written to fade each decade
//! out before that happens, and this measures whether it does.
//!
//! `fwidth` only exists in a fragment shader, so this is a real render pass. The
//! trick is that the fragment shader scales world coordinates by a known
//! metres-per-pixel, which is exactly what `fwidth` then reports -- so one small
//! draw stands in for "the camera is this far away".
//!
//! No window is opened.

const SIZE: u32 = 256;

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
        label: Some("grid-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A full-screen draw of `world_grid`, at a chosen world scale.
///
/// `metres_per_pixel` is pushed through a uniform rather than baked, so one pipeline
/// covers every distance and the shader under test is the shipped one.
const HARNESS: &str = r#"
struct Args { scale: vec4f };
@group(0) @binding(0) var<uniform> args: Args;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    // One pixel steps `scale.x` metres, so fwidth inside `world_grid` sees exactly
    // the footprint a camera at that distance would produce.
    let world_xz = pos.xy * args.scale.x;
    return vec4f(world_grid(world_xz), 1.0);
}
"#;

/// Luminance of every pixel of the grid at a given metres-per-pixel.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, metres_per_pixel: f32) -> Vec<f32> {
    use wgpu::util::DeviceExt;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grid-harness"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{}", include_str!("../../../assets/shaders/common/grid.wgsl"), HARNESS)
                .into(),
        ),
    });

    let args = [metres_per_pixel, 0.0, 0.0, 0.0];
    let args_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("args"),
        contents: bytemuck::cast_slice(&args),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Rgba16Float, not 8-bit: the far-field test measures variance down near the
    // quantisation floor of an 8-bit target, where it would be measuring rounding.
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grid-target"),
        size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("grid-harness"),
        layout: None,
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
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: args_buf.as_entire_binding() }],
    });

    // 8 bytes a pixel, and 256 pixels a row, so the row is already 2048 -- a
    // multiple of the 256-byte copy alignment, no padding to unpick.
    let row_bytes = SIZE * 8;
    assert!(row_bytes.is_multiple_of(256));
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("grid"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &read,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);

    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let out = {
        let data = read.slice(..).get_mapped_range().expect("mapped range");
        let halves: &[half::f16] = bytemuck::cast_slice(&data);
        halves
            .chunks_exact(4)
            .map(|p| {
                let (r, g, b) = (p[0].to_f32(), p[1].to_f32(), p[2].to_f32());
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect::<Vec<f32>>()
    };
    read.unmap();
    out
}

/// Standard deviation of luminance -- how much visible structure the grid has.
fn contrast(lum: &[f32]) -> f32 {
    let mean = lum.iter().sum::<f32>() / lum.len() as f32;
    (lum.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / lum.len() as f32).sqrt()
}

fn mean(lum: &[f32]) -> f32 {
    lum.iter().sum::<f32>() / lum.len() as f32
}

#[test]
fn the_grid_is_legible_up_close() {
    // The whole point: standing on the terrain, the grid has to read as a ruler.
    let Some((device, queue)) = gpu() else { return };
    // 2 cm a pixel: a 1 m square spans 50 px, which is what a metre looks like from
    // a couple of metres away.
    let lum = render(&device, &queue, 0.02);
    let c = contrast(&lum);
    assert!(c > 0.02, "the grid is nearly invisible up close: contrast {c}");
    // And it is a mid grey, not a black-and-white dazzle that would fight the
    // lighting the surface is there to show.
    let m = mean(&lum);
    assert!((0.15..0.45).contains(&m), "mean luminance {m} is not a mid grey");
}

/// Metres per pixel, from standing on the ground to the far field of a 40 km view.
const SCALES: [f32; 8] = [0.02, 0.1, 0.3, 1.0, 3.0, 8.0, 20.0, 60.0];

#[test]
fn no_distance_turns_the_grid_into_a_flat_wash() {
    // The failure mode that makes a naive grid worse than flat grey: once cells
    // approach a pixel the lines merge and cover the surface, so contrast *spikes*
    // and then the whole thing crawls under TAA jitter.
    //
    // Note this is not a monotonic falloff, and expecting one was wrong: the three
    // decades hand over to each other, so at 2 m a pixel the 10 m grid is crisp and
    // legitimately has more contrast than the 1 m checker does up close. What must
    // hold is that no scale is *dramatically* busier than the design intent.
    let Some((device, queue)) = gpu() else { return };
    let baseline = contrast(&render(&device, &queue, 0.02));
    for s in SCALES {
        let c = contrast(&render(&device, &queue, s));
        assert!(
            c < baseline * 2.5,
            "at {s} m a pixel contrast is {c} against {baseline} up close -- the lines have merged"
        );
    }
}

#[test]
fn the_grid_fades_out_entirely_once_it_cannot_be_resolved() {
    // Past the last decade there is nothing left to draw, and drawing it anyway is
    // aliasing. 60 m a pixel is beyond even the 100 m lines.
    let Some((device, queue)) = gpu() else { return };
    let c = contrast(&render(&device, &queue, 60.0));
    assert!(c < 0.002, "structure survives at 60 m a pixel: contrast {c}, which will alias");
}

#[test]
fn the_average_tone_holds_at_every_distance() {
    // The other half of it: whatever each decade does, the surface must not change
    // brightness with distance. Drift here is the "distant terrain looks like a
    // different material" bug, and it is what a grid that fades to its darkest line
    // colour rather than its own average would produce.
    let Some((device, queue)) = gpu() else { return };
    let near = mean(&render(&device, &queue, 0.02));
    for s in SCALES {
        let m = mean(&render(&device, &queue, s));
        assert!(
            (m - near).abs() < 0.05,
            "at {s} m a pixel the surface sits at {m} against {near} up close"
        );
    }
}

#[test]
fn the_grid_never_produces_a_nan_or_a_negative() {
    // `grid_lines` divides by `fwidth`, which is zero on a surface that does not
    // vary. A NaN in albedo propagates through the lighting and paints a black hole
    // that TAA then smears across the frame.
    let Some((device, queue)) = gpu() else { return };
    // Includes 0: a degenerate scale makes every fwidth exactly zero, which is the
    // case the `max(fw, 1e-6)` guard exists for.
    for s in [0.0f32, 1e-6, 0.001, 1.0, 1000.0, 1.0e6] {
        let lum = render(&device, &queue, s);
        assert!(
            lum.iter().all(|v| v.is_finite() && *v >= 0.0),
            "{s} m a pixel produced a non-finite or negative luminance"
        );
    }
}

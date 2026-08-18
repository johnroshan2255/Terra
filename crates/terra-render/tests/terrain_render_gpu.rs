//! The terrain, rendered offscreen and looked at.
//!
//! Every other check in this crate reads the source or probes one function. That is how a
//! slope band tuned outside the range the generator produces survived: the code was
//! correct, the shader compiled, the maths was right, and the result was a flat image.
//! Nothing rendered a frame.
//!
//! This does. It builds a real `Terrain` with a real heightfield and a real material,
//! draws one frame to a texture, and writes it to `target/`. `#[ignore]`d because it is
//! for looking at rather than asserting on -- the assertions live beside it and are about
//! statistics of the pixels, which is the part a machine can judge.
//!
//! No window is opened. `Terrain::new_headless` exists so this is possible at all.

use glam::Vec3;
use terra_core::WorldSize;
use terra_render::camera::Camera;
use terra_render::clouds::Clouds;
use terra_render::context::{DEPTH_FORMAT, SCENE_FORMAT};
use terra_render::lighting::Lighting;
use terra_render::material::Materials;
use terra_render::terrain::Terrain;

const W: u32 = 960;
const H: u32 = 540;

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
        label: Some("terrain-render-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A 1x1x1 stand-in for the volumetric fog volume, which `Lighting` binds and this does
/// not sample.
fn fog_view(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test-fog"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// The heightfield from a real project if it is there, otherwise a generated one.
///
/// Preferring the real file matters: the whole point is to render what the user is
/// looking at, not an idealised surface.
fn heights(size: WorldSize, res: u32, extent: f32) -> (Vec<f32>, &'static str) {
    let p = format!(
        "{}/Library/Application Support/in.synctric.Terra/projects/New_World/world/source/global_height.r16",
        std::env::var("HOME").unwrap_or_default()
    );
    if let Ok(bytes) = std::fs::read(&p) {
        let n = bytes.len() / 2;
        if n == (res * res) as usize {
            let h: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| {
                    u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0 * terra_core::HEIGHT_RANGE_M
                })
                .collect();
            // A world that was created but never generated or sculpted is a single
            // repeated value. Rendering it looks exactly like the flatness this file
            // exists to investigate, which is how two earlier measurements here --
            // "the checkerboard is averaged away", "the albedo has no grain" -- were
            // recorded as sampling defects when they were correct minification over a
            // plane. Refuse it rather than silently reproduce that.
            let flat = h.iter().all(|v| (v - h[0]).abs() < 1e-3);
            if !flat {
                return (h, "the project's own heightfield");
            }
            println!(
                "  note: {} is uniform at {:.1} m -- never generated or sculpted; \
                 using the synthesised field instead",
                p, h[0]
            );
        }
    }
    // `terra-render` cannot depend on `terra-gen` -- the dependency direction is strictly
    // downward -- so the stand-in is a few summed sines. Not a substitute for the real
    // generator, but it has slopes in the same band and that is what is being rendered.
    let _ = size;
    let mut out = Vec::with_capacity((res * res) as usize);
    for z in 0..res {
        for x in 0..res {
            let (u, v) = (x as f32 / res as f32, z as f32 / res as f32);
            let k = std::f32::consts::TAU;
            let h = 0.5
                + 0.30 * (u * k * 1.5).sin() * (v * k * 1.2).cos()
                + 0.12 * (u * k * 3.7 + 1.0).sin() * (v * k * 4.1).cos()
                + 0.05 * (u * k * 9.0).sin() * (v * k * 8.0 + 2.0).cos();
            out.push(terra_core::BASE_ELEVATION_M + h.clamp(0.0, 1.0) * 600.0);
        }
    }
    let _ = extent;
    (out, "a synthesised heightfield")
}

/// Ground height at a world position, by nearest texel.
///
/// Cameras in this file are placed *above the ground under them*, never at an absolute
/// altitude. An absolute `eye_y` chosen from the field's mean put the camera 196 m up
/// over ground that was at the 256 m floor, which makes every render far-field -- and
/// far-field flatness is correct minification, not a defect.
fn ground_at(h: &[f32], res: u32, extent: f32, x: f32, z: f32) -> f32 {
    let to_texel = |w: f32| {
        (((w / extent) + 0.5) * (res - 1) as f32).round().clamp(0.0, (res - 1) as f32) as usize
    };
    h[to_texel(z) * res as usize + to_texel(x)]
}

/// Eye position `up` metres above the ground at `(x, z)`, using the same heightfield
/// [`render_mode`] will upload.
fn eye_above_ground(x: f32, z: f32, up: f32) -> Vec3 {
    let size = WorldSize::Medium;
    let res = size.tier0_res();
    let extent = size.extent_m() as f32;
    let (h, _) = heights(size, res, extent);
    Vec3::new(x, ground_at(&h, res, extent, x, z) + up, z)
}

/// Render one frame and return it as RGBA8.
/// One device throughout. Bind group layouts are not portable between devices, so
/// building the palette on one and the pipeline on another fails validation with a
/// binding mismatch that looks like a shader bug and is not.
#[allow(clippy::too_many_arguments)]
fn render_mode(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    materials: &Materials,
    label: &str,
    eye: Vec3,
    pitch: f32,
    mode: terra_render::ViewMode,
) -> Vec<u8> {
    let fog = fog_view(device);
    let mut lighting = Lighting::new(device, Default::default(), &fog);
    let env = terra_render::environment::EnvironmentGpu::new(device);
    let clouds = Clouds::new_headless(device, &env, W / 2, H / 2);

    let size = WorldSize::Medium;
    let mut terrain =
        Terrain::new_headless(device, queue, false, size, materials, &lighting, &clouds);
    let (h, source) = heights(size, terrain.resolution(), terrain.extent_m());
    println!("  heightfield: {source}");
    terrain.set_heights(queue, h);

    let cam = Camera { pos: eye, yaw: 0.9, pitch, ..Camera::default() };
    let aspect = W as f32 / H as f32;
    lighting.upload(queue, &cam, aspect, [0.0, 1.0, 0.0, 1.0], [W as f32, H as f32]);
    terrain.upload_camera(queue, &cam, aspect);
    terrain.set_brush(queue, None, 0.0);

    let colour = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let colour_view = colour.create_view(&Default::default());
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &colour_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // A flat mid grey, so anything the terrain does not cover is obvious.
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.06, b: 0.08, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                // Reversed-Z clears to 0.
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        terrain.draw(&mut pass, &lighting, &clouds, mode);
    }

    // Readback. `bytes_per_row` has to be 256-aligned, so the copy is padded and the
    // padding stripped below.
    let bpp = 8u32; // Rgba16Float
    let unpadded = W * bpp;
    let padded = unpadded.div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &colour,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let view = staging.slice(..).get_mapped_range().expect("mapped range");

    // Rgba16Float to sRGB bytes, so the PNG looks like what a screen would show.
    let mut out = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        let row = &view[(y * padded) as usize..(y * padded + unpadded) as usize];
        for px in row.chunks_exact(8) {
            for c in 0..3 {
                let h = half::f16::from_le_bytes([px[c * 2], px[c * 2 + 1]]).to_f32();
                let lin = h.clamp(0.0, 1.0);
                let srgb = if lin <= 0.0031308 {
                    lin * 12.92
                } else {
                    1.055 * lin.powf(1.0 / 2.4) - 0.055
                };
                out.push((srgb * 255.0).round() as u8);
            }
            out.push(255);
        }
    }
    out
}

/// Lit, which is what a user sees.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    materials: &Materials,
    label: &str,
    eye: Vec3,
    pitch: f32,
) -> Vec<u8> {
    render_mode(device, queue, materials, label, eye, pitch, terra_render::ViewMode::Lit)
}

/// One material folder from the repository's shared set.
fn shared(device: &wgpu::Device, queue: &wgpu::Queue, names: &[&str]) -> Materials {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/texture");
    let tmp = std::env::temp_dir().join("terra-render-palette");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for n in names {
        let src = root.join(n);
        if src.exists() {
            let _ = terra_render::texture_set::install(&src, &tmp);
        }
    }
    Materials::load(device, queue, &tmp)
}

/// Write a PNG and report what is in it.
fn save(pixels: &[u8], name: &str) {
    if pixels.is_empty() {
        return;
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
    let img = image::RgbaImage::from_raw(W, H, pixels.to_vec()).expect("dimensions");
    img.save(dir.join(name)).expect("write");
    println!("  wrote target/{name}");
}

/// How much the image varies, as the mean absolute difference between neighbouring
/// pixels' luma.
///
/// The number that distinguishes "a shaded surface" from "a flat wash". A photograph of
/// a texture laid on a plane has local variation but no *large-scale* structure; a shaded
/// landform has both.
fn local_contrast(p: &[u8]) -> f32 {
    let luma =
        |i: usize| 0.2126 * p[i] as f32 + 0.7152 * p[i + 1] as f32 + 0.0722 * p[i + 2] as f32;
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in 0..H as usize {
        for x in 0..(W as usize - 1) {
            let i = (y * W as usize + x) * 4;
            sum += (luma(i) - luma(i + 4)).abs();
            n += 1.0;
        }
    }
    sum / n
}

/// Local contrast over the nearest ground only -- the bottom of the frame.
///
/// The whole-image figure is dominated by the far field, where a pixel covers many metres
/// and flatness is *correct*. Grain can only appear where the ground is close, so that is
/// where it has to be measured.
fn near_contrast(p: &[u8]) -> f32 {
    let luma =
        |i: usize| 0.2126 * p[i] as f32 + 0.7152 * p[i + 1] as f32 + 0.0722 * p[i + 2] as f32;
    let from = (H as f32 * 0.85) as usize;
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in from..H as usize {
        for x in 0..(W as usize - 1) {
            let i = (y * W as usize + x) * 4;
            sum += (luma(i) - luma(i + 4)).abs();
            n += 1.0;
        }
    }
    sum / n
}

/// Spread of *column* means, which is large-scale structure rather than texture detail.
fn column_spread(p: &[u8]) -> f32 {
    let mut means = Vec::with_capacity(W as usize);
    for x in 0..W as usize {
        let mut s = 0.0;
        for y in 0..H as usize {
            let i = (y * W as usize + x) * 4;
            s += 0.2126 * p[i] as f32 + 0.7152 * p[i + 1] as f32 + 0.0722 * p[i + 2] as f32;
        }
        means.push(s / H as f32);
    }
    let mean = means.iter().sum::<f32>() / means.len() as f32;
    (means.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / means.len() as f32).sqrt()
}

#[test]
fn the_terrain_renders_something_other_than_a_flat_wash() {
    let Some((device, queue)) = gpu() else { return };
    let m = shared(&device, &queue, &["Grass001", "Ground024", "Rock042"]);
    if m.count() == 0 {
        eprintln!("no shared textures available; skipping");
        return;
    }
    println!("palette: {} materials", m.count());
    for l in &m.layers {
        println!("  {} -> {}", l.name, terra_render::material::role_label(l.role));
    }
    let px = render(&device, &queue, &m, "lit", Vec3::new(-600.0, 620.0, -600.0), -0.35);
    if px.is_empty() {
        return;
    }
    save(&px, "terrain-lit.png");
    let c = local_contrast(&px);
    let s = column_spread(&px);
    println!("  local contrast {c:.2}   column spread {s:.2}");
    // Large-scale structure is what is verified working: the landform shades, the roles
    // vary across it, and the macro noise breaks up the result. A flat wash would have
    // near zero of this.
    assert!(s > 1.0, "the image has no large-scale structure ({s:.2}) -- no landform");
    // Deliberately *not* asserted on `local_contrast`. It measures pixel-to-pixel grain,
    // and the terrain currently delivers far less of it than the source texture holds --
    // see `the_albedo_grain_deficit`, which documents the open question rather than
    // pretending the current number is the intended one.
}

/// How local contrast varies with tiling scale and camera height.
///
/// `cargo test -p terra-render --test terrain_render_gpu -- --ignored --nocapture tiling`
#[test]
#[ignore]
fn tiling_experiment() {
    let Some((device, queue)) = gpu() else { return };
    println!("  repeat    eye_y    local contrast   texels/pixel");
    for repeat in [3.5f32, 12.0, 40.0, 120.0] {
        for (eye_y, pitch) in [(620.0f32, -0.35f32), (300.0, -0.10)] {
            let mut m = shared(&device, &queue, &["Grass001", "Ground024"]);
            for p in m.params.iter_mut() {
                p.tiling_m = repeat;
            }
            m.upload_params(&queue);
            let px = render(&device, &queue, &m, "tiling", Vec3::new(-600.0, eye_y, -600.0), pitch);
            if px.is_empty() {
                return;
            }
            // Roughly how many texels of a 512-square map land in one pixel, for ground
            // directly ahead. Above about 4 the texture is past what a pixel can show and
            // correct filtering collapses it towards its average colour.
            let ground_per_px = eye_y * 1.1 / H as f32;
            let texels = ground_per_px / (repeat / 512.0);
            println!(
                "  {repeat:6.1}   {eye_y:5.0}    {:12.2}   {texels:12.0}",
                local_contrast(&px)
            );
            save(&px, &format!("terrain-tile{repeat:.0}-{eye_y:.0}.png"));
        }
    }
}

/// Renders from several viewpoints, for looking at.
///
/// `cargo test -p terra-render --test terrain_render_gpu -- --ignored --nocapture`
#[test]
#[ignore]
fn shots() {
    let Some((device, queue)) = gpu() else { return };
    for (label, names) in [
        ("one-rock", &["Rock042"][..]),
        ("grass-ground-rock", &["Grass001", "Ground024", "Rock042"][..]),
    ] {
        let m = shared(&device, &queue, names);
        println!("{label}: {} materials", m.count());
        for l in &m.layers {
            println!("  {} -> {}", l.name, terra_render::material::role_label(l.role));
        }
        for (view, eye, pitch) in [
            ("high", Vec3::new(-900.0, 900.0, -900.0), -0.5f32),
            ("mid", Vec3::new(-300.0, 520.0, -300.0), -0.18),
            // Close to the ground, which is where the screenshots that prompted this were
            // taken from and where the albedo and the normal map are actually resolvable.
            ("ground", eye_above_ground(-40.0, -40.0, 5.0), -0.05),
        ] {
            let px = render(&device, &queue, &m, label, eye, pitch);
            if px.is_empty() {
                continue;
            }
            println!(
                "  {view}: local contrast {:.2}  column spread {:.2}",
                local_contrast(&px),
                column_spread(&px)
            );
            save(&px, &format!("terrain-{label}-{view}.png"));
        }
    }
}

/// Splits "is the albedo flat" from "is the shading flat".
///
/// Unlit returns the albedo alone, with no lighting, no normal map and no fog. If the
/// grain appears here and not in Lit, the texture is fine and the shading is eating it;
/// if it is flat here too, the albedo is never reaching the pixel.
#[test]
#[ignore]
fn unlit_versus_lit() {
    let Some((device, queue)) = gpu() else { return };
    let m = shared(&device, &queue, &["Grass001", "Ground024"]);
    if m.count() == 0 {
        return;
    }
    let eye = eye_above_ground(-40.0, -40.0, 5.0);
    for (name, mode) in
        [("unlit", terra_render::ViewMode::Unlit), ("lit", terra_render::ViewMode::Lit)]
    {
        let px = render_mode(&device, &queue, &m, name, eye, -0.05, mode);
        if px.is_empty() {
            continue;
        }
        println!(
            "  {name:<6} whole-image {:.2}   nearest ground {:.2}",
            local_contrast(&px),
            near_contrast(&px)
        );
        save(&px, &format!("terrain-{name}-ground.png"));
    }
}

/// The albedo's grain survives to the pixel when the camera is near the ground.
///
/// This test was originally written to record a defect -- "the albedo arrives far flatter
/// than the texture that fed it" -- and the defect was not real. The camera was placed at
/// an absolute `eye_y` taken from the heightfield's *mean*, which put it 196 m above
/// ground that sat at the 256 m floor. Every render was therefore far-field, where a
/// material with a few-metre repeat is far below one texel per pixel and averaging to a
/// flat wash is exactly what correct minification does.
///
/// Placed 5 m above the ground under it, the near band keeps most of the source contrast.
/// What that establishes, alongside [`a_checkerboard_survives_the_material_path`]: the uv,
/// the tiling scale, the layer index, the mip chain and the filtering are all sound. The
/// flat look reported from the editor is not this.
#[test]
fn the_albedo_grain_deficit() {
    let Some((device, queue)) = gpu() else { return };
    let m = shared(&device, &queue, &["Grass001", "Ground024"]);
    if m.count() == 0 {
        return;
    }
    // Close to the ground, where the nearest texels are about three per pixel and nearly
    // all of the texture's grain should survive.
    let px = render(&device, &queue, &m, "grain", eye_above_ground(-40.0, -40.0, 5.0), -0.05);
    if px.is_empty() {
        return;
    }
    let near = near_contrast(&px);
    println!("  nearest-ground contrast {near:.2}, source texture neighbour difference ~9.0");
    assert!(
        near > 3.0,
        "the nearest ground shows {near:.2} of grain against about 9.0 in the texture that \
         fed it -- five metres above the ground, most of it should survive"
    );
}

/// Pushes a **known** texture through `Materials` and looks for it.
///
/// A hard 8-texel black-and-white checkerboard has the maximum possible neighbour
/// contrast, so it separates "the uv is wrong" from "the texture is low-contrast at the
/// scale it is viewed". It arrives at the pixel with nearly all of its CPU-side contrast
/// intact, which retires the sampling path as a suspect for the flat look: whatever is
/// being sampled, it is the texture that was loaded, at the scale it was asked for.
#[test]
fn a_checkerboard_survives_the_material_path() {
    let Some((device, queue)) = gpu() else { return };
    let tmp = std::env::temp_dir().join("terra-checker");
    let _ = std::fs::remove_dir_all(&tmp);
    let d = tmp.join("Checker");
    std::fs::create_dir_all(&d).unwrap();

    // 512 square, 8-texel squares, full black and white.
    const N: u32 = 512;
    let mut img = image::RgbaImage::new(N, N);
    for y in 0..N {
        for x in 0..N {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 255 } else { 0 };
            img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    img.save(d.join("Checker_Color.png")).unwrap();
    // Flat normal, so nothing but the albedo is under test.
    image::RgbaImage::from_pixel(N, N, image::Rgba([128, 128, 255, 255]))
        .save(d.join("Checker_NormalGL.png"))
        .unwrap();

    let cpu = terra_render::texture_set::load(&d, 512).expect("load");
    let lum: Vec<f32> = cpu.albedo.chunks_exact(4).map(|p| p[0] as f32).collect();
    let mut nd = 0.0;
    let mut c = 0.0;
    for y in 0..512usize {
        for x in 0..511usize {
            nd += (lum[y * 512 + x] - lum[y * 512 + x + 1]).abs();
            c += 1.0;
        }
    }
    println!("  checkerboard on the CPU: neighbour difference {:.1}", nd / c);

    let m = Materials::load(&device, &queue, &tmp);
    println!("  palette: {} materials", m.count());
    // A one metre repeat, so from ten metres up the squares are centimetres and plainly
    // resolvable rather than fighting minification.
    let mut m = m;
    for p in m.params.iter_mut() {
        p.tiling_m = 4.0;
    }
    m.upload_params(&queue);

    let px = render_mode(
        &device,
        &queue,
        &m,
        "checker",
        eye_above_ground(-40.0, -40.0, 5.0),
        -0.05,
        terra_render::ViewMode::Unlit,
    );
    if px.is_empty() {
        return;
    }
    save(&px, "terrain-checker.png");
    let near = near_contrast(&px);
    println!("  rendered nearest-ground contrast {near:.2}");
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        near > 10.0,
        "a black-and-white checkerboard rendered at {near:.2} of contrast from five metres \
         up -- the sampling path is losing the texture it was given"
    );
}

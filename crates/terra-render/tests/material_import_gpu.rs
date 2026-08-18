//! Importing a material folder, end to end, against a real GPU.
//!
//! The unit tests in `texture_set` cover which files play which role using
//! zero-byte placeholders, which is enough for the naming rules and nothing
//! more. This decodes real PNGs, uploads them, and asserts the palette the
//! terrain shader binds actually holds the layer -- the step that was broken.
//!
//! The bug being guarded: `import_asset` copied picked *files* flat into
//! `assets/textures/`, while `discover` only ever looks at subdirectories. Files
//! landed on disk, no material appeared, and nothing was reported. Anything that
//! reintroduces a flat-file import fails `a_flat_pile_of_maps_is_not_a_material`.
//!
//! No window is opened.

use std::path::{Path, PathBuf};
use terra_render::material::Materials;
use terra_render::texture_set;

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
        label: Some("material-import-test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("terra-import-gpu-{tag}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A real, decodable 8-bit image. Small on purpose: `load` resizes to `TILE`
/// anyway, so the pixels only have to survive a decode.
///
/// JPEG carries no alpha, so a `.jpg` is written as RGB. The format follows the
/// extension, which is what lets one helper stand in for a mixed-format download.
fn png(path: &Path, rgba: [u8; 4]) {
    let jpeg = path
        .extension()
        .is_some_and(|x| x.eq_ignore_ascii_case("jpg") || x.eq_ignore_ascii_case("jpeg"));
    if jpeg {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([rgba[0], rgba[1], rgba[2]]));
        img.save(path).unwrap();
    } else {
        image::RgbaImage::from_pixel(8, 8, image::Rgba(rgba)).save(path).unwrap();
    }
}

/// The ambientCG layout, with maps that actually decode.
fn material_folder(parent: &Path, name: &str) -> PathBuf {
    let d = parent.join(name);
    std::fs::create_dir_all(&d).unwrap();
    png(&d.join("Color.png"), [180, 140, 90, 255]);
    png(&d.join("NormalGL.png"), [128, 128, 255, 255]);
    png(&d.join("Roughness.png"), [200, 200, 200, 255]);
    png(&d.join("Displacement.png"), [128, 128, 128, 255]);
    d
}

#[test]
fn an_imported_folder_reaches_the_palette_the_shader_binds() {
    let Some((device, queue)) = gpu() else { return };
    let tmp = scratch("e2e");
    let src = material_folder(&tmp, "Ground024");
    let textures = tmp.join("project/assets/textures");

    let out = texture_set::install(&src, &textures).unwrap();
    assert_eq!(out.materials, vec!["Ground024"]);

    // The palette is what the terrain actually samples. Before the fix this was
    // 0 after an import, which is what made the Material pane read
    // "No material selected" on a double-click.
    let materials = Materials::load(&device, &queue, &textures);
    assert_eq!(materials.count(), 1, "imported material never reached the palette");
    assert_eq!(materials.layers[0].name, "Ground024");
    assert_eq!(materials.params.len(), 1);

    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_flat_pile_of_maps_is_not_a_material() {
    // Exactly what the old import produced: the right files, no folder.
    let Some((device, queue)) = gpu() else { return };
    let tmp = scratch("flat");
    let textures = tmp.join("textures");
    std::fs::create_dir_all(&textures).unwrap();
    png(&textures.join("Color.png"), [180, 140, 90, 255]);
    png(&textures.join("NormalGL.png"), [128, 128, 255, 255]);

    let materials = Materials::load(&device, &queue, &textures);
    assert_eq!(materials.count(), 0, "loose files must not silently become a material");
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn importing_a_pack_gives_one_palette_slot_per_set() {
    let Some((device, queue)) = gpu() else { return };
    let tmp = scratch("pack");
    let pack = tmp.join("Pack");
    std::fs::create_dir_all(&pack).unwrap();
    material_folder(&pack, "Grass001");
    material_folder(&pack, "Rock042");
    let textures = tmp.join("textures");

    let out = texture_set::install(&pack, &textures).unwrap();
    assert_eq!(out.materials, vec!["Grass001", "Rock042"]);

    let materials = Materials::load(&device, &queue, &textures);
    assert_eq!(materials.count(), 2);
    // Sorted, so the palette order a user paints against is stable between runs.
    assert_eq!(materials.layers[0].name, "Grass001");
    assert_eq!(materials.layers[1].name, "Rock042");
    std::fs::remove_dir_all(&tmp).unwrap();
}

/// A real RGB OpenEXR file, written through the same crate that reads it.
fn exr(path: &Path, rgb: [f32; 3]) {
    let img = image::Rgb32FImage::from_pixel(8, 8, image::Rgb(rgb));
    image::DynamicImage::ImageRgb32F(img).save(path).unwrap();
}

/// A **single-channel** OpenEXR, which is what Poly Haven serves for roughness and
/// ambient occlusion, and which `image::open` cannot read at all.
fn exr_gray(path: &Path, value: f32) {
    use exr::prelude::*;
    let channel = AnyChannel::new("Y", FlatSamples::F32(vec![value; 8 * 8]));
    let layer = Layer::new(
        (8, 8),
        LayerAttributes::named("gray"),
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(exr::prelude::SmallVec::from_vec(vec![channel])),
    );
    Image::from_layer(layer).write().to_file(path).unwrap();
}

#[test]
fn a_poly_haven_mixed_format_set_loads_every_map() {
    // Exactly what a complete Poly Haven download looks like: JPEG albedo, PNG
    // displacement, and the normal and roughness as **OpenEXR**. A PNG-only reader
    // silently drops the two maps that carry the relief, so the material renders as
    // an image laid over the ground.
    let tmp = scratch("polyhaven-mixed");
    let d = tmp.join("rocky_terrain_03");
    std::fs::create_dir_all(&d).unwrap();
    png(&d.join("rocky_terrain_03_diff_4k.jpg"), [180, 140, 90, 255]);
    png(&d.join("rocky_terrain_03_disp_4k.png"), [128, 128, 128, 255]);
    exr(&d.join("rocky_terrain_03_nor_gl_4k.exr"), [0.5, 0.75, 1.0]);
    // Single channel, as Poly Haven actually ships it.
    exr_gray(&d.join("rocky_terrain_03_rough_4k.exr"), 0.6);
    png(&d.join("rocky_terrain_03_spec_4k.png"), [10, 10, 10, 255]);

    // Nothing missing: the EXRs count.
    let missing = texture_set::missing_maps(&d);
    assert!(
        !missing.contains(&"normal") && !missing.contains(&"roughness"),
        "EXR maps were not picked up: {missing:?}"
    );

    let set = texture_set::load(&d, 8).expect("the set must load");
    // Green 0.75 -> 191, and it must pass through as GL rather than being flipped.
    assert!((set.normal[1] as i32 - 191).abs() <= 2, "normal green came back {}", set.normal[1]);
    // 0.6 -> 153. A uniform 200 would mean it fell back to the default, which is
    // what happened before single-channel EXRs were handled.
    assert!((set.roughness[0] as i32 - 153).abs() <= 2, "roughness came back {}", set.roughness[0]);
    assert_ne!(set.roughness[0], 200, "roughness fell back to the flat default");
    // The albedo is a JPEG, so it is still treated as sRGB.
    assert!(!set.albedo_is_linear);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn an_exr_albedo_is_flagged_as_already_linear() {
    // EXR is linear float. Decoding it as sRGB applies the transfer twice and the
    // material comes out visibly dark, so the flag is what stops `bake_set` doing it.
    let tmp = scratch("exr-albedo");
    let d = tmp.join("Rock");
    std::fs::create_dir_all(&d).unwrap();
    exr(&d.join("Rock_diff.exr"), [0.5, 0.5, 0.5]);
    let set = texture_set::load(&d, 8).expect("an EXR albedo is still an albedo");
    assert!(set.albedo_is_linear, "an EXR albedo must not be sRGB-decoded again");

    // And a JPEG one is not.
    let d2 = tmp.join("Rock2");
    std::fs::create_dir_all(&d2).unwrap();
    png(&d2.join("Rock2_diff.jpg"), [128, 128, 128, 255]);
    assert!(!texture_set::load(&d2, 8).unwrap().albedo_is_linear);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_format_that_cannot_be_decoded_is_named_rather_than_called_missing() {
    // "no normal map" would send someone to re-download a file already on disk. The
    // two cases need different words because they need different actions.
    let tmp = scratch("undecodable");
    let d = tmp.join("Rock");
    std::fs::create_dir_all(&d).unwrap();
    png(&d.join("Rock_diff.png"), [180, 140, 90, 255]);
    std::fs::write(d.join("Rock_nor_gl.tga"), b"not decodable here").unwrap();

    let unreadable = texture_set::undecodable_maps(&d);
    assert_eq!(unreadable, vec!["Rock_nor_gl.tga".to_string()]);
    // And the install surfaces it.
    let dest = tmp.join("dest");
    let out = texture_set::install(&d, &dest).unwrap();
    assert_eq!(out.unreadable.len(), 1);
    assert!(out.unreadable[0].1.iter().any(|f| f.contains("tga")));
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_dx_normal_map_is_converted_rather_than_used_as_is() {
    // Green is flipped on load, so the shader only ever sees GL. Using the DX map
    // unconverted inverts the shading on every slope, which is the failure the
    // plain-`normal` fallback used to cause silently.
    let tmp = scratch("dx");
    let d = tmp.join("Rock");
    std::fs::create_dir_all(&d).unwrap();
    png(&d.join("Color.png"), [180, 140, 90, 255]);
    // Green well off centre so the flip is unambiguous: 200 -> 55.
    png(&d.join("NormalDX.png"), [128, 200, 255, 255]);

    let set = texture_set::load(&d, 8).expect("a colour map is all that is required");
    assert_eq!(set.normal[1], 55, "DX green was not flipped to GL");
    // The other channels are left alone -- only green differs between the two
    // conventions.
    assert_eq!(set.normal[0], 128);
    assert_eq!(set.normal[2], 255);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_gl_normal_map_is_left_alone() {
    let tmp = scratch("gl");
    let d = tmp.join("Rock");
    std::fs::create_dir_all(&d).unwrap();
    png(&d.join("Color.png"), [180, 140, 90, 255]);
    png(&d.join("NormalGL.png"), [128, 200, 255, 255]);

    let set = texture_set::load(&d, 8).unwrap();
    assert_eq!(set.normal[1], 200, "a GL map must pass through unchanged");
    std::fs::remove_dir_all(&tmp).unwrap();
}

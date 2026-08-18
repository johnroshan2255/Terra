//! Reading downloaded PBR texture sets off disk.
//!
//! One subfolder of a project's `assets/textures/` is one material. Whatever is
//! in that folder when the editor starts is what appears in the palette -- there
//! is no registry to edit and no id to allocate, because the thing an artist
//! actually does is drop a folder in and expect it to show up.
//!
//! The repository's own `assets/texture/` (singular) is a different directory
//! serving a different purpose: it is the shared set the menu backdrop renders
//! with. Dropping a material there does not put it in any project.
//!
//! Map roles are recognised by substring, case-insensitively, so the sets work
//! as downloaded. ambientCG ships `Ground024_2K-PNG_Color.png`, Poly Haven
//! ships `rock_diff_2k.png`, and both land in the right slot:
//!
//!   albedo       color, albedo, basecolor, base_color, diff
//!   normal       normalgl, nrm, normal            (GL convention, green up)
//!   roughness    rough
//!   occlusion    ambientocclusion, occlusion, _ao
//!   height       displacement, height, disp
//!
//! Only albedo is required. A set without a height map still works, but it
//! blends by mask alone -- height is what buys the interlocking transition, so
//! it is worth downloading the displacement map even when it costs another
//! 8 MB.
//!
//! Readable formats are PNG, JPEG, **OpenEXR** and TIFF. EXR is here because a
//! complete Poly Haven download is a *mix*: the albedo arrives as JPEG while the
//! normal and roughness arrive as EXR, so a PNG-only reader silently loses the two
//! maps that carry the relief and the material renders as an image laid over the
//! ground. Two wrinkles come with it, both handled:
//!
//! * EXR is **linear**, so an EXR albedo must not be sRGB-decoded a second time --
//!   see [`TextureSet::albedo_is_linear`].
//! * `image` can only decode EXRs carrying **RGB** channels, and Poly Haven ships
//!   roughness and occlusion as *single channel*. Those go through `read_gray_exr`.
//!
//! Anything still unreadable -- TGA, PSD -- is named by [`undecodable_maps`] rather
//! than reported as absent, because the fix is to convert a file that is already
//! there rather than to go and find one.

use image::imageops::FilterType;
use std::path::{Path, PathBuf};

/// File extensions a map can arrive as. Matches the `image` features enabled in
/// the workspace manifest -- adding one here without enabling its decoder makes
/// a file discoverable and then unreadable, which is worse than skipping it.
///
/// `exr` is here because Poly Haven serves normal and roughness maps as OpenEXR
/// while serving the albedo as JPEG, so a complete download is a mix of formats and
/// a PNG-only reader silently loses the two maps that matter most for relief.
pub const MAP_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "exr", "tif", "tiff"];

/// Extensions whose contents are **linear**, not sRGB-encoded.
///
/// Only matters for the albedo: every other map is consumed linearly whatever it
/// arrived as. OpenEXR is float and linear by definition, so decoding it as though
/// it were sRGB applies the transfer twice and the material comes out visibly dark.
///
/// TIFF is deliberately not here. It can hold either, 8-bit TIFFs are usually sRGB,
/// and there is no way to tell from the extension -- so it takes the common case.
const LINEAR_EXTENSIONS: [&str; 1] = ["exr"];

/// Whether `path` carries linear data rather than sRGB.
fn is_linear_format(path: &Path) -> bool {
    path.extension()
        .is_some_and(|x| LINEAR_EXTENSIONS.contains(&x.to_string_lossy().to_lowercase().as_str()))
}

/// Filename substrings that identify the one required map. Public so the editor
/// can name them when it has to explain why a folder was rejected.
pub const ALBEDO_KEYS: [&str; 6] =
    ["color", "albedo", "basecolor", "base_color", "_diff", "diffuse"];

/// The maps making up one material, already decoded and resized to a square.
pub struct TextureSet {
    pub name: String,
    /// `size * size * 4`, sRGB-encoded unless [`Self::albedo_is_linear`].
    pub albedo: Vec<u8>,
    /// Whether `albedo` is already linear, which it is when it came from an EXR.
    /// `material::bake_set` skips its sRGB decode in that case; applying the
    /// transfer to data that never had it darkens the whole material.
    pub albedo_is_linear: bool,
    /// Tangent-space normal, GL convention. `size * size * 4`.
    pub normal: Vec<u8>,
    /// Single channel each, `size * size`.
    pub roughness: Vec<u8>,
    pub occlusion: Vec<u8>,
    pub height: Vec<u8>,
    pub size: u32,
}

/// Which files in a folder play which role.
struct Maps {
    albedo: PathBuf,
    normal: Option<PathBuf>,
    /// Whether `normal` is a DirectX-convention map, so green has to be flipped
    /// on load. Sets that ship only DX are common enough -- anything aimed at
    /// Unreal -- that rejecting them loses real detail for no reason.
    normal_is_dx: bool,
    roughness: Option<PathBuf>,
    occlusion: Option<PathBuf>,
    height: Option<PathBuf>,
}

/// Whether `dir` holds a usable material, i.e. at least a readable albedo.
pub fn is_material(dir: &Path) -> bool {
    find_maps(dir).is_some()
}

/// Folder names that say nothing about what the material is.
///
/// Downloads arrive as `rocky_terrain_03/textures/*.png` often enough that naming
/// the material after its folder gives "textures" -- which is useless in the
/// palette and, worse, decides the material's automatic *role*, because
/// `material::role_of` reads the name. "textures" matches nothing and lands in the
/// soil base coat, so a rock set ends up carpeting the flats.
const GENERIC_FOLDER_NAMES: [&str; 8] =
    ["textures", "texture", "maps", "map", "tex", "pbr", "material", "materials"];

/// Maps a folder does not have, most visually important first.
///
/// Absent maps are substituted with flat defaults, which is the right behaviour and
/// a terrible thing to do quietly. A missing **normal** map in particular is the
/// difference between a surface and a photograph of one: with a flat normal there is
/// no relief for the light to catch, so the material reads as an image laid over the
/// ground however good the albedo is. Parallax cannot rescue it either -- shifting
/// the lookup is only convincing once the shading responds to the shift.
pub fn missing_maps(dir: &Path) -> Vec<&'static str> {
    let Some(m) = find_maps(dir) else { return Vec::new() };
    let mut out = Vec::new();
    if m.normal.is_none() {
        out.push("normal");
    }
    if m.roughness.is_none() {
        out.push("roughness");
    }
    if m.height.is_none() {
        out.push("height");
    }
    if m.occlusion.is_none() {
        out.push("occlusion");
    }
    out
}

/// Map files present in `dir` that name a role but cannot be decoded.
///
/// A different problem from a missing map, and a different fix: the file is sitting
/// right there and only its format is wrong, so "no normal map" would send someone
/// back to download something they already have. `missing_maps` cannot tell the two
/// apart, because an undecodable file never reaches `find_maps` in the first place.
pub fn undecodable_maps(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let role_keys: Vec<&str> = ALBEDO_KEYS
        .iter()
        .copied()
        .chain(["normal", "_nor_", "nrm", "rough", "occlusion", "_ao", "displacement", "_disp"])
        .collect();
    let mut out: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            let names_a_role = role_keys.iter().any(|k| name.contains(k));
            let decodes = p.extension().is_some_and(|x| {
                MAP_EXTENSIONS.contains(&x.to_string_lossy().to_lowercase().as_str())
            });
            names_a_role && !decodes
        })
        .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
        .collect();
    out.sort();
    out
}

/// A material name for `dir`, preferring the map filenames when the folder name is
/// generic.
///
/// `rocky_terrain_03/textures/rocky_terrain_03_diff_4k.jpg` should be called
/// `rocky_terrain_03`, not `textures`: it is what the palette shows and it is what
/// picks the role, so "rock" being in there is the difference between the set
/// appearing on cliffs and it covering every flat field in the world.
pub fn material_name(dir: &Path) -> String {
    let folder = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
    let generic = GENERIC_FOLDER_NAMES.contains(&folder.to_lowercase().as_str());
    if !generic && !folder.is_empty() {
        return folder;
    }
    derived_name(dir).unwrap_or(folder)
}

/// The common asset name across a folder's map files, with the role suffix removed.
fn derived_name(dir: &Path) -> Option<String> {
    let maps = find_maps(dir)?;
    let mut stems: Vec<String> =
        [Some(maps.albedo), maps.normal, maps.roughness, maps.occlusion, maps.height]
            .into_iter()
            .flatten()
            .filter_map(|p| p.file_stem().map(|s| asset_stem(&s.to_string_lossy())))
            .filter(|s| !s.is_empty())
            .collect();
    stems.sort();
    stems.dedup();

    let first = stems.first()?.clone();
    // Longest common prefix across the maps, so a set whose files disagree past the
    // asset name still yields the part they share.
    let common = stems.iter().fold(first, |acc, s| {
        let n = acc.chars().zip(s.chars()).take_while(|(a, b)| a.eq_ignore_ascii_case(b)).count();
        acc.chars().take(n).collect()
    });
    let trimmed = common.trim_matches(|c: char| c == '_' || c == '-' || c == ' ');
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// One map filename, cut at whichever role keyword it carries.
///
/// `rocky_terrain_03_diff_4k` -> `rocky_terrain_03`. Cut at the *earliest* match, so
/// a name containing two role words does not keep the tail of the first.
fn asset_stem(stem: &str) -> String {
    let lower = stem.to_lowercase();
    const ROLE_KEYS: [&str; 18] = [
        "color",
        "albedo",
        "basecolor",
        "base_color",
        "_diff",
        "diffuse",
        "normalgl",
        "normal_gl",
        "_nor_gl",
        "normal",
        "_nor_",
        "nrm",
        "rough",
        "ambientocclusion",
        "occlusion",
        "_ao",
        "displacement",
        "_disp",
    ];
    let cut = ROLE_KEYS.iter().filter_map(|k| lower.find(k)).min().unwrap_or(stem.len());
    stem[..cut].trim_matches(|c: char| c == '_' || c == '-' || c == ' ').to_string()
}

/// Why `dir` is not a material, phrased for a user who just tried to import it.
/// `None` means it is one.
///
/// Distinguishes the two failures that look identical from the outside: a folder
/// with no colour map at all, and one whose colour map is in a format that does
/// not decode here.
pub fn reject_reason(dir: &Path) -> Option<String> {
    if is_material(dir) {
        return None;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Some("could not be read".into());
    };
    let names: Vec<String> =
        entries.flatten().map(|e| e.file_name().to_string_lossy().to_lowercase()).collect();

    // An albedo under an extension we cannot decode is the more useful message:
    // the naming is right and only the format is wrong.
    let undecodable: Vec<&String> = names
        .iter()
        .filter(|n| ALBEDO_KEYS.iter().any(|k| n.contains(k)))
        .filter(|n| !MAP_EXTENSIONS.iter().any(|x| n.ends_with(&format!(".{x}"))))
        .collect();
    if let Some(n) = undecodable.first() {
        return Some(format!(
            "its colour map ({n}) is not a PNG or JPEG -- only those two decode, so convert it"
        ));
    }
    Some(format!(
        "no colour map found. One file's name has to contain one of: {}",
        ALBEDO_KEYS.join(", ")
    ))
}

/// Material folders under `dir`, sorted by name so the palette order is stable
/// between runs. Files loose in `dir` are ignored -- an archive sitting there
/// unextracted is reported, because silently not appearing in the palette is
/// the most confusing thing this could do.
pub fn discover(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    let mut archives = 0;
    for e in entries.flatten() {
        let path = e.path();
        if path.is_dir() {
            // `.cache` and any other dot-folder is ours, not a material.
            let hidden =
                path.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with('.'));
            if hidden {
                continue;
            }
            match reject_reason(&path) {
                None => found.push(path),
                // Previously silent, and it was the most confusing outcome
                // available: the folder is listed by the content browser, which
                // only checks that it is a directory, and then never appears in
                // the palette.
                Some(why) => log::warn!(
                    "{}: not loaded as a material -- {why}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ),
            }
        } else if path.extension().is_some_and(|x| x.eq_ignore_ascii_case("zip")) {
            // Only worth mentioning if nothing was extracted from it. Sets are
            // commonly downloaded as `Grass001_2K-PNG.zip` and extracted to
            // `Grass001/`, and warning about the archive still sitting there
            // afterwards is noise on every single start.
            let stem = path.file_stem().map(|s| s.to_string_lossy().to_lowercase());
            let extracted = stem.as_ref().is_some_and(|stem| {
                std::fs::read_dir(dir).ok().is_some_and(|mut e| {
                    e.any(|d| {
                        d.ok().filter(|d| d.path().is_dir()).is_some_and(|d| {
                            let n = d.file_name().to_string_lossy().to_lowercase();
                            stem.starts_with(&n)
                        })
                    })
                })
            });
            if !extracted {
                archives += 1;
            }
        }
    }
    if archives > 0 {
        log::warn!(
            "{}: {archives} archive(s) not extracted -- unzip each into its own subfolder to \
             have it appear in the palette",
            dir.display()
        );
    }
    found.sort();
    found
}

/// Locate the maps inside one material folder, if it has at least an albedo.
fn find_maps(dir: &Path) -> Option<Maps> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| {
                let x = x.to_string_lossy().to_lowercase();
                MAP_EXTENSIONS.contains(&x.as_str())
            })
        })
        .collect();
    // `read_dir` order is unspecified, and two files can match one role -- a set
    // shipping both `Color.png` and `Color_Opacity.png`, say. Sorting makes which
    // one wins the same on every machine and every run.
    files.sort();

    // Lower-cased stem for matching, so `_Color` and `_color` behave the same.
    let stem =
        |p: &PathBuf| p.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let pick = |keys: &[&str]| -> Option<PathBuf> {
        files
            .iter()
            .find(|p| {
                let s = stem(p);
                keys.iter().any(|k| s.contains(k))
            })
            .cloned()
    };

    // Normal maps come in two conventions and sets often ship both. GL has
    // green pointing up, which is what the shader expects; taking DX by
    // accident inverts every slope's shading and is maddening to diagnose.
    //
    // The plain-`normal` fallback cannot be used as-is, because "normaldx"
    // contains "normal" -- so a set shipping only DX used to land in the GL slot
    // silently, which is exactly the bug the comment above warns about. Match DX
    // deliberately instead and flip green at load.
    const DX_KEYS: [&str; 3] = ["normaldx", "normal_dx", "_nor_dx"];
    let is_dx = |p: &PathBuf| {
        let s = stem(p);
        DX_KEYS.iter().any(|k| s.contains(k))
    };
    let normal = pick(&["normalgl", "normal_gl", "_nor_gl"])
        .or_else(|| pick(&["normal", "_nor_", "nrm"]).filter(|p| !is_dx(p)))
        .or_else(|| pick(&DX_KEYS));
    let normal_is_dx = normal.as_ref().is_some_and(is_dx);

    Some(Maps {
        albedo: pick(&ALBEDO_KEYS)?,
        normal,
        normal_is_dx,
        roughness: pick(&["rough"]),
        occlusion: pick(&["ambientocclusion", "occlusion", "_ao", "ambient_occlusion"]),
        height: pick(&["displacement", "height", "_disp", "bump"]),
    })
}

/// What one import attempt produced.
#[derive(Debug, Default)]
pub struct Installed {
    /// Folder names now present under the destination, in the order copied.
    pub materials: Vec<String>,
    /// Folders skipped, each with a reason already phrased for a user.
    pub rejected: Vec<(String, String)>,
    /// Materials that imported but are missing maps, with which ones. A set with no
    /// normal map renders as a flat image however good its albedo is, so this is
    /// reported rather than left to be discovered by looking at it.
    pub incomplete: Vec<(String, Vec<&'static str>)>,
    /// Files in the source that name a map role but are in a format that does not
    /// decode, so they were left behind. Reported separately from `incomplete`
    /// because the fix is to convert a file rather than to go and find one.
    pub unreadable: Vec<(String, Vec<String>)>,
}

/// Copy `src` into `dest_root` as one or more material folders.
///
/// Two shapes are accepted, because both are what a user actually picks:
///
/// * `src` is itself a material -- the usual `Ground024/` with maps loose in it
/// * `src` is a *pack* holding several material folders, which is how multi-set
///   downloads arrive
///
/// Only map files are copied. Readmes, licences and leftover archives are left
/// behind rather than dragged into the project, and a `.zip` in particular would
/// otherwise trip `discover`'s unextracted-archive warning forever.
pub fn install(src: &Path, dest_root: &Path) -> std::io::Result<Installed> {
    std::fs::create_dir_all(dest_root)?;
    let mut out = Installed::default();
    let label = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().to_string();

    if is_material(src) {
        let name = copy_material(src, dest_root)?;
        note_gaps(&mut out, &name, src, &dest_root.join(&name));
        out.materials.push(name);
        return Ok(out);
    }

    let mut subs: Vec<PathBuf> = std::fs::read_dir(src)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| !p.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with('.')))
        .collect();
    subs.sort();

    if subs.is_empty() {
        let why = reject_reason(src).unwrap_or_else(|| "is not a material".into());
        out.rejected.push((label(src), why));
        return Ok(out);
    }

    for sub in subs {
        match reject_reason(&sub) {
            None => {
                let name = copy_material(&sub, dest_root)?;
                note_gaps(&mut out, &name, &sub, &dest_root.join(&name));
                out.materials.push(name);
            }
            Some(why) => out.rejected.push((label(&sub), why)),
        }
    }
    Ok(out)
}

/// Record what one installed material is missing or could not read.
///
/// `src` for the unreadable files, because `copy_material` leaves those behind, and
/// `dest` for the absent ones, because that is the folder the palette will load.
fn note_gaps(out: &mut Installed, name: &str, src: &Path, dest: &Path) {
    let missing = missing_maps(dest);
    if !missing.is_empty() {
        out.incomplete.push((name.to_string(), missing));
    }
    let unreadable = undecodable_maps(src);
    if !unreadable.is_empty() {
        out.unreadable.push((name.to_string(), unreadable));
    }
}

/// Copy one known-good material folder in, under a name that does not collide.
fn copy_material(src: &Path, dest_root: &Path) -> std::io::Result<String> {
    // From the map filenames when the folder is called something like "textures",
    // because the name is both what the palette shows and what picks the role.
    let base = match material_name(src) {
        n if n.is_empty() => "Material".to_string(),
        n => n,
    };
    let name = unique_name(dest_root, &base);
    let dest = dest_root.join(&name);
    std::fs::create_dir_all(&dest)?;

    for f in std::fs::read_dir(src)?.flatten().map(|e| e.path()) {
        let is_map = f.is_file()
            && f.extension().is_some_and(|x| {
                MAP_EXTENSIONS.contains(&x.to_string_lossy().to_lowercase().as_str())
            });
        if is_map && let Some(n) = f.file_name() {
            std::fs::copy(&f, dest.join(n))?;
        }
    }
    Ok(name)
}

/// A folder name free in `parent`. Importing the same set twice should give two
/// materials rather than overwriting the first, which would silently change what
/// is already painted onto the terrain.
fn unique_name(parent: &Path, base: &str) -> String {
    if !parent.join(base).exists() {
        return base.to_string();
    }
    (2..10_000)
        .map(|n| format!("{base}_{n}"))
        .find(|c| !parent.join(c).exists())
        .unwrap_or_else(|| base.to_string())
}

/// Decode one material folder down to `size`.
///
/// Downsampling here rather than at 2K is deliberate: the array is sampled
/// triplanar, three fetches per layer, so a 2K set would cost VRAM and
/// bandwidth for detail that is gone by the second mip anyway.
pub fn load(dir: &Path, size: u32) -> Option<TextureSet> {
    let maps = find_maps(dir)?;
    let name = material_name(dir);
    let missing = missing_maps(dir);
    if !missing.is_empty() {
        // `normal` first in that list, and it is the one that matters: without it the
        // shading has no relief to respond to and the material reads as a photograph
        // laid over the ground.
        log::warn!("material '{name}': no {} map -- using flat defaults", missing.join(", no "));
    }

    let rgba = |path: &Path| -> Option<Vec<u8>> {
        match image::open(path) {
            Ok(img) => {
                Some(img.resize_exact(size, size, FilterType::Lanczos3).to_rgba8().into_raw())
            }
            Err(e) => {
                log::error!("{}: {e}", path.display());
                None
            }
        }
    };
    let gray = |path: &Option<PathBuf>, default: u8| -> Vec<u8> {
        let Some(p) = path else {
            return vec![default; (size * size) as usize];
        };
        // `image::open` first, then the single-channel EXR path. Poly Haven serves
        // roughness and ambient occlusion as one-channel EXRs, which are perfectly
        // valid and which `image` refuses -- it can only decode EXRs carrying RGB
        // channels. Left to fall through, the map came back as a uniform default and
        // the material lost its roughness variation with only a log line to say so.
        match image::open(p).or_else(|e| read_gray_exr(p).ok_or(e)) {
            Ok(img) => img.resize_exact(size, size, FilterType::Lanczos3).to_luma8().into_raw(),
            Err(e) => {
                log::error!("{}: {e}", p.display());
                vec![default; (size * size) as usize]
            }
        }
    };

    let albedo = rgba(&maps.albedo)?;
    // A flat normal is (0.5, 0.5, 1.0) -- pointing straight out of the surface.
    let mut normal = maps
        .normal
        .as_deref()
        .and_then(rgba)
        .unwrap_or_else(|| [128u8, 128, 255, 255].repeat((size * size) as usize));
    // DX to GL is one channel flip: the conventions differ only in the sign of
    // green. Doing it here means the shader only ever sees GL.
    if maps.normal_is_dx {
        log::info!("{}: normal map is DirectX convention, flipping green", name);
        for px in normal.chunks_exact_mut(4) {
            px[1] = 255 - px[1];
        }
    }

    Some(TextureSet {
        name,
        albedo_is_linear: is_linear_format(&maps.albedo),
        albedo,
        normal,
        roughness: gray(&maps.roughness, 200),
        occlusion: gray(&maps.occlusion, 255),
        // Mid-grey means "no relief information": every texel claims the same
        // height, so the blend degrades gracefully to a plain mask fade.
        height: gray(&maps.height, 128),
        size,
    })
}

/// Read a single-channel EXR as a greyscale image.
///
/// `image`'s EXR decoder requires RGB channels and errors with "image does not
/// contain non-deep rgb channels" on a one-channel file, which is what a roughness or
/// occlusion map from Poly Haven actually is. This reads whatever the first channel
/// happens to be called -- `Y`, `R`, `A`, anything -- because for a mask the channel
/// name carries no meaning worth honouring.
///
/// Values are linear floats and are clamped to 0..1 on the way to 8 bits: a mask
/// outside that range has no defined meaning, and the alternative is a wrapped byte.
fn read_gray_exr(path: &Path) -> Option<image::DynamicImage> {
    if !is_linear_format(path) {
        return None;
    }
    let img = exr::prelude::read_first_flat_layer_from_file(path).ok()?;
    let size = img.layer_data.size;
    let (w, h) = (size.0 as u32, size.1 as u32);
    if w == 0 || h == 0 {
        return None;
    }
    let channel = img.layer_data.channel_data.list.first()?;
    let mut out = Vec::with_capacity((w * h) as usize);
    match &channel.sample_data {
        exr::prelude::FlatSamples::F32(v) => {
            out.extend(v.iter().map(|s| (s.clamp(0.0, 1.0) * 255.0).round() as u8))
        }
        exr::prelude::FlatSamples::F16(v) => {
            out.extend(v.iter().map(|s| (s.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8))
        }
        exr::prelude::FlatSamples::U32(v) => out.extend(v.iter().map(|s| (*s).min(255) as u8)),
    }
    if out.len() < (w * h) as usize {
        return None;
    }
    out.truncate((w * h) as usize);
    Some(image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, out)?))
}

/// Fingerprint of the folder's contents, used to decide whether the decoded
/// cache is still good. Paths, sizes and modification times -- enough to catch
/// a replaced or re-downloaded map without hashing hundreds of megabytes.
pub fn fingerprint(dirs: &[PathBuf], size: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    eat(&size.to_le_bytes());
    for d in dirs {
        eat(d.to_string_lossy().as_bytes());
        let Ok(entries) = std::fs::read_dir(d) else { continue };
        let mut files: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        files.sort();
        for f in files {
            eat(f.to_string_lossy().as_bytes());
            if let Ok(m) = std::fs::metadata(&f) {
                eat(&m.len().to_le_bytes());
                if let Ok(t) = m.modified().and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)
                }) {
                    eat(&t.as_secs().to_le_bytes());
                }
            }
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"not really a png").unwrap();
    }

    /// A scratch directory of its own per test, so they can run in parallel.
    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("terra-tex-{tag}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The ambientCG layout: maps loose in a folder named after the set.
    fn material_folder(parent: &Path, name: &str) -> PathBuf {
        let d = parent.join(name);
        std::fs::create_dir_all(&d).unwrap();
        for f in ["Color.png", "NormalGL.png", "Roughness.png", "Displacement.png"] {
            touch(&d, f);
        }
        d
    }

    // --- normal map convention ------------------------------------------------

    #[test]
    fn a_dx_only_set_is_taken_as_dx_rather_than_mistaken_for_gl() {
        // The bug this guards: "normaldx".contains("normal") is true, so the
        // plain-`normal` fallback used to swallow a DX map and hand it to a
        // shader expecting GL -- inverting the shading on every slope.
        let tmp = scratch("dxonly");
        touch(&tmp, "Rock_Color.png");
        touch(&tmp, "Rock_NormalDX.png");
        let m = find_maps(&tmp).expect("albedo found");
        assert!(m.normal.as_ref().unwrap().to_string_lossy().contains("NormalDX"));
        assert!(m.normal_is_dx, "a DX map has to be flagged so green gets flipped");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn gl_is_preferred_and_not_flagged_when_a_set_ships_both() {
        let tmp = scratch("bothconv");
        touch(&tmp, "Rock_Color.png");
        touch(&tmp, "Rock_NormalDX.png");
        touch(&tmp, "Rock_NormalGL.png");
        let m = find_maps(&tmp).expect("albedo found");
        assert!(m.normal.as_ref().unwrap().to_string_lossy().contains("NormalGL"));
        assert!(!m.normal_is_dx);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_plain_normal_map_is_still_treated_as_gl() {
        let tmp = scratch("plainnorm");
        touch(&tmp, "Rock_Color.png");
        touch(&tmp, "Rock_Normal.png");
        let m = find_maps(&tmp).expect("albedo found");
        assert!(m.normal.is_some());
        assert!(!m.normal_is_dx, "an unqualified normal map is the GL convention");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    // --- rejection reasons ----------------------------------------------------

    #[test]
    fn a_folder_with_no_colour_map_says_what_is_missing() {
        let tmp = scratch("noalbedo");
        touch(&tmp, "Rock_NormalGL.png");
        touch(&tmp, "readme.txt");
        let why = reject_reason(&tmp).expect("not a material");
        assert!(why.contains("no colour map"), "{why}");
        // The message has to name what to rename a file to, or it is not
        // actionable -- this is the failure the user actually hit.
        assert!(why.contains("albedo") && why.contains("basecolor"), "{why}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn an_undecodable_colour_map_is_reported_as_a_format_problem() {
        // TGA and EXR are everywhere in texture packs and neither decodes here,
        // so "no colour map" would be actively misleading: the naming is right.
        let tmp = scratch("tga");
        touch(&tmp, "Rock_Color.tga");
        let why = reject_reason(&tmp).expect("not a material");
        assert!(why.contains("not a PNG or JPEG"), "{why}");
        assert!(why.contains("rock_color.tga"), "{why}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_real_material_has_no_rejection_reason() {
        let tmp = scratch("ok");
        let d = material_folder(&tmp, "Ground024");
        assert!(reject_reason(&d).is_none());
        assert!(is_material(&d));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    // --- install --------------------------------------------------------------

    #[test]
    fn installing_a_material_folder_copies_its_maps() {
        let tmp = scratch("inst-one");
        let src = material_folder(&tmp, "Ground024");
        touch(&src, "licence.txt");
        let dest = tmp.join("dest");

        let out = install(&src, &dest).unwrap();
        assert_eq!(out.materials, vec!["Ground024"]);
        assert!(out.rejected.is_empty());

        // And it is a material where it landed, which is the whole point.
        assert!(is_material(&dest.join("Ground024")));
        assert!(dest.join("Ground024/Color.png").exists());
        // Non-map files are left behind rather than dragged in.
        assert!(!dest.join("Ground024/licence.txt").exists());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn installing_a_pack_of_folders_takes_every_material_in_it() {
        // Multi-set downloads arrive as one folder of folders, and picking the
        // pack has to work as well as picking a single set.
        let tmp = scratch("inst-pack");
        let pack = tmp.join("TexturePack");
        std::fs::create_dir_all(&pack).unwrap();
        material_folder(&pack, "Grass001");
        material_folder(&pack, "Rock042");
        // One dud, to prove the good ones still land.
        let dud = pack.join("Notes");
        std::fs::create_dir_all(&dud).unwrap();
        touch(&dud, "todo.txt");
        let dest = tmp.join("dest");

        let out = install(&pack, &dest).unwrap();
        assert_eq!(out.materials, vec!["Grass001", "Rock042"]);
        assert_eq!(out.rejected.len(), 1);
        assert_eq!(out.rejected[0].0, "Notes");
        assert!(is_material(&dest.join("Grass001")));
        assert!(is_material(&dest.join("Rock042")));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn installing_the_same_set_twice_does_not_overwrite_the_first() {
        // Overwriting would silently change what is already painted onto the
        // terrain, since layers are addressed by palette index.
        let tmp = scratch("inst-dup");
        let src = material_folder(&tmp, "Ground024");
        let dest = tmp.join("dest");

        assert_eq!(install(&src, &dest).unwrap().materials, vec!["Ground024"]);
        assert_eq!(install(&src, &dest).unwrap().materials, vec!["Ground024_2"]);
        assert_eq!(discover(&dest).len(), 2);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn installing_a_folder_of_loose_images_that_is_not_a_material_is_reported() {
        let tmp = scratch("inst-bad");
        let src = tmp.join("Screenshots");
        std::fs::create_dir_all(&src).unwrap();
        touch(&src, "shot1.png");
        let dest = tmp.join("dest");

        let out = install(&src, &dest).unwrap();
        assert!(out.materials.is_empty());
        assert_eq!(out.rejected.len(), 1);
        assert!(out.rejected[0].1.contains("no colour map"));
        // Nothing half-copied: a rejected import leaves the palette untouched.
        assert_eq!(discover(&dest).len(), 0);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn an_installed_folder_is_discovered_as_a_material() {
        // The end-to-end property that was broken: import, then discover finds
        // it. Previously the import wrote loose files and discover ignored them.
        let tmp = scratch("inst-e2e");
        let src = material_folder(&tmp, "Ground024");
        let dest = tmp.join("textures");
        install(&src, &dest).unwrap();

        let found = discover(&dest);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].file_name().unwrap(), "Ground024");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn role_assignment_does_not_depend_on_directory_order() {
        // Two files match "color"; whichever sorts first has to win every time,
        // or the palette differs between runs on the same folder.
        let tmp = scratch("order");
        touch(&tmp, "Color.png");
        touch(&tmp, "Color_Opacity.png");
        let first = find_maps(&tmp).unwrap().albedo;
        for _ in 0..8 {
            assert_eq!(find_maps(&tmp).unwrap().albedo, first);
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    // --- naming and completeness ---

    #[test]
    fn a_poly_haven_pack_is_named_after_its_maps_not_its_folder() {
        // The real case that produced a rock set carpeting every flat field: Poly Haven
        // serves `rocky_terrain_03/textures/rocky_terrain_03_diff_4k.jpg`, so importing
        // it named the material "textures" -- which matches no role keyword and lands
        // in the soil base coat.
        let tmp = scratch("polyhaven-pack");
        let inner = tmp.join("textures");
        std::fs::create_dir_all(&inner).unwrap();
        for f in [
            "rocky_terrain_03_diff_4k.jpg",
            "rocky_terrain_03_disp_4k.png",
            "rocky_terrain_03_spec_4k.png",
        ] {
            touch(&inner, f);
        }
        assert_eq!(material_name(&inner), "rocky_terrain_03");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_real_folder_name_is_kept_as_it_is() {
        // Only the generic ones are overridden. Renaming `Ground024` would throw away
        // the one piece of information the user actually chose.
        let tmp = scratch("keepname");
        let d = material_folder(&tmp, "Ground024");
        assert_eq!(material_name(&d), "Ground024");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn installing_a_generic_folder_gives_the_material_a_useful_role() {
        // End to end: the derived name is what lands on disk, so `material::role_of`
        // sees "rocky_terrain_03" and puts it on cliffs instead of on the flats.
        let tmp = scratch("install-generic");
        let inner = tmp.join("textures");
        std::fs::create_dir_all(&inner).unwrap();
        touch(&inner, "rocky_terrain_03_diff_4k.jpg");
        touch(&inner, "rocky_terrain_03_nor_gl_4k.png");
        let dest = tmp.join("dest");

        let out = install(&inner, &dest).unwrap();
        assert_eq!(out.materials, vec!["rocky_terrain_03"]);
        assert!(dest.join("rocky_terrain_03").is_dir());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_set_with_no_normal_map_is_reported_as_incomplete() {
        // The single most consequential missing map: with a flat normal there is no
        // relief for the light to catch, so the material reads as an image laid over
        // the ground however good its albedo is.
        let tmp = scratch("nonormal");
        let d = tmp.join("rocky_terrain_03");
        std::fs::create_dir_all(&d).unwrap();
        touch(&d, "rocky_terrain_03_diff_4k.jpg");
        touch(&d, "rocky_terrain_03_disp_4k.png");
        touch(&d, "rocky_terrain_03_spec_4k.png");

        let missing = missing_maps(&d);
        assert_eq!(missing.first(), Some(&"normal"), "normal has to lead: {missing:?}");
        assert!(missing.contains(&"roughness"));
        // The height map is present, so it must not be listed.
        assert!(!missing.contains(&"height"), "{missing:?}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn install_surfaces_the_missing_maps() {
        let tmp = scratch("install-incomplete");
        let d = tmp.join("rocky_terrain_03");
        std::fs::create_dir_all(&d).unwrap();
        touch(&d, "rocky_terrain_03_diff_4k.jpg");
        let dest = tmp.join("dest");

        let out = install(&d, &dest).unwrap();
        assert_eq!(out.materials, vec!["rocky_terrain_03"]);
        assert_eq!(out.incomplete.len(), 1, "an albedo-only set is incomplete");
        assert!(out.incomplete[0].1.contains(&"normal"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_complete_set_is_not_reported_as_incomplete() {
        let tmp = scratch("complete");
        let d = tmp.join("Ground024");
        std::fs::create_dir_all(&d).unwrap();
        for f in [
            "Color.png",
            "NormalGL.png",
            "Roughness.png",
            "Displacement.png",
            "AmbientOcclusion.png",
        ] {
            touch(&d, f);
        }
        assert!(missing_maps(&d).is_empty(), "{:?}", missing_maps(&d));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn the_role_suffix_is_stripped_from_a_derived_name() {
        assert_eq!(asset_stem("rocky_terrain_03_diff_4k"), "rocky_terrain_03");
        assert_eq!(asset_stem("Ground024_2K-PNG_Color"), "Ground024_2K-PNG");
        assert_eq!(asset_stem("rock_nor_gl_2k"), "rock");
        // Nothing to strip leaves it alone.
        assert_eq!(asset_stem("mystery"), "mystery");
    }

    #[test]
    fn recognises_ambientcg_naming() {
        let tmp = std::env::temp_dir().join("terra-maps-acg");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        for f in [
            "Ground024_2K-PNG_Color.png",
            "Ground024_2K-PNG_NormalGL.png",
            "Ground024_2K-PNG_NormalDX.png",
            "Ground024_2K-PNG_Roughness.png",
            "Ground024_2K-PNG_AmbientOcclusion.png",
            "Ground024_2K-PNG_Displacement.png",
        ] {
            touch(&tmp, f);
        }
        let m = find_maps(&tmp).expect("albedo found");
        assert!(m.albedo.to_string_lossy().contains("Color"));
        // GL, not DX: picking the wrong one inverts every slope.
        assert!(m.normal.unwrap().to_string_lossy().contains("NormalGL"));
        assert!(m.roughness.unwrap().to_string_lossy().contains("Roughness"));
        assert!(m.occlusion.unwrap().to_string_lossy().contains("AmbientOcclusion"));
        assert!(m.height.unwrap().to_string_lossy().contains("Displacement"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn recognises_polyhaven_naming() {
        let tmp = std::env::temp_dir().join("terra-maps-ph");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        for f in ["rock_diff_2k.png", "rock_nor_gl_2k.png", "rock_rough_2k.png", "rock_disp_2k.png"]
        {
            touch(&tmp, f);
        }
        let m = find_maps(&tmp).expect("albedo found");
        assert!(m.albedo.to_string_lossy().contains("diff"));
        assert!(m.normal.unwrap().to_string_lossy().contains("nor_gl"));
        assert!(m.height.unwrap().to_string_lossy().contains("disp"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_folder_without_an_albedo_is_not_a_material() {
        let tmp = std::env::temp_dir().join("terra-maps-empty");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        touch(&tmp, "readme.txt");
        touch(&tmp, "preview_nor_gl.png");
        assert!(find_maps(&tmp).is_none());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn discovery_skips_dot_folders_and_sorts() {
        let tmp = std::env::temp_dir().join("terra-maps-scan");
        let _ = std::fs::remove_dir_all(&tmp);
        for sub in ["Zinc", "Alpha", ".cache"] {
            std::fs::create_dir_all(tmp.join(sub)).unwrap();
            touch(&tmp.join(sub), "Color.png");
        }
        let found = discover(&tmp);
        let names: Vec<_> =
            found.iter().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert_eq!(names, vec!["Alpha", "Zinc"]);
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}

//! Reading downloaded PBR texture sets off disk.
//!
//! One subfolder of `assets/texture/` is one material. Whatever is in that
//! folder when the editor starts is what appears in the palette -- there is no
//! registry to edit and no id to allocate, because the thing an artist actually
//! does is drop a folder in and expect it to show up.
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

use image::imageops::FilterType;
use std::path::{Path, PathBuf};

/// The maps making up one material, already decoded and resized to a square.
pub struct TextureSet {
    pub name: String,
    /// sRGB-encoded, `size * size * 4`.
    pub albedo: Vec<u8>,
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
    roughness: Option<PathBuf>,
    occlusion: Option<PathBuf>,
    height: Option<PathBuf>,
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
            if !hidden && find_maps(&path).is_some() {
                found.push(path);
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
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| {
                let x = x.to_string_lossy().to_lowercase();
                x == "png" || x == "jpg" || x == "jpeg"
            })
        })
        .collect();

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
    let normal =
        pick(&["normalgl", "normal_gl", "_nor_gl"]).or_else(|| pick(&["normal", "_nor_", "nrm"]));

    Some(Maps {
        albedo: pick(&["color", "albedo", "basecolor", "base_color", "_diff", "diffuse"])?,
        normal,
        roughness: pick(&["rough"]),
        occlusion: pick(&["ambientocclusion", "occlusion", "_ao", "ambient_occlusion"]),
        height: pick(&["displacement", "height", "_disp", "bump"]),
    })
}

/// Decode one material folder down to `size`.
///
/// Downsampling here rather than at 2K is deliberate: the array is sampled
/// triplanar, three fetches per layer, so a 2K set would cost VRAM and
/// bandwidth for detail that is gone by the second mip anyway.
pub fn load(dir: &Path, size: u32) -> Option<TextureSet> {
    let maps = find_maps(dir)?;
    let name = dir.file_name()?.to_string_lossy().to_string();

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
        match image::open(p) {
            Ok(img) => img.resize_exact(size, size, FilterType::Lanczos3).to_luma8().into_raw(),
            Err(e) => {
                log::error!("{}: {e}", p.display());
                vec![default; (size * size) as usize]
            }
        }
    };

    let albedo = rgba(&maps.albedo)?;
    // A flat normal is (0.5, 0.5, 1.0) -- pointing straight out of the surface.
    let normal = maps
        .normal
        .as_deref()
        .and_then(rgba)
        .unwrap_or_else(|| [128u8, 128, 255, 255].repeat((size * size) as usize));

    Some(TextureSet {
        name,
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

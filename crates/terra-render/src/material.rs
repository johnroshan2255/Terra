//! Terrain surface materials.
//!
//! A layered stack in the style every realistic terrain renderer uses: a base
//! layer of soil covering the whole map, and further layers -- grass, rock,
//! gravel, snow, mud -- laid over it and revealed by masks. What makes the
//! result read as ground rather than as tinted geometry is that each layer
//! carries a *height* channel, so where two layers meet the one standing
//! proud wins per-texel instead of the two cross-fading. Grass then grows
//! through the dirt in clumps along a transition, which is what a linear blend
//! can never produce.
//!
//! Content comes from `assets/texture/`: one subfolder per material, discovered
//! at startup (see [`crate::texture_set`]). Nothing is registered by hand -- drop
//! a downloaded set in the folder and it is in the palette next run. When the
//! folder is empty the layers are generated from noise instead, so a fresh
//! clone with no downloads still renders something.
//!
//! Layout, per layer, as two array slices:
//!
//!   albedo   rgba8-srgb   rgb = albedo          a = height
//!   surface  rgba8-unorm  rg  = normal xy       b = roughness   a = ambient occlusion
//!
//! Height lives in the albedo alpha because sRGB formats leave alpha linear,
//! so it survives the transfer function untouched and costs no extra fetch.

use crate::texture_set::{self, TextureSet};
use rayon::prelude::*;
use std::path::Path;

/// Side length of one material tile, in texels.
///
/// 512 is enough for the grain to hold up at the near clip and small enough
/// that the whole set plus mips is ~17 MB.
pub const TILE: u32 = 512;

/// Built-in layer slots, used by the generated fallback and by the automatic
/// (slope- and erosion-driven) weights when no set has claimed a role.
pub const SOIL: u32 = 0;
pub const GRASS: u32 = 1;
pub const ROCK: u32 = 2;
pub const GRAVEL: u32 = 3;
pub const SNOW: u32 = 4;
pub const MUD: u32 = 5;
pub const LAYER_COUNT: u32 = 6;

/// Hard ceiling on palette size. The shader carries one weight per slot in a
/// fixed-size array and packs the painted weights into two RGBA8 textures, so
/// this is the number those two facts allow.
pub const MAX_LAYERS: u32 = 8;

/// Anisotropy for the albedo array.
///
/// The single most expensive knob in the terrain shader: each step multiplies
/// the work of every albedo fetch, and there are six of them per fragment.
/// Measured on an M4 at 1600x900, 8 costs about 1.9 ms a frame over 1, and 4
/// keeps almost all of the visible sharpness for half of that.
pub const MATERIAL_ANISOTROPY: u16 = 4;

/// Edge of one material thumbnail shown in the palette.
pub const THUMB: u32 = 64;

/// Role index meaning "this layer is never placed automatically".
pub const ROLE_NONE: u32 = 6;

/// Where a material goes when nobody has painted.
///
/// The automatic weights are written in terms of ground behaviour -- steep
/// faces get rock, silted flats get grass -- but a folder called `Ground087`
/// says nothing about that. The role is guessed from the folder name, which is
/// crude and works, because texture sites name their sets after what they are.
/// Painting overrides it regardless, so a wrong guess costs a brush stroke.
pub fn role_of(name: &str) -> u32 {
    let n = name.to_lowercase();
    let has = |keys: &[&str]| keys.iter().any(|k| n.contains(k));
    if has(&["grass", "meadow", "lawn", "moss"]) {
        GRASS
    } else if has(&["rock", "cliff", "stone", "granite", "slate"]) {
        ROCK
    } else if has(&["gravel", "pebble", "scree", "shingle"]) {
        GRAVEL
    } else if has(&["snow", "ice"]) {
        SNOW
    } else if has(&["mud", "road", "track", "asphalt", "dirt_road"]) {
        MUD
    } else {
        // ground, soil, dirt, sand, and anything unrecognised: the base coat.
        SOIL
    }
}

/// What the UI needs to show a palette entry.
#[derive(Clone)]
pub struct LayerInfo {
    pub name: String,
    /// RGBA, `THUMB * THUMB * 4`.
    pub thumbnail: Vec<u8>,
    /// Role this layer fills automatically, or [`ROLE_NONE`].
    pub role: u32,
    /// Mean linear albedo of the whole texture.
    ///
    /// Taken from the 1x1 mip, which *is* the average by construction. Grass
    /// blades tint toward the ground they stand on as they dissolve, and this
    /// is the colour they have to match.
    pub average_color: [f32; 3],
}

/// Human-readable role, for the palette.
pub fn role_label(role: u32) -> &'static str {
    match role {
        SOIL => "soil",
        GRASS => "grass",
        ROCK => "rock",
        GRAVEL => "gravel",
        SNOW => "snow",
        MUD => "track",
        _ => "paint only",
    }
}

/// Assign each layer its automatic role, giving a role to the first layer that
/// claims it and marking later duplicates paint-only.
///
/// Two grass sets both placed automatically would fight over the same ground
/// and neither would win predictably; the first is the default and the second
/// is a brush.
fn assign_roles(names: &[String]) -> Vec<u32> {
    let mut taken = [false; ROLE_NONE as usize];
    names
        .iter()
        .map(|n| {
            let r = role_of(n);
            if taken[r as usize] {
                ROLE_NONE
            } else {
                taken[r as usize] = true;
                r
            }
        })
        .collect()
}

/// Generated material set, shared by every terrain in the session.
///
/// One instance is built at startup and handed to each [`crate::terrain::Terrain`];
/// the textures are read-only, so the menu backdrop and the open world sample
/// the same memory.
pub struct Materials {
    pub bind_group: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
    /// Palette entries, parallel to the texture array layers.
    pub layers: Vec<LayerInfo>,
}

impl Materials {
    pub fn count(&self) -> u32 {
        self.layers.len() as u32
    }
}

impl Materials {
    /// Build the palette from `dir`, falling back to generated layers when it
    /// holds no material folders.
    ///
    /// Decoding six 2K sets is seconds of work, so the resized result is cached
    /// under `<dir>/.cache` and keyed by a fingerprint of the source files.
    /// Replacing a map invalidates it; a normal startup does not pay for it
    /// twice.
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue, dir: &Path) -> Self {
        let dirs = texture_set::discover(dir);
        let (baked, names) = if dirs.is_empty() {
            log::info!("{}: no material folders found, using generated layers", dir.display());
            (generated_layers(), generated_names())
        } else {
            let taken = dirs.len().min(MAX_LAYERS as usize);
            if dirs.len() > taken {
                log::warn!(
                    "{} material folders found but only {taken} slots; ignoring the rest",
                    dirs.len()
                );
            }
            let dirs = &dirs[..taken];
            let names: Vec<String> = dirs
                .iter()
                .map(|d| d.file_name().unwrap_or_default().to_string_lossy().to_string())
                .collect();
            (load_or_bake(dir, dirs), names)
        };

        let roles = assign_roles(&names);
        for (n, r) in names.iter().zip(&roles) {
            log::info!("material '{n}' -> {}", role_label(*r));
        }
        let layers: Vec<LayerInfo> = names
            .into_iter()
            .zip(roles)
            .zip(baked.iter())
            .map(|((name, role), b)| LayerInfo {
                name,
                thumbnail: b.thumbnail(),
                role,
                average_color: b.average_color(),
            })
            .collect();

        Self::from_baked(device, queue, baked, layers)
    }

    fn from_baked(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        baked: Vec<Baked>,
        layers: Vec<LayerInfo>,
    ) -> Self {
        let mips = TILE.ilog2() + 1;
        let count = baked.len().max(1) as u32;

        let albedo = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material-albedo"),
            size: wgpu::Extent3d { width: TILE, height: TILE, depth_or_array_layers: count },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let surface = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material-surface"),
            size: wgpu::Extent3d { width: TILE, height: TILE, depth_or_array_layers: count },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer, b) in baked.iter().enumerate() {
            b.upload(queue, &albedo, &surface, layer as u32);
        }

        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let surface_view = surface.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Two samplers, because the two arrays do not deserve the same budget.
        //
        // Terrain is viewed at grazing angles almost everywhere, and trilinear
        // alone turns the ground to mush a few hundred metres out -- so albedo
        // gets anisotropy. The surface array is normals, roughness and
        // occlusion, none of which the eye tracks the way it tracks colour, and
        // it is fetched just as often. Giving it the same anisotropy doubled
        // the most expensive thing this shader does, for detail nobody sees.
        let aniso = |label, clamp| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: clamp,
                ..Default::default()
            })
        };
        let sampler = aniso("material-sampler", MATERIAL_ANISOTROPY);
        let sampler_fast = aniso("material-sampler-fast", 1);

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&surface_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler_fast),
                },
            ],
        });

        Self { bind_group, layout, layers }
    }
}

// ---------------------------------------------------------------------------
// Assembling layers
// ---------------------------------------------------------------------------

fn generated_layers() -> Vec<Baked> {
    (0..LAYER_COUNT).into_par_iter().map(bake_layer).collect()
}

fn generated_names() -> Vec<String> {
    ["Soil", "Grass", "Rock", "Gravel", "Snow", "Mud"].iter().map(|s| s.to_string()).collect()
}

/// Decoded layers for `dirs`, from the cache when it is still valid.
fn load_or_bake(root: &Path, dirs: &[std::path::PathBuf]) -> Vec<Baked> {
    let key = texture_set::fingerprint(dirs, TILE);
    let cache = root.join(".cache").join(format!("materials-{key:016x}.bin"));

    if let Some(v) = read_cache(&cache, dirs.len()) {
        log::info!("materials: {} layers from cache", v.len());
        return v;
    }

    let t0 = std::time::Instant::now();
    let baked: Vec<Baked> = dirs
        .par_iter()
        .map(|d| match texture_set::load(d, TILE) {
            Some(set) => bake_set(&set),
            None => {
                log::error!("{}: could not read texture set, substituting soil", d.display());
                bake_layer(SOIL)
            }
        })
        .collect();
    log::info!("materials: decoded {} sets in {:.1} s", baked.len(), t0.elapsed().as_secs_f32());

    if let Err(e) = write_cache(&cache, &baked) {
        // Not fatal: it only means the next start pays for decoding again.
        log::warn!("{}: could not write material cache: {e}", cache.display());
    }
    baked
}

/// Pack one decoded set into the two-array layout the shader samples.
fn bake_set(set: &TextureSet) -> Baked {
    let n = set.size as usize;
    let mut rgba = vec![0.0f32; n * n * 4];
    let mut surf = vec![0.0f32; n * n * 4];

    for i in 0..n * n {
        // Albedo arrives sRGB-encoded and is re-encoded on the way out, so it
        // is decoded to linear here for the mip filtering in between. Box
        // filtering gamma-encoded values darkens every downsample.
        for k in 0..3 {
            rgba[i * 4 + k] = srgb_to_linear(set.albedo[i * 4 + k] as f32 / 255.0);
        }
        rgba[i * 4 + 3] = set.height[i] as f32 / 255.0;

        surf[i * 4] = set.normal[i * 4] as f32 / 255.0;
        surf[i * 4 + 1] = set.normal[i * 4 + 1] as f32 / 255.0;
        surf[i * 4 + 2] = set.roughness[i] as f32 / 255.0;
        surf[i * 4 + 3] = set.occlusion[i] as f32 / 255.0;
    }

    mip_chain(rgba, surf, n)
}

// ---------------------------------------------------------------------------
// Generated fallback
// ---------------------------------------------------------------------------

/// One layer's full mip chain, ready to upload.
struct Baked {
    /// `(width, albedo bytes, surface bytes)` per mip level, level 0 first.
    levels: Vec<(u32, Vec<u8>, Vec<u8>)>,
}

impl Baked {
    fn upload(
        &self,
        queue: &wgpu::Queue,
        albedo: &wgpu::Texture,
        surface: &wgpu::Texture,
        layer: u32,
    ) {
        for (mip, (w, a, s)) in self.levels.iter().enumerate() {
            let size = wgpu::Extent3d { width: *w, height: *w, depth_or_array_layers: 1 };
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(*w),
            };
            fn target(tex: &wgpu::Texture, mip: u32, layer: u32) -> wgpu::TexelCopyTextureInfo<'_> {
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: mip,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                }
            }
            queue.write_texture(target(albedo, mip as u32, layer), a, layout, size);
            queue.write_texture(target(surface, mip as u32, layer), s, layout, size);
        }
    }
}

/// A layer as a set of dials rather than a bespoke function.
///
/// `relief` is the one that matters most: it is the amplitude of the height
/// channel, and therefore how aggressively this layer punches through its
/// neighbour at a transition. Grass sits high so it grows through soil in
/// clumps; snow sits low so it fills in around what it settles on.
struct Recipe {
    /// Linear albedo, and the second colour the macro variation mixes toward.
    base: [f32; 3],
    alt: [f32; 3],
    /// Tiles-per-texture of the coarse colour/height variation.
    clump: i32,
    /// Tiles-per-texture of the fine grain.
    grain: i32,
    relief: f32,
    /// How strongly the height channel darkens the albedo in its own crevices.
    /// Cheap baked occlusion, and most of what stops a texture looking printed.
    cavity: f32,
    roughness: f32,
    /// Multiplier on the derived normal. Higher reads as coarser material.
    bump: f32,
}

fn recipe(layer: u32) -> Recipe {
    match layer {
        // Fine, fairly uniform earth. Low relief: it is the thing other layers
        // break through, so it must not fight them.
        SOIL => Recipe {
            base: [0.135, 0.098, 0.062],
            alt: [0.088, 0.064, 0.042],
            clump: 5,
            grain: 44,
            relief: 0.34,
            cavity: 0.45,
            roughness: 0.92,
            bump: 1.0,
        },
        // Clumpy and high-relief. This is the layer the whole height-blend
        // exists to serve.
        GRASS => Recipe {
            base: [0.105, 0.150, 0.062],
            alt: [0.062, 0.092, 0.036],
            clump: 9,
            grain: 60,
            relief: 0.95,
            cavity: 0.70,
            roughness: 0.85,
            bump: 1.5,
        },
        // Cracked and directional; the ridged basis gives fractures rather
        // than lumps.
        ROCK => Recipe {
            base: [0.150, 0.147, 0.140],
            alt: [0.088, 0.086, 0.082],
            clump: 3,
            grain: 26,
            relief: 0.85,
            cavity: 0.65,
            roughness: 0.78,
            bump: 1.8,
        },
        // Loose stones: many small, high-contrast lumps.
        GRAVEL => Recipe {
            base: [0.205, 0.192, 0.172],
            alt: [0.120, 0.112, 0.100],
            clump: 7,
            grain: 34,
            relief: 0.80,
            cavity: 0.60,
            roughness: 0.70,
            bump: 1.6,
        },
        // Nearly smooth, and bright enough that any relief reads as shadow.
        SNOW => Recipe {
            base: [0.780, 0.810, 0.860],
            alt: [0.700, 0.740, 0.820],
            clump: 4,
            grain: 20,
            relief: 0.22,
            cavity: 0.20,
            roughness: 0.35,
            bump: 0.7,
        },
        // Wet earth, smoothed by traffic. Low roughness is what makes a track
        // catch the sun the way packed mud does.
        _ => Recipe {
            base: [0.098, 0.076, 0.052],
            alt: [0.062, 0.048, 0.034],
            clump: 6,
            grain: 38,
            relief: 0.40,
            cavity: 0.50,
            roughness: 0.42,
            bump: 0.9,
        },
    }
}

fn bake_layer(layer: u32) -> Baked {
    let r = recipe(layer);
    let n = TILE as usize;
    let seed = layer * 7919 + 13;

    // Height first and on its own: the albedo and the normal are both derived
    // from it, which is what keeps the crevices, the shading and the blend
    // mask agreeing with each other.
    let height: Vec<f32> = (0..n * n)
        .into_par_iter()
        .map(|i| {
            let u = (i % n) as f32 / n as f32;
            let v = (i / n) as f32 / n as f32;
            layer_height(layer, u, v, &r, seed)
        })
        .collect();

    // Linear albedo and the packed surface data at full resolution. Mips are
    // box-filtered from these floats rather than from the encoded bytes, so
    // the sRGB curve is applied once, at the end, per level.
    let mut rgba = vec![0.0f32; n * n * 4];
    let mut surf = vec![0.0f32; n * n * 4];
    for i in 0..n * n {
        let x = i % n;
        let z = i / n;
        let h = height[i];

        let u = x as f32 / n as f32;
        let v = z as f32 / n as f32;
        // Macro tint, so a tile is not one flat colour at distance.
        let t = fbm(u, v, r.clump, 3, seed ^ 0x51ED).clamp(0.0, 1.0);
        let mut c = [0.0f32; 3];
        for (k, ch) in c.iter_mut().enumerate() {
            *ch = r.base[k] + (r.alt[k] - r.base[k]) * t;
        }
        // Bake occlusion from the layer's own height. Pits go dark; tops stay.
        let cavity = 1.0 - r.cavity * (1.0 - h);
        let ao = 0.55 + 0.45 * h;

        // Slope of the height field, in texels, becomes the tangent normal.
        let hx = height[z * n + (x + 1) % n] - height[z * n + (x + n - 1) % n];
        let hz = height[((z + 1) % n) * n + x] - height[((z + n - 1) % n) * n + x];
        let nx = -hx * r.bump * 4.0;
        let nz = -hz * r.bump * 4.0;

        rgba[i * 4] = c[0] * cavity;
        rgba[i * 4 + 1] = c[1] * cavity;
        rgba[i * 4 + 2] = c[2] * cavity;
        rgba[i * 4 + 3] = h;

        surf[i * 4] = nx * 0.5 + 0.5;
        surf[i * 4 + 1] = nz * 0.5 + 0.5;
        surf[i * 4 + 2] = (r.roughness * (0.85 + 0.3 * (1.0 - h))).clamp(0.0, 1.0);
        surf[i * 4 + 3] = ao;
    }

    mip_chain(rgba, surf, n)
}

/// The per-layer height field, 0..1. Each material gets the basis whose shape
/// matches how it actually sits on the ground.
fn layer_height(layer: u32, u: f32, v: f32, r: &Recipe, seed: u32) -> f32 {
    let coarse = fbm(u, v, r.clump, 4, seed);
    let fine = fbm(u, v, r.grain, 3, seed ^ 0xBEEF);

    let h = match layer {
        // Clumps with blade-scale break-up, then a curve that lifts the tufts
        // and keeps the gaps low -- exactly the profile the blend needs.
        GRASS => {
            let clumps = (coarse * 1.25 - 0.12).clamp(0.0, 1.0);
            (clumps * 0.75 + fine * 0.25).powf(0.65)
        }
        // Ridged: the creases become fractures rather than dents.
        ROCK => {
            let ridged = 1.0 - (coarse * 2.0 - 1.0).abs();
            (ridged * 0.7 + fine * 0.3).powf(1.4)
        }
        // Many small stones: square the fine octave to separate them.
        GRAVEL => {
            let stones = fine * fine * 1.6;
            (coarse * 0.35 + stones * 0.65).clamp(0.0, 1.0)
        }
        // Wind-smoothed, with the fine octave almost absent.
        SNOW => coarse * 0.85 + fine * 0.15,
        // Earth: mostly grain, with a little large-scale unevenness, plus
        // sparse grit sitting on top.
        _ => {
            let grit = (fine - 0.72).max(0.0) * 3.0;
            (coarse * 0.4 + fine * 0.6 + grit * 0.25).clamp(0.0, 1.0)
        }
    };

    // `relief` is applied as a contrast curve about the midpoint, so a low
    // relief layer flattens toward 0.5 instead of toward black.
    (0.5 + (h - 0.5) * r.relief).clamp(0.0, 1.0)
}

/// Box-filtered mip chain, encoded per level.
fn mip_chain(rgba: Vec<f32>, surf: Vec<f32>, n: usize) -> Baked {
    let mut levels = Vec::new();
    let (mut w, mut a, mut s) = (n, rgba, surf);
    loop {
        levels.push((w as u32, encode_srgb(&a), encode_unorm(&s)));
        if w == 1 {
            break;
        }
        a = downsample(&a, w);
        s = downsample(&s, w);
        w /= 2;
    }
    Baked { levels }
}

impl Baked {
    /// Mean colour, read straight out of the 1x1 mip.
    fn average_color(&self) -> [f32; 3] {
        let Some((_, albedo, _)) = self.levels.iter().find(|(w, _, _)| *w == 1) else {
            return [0.1, 0.15, 0.06];
        };
        // Stored sRGB-encoded; the shader wants linear.
        let dec = |v: u8| {
            let c = v as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        };
        [dec(albedo[0]), dec(albedo[1]), dec(albedo[2])]
    }

    /// Palette swatch, taken from whichever mip is already close to thumbnail
    /// size so nothing has to be resampled again.
    fn thumbnail(&self) -> Vec<u8> {
        let level = self
            .levels
            .iter()
            .min_by_key(|(w, _, _)| w.abs_diff(THUMB))
            .expect("a chain always has a level");
        let (w, albedo, _) = level;
        let w = *w as usize;
        let t = THUMB as usize;
        let mut out = vec![255u8; t * t * 4];
        for z in 0..t {
            for x in 0..t {
                let sx = x * w / t;
                let sz = z * w / t;
                for k in 0..3 {
                    out[(z * t + x) * 4 + k] = albedo[(sz * w + sx) * 4 + k];
                }
            }
        }
        out
    }
}

/// Cache format: a plain length-prefixed dump. It is derived data keyed by a
/// fingerprint of its inputs, so it needs no version field -- a format change
/// means changing TILE or the packing, both of which change the key.
fn write_cache(path: &Path, baked: &[Baked]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(baked.len() as u32).to_le_bytes());
    for b in baked {
        out.extend_from_slice(&(b.levels.len() as u32).to_le_bytes());
        for (w, a, s) in &b.levels {
            out.extend_from_slice(&w.to_le_bytes());
            out.extend_from_slice(&(a.len() as u32).to_le_bytes());
            out.extend_from_slice(a);
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s);
        }
    }
    std::fs::write(path, out)
}

fn read_cache(path: &Path, expect: usize) -> Option<Vec<Baked>> {
    let bytes = std::fs::read(path).ok()?;
    let mut p = 0usize;
    let u32_at = |p: &mut usize| -> Option<u32> {
        let v = bytes.get(*p..*p + 4)?;
        *p += 4;
        Some(u32::from_le_bytes(v.try_into().ok()?))
    };
    let count = u32_at(&mut p)? as usize;
    if count != expect {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let n = u32_at(&mut p)? as usize;
        let mut levels = Vec::with_capacity(n);
        for _ in 0..n {
            let w = u32_at(&mut p)?;
            let la = u32_at(&mut p)? as usize;
            let a = bytes.get(p..p + la)?.to_vec();
            p += la;
            let ls = u32_at(&mut p)? as usize;
            let sv = bytes.get(p..p + ls)?.to_vec();
            p += ls;
            levels.push((w, a, sv));
        }
        out.push(Baked { levels });
    }
    Some(out)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

// --- noise -----------------------------------------------------------------
//
// Periodic value noise. The lattice wraps at the octave's own period, which is
// what makes the result tile: a texture with a seam is worse than no texture,
// because the seam draws a straight line across a landscape that has none.
//
// This is material grain, not terrain. The heightfield policy in
// `assets/shaders/common/noise.wgsl` -- ridged multifractal, never fBm -- is
// about the landform; it does not govern what a square metre of dirt looks
// like up close.

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x1657_4B0D)
        ^ (y as u32).wrapping_mul(0x27D4_EB2D)
        ^ seed.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    h as f32 / u32::MAX as f32
}

/// Value noise on a lattice of `period` cells across the unit square.
fn vnoise(u: f32, v: f32, period: i32, seed: u32) -> f32 {
    let x = u * period as f32;
    let y = v * period as f32;
    let xi = x.floor();
    let yi = y.floor();
    let fx = x - xi;
    let fy = y - yi;
    // Quintic: C2 continuous, so the derived normal has no cell-edge creases.
    let sx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let sy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    let wrap = |i: i32| i.rem_euclid(period.max(1));
    let x0 = wrap(xi as i32);
    let y0 = wrap(yi as i32);
    let x1 = wrap(xi as i32 + 1);
    let y1 = wrap(yi as i32 + 1);

    let a = hash(x0, y0, seed);
    let b = hash(x1, y0, seed);
    let c = hash(x0, y1, seed);
    let d = hash(x1, y1, seed);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}

/// Octave sum. Both frequency and lattice period double per octave, so every
/// octave tiles and therefore so does the sum.
fn fbm(u: f32, v: f32, period: i32, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let mut p = period.max(1);
    for o in 0..octaves {
        sum += amp * vnoise(u, v, p, seed.wrapping_add(o * 131));
        norm += amp;
        amp *= 0.5;
        p *= 2;
    }
    sum / norm
}

// --- encoding --------------------------------------------------------------

fn downsample(src: &[f32], w: usize) -> Vec<f32> {
    let h = w / 2;
    let mut out = vec![0.0f32; h * h * 4];
    for z in 0..h {
        for x in 0..h {
            for k in 0..4 {
                let s = |dx: usize, dz: usize| src[((z * 2 + dz) * w + x * 2 + dx) * 4 + k];
                out[(z * h + x) * 4 + k] = (s(0, 0) + s(1, 0) + s(0, 1) + s(1, 1)) * 0.25;
            }
        }
    }
    out
}

fn encode_srgb(src: &[f32]) -> Vec<u8> {
    src.iter()
        .enumerate()
        .map(|(i, &c)| {
            // Alpha is the height channel and stays linear -- sRGB formats do
            // not apply the transfer function to it.
            let v = if i % 4 == 3 { c } else { linear_to_srgb(c) };
            (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        })
        .collect()
}

fn encode_unorm(src: &[f32]) -> Vec<u8> {
    src.iter().map(|&c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect()
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.max(0.0).powf(1.0 / 2.4) - 0.055 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_tiles_across_the_seam() {
        // The left and right edges must agree, or every tile boundary draws a
        // line across the terrain.
        for i in 0..64 {
            let v = i as f32 / 64.0;
            let left = fbm(0.0, v, 8, 4, 99);
            let right = fbm(1.0, v, 8, 4, 99);
            assert!((left - right).abs() < 1e-4, "seam at v={v}: {left} vs {right}");
            let top = fbm(v, 0.0, 8, 4, 99);
            let bottom = fbm(v, 1.0, 8, 4, 99);
            assert!((top - bottom).abs() < 1e-4, "seam at u={v}: {top} vs {bottom}");
        }
    }

    #[test]
    fn every_layer_has_relief_to_blend_with() {
        // A layer whose height is constant cannot punch through anything, and
        // the whole stack degrades to a linear cross-fade.
        for layer in 0..LAYER_COUNT {
            let r = recipe(layer);
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for i in 0..4096 {
                let u = (i % 64) as f32 / 64.0;
                let v = (i / 64) as f32 / 64.0;
                let h = layer_height(layer, u, v, &r, layer * 7919 + 13);
                assert!((0.0..=1.0).contains(&h), "layer {layer} height out of range: {h}");
                lo = lo.min(h);
                hi = hi.max(h);
            }
            assert!(hi - lo > 0.1, "layer {layer} is nearly flat: {lo}..{hi}");
        }
    }

    #[test]
    fn grass_stands_prouder_than_soil() {
        // The premise of the stack: at a boundary, grass wins.
        let mean = |layer: u32| {
            let r = recipe(layer);
            let seed = layer * 7919 + 13;
            let n = 64;
            let mut sum = 0.0;
            for i in 0..n * n {
                let u = (i % n) as f32 / n as f32;
                let v = (i / n) as f32 / n as f32;
                sum += layer_height(layer, u, v, &r, seed);
            }
            sum / (n * n) as f32
        };
        assert!(mean(GRASS) > mean(SOIL), "grass must sit above soil to break through it");
    }
}

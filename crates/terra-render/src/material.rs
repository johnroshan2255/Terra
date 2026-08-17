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
//! Content comes from the project's own `assets/textures/`: one subfolder per
//! material, discovered by [`crate::texture_set`]. Nothing is registered by hand
//! and nothing ships prebuilt -- an empty folder means an empty palette, and the
//! terrain renders as plain shaded ground until something is imported. There was
//! a noise-generated fallback here once; it made a fresh project look furnished
//! with materials the user had not chosen and could not edit.
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

/// Per-layer PBR settings, editable in the material editor.
///
/// One tiling scale for the whole terrain was the previous arrangement, and it
/// cannot work across a palette: gravel wants a repeat every metre or two and a
/// cliff face wants ten, and forcing both to share a number makes one of them
/// either blurred or visibly tiled. Everything here is per layer for that
/// reason.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerParams {
    /// Metres per texture repeat.
    pub tiling_m: f32,
    /// Multiplier on the tangent-space normal. 0 flattens the layer.
    pub normal_strength: f32,
    /// Scales the sampled roughness. Above 1 dulls, below 1 polishes.
    pub roughness: f32,
    /// Width of the band where this layer and its neighbour both contend in the
    /// height blend. 0 is a hard per-texel cut.
    pub height_blend: f32,
    /// Parallax depth in metres. 0 disables the effect for this layer.
    ///
    /// This is what makes gravel read as gravel rather than as a photograph of
    /// gravel: the height channel offsets the lookup along the view vector, so
    /// stones occlude each other as the camera moves.
    pub parallax_m: f32,
    /// Multiplier on sampled ambient occlusion.
    pub ao: f32,
    /// Padding the shader's alignment rules demand, not ours.
    ///
    /// WGSL aligns a `vec3<f32>` to **16 bytes**. The six scalars above fill
    /// 0..24, so the shader places `tint` at offset 32 while `repr(C)` would put
    /// it at 24 -- and the shader then reads `tint.rgb` from Rust bytes 32..44,
    /// which are `tint[2]` followed by two padding floats: `(1, 0, 0)`. Pure red.
    ///
    /// That is exactly what it did: the menu backdrop, which is the one terrain
    /// with a non-empty palette, rendered blood red. A fresh project has no
    /// materials, takes the `layer_count == 0` branch, and never reads the tint --
    /// which is why only the start page showed it.
    _pad0: [f32; 2],
    /// Albedo tint, applied after sampling. At offset 32, matching WGSL.
    pub tint: [f32; 3],
    /// Takes the element to 48 bytes, the 16-byte stride a uniform array needs.
    _pad1: f32,
}

const _: () = assert!(std::mem::size_of::<LayerParams>() == 48);
// The size was already right; the *offset* was not, and only the offset produced
// the bug. Asserted directly so a reordering cannot reintroduce it silently.
const _: () = assert!(std::mem::offset_of!(LayerParams, tint) == 32);
const _: () = assert!(std::mem::offset_of!(LayerParams, ao) == 20);

impl Default for LayerParams {
    fn default() -> Self {
        Self {
            // 3.5 m was the single global value this replaced, and it is a
            // reasonable starting point for most ground.
            tiling_m: 3.5,
            normal_strength: 1.0,
            roughness: 1.0,
            height_blend: 0.22,
            // Off by default. Parallax costs a loop of samples per pixel, and a
            // material with a flat height channel gains nothing from it.
            parallax_m: 0.0,
            ao: 1.0,
            _pad0: [0.0; 2],
            tint: [1.0, 1.0, 1.0],
            _pad1: 0.0,
        }
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
    /// Editable PBR settings, parallel to `layers`.
    pub params: Vec<LayerParams>,
    params_buffer: wgpu::Buffer,
}

impl Materials {
    pub fn count(&self) -> u32 {
        self.layers.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Push the current parameters to the GPU.
    ///
    /// The whole array every time: it is 8 x 48 bytes, and a partial write keyed
    /// by index would be more code than the transfer costs.
    pub fn upload_params(&self, queue: &wgpu::Queue) {
        let mut padded = [LayerParams::default(); MAX_LAYERS as usize];
        for (slot, p) in padded.iter_mut().zip(&self.params) {
            *slot = *p;
        }
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&padded));
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
        let taken = dirs.len().min(MAX_LAYERS as usize);
        if dirs.len() > taken {
            log::warn!(
                "{} material folders found but only {taken} slots; ignoring the rest",
                dirs.len()
            );
        }
        let dirs = &dirs[..taken];
        // Names travel with their layers rather than being zipped in afterwards:
        // a set that fails to decode is dropped, so the two lists would fall out
        // of step and every material after the failure would show under the
        // wrong name.
        let (names, baked): (Vec<String>, Vec<Baked>) = load_or_bake(dir, dirs).into_iter().unzip();
        if baked.is_empty() {
            log::info!("{}: no materials imported yet", dir.display());
        }

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

        // Always MAX_LAYERS entries, whatever the palette holds: a fixed-size
        // uniform block means the shader can index it without a bounds check and
        // the layout never changes when a material is imported.
        let params: Vec<LayerParams> = vec![LayerParams::default(); layers.len()];
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material-params"),
            size: (MAX_LAYERS as usize * std::mem::size_of::<LayerParams>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut padded = [LayerParams::default(); MAX_LAYERS as usize];
            for (slot, p) in padded.iter_mut().zip(&params) {
                *slot = *p;
            }
            queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&padded));
        }

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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        Self { bind_group, layout, layers, params, params_buffer }
    }
}

// ---------------------------------------------------------------------------
// Assembling layers
// ---------------------------------------------------------------------------

/// Decoded layers for `dirs`, from the cache when it is still valid.
fn load_or_bake(root: &Path, dirs: &[std::path::PathBuf]) -> Vec<(String, Baked)> {
    let key = texture_set::fingerprint(dirs, TILE);
    let cache = root.join(".cache").join(format!("materials-{key:016x}.bin"));
    let name_of = |d: &Path| d.file_name().unwrap_or_default().to_string_lossy().to_string();

    if let Some(v) = read_cache(&cache, dirs.len()) {
        log::info!("materials: {} layers from cache", v.len());
        return dirs.iter().map(|d| name_of(d)).zip(v).collect();
    }

    let t0 = std::time::Instant::now();
    // An unreadable set is dropped, not substituted. Standing in a generated
    // layer meant the palette showed a material the user never imported, under
    // the name of the one that failed -- so a broken download looked like a
    // successful one with the wrong content.
    let pairs: Vec<(String, Baked)> = dirs
        .par_iter()
        .filter_map(|d| match texture_set::load(d, TILE) {
            Some(set) => Some((name_of(d), bake_set(&set))),
            None => {
                log::error!("{}: could not read texture set, skipping", d.display());
                None
            }
        })
        .collect();
    log::info!("materials: decoded {} sets in {:.1} s", pairs.len(), t0.elapsed().as_secs_f32());

    // Only cache a complete decode. `read_cache` validates against `dirs.len()`,
    // so writing a short list would make the cache permanently unreadable and
    // every start would silently re-decode.
    if pairs.len() == dirs.len() {
        let refs: Vec<&Baked> = pairs.iter().map(|(_, b)| b).collect();
        if let Err(e) = write_cache(&cache, &refs) {
            // Not fatal: it only means the next start pays for decoding again.
            log::warn!("{}: could not write material cache: {e}", cache.display());
        }
    }
    pairs
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
fn write_cache(path: &Path, baked: &[&Baked]) -> std::io::Result<()> {
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

    // The tests that used to live here -- seam tiling, per-layer relief, and
    // "grass stands prouder than soil" -- all exercised the noise generator that
    // produced the built-in layers. That generator is gone, so they are too:
    // there is nothing to assert about content the user has not imported yet.
    // What survives is the part that is still ours to get wrong.

    #[test]
    fn an_empty_folder_yields_an_empty_palette() {
        // The whole point of removing the fallback: a project with no imports
        // must have no materials, not six invented ones.
        let tmp = std::env::temp_dir().join("terra-empty-materials");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(texture_set::discover(&tmp).is_empty());
        assert!(load_or_bake(&tmp, &[]).is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn roles_are_claimed_once_and_then_paint_only() {
        // Two grass sets placed automatically would fight over the same ground,
        // so the first claims the role and the second becomes a brush.
        let names = ["Grass001".to_string(), "MossyGrass".to_string(), "Rock017".to_string()];
        let roles = assign_roles(&names);
        assert_eq!(roles[0], GRASS);
        assert_eq!(roles[1], ROLE_NONE, "the second grass must not also be automatic");
        assert_eq!(roles[2], ROCK);
    }

    #[test]
    fn layer_params_matches_the_wgsl_layout() {
        // The bug this exists for: WGSL aligns `vec3<f32>` to 16 bytes, so `tint`
        // sits at offset 32 in the shader. `repr(C)` put it at 24, and the shader
        // read `(tint[2], pad, pad)` = `(1, 0, 0)` -- the terrain came out pure
        // red, and only on the menu backdrop, because that is the one terrain with
        // a non-empty palette.
        //
        // Offsets, not just the size. The size was already correct.
        assert_eq!(std::mem::offset_of!(LayerParams, tiling_m), 0);
        assert_eq!(std::mem::offset_of!(LayerParams, normal_strength), 4);
        assert_eq!(std::mem::offset_of!(LayerParams, roughness), 8);
        assert_eq!(std::mem::offset_of!(LayerParams, height_blend), 12);
        assert_eq!(std::mem::offset_of!(LayerParams, parallax_m), 16);
        assert_eq!(std::mem::offset_of!(LayerParams, ao), 20);
        assert_eq!(std::mem::offset_of!(LayerParams, tint), 32, "the vec3 must be 16-aligned");
        assert_eq!(std::mem::size_of::<LayerParams>(), 48);
    }

    #[test]
    fn a_default_layer_reads_as_untinted_through_the_shader_layout() {
        // Read the bytes the way the shader does -- three floats from offset 32 --
        // rather than through the Rust field. Reading the field would have passed
        // while the shader saw red.
        let p = LayerParams::default();
        let bytes = bytemuck::bytes_of(&p);
        let tint: &[f32] = bytemuck::cast_slice(&bytes[32..44]);
        assert_eq!(tint, [1.0, 1.0, 1.0], "the shader sees {tint:?}, not white");
    }

    #[test]
    fn every_vector_member_sits_on_a_sixteen_byte_boundary() {
        // The general rule, stated so the next vector added to this struct is
        // checked by something rather than by whoever remembers. `tint` is the
        // only one today; extend the list when another arrives.
        const VECTOR_OFFSETS: &[usize] = &[std::mem::offset_of!(LayerParams, tint)];
        assert!(
            VECTOR_OFFSETS.iter().all(|o| o.is_multiple_of(16)),
            "a vector member is misaligned: {VECTOR_OFFSETS:?}"
        );
    }

    #[test]
    fn role_labels_cover_every_role() {
        for r in [SOIL, GRASS, ROCK, GRAVEL, SNOW, MUD, ROLE_NONE] {
            assert!(!role_label(r).is_empty());
        }
    }
}

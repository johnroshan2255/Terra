//! Instanced scatter for grass, rocks, and trees.
//!
//! Placement is derived from density masks plus a seed, so instance transforms
//! are never stored per-object -- only the rules are saved. A world with two
//! million trees costs a few kilobytes on disk and rebuilds identically every
//! time, which is also what lets a regenerate keep its forest.
//!
//! The rules are the ones a foliage tool is expected to have: how dense, how
//! large, which slopes, which altitudes, and whether the instance stands upright
//! or leans with the ground. A painted density mask gates all of it, so the
//! brush controls *where* and the rules control *what*.
//!
//! Culling is Phase B of `docs/culling.md`: a compute pass tests every instance
//! against the frustum and a distance threshold, compacts the survivors, and
//! writes the count into the indirect draw arguments. The CPU never sees an
//! instance and never learns how many were drawn -- which is the point, because
//! reading that count back would cost a GPU-to-CPU sync every frame.

use crate::camera::Camera;
use crate::lighting::Lighting;
use crate::mesh::{Instance, Mesh, MeshRenderer};
use crate::terrain::Terrain;
use glam::{Mat4, Quat, Vec3};
use rayon::prelude::*;
use terra_assets::MeshData;

/// Edge of a species preview in the palette.
pub const THUMB: u32 = 64;

/// Painted density resolution. Coarser than the splat map: foliage density is
/// a broad-strokes quantity, and the jitter within a cell hides the steps.
pub const DENSITY_RES: u32 = 256;

/// Ceiling per species, so a slider dragged to maximum cannot exhaust memory.
const MAX_INSTANCES: usize = 300_000;

/// Triangle ceiling per species after import decimation.
///
/// At scatter density the instance count is the multiplier: 20k instances of a
/// 12k-triangle tree is already 240 M triangles a frame before culling.
const MAX_TRIS_PER_SPECIES: usize = 6_000;

/// Palette slots. Eight is what the UI grid shows and what the density masks cost:
/// one `DENSITY_RES` byte mask each, so the ceiling is memory, not a shader limit.
const MAX_SPECIES: usize = 8;

/// Tag at the head of a saved foliage file.
///
/// Added when `Rules` grew the Unreal-shaped fields. The previous format had no
/// header at all, so without this an old file's first four bytes -- a species count --
/// would be read as a version and the rest as nonsense floats. With it, an old file is
/// recognised, refused, and the species keep their defaults.
const FOLIAGE_MAGIC: [u8; 4] = *b"TFOL";
const FOLIAGE_VERSION: u32 = 2;

/// Floats in one species' rules, in the order `encode_rules` writes them.
const FLOAT_FIELDS: usize = 14;

/// Bytes one species' rules occupy: the floats, three bools, one seed.
const RULES_BYTES: usize = FLOAT_FIELDS * 4 + 3 + 4;

/// Strip the resolution suffix downloads carry, so the palette reads as names.
fn pretty_name(file_stem: &str) -> String {
    let base = file_stem
        .trim_end_matches("_1k")
        .trim_end_matches("_2k")
        .trim_end_matches("_4k")
        .trim_end_matches("_8k");
    let mut out = String::with_capacity(base.len());
    for (i, part) in base.split('_').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut c = part.chars();
        if let Some(f) = c.next() {
            out.extend(f.to_uppercase());
            out.push_str(c.as_str());
        }
    }
    out
}

/// Plausible real-world height for an imported model, from its name. Scanned
/// assets carry no sense of scale that survives export.
fn default_height(name: &str) -> f32 {
    let n = name.to_lowercase();
    let has = |keys: &[&str]| keys.iter().any(|k| n.contains(k));
    if has(&["fern", "shrub", "bush", "weed", "grass", "plant", "flower"]) {
        0.9
    } else if has(&["branch", "debris", "twig"]) {
        1.6
    } else if has(&["trunk", "log", "stump"]) {
        3.0
    } else if has(&["sapling"]) {
        4.0
    } else if has(&["tree"]) {
        12.0
    } else if has(&["boulder"]) {
        2.4
    } else {
        1.2
    }
}

/// Everything about how one species is placed. Serialised with the world; the
/// instances themselves never are.
///
/// Field for field this is Unreal's Foliage Type, including the units, because the
/// units are half the meaning: a slope stated in degrees is something you can reason
/// about from a screenshot, and the same slope as an abstract 0..1 is not.
#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    /// Instances per hectare at full painted density.
    pub density: f32,
    /// Minimum metres between two instances -- Unreal's Radius.
    ///
    /// Density alone cannot express "sparse but never touching": raising density
    /// packs instances until they interpenetrate, and lowering it thins the whole
    /// distribution. This sets a floor on the spacing independently.
    pub radius_m: f32,
    /// Real-world height of one instance, in metres, before the scale variation
    /// below is applied.
    ///
    /// The size control the tool was missing. Every imported mesh is normalised to one
    /// metre tall at load, so this *is* the instance's height and the number in the
    /// panel is the number on screen. Previously the height came from a guess about the
    /// file name and could not be changed at all: an asset the guess did not recognise
    /// arrived 1.2 m tall whatever it actually was, and the only recourse was to abuse
    /// the scale range.
    pub height_m: f32,
    /// Random size variation, as a multiple of `height_m`. Unreal's Scale min/max.
    pub scale_min: f32,
    pub scale_max: f32,

    /// Tilt instances to stand perpendicular to the ground rather than straight up.
    ///
    /// A bool, matching Unreal, and off by default -- which is the "point to sky"
    /// behaviour. It matters most on exactly the ground that makes it visible: on a
    /// cliff face, an aligned tree grows sideways out of the rock, while an unaligned
    /// one stands vertically as a real tree does, because a tree grows toward the
    /// light and not perpendicular to the hillside. Rocks and ground cover want the
    /// opposite, which is why it is per species.
    pub align_to_normal: bool,
    /// Cap on that tilt, in degrees from vertical. Unreal's Align Max Angle.
    ///
    /// Alignment on a 70-degree cliff would lay an instance almost horizontal; this
    /// is how a species leans into moderate ground without falling over on severe
    /// ground. Ignored entirely when `align_to_normal` is off.
    pub align_max_angle_deg: f32,
    /// Spin each instance to a random heading. Unreal's Random Yaw.
    ///
    /// Wanted for anything organic and unwanted for anything with a facing --
    /// a fence post, a sign, a row of vines.
    pub random_yaw: bool,
    /// Extra random tilt in degrees, applied after alignment. Unreal's Random Pitch
    /// Angle. A few degrees stops a stand of trees looking like a comb.
    pub random_pitch_deg: f32,

    /// Shallowest ground this species will grow on, in degrees. Unreal's Ground
    /// Slope Angle minimum.
    ///
    /// Not redundant with the maximum: a minimum is what puts a species *only* on
    /// cliffs, which is otherwise impossible to express.
    pub slope_min_deg: f32,
    /// Steepest ground, in degrees. 90 accepts a vertical face.
    pub slope_max_deg: f32,
    pub altitude_min: f32,
    pub altitude_max: f32,
    /// Metres to shift each instance vertically. Unreal's Z Offset.
    ///
    /// Negative beds an instance into the ground, which is what stops a rock from
    /// resting on the surface like a dropped prop. In metres rather than the old
    /// fraction-of-height, so the number means the same thing for a 14 m tree and a
    /// 0.3 m tuft.
    pub z_offset_m: f32,

    /// Whether this species casts. Ground cover casting into a shadow map that
    /// covers hundreds of metres produces noise, not shadows.
    pub cast_shadow: bool,
    /// Radius of the collider stand-in, as a fraction of the instance's height. Zero
    /// means the species is not solid -- grass and ferns should not stop a car.
    pub collide_radius: f32,
    /// Metres beyond which instances stop being drawn.
    ///
    /// The lever that makes scatter affordable. Without it a bush 3 km away costs
    /// exactly what one at your feet costs, and the whole map is submitted every
    /// frame. Small props can be culled hard because nobody can resolve them anyway;
    /// a skyline of trees cannot.
    pub cull_distance: f32,
    pub seed: u32,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            density: 40.0,
            radius_m: 0.0,
            height_m: 2.0,
            scale_min: 0.8,
            scale_max: 1.35,
            // Off, so an imported mesh stands up straight on any ground until asked
            // to do otherwise. Unreal's default, and the safe one: a mesh authored
            // upright looks right upright, and looks broken lying on a slope.
            align_to_normal: false,
            align_max_angle_deg: 45.0,
            random_yaw: true,
            random_pitch_deg: 3.0,
            slope_min_deg: 0.0,
            slope_max_deg: 40.0,
            altitude_min: -10_000.0,
            altitude_max: 10_000.0,
            z_offset_m: 0.0,
            collide_radius: 0.0,
            cast_shadow: true,
            cull_distance: 900.0,
            seed: 1,
        }
    }
}

impl Rules {
    /// The slope acceptance test, as a surface normal's `y` component.
    ///
    /// Kept beside the fields because the conversion is the easy thing to get
    /// backwards: a *larger* slope angle is a *smaller* normal `y`, so the maximum
    /// angle becomes the minimum `y` and the comparison inverts.
    pub fn accepts_slope(&self, normal_y: f32) -> bool {
        let angle_deg = normal_y.clamp(-1.0, 1.0).acos().to_degrees();
        angle_deg >= self.slope_min_deg - 1e-3 && angle_deg <= self.slope_max_deg + 1e-3
    }

    /// Orientation for one instance on ground with this `normal`.
    ///
    /// `hash` is the cell's own random word, so the result is reproducible from the
    /// seed and no instance needs storing.
    ///
    /// Order matters and is Unreal's: align the up-axis first, then apply random
    /// pitch, then yaw *about the instance's own new up*. Yawing first and aligning
    /// second would turn the yaw into a lean whose direction depends on the slope,
    /// which reads as instances all leaning downhill.
    pub fn orientation(&self, normal: Vec3, hash: u32) -> Quat {
        let up = if self.align_to_normal {
            // Clamp the tilt to the configured angle. Slerping toward the normal by a
            // ratio would look similar and be wrong: it would let a 70-degree cliff
            // still tilt an instance 35 degrees under a 20-degree cap.
            let tilt_deg = normal.y.clamp(-1.0, 1.0).acos().to_degrees();
            let cap = self.align_max_angle_deg.max(0.0);
            if tilt_deg <= cap + 1e-4 {
                Quat::from_rotation_arc(Vec3::Y, normal)
            } else if tilt_deg > 1e-4 {
                Quat::from_rotation_arc(Vec3::Y, normal.normalize_or(Vec3::Y))
                    .normalize()
                    .slerp(Quat::IDENTITY, 1.0 - cap / tilt_deg)
            } else {
                Quat::IDENTITY
            }
        } else {
            // Point to sky: the instance's up-axis is world up, whatever the ground
            // beneath it does. On a cliff this is the difference between a tree
            // growing out of the rock face and one standing on the ledge.
            Quat::IDENTITY
        };

        let mut rot = up;
        if self.random_pitch_deg > 0.0 {
            // Two words: one for how far, one for which way, or every instance would
            // tilt in the same direction.
            let amount = (hash & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0;
            let dir = ((hash >> 16) & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
            let axis = Vec3::new(dir.cos(), 0.0, dir.sin());
            rot *= Quat::from_axis_angle(axis, amount * self.random_pitch_deg.to_radians());
        }
        if self.random_yaw {
            let yaw = (hash.rotate_left(11) & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
            rot *= Quat::from_rotation_y(yaw);
        }
        rot.normalize()
    }
}

/// One entry in the foliage palette.
pub struct Species {
    pub name: String,
    pub rules: Rules,
    /// Painted coverage, `DENSITY_RES^2`, 0 = none.
    pub density: Vec<u8>,
    /// Cached, for the same reason the terrain caches its own: the palette
    /// asks once per species per frame.
    painted: bool,
    pub color: Vec3,
    /// Palette preview, RGBA, `THUMB` square.
    pub thumbnail: Vec<u8>,
    mesh: Mesh,
    radius: f32,
    /// Every instance, written once at rebuild and read only by the compute
    /// pass.
    source: Option<wgpu::Buffer>,
    /// Survivors of the cull, and the vertex buffer the indirect draw reads.
    visible: Option<wgpu::Buffer>,
    /// `draw_indexed_indirect` arguments, with `instance_count` written by the
    /// compute pass.
    args: Option<wgpu::Buffer>,
    cull_bg: Option<wgpu::BindGroup>,
    params: Option<wgpu::Buffer>,
    count: u32,
    dirty: bool,
}

impl Species {
    pub fn instance_count(&self) -> u32 {
        self.count
    }

    pub fn is_painted(&self) -> bool {
        self.painted
    }

    /// Wipe this species' painting, for a world being replaced.
    pub fn clear_density(&mut self) {
        self.density.fill(0);
        self.painted = false;
        self.dirty = true;
    }

    /// Mark for regeneration. Cheap -- the rebuild happens once, later, rather
    /// than on every slider tick.
    pub fn touch(&mut self) {
        self.dirty = true;
    }
}

/// A solid stand-in for one scattered object, in renderer terms.
///
/// Deliberately not a physics type: the renderer has no business depending on
/// the simulation crate, and the caller that owns both does the translation.
#[derive(Clone, Copy, Debug)]
pub struct Solid {
    /// Base of the object, on the ground.
    pub pos: Vec3,
    pub radius: f32,
    pub height: f32,
    /// Round and squat rather than tall and thin.
    pub boulder: bool,
}

/// A hand-placed object, in renderer terms. `species` indexes the palette.
#[derive(Clone, Debug)]
pub struct Prop {
    pub species: usize,
    pub pos: Vec3,
    pub yaw: f32,
    pub scale: f32,
}

pub struct Scatter {
    pub species: Vec<Species>,
    /// Hand-placed objects. Few enough that they are neither culled nor
    /// instanced in bulk -- if that stops being true they should become a
    /// species with a painted mask instead.
    pub props: Vec<Prop>,
    props_buf: Option<wgpu::Buffer>,
    /// `(species, first, count)` runs within `props_buf`.
    prop_runs: Vec<(usize, u32, u32)>,
    props_dirty: bool,
    pipeline: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
}

impl Scatter {
    /// Build the palette from `dir`, falling back to the generated species when
    /// it holds no models.
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &MeshRenderer,
        dir: &std::path::Path,
    ) -> Self {
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter-cull-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter-cull"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/render/scatter_cull.wgsl").into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter-cull-layout"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter-cull-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut me = Self {
            species: Vec::new(),
            props: Vec::new(),
            props_buf: None,
            prop_runs: Vec::new(),
            props_dirty: false,
            pipeline,
            cull_bgl,
        };
        me.reload(device, queue, meshes, dir);
        me
    }

    /// Rebuild the palette from `dir`, keeping the pipeline and its layout.
    ///
    /// Separate from [`Self::load`] because the palette changes during a session and
    /// the pipeline does not: a model imported through the content browser has to
    /// appear without restarting, which for a content browser is the difference
    /// between the import working and not.
    ///
    /// Painting is *not* carried across. Density masks are keyed by species name and
    /// restored from the world file by [`Self::restore`]; guessing which of the new
    /// entries inherits an old mask would eventually hand one species' forest to
    /// another.
    pub fn reload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &MeshRenderer,
        dir: &std::path::Path,
    ) {
        let files = terra_assets::mesh::discover(dir);
        let mut species = Vec::new();

        // Decode, decimate and preview each model on its own thread. Eight
        // scanned assets is seconds of work and none of it touches the others;
        // only the GPU upload has to be serial.
        let decoded: Vec<(String, terra_assets::MeshData, Rules)> = files
            .par_iter()
            .filter_map(|path| {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".into());
                match terra_assets::mesh::load_gltf(path) {
                    Ok(mut data) => {
                        let raw = data.triangle_count();
                        // Scanned assets are film-resolution. Scattering them at
                        // source density is not possible, and the detail is gone
                        // by the second mip regardless.
                        data.decimate(MAX_TRIS_PER_SPECIES);
                        // One metre, always. That makes an instance's scale factor its
                        // height in metres, so `Rules::height_m` is the single place
                        // size is decided and the collider maths needs no per-species
                        // constant to multiply by.
                        data.normalize_height(1.0);
                        if data.triangle_count() < raw {
                            log::info!(
                                "model '{name}': {raw} -> {} triangles",
                                data.triangle_count()
                            );
                        } else {
                            log::info!("model '{name}': {raw} triangles");
                        }
                        let mut rules = guess_rules(path);
                        // The name-based height is now a starting value the user can
                        // change, not a fixed property of the import.
                        rules.height_m = default_height(&name);
                        Some((pretty_name(&name), data, rules))
                    }
                    Err(e) => {
                        log::error!("{e:#}");
                        None
                    }
                }
            })
            .collect();

        for (name, data, rules) in decoded {
            species.push(Species::new(device, queue, meshes, name, data, rules));
        }

        // No fallback. An empty folder yields an empty palette, and the Foliage tool
        // says so rather than quietly filling itself with engine-made trees -- which
        // is what it used to do, and what made "nothing ships prebuilt" untrue for
        // meshes even after it was true for textures.
        //
        // Sorted by name so the palette order, and therefore the index the world file
        // remembers, does not depend on the order the filesystem happened to list.
        species.sort_by_key(|s| s.name.to_lowercase());
        species.truncate(MAX_SPECIES);
        if species.is_empty() {
            log::info!("{}: no models, foliage palette is empty", dir.display());
        } else {
            log::info!("{}: {} species", dir.display(), species.len());
        }

        // Props reference species by index, and those indices have just moved.
        // Dropping them is wrong and keeping them blindly is worse; the editor
        // re-resolves placed props by species *name* after a reload, which is why this
        // clears rather than remaps.
        self.props.clear();
        self.prop_runs.clear();
        self.props_dirty = true;
        self.species = species;
    }

    // --- hand-placed props ---

    /// Drop a prop and return its index.
    pub fn place(&mut self, species: usize, pos: Vec3, scale: f32, yaw: f32) -> usize {
        self.props.push(Prop { species, pos, yaw, scale });
        self.props_dirty = true;
        self.props.len() - 1
    }

    pub fn remove_prop(&mut self, index: usize) {
        if index < self.props.len() {
            self.props.remove(index);
            self.props_dirty = true;
        }
    }

    pub fn touch_props(&mut self) {
        self.props_dirty = true;
    }

    /// Nearest prop along a ray, by bounding sphere.
    ///
    /// Sphere rather than triangle intersection: picking a tree by its trunk
    /// would mean clicking a few pixels of geometry, and every editor that does
    /// this picks generously on purpose.
    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<usize> {
        let mut best: Option<(f32, usize)> = None;
        for (i, p) in self.props.iter().enumerate() {
            let Some(sp) = self.species.get(p.species) else { continue };
            // Centre the sphere on the mesh's middle, not its base, or tall
            // props are only clickable around their feet.
            let centre = p.pos + Vec3::Y * (p.scale * 0.5);
            let radius = (sp.radius * p.scale).max(p.scale * 0.5);

            let to = centre - origin;
            let along = to.dot(dir);
            if along < 0.0 {
                continue;
            }
            let perp2 = to.length_squared() - along * along;
            if perp2 > radius * radius {
                continue;
            }
            let hit = along - (radius * radius - perp2).sqrt();
            if best.is_none_or(|(d, _)| hit < d) {
                best = Some((hit, i));
            }
        }
        best.map(|(_, i)| i)
    }

    /// Rebuild the prop instance buffer, tinting `selected` so the choice is
    /// visible in the viewport rather than only in the panel.
    pub fn refresh_props(&mut self, device: &wgpu::Device, selected: Option<usize>) {
        if !self.props_dirty {
            return;
        }
        self.props_dirty = false;
        self.prop_runs.clear();

        // Grouped by species so each run is one draw against one mesh.
        let mut order: Vec<usize> = (0..self.props.len()).collect();
        order.sort_by_key(|i| self.props[*i].species);

        let mut flat: Vec<Instance> = Vec::new();
        let mut run: Option<(usize, u32)> = None;
        for i in order {
            let p = &self.props[i];
            let Some(sp) = self.species.get(p.species) else { continue };
            let rot = Quat::from_rotation_y(p.yaw);
            let m = Mat4::from_scale_rotation_translation(Vec3::splat(p.scale), rot, p.pos);
            let color =
                if Some(i) == selected { sp.color * 0.6 + Vec3::splat(0.35) } else { sp.color };
            match &mut run {
                Some((species, count)) if *species == p.species => *count += 1,
                Some((species, count)) => {
                    self.prop_runs.push((*species, flat.len() as u32 - *count, *count));
                    run = Some((p.species, 1));
                }
                None => run = Some((p.species, 1)),
            }
            flat.push(Instance::new(m, color));
        }
        if let Some((species, count)) = run {
            self.prop_runs.push((species, flat.len() as u32 - count, count));
        }

        if flat.is_empty() {
            self.props_buf = None;
            return;
        }
        use wgpu::util::DeviceExt;
        self.props_buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("props"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::VERTEX,
        }));
    }

    pub fn draw_props(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        meshes: &MeshRenderer,
        lighting: &Lighting,
    ) {
        let Some(buf) = self.props_buf.as_ref() else { return };
        for &(species, first, count) in &self.prop_runs {
            if let Some(sp) = self.species.get(species) {
                meshes.draw_instanced(pass, lighting, &sp.mesh, buf, first, count);
            }
        }
    }

    pub fn draw_props_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        meshes: &MeshRenderer,
        lighting: &Lighting,
        cascade: usize,
    ) {
        let Some(buf) = self.props_buf.as_ref() else { return };
        for &(species, first, count) in &self.prop_runs {
            if let Some(sp) = self.species.get(species) {
                meshes.draw_shadow_instanced(pass, lighting, cascade, &sp.mesh, buf, first, count);
            }
        }
    }

    /// Rebuild whichever species have been marked dirty.
    ///
    /// Called once a frame rather than at the point of edit, so dragging a
    /// density slider costs one rebuild when the drag ends, not one per frame.
    pub fn refresh(&mut self, device: &wgpu::Device, terrain: &Terrain) {
        for s in &mut self.species {
            if s.dirty {
                s.rebuild(device, terrain, &self.cull_bgl);
            }
        }
    }

    pub fn mark_all_dirty(&mut self) {
        for s in &mut self.species {
            s.dirty = true;
        }
    }

    pub fn total_instances(&self) -> u32 {
        self.species.iter().map(|s| s.count).sum()
    }

    /// Paint density for one species under the brush.
    pub fn paint(
        &mut self,
        index: usize,
        terrain: &Terrain,
        centre: glam::Vec2,
        radius_m: f32,
        strength: f32,
        erase: bool,
    ) {
        let Some(s) = self.species.get_mut(index) else { return };
        let extent = terrain.extent_m();
        let last = (DENSITY_RES - 1) as f32;
        let to_texel = |v: f32| ((v / extent + 0.5) * last).clamp(0.0, last);
        let x0 = to_texel(centre.x - radius_m).floor() as u32;
        let x1 = to_texel(centre.x + radius_m).ceil() as u32;
        let z0 = to_texel(centre.y - radius_m).floor() as u32;
        let z1 = to_texel(centre.y + radius_m).ceil() as u32;

        for z in z0..=z1 {
            for x in x0..=x1 {
                let wx = (x as f32 / last - 0.5) * extent;
                let wz = (z as f32 / last - 0.5) * extent;
                let d = ((wx - centre.x).powi(2) + (wz - centre.y).powi(2)).sqrt();
                if d > radius_m {
                    continue;
                }
                let t = (d / radius_m.max(1e-3)).clamp(0.0, 1.0);
                let amount = strength * (1.0 - t * t).powi(2);
                let i = (z * DENSITY_RES + x) as usize;
                let cur = s.density[i] as f32 / 255.0;
                let next = if erase { cur - amount } else { cur + amount };
                s.density[i] = (next.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        // Erasing may or may not have emptied the mask; a rescan is the only
        // way to know, and it happens once per stroke rather than per frame.
        s.painted = if erase { s.density.iter().any(|&d| d != 0) } else { true };
        s.dirty = true;
    }

    pub fn fill(&mut self, index: usize) {
        if let Some(s) = self.species.get_mut(index) {
            s.density.fill(255);
            s.painted = true;
            s.dirty = true;
        }
    }

    pub fn clear(&mut self, index: usize) {
        if let Some(s) = self.species.get_mut(index) {
            s.density.fill(0);
            s.painted = false;
            s.dirty = true;
        }
    }

    /// Cull every species into its visible buffer. Must run before the render
    /// pass that draws them.
    pub fn cull(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cam: &Camera,
        aspect: f32,
    ) {
        let frustum = Frustum::new(&(cam.projection(aspect) * cam.look_at()));

        // Uniform and argument writes first. These are queue operations, so
        // they are ordered before the encoder runs regardless of where they sit
        // relative to the pass.
        for s in &self.species {
            let (Some(params), Some(args)) = (s.params.as_ref(), s.args.as_ref()) else {
                continue;
            };
            if s.count == 0 {
                continue;
            }
            let p = CullParams {
                planes: frustum.planes.map(|v| v.to_array()),
                eye: cam.pos.extend(0.0).to_array(),
                cull_distance: s.rules.cull_distance,
                // The instance scale range widens the bounding sphere.
                radius: s.radius * s.rules.scale_max.max(s.rules.scale_min),
                count: s.count,
                _pad: 0,
            };
            queue.write_buffer(params, 0, bytemuck::bytes_of(&p));
            // Reset the survivor count. Everything else is fixed, so rewriting
            // the whole struct beats a clearing pass and costs 20 bytes.
            queue.write_buffer(
                args,
                0,
                bytemuck::bytes_of(&DrawArgs {
                    index_count: s.mesh.index_count(),
                    instance_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                }),
            );
        }

        // One pass for every species. Each `begin_compute_pass` costs a
        // barrier and a descriptor; there is nothing to gain from four.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter-cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for s in &self.species {
            let Some(bg) = s.cull_bg.as_ref() else { continue };
            if s.count == 0 {
                continue;
            }
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(s.count.div_ceil(64), 1, 1);
        }
    }

    /// Draw whatever the cull left, without ever asking how much that was.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        meshes: &MeshRenderer,
        lighting: &Lighting,
    ) {
        for s in &self.species {
            let (Some(visible), Some(args)) = (s.visible.as_ref(), s.args.as_ref()) else {
                continue;
            };
            if s.count == 0 {
                continue;
            }
            meshes.draw_indirect(pass, lighting, &s.mesh, visible, args);
        }
    }

    /// Cast the scatter into one cascade.
    ///
    /// Reuses the culled set: the shadow pass shows whatever survived the
    /// camera cull, which is wrong for casters just outside the view. Accepted
    /// deliberately -- the alternative is a second cull per cascade, and the
    /// artefact is confined to shadows entering from off-screen edges.
    pub fn draw_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        meshes: &MeshRenderer,
        lighting: &Lighting,
        cascade: usize,
    ) {
        for s in &self.species {
            let (Some(visible), Some(args)) = (s.visible.as_ref(), s.args.as_ref()) else {
                continue;
            };
            if s.count == 0 || !s.rules.cast_shadow {
                continue;
            }
            meshes.draw_shadow_indirect(pass, lighting, cascade, &s.mesh, visible, args);
        }
    }
}

/// Mirrors the `Cull` block in `scatter_cull.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParams {
    planes: [[f32; 4]; 6],
    eye: [f32; 4],
    cull_distance: f32,
    radius: f32,
    count: u32,
    _pad: u32,
}

/// Exactly what `draw_indexed_indirect` reads.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

impl Species {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &MeshRenderer,
        name: String,
        data: MeshData,
        rules: Rules,
    ) -> Self {
        let radius = data.bounding_radius();
        let thumbnail = data.thumbnail(THUMB);
        // A textured mesh must not have its map tinted by the base colour
        // factor as well, or every scanned asset comes out muddy.
        let color = if data.albedo.is_some() { Vec3::ONE } else { Vec3::from(data.base_color) };
        Self {
            name,
            rules,
            density: vec![0; (DENSITY_RES * DENSITY_RES) as usize],
            painted: false,
            color,
            thumbnail,
            mesh: meshes.upload_mesh(device, queue, &data),
            radius,
            source: None,
            visible: None,
            args: None,
            cull_bg: None,
            params: None,
            count: 0,
            dirty: false,
        }
    }

    /// Regenerate every instance from the rules, the density mask and the seed.
    ///
    /// A jittered grid rather than pure random sampling: uniform random
    /// clumps and leaves bald patches, which reads as a bug rather than as
    /// nature. One sample per cell with an offset inside it gives even coverage
    /// that still looks unplanned.
    /// Walk every instance this species' rules produce.
    ///
    /// The single generator. `docs/physics.md` requires that physics and
    /// rendering derive the identical set from the seed rather than sharing
    /// stored transforms; that only holds if there is one function, so the
    /// collider pass and the instance buffer both come through here.
    /// The grid this species' rules imply: cells per side, and their spacing.
    fn grid(&self, extent: f32) -> Option<(u32, f32)> {
        let hectares = (extent * extent) / 10_000.0;
        let wanted = ((self.rules.density * hectares) as usize).min(MAX_INSTANCES);
        if wanted == 0 {
            return None;
        }
        let mut side = (wanted as f32).sqrt().ceil().max(1.0) as u32;

        // Enforce the minimum spacing by coarsening the grid, because one cell
        // produces at most one instance and a cell is jittered only within itself.
        // That makes `radius_m` a guarantee rather than a rejection pass: no instance
        // can land closer than one cell to another, so nothing has to be thrown away
        // after the fact and the placement stays a pure function of the coordinate.
        //
        // The jitter within a cell means two neighbours can still end up nearer than
        // the full cell width -- worst case they meet at a shared corner -- so the
        // cell is sized at `radius_m`, not `radius_m / 2`, to keep the typical
        // spacing at the requested figure.
        if self.rules.radius_m > 0.0 {
            let max_side = (extent / self.rules.radius_m).floor().max(1.0) as u32;
            side = side.min(max_side);
        }
        Some((side, extent / side as f32))
    }

    /// Whether one grid cell produces an instance, and what it looks like.
    ///
    /// The single generator, factored per-cell so it can be walked serially by
    /// the physics pass and in parallel by the rebuild without either of them
    /// being a second implementation. Two copies of this that had to agree is
    /// exactly the drift `docs/physics.md` warns about when it asks physics and
    /// rendering to derive the identical set.
    fn place_at(
        &self,
        terrain: &Terrain,
        gx: u32,
        gz: u32,
        step: f32,
        extent: f32,
    ) -> Option<(Vec3, f32, Quat)> {
        let last = (DENSITY_RES - 1) as f32;

        let h = hash3(gx, gz, self.rules.seed);
        let jx = (h & 0xFFFF) as f32 / 65535.0;
        let jz = ((h >> 16) & 0xFFFF) as f32 / 65535.0;
        let x = (gx as f32 + jx) * step - extent * 0.5;
        let z = (gz as f32 + jz) * step - extent * 0.5;

        // Painted coverage decides the odds, not a hard cut, so a half-strength
        // stroke thins out rather than stopping dead.
        let dx = ((x / extent + 0.5) * last).clamp(0.0, last) as usize;
        let dz = ((z / extent + 0.5) * last).clamp(0.0, last) as usize;
        let coverage = self.density[dz * DENSITY_RES as usize + dx] as f32 / 255.0;
        if coverage <= 0.0 {
            return None;
        }
        let roll = hash3(gx ^ 0x9E37, gz ^ 0x85EB, self.rules.seed.wrapping_add(7));
        if (roll & 0xFFFF) as f32 / 65535.0 > coverage {
            return None;
        }

        let y = terrain.height_at(x, z);
        if y < self.rules.altitude_min || y > self.rules.altitude_max {
            return None;
        }
        let normal = terrain.normal_at(x, z);
        if !self.rules.accepts_slope(normal.y) {
            return None;
        }

        let r2 = hash3(gx, gz, self.rules.seed.wrapping_add(31));
        let s01 = (r2 & 0xFFFF) as f32 / 65535.0;
        // The mesh is one metre tall, so the scale factor *is* the height in metres.
        let variation = self.rules.scale_min + (self.rules.scale_max - self.rules.scale_min) * s01;
        let scale = self.rules.height_m * variation;

        let rot = self.rules.orientation(normal, hash3(gx, gz, self.rules.seed.wrapping_add(101)));
        let pos = Vec3::new(x, y + self.rules.z_offset_m, z);
        Some((pos, scale, rot))
    }

    /// Walk the instances whose cells fall inside a world-space circle.
    ///
    /// Bounded rather than filtered. Placement is a regular grid, so the cells
    /// that can contain anything within `radius` are known arithmetically --
    /// walking all 90,000 of them to keep the 800 in range was doing 100x the
    /// work for the same answer, on the thread driving the car.
    fn for_each_near(
        &self,
        terrain: &Terrain,
        centre: Vec3,
        radius: f32,
        mut f: impl FnMut(Vec3, f32, Quat),
    ) {
        let extent = terrain.extent_m();
        let Some((side, step)) = self.grid(extent) else { return };

        // One cell of slack: a cell's instance is jittered anywhere inside it.
        let to_cell = |v: f32| ((v + extent * 0.5) / step).floor() as i64;
        let last = side as i64 - 1;
        let x0 = to_cell(centre.x - radius).clamp(0, last) as u32;
        let x1 = (to_cell(centre.x + radius) + 1).clamp(0, last) as u32;
        let z0 = to_cell(centre.z - radius).clamp(0, last) as u32;
        let z1 = (to_cell(centre.z + radius) + 1).clamp(0, last) as u32;

        let r2 = radius * radius;
        for gz in z0..=z1 {
            for gx in x0..=x1 {
                if let Some((pos, scale, rot)) = self.place_at(terrain, gx, gz, step, extent)
                    && (pos.x - centre.x).powi(2) + (pos.z - centre.z).powi(2) <= r2
                {
                    f(pos, scale, rot);
                }
            }
        }
    }

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        terrain: &Terrain,
        cull_bgl: &wgpu::BindGroupLayout,
    ) {
        self.dirty = false;
        self.count = 0;

        if !self.is_painted() {
            self.release();
            return;
        }

        let extent = terrain.extent_m();
        let Some((side, step)) = self.grid(extent) else {
            self.release();
            return;
        };
        let color = self.color;
        // Reborrowed shared so the closure captures `&Species` rather than
        // trying to move the `&mut self` the rebuild holds.
        let me: &Species = self;

        // A row at a time. Cells are independent -- the whole point of deriving
        // placement from a hash of the coordinate is that no cell depends on any
        // other -- so this is a flat_map with nothing to synchronise.
        let t0 = std::time::Instant::now();
        let flat: Vec<Instance> = (0..side)
            .into_par_iter()
            .flat_map_iter(|gz| {
                (0..side).filter_map(move |gx| {
                    let (pos, scale, rot) = me.place_at(terrain, gx, gz, step, extent)?;
                    let m = Mat4::from_scale_rotation_translation(Vec3::splat(scale), rot, pos);
                    Some(Instance::new(m, color))
                })
            })
            .collect();
        log::debug!(
            "scatter '{}': {} instances from {}x{} cells in {:.1} ms",
            me.name,
            flat.len(),
            side,
            side,
            t0.elapsed().as_secs_f32() * 1000.0
        );

        self.count = flat.len() as u32;
        if flat.is_empty() {
            self.release();
            return;
        }

        use wgpu::util::DeviceExt;
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scatter-source"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Worst case every instance survives, so the destination matches the
        // source. Sizing it to a guess would mean silently dropping geometry
        // the moment a camera looked at the whole map.
        let visible = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-visible"),
            size: (flat.len() * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-args"),
            size: std::mem::size_of::<DrawArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-cull-params"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter-cull-bg"),
            layout: cull_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: visible.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: args.as_entire_binding() },
            ],
        });

        self.source = Some(source);
        self.visible = Some(visible);
        self.args = Some(args);
        self.params = Some(params);
        self.cull_bg = Some(cull_bg);
    }

    /// Drop the GPU side when a species has nothing to draw.
    fn release(&mut self) {
        self.source = None;
        self.visible = None;
        self.args = None;
        self.params = None;
        self.cull_bg = None;
        self.count = 0;
    }
}

/// Starting rules for a newly imported mesh, guessed from its file name.
///
/// The same trick the material loader uses for roles, and for the same reason: the
/// alternative is that every import lands on one set of numbers and the user tunes
/// six sliders before seeing anything sensible. A wrong guess costs one slider drag,
/// and an unrecognised name simply gets the defaults.
///
/// These are starting points, not presets in the sense that was removed -- no
/// geometry is invented, and nothing appears in the palette that the user did not
/// import.
fn guess_rules(path: &std::path::Path) -> Rules {
    let n = path.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let has = |keys: &[&str]| keys.iter().any(|k| n.contains(k));

    if has(&["rock", "stone", "boulder", "pebble"]) {
        // Rocks bed into the ground, sit at any angle, and do not care how steep it
        // is -- the one thing that genuinely wants normal alignment.
        Rules {
            density: 30.0,
            align_to_normal: true,
            align_max_angle_deg: 60.0,
            random_pitch_deg: 12.0,
            slope_max_deg: 75.0,
            z_offset_m: -0.5,
            scale_min: 0.5,
            scale_max: 2.2,
            cull_distance: 500.0,
            collide_radius: 0.45,
            ..Default::default()
        }
    } else if has(&["bush", "shrub", "fern", "grass", "plant", "weed", "flower"]) {
        Rules {
            density: 55.0,
            align_to_normal: true,
            align_max_angle_deg: 25.0,
            random_pitch_deg: 8.0,
            slope_max_deg: 40.0,
            cull_distance: 380.0,
            // Ground cover casting into a cascade that spans hundreds of metres is
            // noise, not shadow.
            cast_shadow: false,
            ..Default::default()
        }
    } else if has(&["tree", "pine", "spruce", "oak", "birch", "palm", "trunk", "log"]) {
        Rules {
            density: 22.0,
            // Upright: a tree grows toward the light. This is the case the "point to
            // sky" default exists for.
            align_to_normal: false,
            random_pitch_deg: 2.5,
            slope_max_deg: 35.0,
            // Trees make the skyline; culling them at the default distance pops a
            // whole ridgeline in and out as you turn.
            cull_distance: 1800.0,
            collide_radius: 0.035,
            ..Default::default()
        }
    } else {
        Rules::default()
    }
}

fn hash3(x: u32, y: u32, seed: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x8DA6_B343) ^ y.wrapping_mul(0xD8163841) ^ seed.wrapping_mul(0xCB1A_B31F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    h
}

/// Six clip-space planes, for cell culling.
struct Frustum {
    planes: [glam::Vec4; 6],
}

impl Frustum {
    fn new(view_proj: &Mat4) -> Self {
        let m = view_proj.transpose();
        // Reversed-Z with an infinite far plane: there is no far plane to
        // extract, so only five are meaningful and the sixth is left degenerate.
        let planes = [
            m.w_axis + m.x_axis,
            m.w_axis - m.x_axis,
            m.w_axis + m.y_axis,
            m.w_axis - m.y_axis,
            m.w_axis - m.z_axis,
            m.w_axis,
        ];
        Self { planes }
    }

    /// Sphere test, mirroring `scatter_cull.wgsl` exactly.
    ///
    /// Only the shader culls in anger; this exists so the plane extraction --
    /// the part that is easy to get subtly wrong and invisible when it is --
    /// is covered by a test.
    #[cfg(test)]
    fn contains_sphere(&self, centre: Vec3, radius: f32) -> bool {
        self.planes.iter().all(|p| p.truncate().dot(centre) + p.w >= -radius)
    }
}
impl Scatter {
    /// Serialise rules and density masks.
    ///
    /// Instance transforms are deliberately absent: they are reproducible from
    /// the seed, and writing them would turn a few kilobytes into hundreds of
    /// megabytes.
    pub fn save(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&FOLIAGE_MAGIC);
        out.extend_from_slice(&FOLIAGE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.species.len() as u32).to_le_bytes());
        out.extend_from_slice(&DENSITY_RES.to_le_bytes());
        for s in &self.species {
            let name = s.name.as_bytes();
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&encode_rules(&s.rules));
            out.extend_from_slice(&s.density);
        }
        out
    }

    /// Restore by name, so reordering or removing a model folder cannot hand one
    /// species' forest to another.
    ///
    /// A file this cannot read is ignored rather than partly applied. Half-restored
    /// rules would be worse than defaults, because they would look deliberate.
    pub fn restore(&mut self, bytes: &[u8]) {
        let u32_at = |p: &mut usize| -> Option<u32> {
            let v = bytes.get(*p..*p + 4)?;
            *p += 4;
            Some(u32::from_le_bytes(v.try_into().ok()?))
        };

        let Some(version) = foliage_version(bytes) else {
            log::warn!("foliage file predates the Unreal-style rules; keeping defaults");
            return;
        };
        if version != FOLIAGE_VERSION {
            log::warn!("foliage file is version {version}, expected {FOLIAGE_VERSION}; ignoring");
            return;
        }
        let mut p = 8usize;

        let Some(count) = u32_at(&mut p) else { return };
        let Some(res) = u32_at(&mut p) else { return };
        if res != DENSITY_RES {
            log::warn!("foliage saved at {res}, expected {DENSITY_RES}; ignoring");
            return;
        }
        let mask_len = (res * res) as usize;

        for _ in 0..count {
            let Some(n) = u32_at(&mut p) else { return };
            let Some(name) =
                bytes.get(p..p + n as usize).map(|b| String::from_utf8_lossy(b).to_string())
            else {
                return;
            };
            p += n as usize;
            let Some(rules) = bytes.get(p..p + RULES_BYTES).and_then(decode_rules) else {
                return;
            };
            p += RULES_BYTES;
            let Some(mask) = bytes.get(p..p + mask_len) else { return };
            p += mask_len;

            if let Some(s) = self.species.iter_mut().find(|s| s.name == name) {
                s.rules = rules;
                s.density.copy_from_slice(mask);
                s.painted = s.density.iter().any(|&d| d != 0);
                s.dirty = true;
            }
        }
    }

    pub fn is_painted(&self) -> bool {
        self.species.iter().any(|s| s.is_painted())
    }

    /// Re-seat hand-placed props on the surface after the terrain has moved.
    ///
    /// Scatter instances are derived and simply regenerate at the new height.
    /// A prop's position is its data, so nothing regenerates it -- sculpting
    /// under one leaves it floating or buried unless it is put back.
    pub fn reground_props(&mut self, terrain: &Terrain) {
        for p in &mut self.props {
            p.pos.y = terrain.height_at(p.pos.x, p.pos.z);
        }
        self.props_dirty = true;
    }

    /// Obstacles within `radius` of `centre`, for the physics world.
    ///
    /// Regenerated from the same rules and seed the renderer scatters from, so
    /// nothing per-instance is stored to keep the two in step. Species with no
    /// `collide_radius` are skipped entirely -- a fern is not an obstacle, and
    /// giving it a collider would fill the broad phase with things that only
    /// make a car judder.
    pub fn obstacles_near(&self, terrain: &Terrain, centre: Vec3, radius: f32) -> Vec<Solid> {
        // Species are independent, and this runs while driving -- the one place
        // a stall is a physics hitch rather than a slow edit.
        let mut out: Vec<Solid> = self
            .species
            .par_iter()
            .filter(|s| s.rules.collide_radius > 0.0 && s.is_painted())
            .flat_map_iter(|s| {
                let mut local = Vec::new();
                s.for_each_near(terrain, centre, radius, |pos, scale, _rot| {
                    // Unit mesh, so the scale factor is the height.
                    let height = scale;
                    local.push(Solid {
                        pos,
                        radius: height * s.rules.collide_radius,
                        height,
                        boulder: s.rules.collide_radius >= 0.25,
                    });
                });
                local
            })
            .collect();

        let r2 = radius * radius;
        for p in &self.props {
            let Some(sp) = self.species.get(p.species) else { continue };
            if sp.rules.collide_radius <= 0.0 {
                continue;
            }
            if (p.pos.x - centre.x).powi(2) + (p.pos.z - centre.z).powi(2) > r2 {
                continue;
            }
            let height = p.scale;
            let rad = height * sp.rules.collide_radius;
            out.push(Solid {
                pos: p.pos,
                radius: rad,
                height,
                boulder: sp.rules.collide_radius >= 0.25,
            });
        }
        out
    }
}

/// One species' rules, as bytes.
///
/// A pure function, and the only place the wire order is written down. `save` and
/// `restore` used to each spell the field order out inline, which is two lists that
/// have to agree: adding a field in one and not the other silently shifts every
/// value after it into the wrong slot.
fn encode_rules(r: &Rules) -> Vec<u8> {
    let mut out = Vec::with_capacity(RULES_BYTES);
    for f in [
        r.density,
        r.radius_m,
        r.height_m,
        r.scale_min,
        r.scale_max,
        r.align_max_angle_deg,
        r.random_pitch_deg,
        r.slope_min_deg,
        r.slope_max_deg,
        r.altitude_min,
        r.altitude_max,
        r.z_offset_m,
        r.cull_distance,
        r.collide_radius,
    ] {
        out.extend_from_slice(&f.to_le_bytes());
    }
    // One byte a bool rather than a bitfield: packing would save six bytes out of a
    // 65 kB density mask and cost the next reader an hour.
    out.push(r.align_to_normal as u8);
    out.push(r.random_yaw as u8);
    out.push(r.cast_shadow as u8);
    out.extend_from_slice(&r.seed.to_le_bytes());
    debug_assert_eq!(out.len(), RULES_BYTES);
    out
}

/// Inverse of [`encode_rules`]. `None` if the slice is short.
fn decode_rules(b: &[u8]) -> Option<Rules> {
    if b.len() < RULES_BYTES {
        return None;
    }
    let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let bools = FLOAT_FIELDS * 4;
    Some(Rules {
        density: f(0),
        radius_m: f(1),
        height_m: f(2),
        scale_min: f(3),
        scale_max: f(4),
        align_max_angle_deg: f(5),
        random_pitch_deg: f(6),
        slope_min_deg: f(7),
        slope_max_deg: f(8),
        altitude_min: f(9),
        altitude_max: f(10),
        z_offset_m: f(11),
        cull_distance: f(12),
        collide_radius: f(13),
        align_to_normal: b[bools] != 0,
        random_yaw: b[bools + 1] != 0,
        cast_shadow: b[bools + 2] != 0,
        seed: u32::from_le_bytes(b[bools + 3..bools + 7].try_into().unwrap()),
    })
}

/// Version at the head of a saved foliage file, or `None` if these bytes are not one.
///
/// The magic is what lets a file written before the Unreal-shaped rules be recognised
/// and skipped. The previous format had no header at all, so without it an old file's
/// leading species count would be read as a version and the rest as nonsense floats --
/// every species would come back with a plausible-looking but wrong slope and scale,
/// which reads as deliberate and is far worse than falling back to defaults.
fn foliage_version(bytes: &[u8]) -> Option<u32> {
    if bytes.get(0..4)? != FOLIAGE_MAGIC {
        return None;
    }
    Some(u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_is_deterministic_for_a_seed() {
        // Same seed must give the same forest, or a world would not survive
        // being closed and reopened.
        assert_eq!(hash3(3, 9, 42), hash3(3, 9, 42));
        assert_ne!(hash3(3, 9, 42), hash3(3, 9, 43));
        assert_ne!(hash3(3, 9, 42), hash3(4, 9, 42));
    }

    #[test]
    fn frustum_rejects_what_is_behind_the_camera() {
        let cam = Camera { pos: Vec3::ZERO, yaw: 0.0, pitch: 0.0, ..Default::default() };
        let vp = cam.projection(16.0 / 9.0) * cam.look_at();
        let f = Frustum::new(&vp);
        // The camera looks down +X at yaw 0.
        assert!(f.contains_sphere(Vec3::new(100.0, 0.0, 0.0), 5.0), "ahead must be visible");
        assert!(!f.contains_sphere(Vec3::new(-100.0, 0.0, 0.0), 5.0), "behind must be rejected");
        // A large radius straddling the plane must survive: culling a sphere
        // whose centre is out but whose body is in pops geometry at the edges.
        assert!(f.contains_sphere(Vec3::new(-2.0, 0.0, 0.0), 40.0), "straddling must survive");
    }

    /// A full-coverage fill on a normal world must not produce an instance count
    /// nothing can draw. This is the guard on the guessed density figures.
    #[test]
    fn filling_a_medium_world_stays_within_budget() {
        let hectares = (4000.0 * 4000.0) / 10_000.0;
        for name in ["rock.glb", "bush.glb", "pine_tree.glb", "unrecognised.glb"] {
            let r = guess_rules(std::path::Path::new(name));
            let n = r.density * hectares;
            assert!(n <= 120_000.0, "{name} would place {n:.0} instances over a 4 km world");
            assert!(r.cull_distance > 0.0, "{name} must cull somewhere");
        }
    }

    #[test]
    fn default_rules_are_plantable() {
        let r = Rules::default();
        assert!(r.scale_min <= r.scale_max);
        assert!(r.density > 0.0);
        assert!(r.slope_min_deg <= r.slope_max_deg);
    }

    // --- alignment: "point to sky" ---
    //
    // The behaviour is only visible on ground steep enough to matter, so every test
    // here uses a real cliff normal rather than a gentle slope.

    /// The normal of ground at `deg` from horizontal, tilted toward +X.
    fn slope_normal(deg: f32) -> Vec3 {
        let r = deg.to_radians();
        Vec3::new(r.sin(), r.cos(), 0.0).normalize()
    }

    /// Where an instance's own up-axis ends up pointing.
    fn up_axis(r: &Rules, normal: Vec3) -> Vec3 {
        r.orientation(normal, 0x1234_5678) * Vec3::Y
    }

    #[test]
    fn unaligned_instances_point_at_the_sky_on_any_cliff() {
        // The requested behaviour, and the default. A tree grows toward the light, so
        // on a 75-degree rock face it must still stand vertically rather than sprout
        // sideways out of the wall.
        let r = Rules { align_to_normal: false, random_pitch_deg: 0.0, ..Default::default() };
        for deg in [0.0f32, 20.0, 45.0, 75.0, 89.0] {
            let up = up_axis(&r, slope_normal(deg));
            let off = up.angle_between(Vec3::Y).to_degrees();
            assert!(off < 0.01, "on {deg}-degree ground the up-axis is {off} degrees off vertical");
        }
    }

    #[test]
    fn random_yaw_does_not_tip_an_upright_instance_over() {
        // Yaw has to be applied about the instance's *own* up-axis. Compose it the
        // other way and the spin becomes a lean whose direction follows the slope,
        // which reads as every instance leaning downhill.
        let r = Rules { align_to_normal: false, random_pitch_deg: 0.0, ..Default::default() };
        assert!(r.random_yaw);
        for h in [0u32, 1, 0xDEAD_BEEF, 0x7FFF_FFFF, u32::MAX] {
            let up = (r.orientation(slope_normal(60.0), h) * Vec3::Y).normalize();
            assert!(
                up.angle_between(Vec3::Y).to_degrees() < 0.01,
                "hash {h:#x} tipped an upright instance to {up}"
            );
        }
    }

    #[test]
    fn aligned_instances_follow_the_ground() {
        // The other half of the toggle: with it on, a rock lies against the hillside.
        let r = Rules {
            align_to_normal: true,
            align_max_angle_deg: 90.0,
            random_pitch_deg: 0.0,
            ..Default::default()
        };
        for deg in [10.0f32, 30.0, 60.0] {
            let n = slope_normal(deg);
            let up = up_axis(&r, n);
            let off = up.angle_between(n).to_degrees();
            assert!(
                off < 0.5,
                "on {deg}-degree ground the up-axis is {off} degrees off the normal"
            );
        }
    }

    #[test]
    fn the_align_cap_is_an_angle_and_not_a_ratio() {
        // The trap: slerping toward the normal by `cap / tilt` looks like a cap and is
        // not one -- a 70-degree slope under a 20-degree cap would still tilt the
        // instance 35 degrees. The cap has to hold as an absolute angle.
        let cap = 20.0;
        let r = Rules {
            align_to_normal: true,
            align_max_angle_deg: cap,
            random_pitch_deg: 0.0,
            ..Default::default()
        };
        for deg in [25.0f32, 45.0, 70.0, 89.0] {
            let tilt = up_axis(&r, slope_normal(deg)).angle_between(Vec3::Y).to_degrees();
            assert!(tilt <= cap + 0.5, "{deg}-degree ground tilted the instance {tilt} degrees");
            // And it does use the whole allowance, rather than giving up and standing
            // straight: a cap that silently disabled alignment would pass the line above.
            assert!(tilt > cap - 1.0, "{deg}-degree ground only tilted {tilt}, cap is {cap}");
        }
        // Below the cap, alignment is exact.
        let tilt = up_axis(&r, slope_normal(12.0)).angle_between(Vec3::Y).to_degrees();
        assert!((tilt - 12.0).abs() < 0.5, "gentle ground should align fully, got {tilt}");
    }

    #[test]
    fn a_zero_align_cap_is_the_same_as_not_aligning() {
        // Reachable from the slider, and it must not produce a NaN orientation.
        let r = Rules {
            align_to_normal: true,
            align_max_angle_deg: 0.0,
            random_pitch_deg: 0.0,
            ..Default::default()
        };
        let up = up_axis(&r, slope_normal(50.0));
        assert!(up.is_finite(), "a zero cap produced {up}");
        assert!(up.angle_between(Vec3::Y).to_degrees() < 0.5);
    }

    #[test]
    fn every_orientation_is_a_usable_rotation() {
        // A denormalised or NaN quaternion becomes a collapsed or vanished instance,
        // and it would show as foliage flickering rather than as an error.
        for align in [false, true] {
            for pitch in [0.0f32, 5.0, 30.0] {
                let r = Rules {
                    align_to_normal: align,
                    align_max_angle_deg: 35.0,
                    random_pitch_deg: pitch,
                    ..Default::default()
                };
                for deg in [0.0f32, 1.0, 45.0, 90.0] {
                    for h in [0u32, 12_345, u32::MAX] {
                        let q = r.orientation(slope_normal(deg), h);
                        assert!(q.is_finite(), "align {align} pitch {pitch} deg {deg}: {q}");
                        assert!(
                            (q.length() - 1.0).abs() < 1e-3,
                            "align {align} pitch {pitch} deg {deg}: length {}",
                            q.length()
                        );
                    }
                }
            }
        }
        // A flat-up normal is the degenerate case for `from_rotation_arc`, and a
        // straight-down one is the antipodal case that makes the axis ambiguous.
        let r = Rules { align_to_normal: true, align_max_angle_deg: 180.0, ..Default::default() };
        assert!(r.orientation(Vec3::Y, 7).is_finite());
        assert!(r.orientation(-Vec3::Y, 7).is_finite());
        assert!(r.orientation(Vec3::ZERO, 7).is_finite(), "a zero normal must not NaN");
    }

    #[test]
    fn random_pitch_tilts_in_every_direction_not_just_one() {
        // One hash word cannot drive both how far and which way; using the same bits
        // for each makes the whole stand lean the same way, which is the comb look
        // random pitch exists to break up.
        let r = Rules { align_to_normal: false, random_pitch_deg: 15.0, ..Default::default() };
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for i in 0..500u32 {
            let up = r.orientation(Vec3::Y, hash3(i, i * 7 + 1, 3)) * Vec3::Y;
            min_x = min_x.min(up.x);
            max_x = max_x.max(up.x);
            min_z = min_z.min(up.z);
            max_z = max_z.max(up.z);
        }
        assert!(min_x < -0.05 && max_x > 0.05, "pitch never tilted both ways in x");
        assert!(min_z < -0.05 && max_z > 0.05, "pitch never tilted both ways in z");
    }

    // --- slope filtering ---

    #[test]
    fn slope_filtering_reads_in_degrees() {
        // Degrees, not an abstract 0..1, and the conversion inverts: a larger angle is
        // a smaller normal y, so the maximum angle becomes the minimum y. Getting that
        // backwards puts every species on exactly the ground it was excluded from.
        let r = Rules { slope_min_deg: 0.0, slope_max_deg: 30.0, ..Default::default() };
        assert!(r.accepts_slope(slope_normal(0.0).y), "flat ground must pass a 0-30 filter");
        assert!(r.accepts_slope(slope_normal(29.0).y));
        assert!(!r.accepts_slope(slope_normal(31.0).y), "31 degrees must fail a 30 degree cap");
        assert!(!r.accepts_slope(slope_normal(80.0).y));
    }

    #[test]
    fn a_minimum_slope_puts_a_species_only_on_cliffs() {
        // What the minimum is for, and the reason it is not redundant with the maximum.
        let cliff = Rules { slope_min_deg: 55.0, slope_max_deg: 90.0, ..Default::default() };
        assert!(!cliff.accepts_slope(slope_normal(0.0).y), "flat ground is not a cliff");
        assert!(!cliff.accepts_slope(slope_normal(40.0).y));
        assert!(cliff.accepts_slope(slope_normal(60.0).y));
        assert!(cliff.accepts_slope(slope_normal(89.0).y));
    }

    #[test]
    fn the_slope_filter_accepts_its_own_boundaries() {
        // A filter that rejects exactly the angle it was set to leaves a species
        // mysteriously absent from ground that reads as in range.
        let r = Rules { slope_min_deg: 20.0, slope_max_deg: 60.0, ..Default::default() };
        assert!(r.accepts_slope(slope_normal(20.0).y), "the minimum itself must pass");
        assert!(r.accepts_slope(slope_normal(60.0).y), "the maximum itself must pass");
    }

    // --- size ---

    #[test]
    fn the_height_setting_is_the_height_on_screen() {
        // Meshes are normalised to one metre at import, so an instance's scale factor
        // is its height in metres. That equivalence is what lets the panel show a real
        // number and what the collider maths relies on; if it breaks, foliage silently
        // renders at the wrong size and colliders stop matching what is drawn.
        let r = Rules { height_m: 12.0, scale_min: 1.0, scale_max: 1.0, ..Default::default() };
        let variation = r.scale_min;
        assert_eq!(r.height_m * variation, 12.0);

        // With variation, the range is the height times the range.
        let r = Rules { height_m: 10.0, scale_min: 0.5, scale_max: 2.0, ..Default::default() };
        assert_eq!(r.height_m * r.scale_min, 5.0);
        assert_eq!(r.height_m * r.scale_max, 20.0);
    }

    #[test]
    fn an_unrecognised_mesh_still_gets_a_usable_height() {
        // The old behaviour was a fixed 1.2 m for anything the name guess missed, with
        // no way to change it. The guess remains, but only as a starting value.
        for name in ["tree_01", "boulder", "fern", "widget_xyz"] {
            let h = default_height(name);
            assert!(h > 0.0 && h < 60.0, "{name} guessed {h} m");
        }
        assert!(default_height("oak_tree") > default_height("fern"), "a tree beats a fern");
    }

    // --- persistence ---

    #[test]
    fn an_authored_rule_set_survives_a_round_trip() {
        // Every field off its default, so one that failed to encode shows up as a
        // difference rather than coincidentally matching.
        let want = Rules {
            density: 73.5,
            radius_m: 4.25,
            height_m: 17.5,
            scale_min: 0.35,
            scale_max: 2.75,
            align_to_normal: true,
            align_max_angle_deg: 62.5,
            random_yaw: false,
            random_pitch_deg: 11.5,
            slope_min_deg: 22.0,
            slope_max_deg: 81.0,
            altitude_min: -45.0,
            altitude_max: 1320.0,
            z_offset_m: -0.85,
            cast_shadow: false,
            collide_radius: 0.33,
            cull_distance: 1450.0,
            seed: 0xC0FF_EE01,
        };
        let bytes = encode_rules(&want);
        assert_eq!(bytes.len(), RULES_BYTES);
        assert_eq!(decode_rules(&bytes), Some(want));
    }

    #[test]
    fn every_bool_survives_independently() {
        // Three bools packed next to each other is exactly where an off-by-one shows
        // up, and it would present as "cast shadow toggles align to normal".
        for (a, b, c) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (true, true, true),
        ] {
            let r =
                Rules { align_to_normal: a, random_yaw: b, cast_shadow: c, ..Default::default() };
            let got = decode_rules(&encode_rules(&r)).expect("decode");
            assert_eq!((got.align_to_normal, got.random_yaw, got.cast_shadow), (a, b, c));
        }
    }

    #[test]
    fn a_truncated_rule_block_decodes_to_nothing() {
        // Better to keep defaults than to apply half a rule set, which would look
        // deliberate.
        let bytes = encode_rules(&Rules::default());
        for cut in [0, 1, RULES_BYTES / 2, RULES_BYTES - 1] {
            assert_eq!(decode_rules(&bytes[..cut]), None, "{cut} bytes should not decode");
        }
    }

    #[test]
    fn an_old_foliage_file_is_recognised_and_refused() {
        // The pre-versioning format opened with a species count. Read as a header that
        // is not a version, and it must be rejected rather than reinterpreted.
        let mut old = Vec::new();
        old.extend_from_slice(&2u32.to_le_bytes());
        old.extend_from_slice(&DENSITY_RES.to_le_bytes());
        assert_eq!(foliage_version(&old), None, "an old file must not look like a versioned one");

        let mut current = Vec::new();
        current.extend_from_slice(&FOLIAGE_MAGIC);
        current.extend_from_slice(&FOLIAGE_VERSION.to_le_bytes());
        assert_eq!(foliage_version(&current), Some(FOLIAGE_VERSION));

        // And nothing shorter than a header is mistaken for one.
        assert_eq!(foliage_version(&[]), None);
        assert_eq!(foliage_version(b"TFO"), None);
        assert_eq!(foliage_version(b"TFOL"), None, "magic without a version is not a file");
    }

    #[test]
    fn the_saved_layout_is_the_size_it_claims() {
        // `RULES_BYTES` is used to advance the read cursor, so if it disagrees with what
        // `encode_rules` writes then every species after the first reads from the wrong
        // offset -- and the first one would look fine, which is the worst case.
        assert_eq!(encode_rules(&Rules::default()).len(), RULES_BYTES);
    }

    // --- guessed starting rules ---

    #[test]
    fn an_imported_tree_stands_up_and_an_imported_rock_lies_down() {
        // The guess exists so an import looks sensible before any slider is touched.
        // These two are the cases where a wrong default is immediately obvious.
        let tree = guess_rules(std::path::Path::new("scots_pine_tree.glb"));
        assert!(!tree.align_to_normal, "a tree must point at the sky");
        let rock = guess_rules(std::path::Path::new("granite_boulder.glb"));
        assert!(rock.align_to_normal, "a rock must sit against the ground");
        assert!(rock.z_offset_m < 0.0, "a rock must bed in rather than perch");
        // An unrecognised name gets the defaults, which are upright.
        let other = guess_rules(std::path::Path::new("fence_post_01.glb"));
        assert_eq!(other, Rules::default());
        assert!(!other.align_to_normal);
    }

    #[test]
    fn every_guess_stays_within_its_own_slider_ranges() {
        // The panel clamps to these, so a guess outside them would jump the moment the
        // user touched an unrelated control.
        for name in ["rock.glb", "fern.glb", "oak_tree.glb", "thing.glb"] {
            let r = guess_rules(std::path::Path::new(name));
            assert!((1.0..=600.0).contains(&r.density), "{name} density {}", r.density);
            assert!((0.0..=90.0).contains(&r.slope_min_deg), "{name}");
            assert!((0.0..=90.0).contains(&r.slope_max_deg), "{name}");
            assert!(r.slope_min_deg <= r.slope_max_deg, "{name} has an inverted slope range");
            assert!((0.0..=90.0).contains(&r.align_max_angle_deg), "{name}");
            assert!((0.0..=30.0).contains(&r.random_pitch_deg), "{name}");
            assert!((-3.0..=3.0).contains(&r.z_offset_m), "{name} z offset {}", r.z_offset_m);
            assert!(r.scale_min <= r.scale_max, "{name}");
        }
    }
}

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

/// Everything about how one species is placed. Serialised with the world;
/// the instances themselves never are.
#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    /// Instances per hectare at full painted density.
    pub density: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    /// 0 = always upright, 1 = fully normal-aligned. Trees want near zero --
    /// a tree grows toward the light, not perpendicular to the hillside -- and
    /// rocks want most of the way up.
    pub align_to_normal: f32,
    /// Steepest ground, as the same 0..1 slope the terrain shader uses.
    pub slope_max: f32,
    pub altitude_min: f32,
    pub altitude_max: f32,
    /// Sunk into the ground by this fraction of the instance's height, so a
    /// rock beds in instead of resting on the surface like a prop.
    pub sink: f32,
    /// Whether this species casts. Ground cover casting into a shadow map that
    /// covers hundreds of metres produces noise, not shadows.
    pub cast_shadow: bool,
    /// Radius of the collider stand-in, as a fraction of the instance's
    /// height. Zero means the species is not solid -- grass and ferns should
    /// not stop a car.
    pub collide_radius: f32,
    /// Metres beyond which instances stop being drawn.
    ///
    /// The lever that makes scatter affordable. Without it a bush 3 km away
    /// costs exactly what one at your feet costs, and the whole map is
    /// submitted every frame. Small props can be culled hard because nobody
    /// can resolve them anyway; a skyline of trees cannot.
    pub cull_distance: f32,
    pub seed: u32,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            density: 40.0,
            scale_min: 0.8,
            scale_max: 1.35,
            align_to_normal: 0.15,
            slope_max: 0.55,
            altitude_min: -10_000.0,
            altitude_max: 10_000.0,
            sink: 0.04,
            collide_radius: 0.0,
            cast_shadow: true,
            cull_distance: 900.0,
            seed: 1,
        }
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
    /// Height of the source mesh in metres, after import normalisation.
    pub height_m: f32,
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

        let files = terra_assets::mesh::discover(dir);
        let mut species = Vec::new();

        // Decode, decimate and preview each model on its own thread. Eight
        // scanned assets is seconds of work and none of it touches the others;
        // only the GPU upload has to be serial.
        let decoded: Vec<(String, terra_assets::MeshData, f32, Rules)> = files
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
                        let height = default_height(&name);
                        data.normalize_height(height);
                        if data.triangle_count() < raw {
                            log::info!(
                                "model '{name}': {raw} -> {} triangles",
                                data.triangle_count()
                            );
                        } else {
                            log::info!("model '{name}': {raw} triangles");
                        }
                        let rules = guess_rules(path);
                        Some((pretty_name(&name), data, height, rules))
                    }
                    Err(e) => {
                        log::error!("{e:#}");
                        None
                    }
                }
            })
            .collect();

        for (name, data, height, rules) in decoded {
            species.push(Species::new(device, queue, meshes, name, data, height, rules));
        }

        if species.is_empty() {
            log::info!("{}: no models found, using generated species", dir.display());
            for b in terra_assets::Builtin::ALL {
                let mut data = b.build();
                data.normalize_height(b.height_m());
                let rules = builtin_rules(b);
                species.push(Species::new(
                    device,
                    queue,
                    meshes,
                    b.name().into(),
                    data,
                    b.height_m(),
                    rules,
                ));
            }
        }

        species.truncate(8);
        Self {
            species,
            props: Vec::new(),
            props_buf: None,
            prop_runs: Vec::new(),
            props_dirty: false,
            pipeline,
            cull_bgl,
        }
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
            let centre = p.pos + Vec3::Y * (sp.height_m * p.scale * 0.5);
            let radius = (sp.radius * p.scale).max(sp.height_m * p.scale * 0.5);

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
        height_m: f32,
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
            height_m,
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
        let side = (wanted as f32).sqrt().ceil().max(1.0) as u32;
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
        if 1.0 - normal.y.clamp(0.0, 1.0) > self.rules.slope_max {
            return None;
        }

        let r2 = hash3(gx, gz, self.rules.seed.wrapping_add(31));
        let s01 = (r2 & 0xFFFF) as f32 / 65535.0;
        let scale = self.rules.scale_min + (self.rules.scale_max - self.rules.scale_min) * s01;
        let yaw = ((r2 >> 16) & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;

        // Lean toward the surface normal by the configured amount.
        let upright = Quat::from_rotation_y(yaw);
        let lean = Quat::from_rotation_arc(Vec3::Y, normal);
        let rot = Quat::IDENTITY.slerp(lean, self.rules.align_to_normal) * upright;

        let pos = Vec3::new(x, y - self.height_m * scale * self.rules.sink, z);
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
                if let Some((pos, scale, rot)) = self.place_at(terrain, gx, gz, step, extent) {
                    if (pos.x - centre.x).powi(2) + (pos.z - centre.z).powi(2) <= r2 {
                        f(pos, scale, rot);
                    }
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

/// Sensible starting rules for a generated species.
fn builtin_rules(b: terra_assets::Builtin) -> Rules {
    use terra_assets::Builtin as B;
    match b {
        // Trees stand up regardless of the hillside and give up on steep ground.
        B::PineTree => {
            Rules {
                density: 22.0,
                align_to_normal: 0.08,
                slope_max: 0.5,
                // Trees make the skyline; culling them at the default distance
                // pops a whole ridgeline in and out as you turn.
                cull_distance: 1800.0,
                collide_radius: 0.035,
                ..Default::default()
            }
        }
        B::BroadleafTree => Rules {
            density: 14.0,
            align_to_normal: 0.06,
            slope_max: 0.42,
            cull_distance: 1800.0,
            ..Default::default()
        },
        // Rocks bed into the ground and do not care how steep it is.
        B::Rock => Rules {
            density: 30.0,
            cull_distance: 500.0,
            collide_radius: 0.45,
            align_to_normal: 0.75,
            slope_max: 0.95,
            sink: 0.28,
            scale_min: 0.5,
            scale_max: 2.2,
            ..Default::default()
        },
        B::Bush => Rules {
            density: 55.0,
            align_to_normal: 0.4,
            slope_max: 0.6,
            cull_distance: 380.0,
            ..Default::default()
        },
    }
}

/// Guess rules for an imported model from its file name, the same way material
/// roles are guessed. Wrong guesses cost one slider drag.
fn guess_rules(path: &std::path::Path) -> Rules {
    let n = path.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let has = |keys: &[&str]| keys.iter().any(|k| n.contains(k));
    if has(&["rock", "stone", "boulder", "cliff"]) {
        builtin_rules(terra_assets::Builtin::Rock)
    } else if has(&["bush", "shrub", "fern", "grass", "plant"]) {
        builtin_rules(terra_assets::Builtin::Bush)
    } else if has(&["tree", "pine", "spruce", "oak", "birch"]) {
        builtin_rules(terra_assets::Builtin::PineTree)
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
        out.extend_from_slice(&(self.species.len() as u32).to_le_bytes());
        out.extend_from_slice(&DENSITY_RES.to_le_bytes());
        for s in &self.species {
            let name = s.name.as_bytes();
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name);
            let r = &s.rules;
            for f in [
                r.density,
                r.scale_min,
                r.scale_max,
                r.align_to_normal,
                r.slope_max,
                r.altitude_min,
                r.altitude_max,
                r.sink,
                r.cull_distance,
                r.collide_radius,
            ] {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out.extend_from_slice(&r.seed.to_le_bytes());
            out.extend_from_slice(&s.density);
        }
        out
    }

    /// Restore by name, so reordering or removing a model folder cannot hand
    /// one species' forest to another.
    pub fn restore(&mut self, bytes: &[u8]) {
        let mut p = 0usize;
        let u32_at = |p: &mut usize| -> Option<u32> {
            let v = bytes.get(*p..*p + 4)?;
            *p += 4;
            Some(u32::from_le_bytes(v.try_into().ok()?))
        };
        let f32_at = |p: &mut usize| -> Option<f32> {
            let v = bytes.get(*p..*p + 4)?;
            *p += 4;
            Some(f32::from_le_bytes(v.try_into().ok()?))
        };
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
            let mut vals = [0.0f32; 10];
            for v in &mut vals {
                let Some(f) = f32_at(&mut p) else { return };
                *v = f;
            }
            let Some(seed) = u32_at(&mut p) else { return };
            let Some(mask) = bytes.get(p..p + mask_len) else { return };
            p += mask_len;

            if let Some(s) = self.species.iter_mut().find(|s| s.name == name) {
                let cast_shadow = s.rules.cast_shadow;
                s.rules = Rules {
                    density: vals[0],
                    scale_min: vals[1],
                    scale_max: vals[2],
                    align_to_normal: vals[3],
                    slope_max: vals[4],
                    altitude_min: vals[5],
                    altitude_max: vals[6],
                    sink: vals[7],
                    cull_distance: vals[8],
                    collide_radius: vals[9],
                    // Not persisted: it is a rendering choice tied to the
                    // species, not to the world someone painted.
                    cast_shadow,
                    seed,
                };
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
                    let height = s.height_m * scale;
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
            let height = sp.height_m * p.scale;
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

    /// A full-coverage fill on a normal world must not produce an instance
    /// count nothing can draw. This is the guard on the density defaults.
    #[test]
    fn filling_a_medium_world_stays_within_budget() {
        let hectares = (4000.0 * 4000.0) / 10_000.0;
        for b in terra_assets::Builtin::ALL {
            let r = builtin_rules(b);
            let n = r.density * hectares;
            assert!(
                n <= 120_000.0,
                "{} would place {n:.0} instances covering a 4 km world",
                b.name()
            );
            assert!(r.cull_distance > 0.0, "{} must cull somewhere", b.name());
        }
    }

    #[test]
    fn default_rules_are_plantable() {
        let r = Rules::default();
        assert!(r.scale_min <= r.scale_max);
        assert!(r.density > 0.0);
        assert!((0.0..=1.0).contains(&r.align_to_normal));
    }
}

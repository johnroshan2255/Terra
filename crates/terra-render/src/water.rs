//! Water bodies.
//!
//! A blended surface over CDLOD patches, drawn after everything opaque. The shape of
//! the feature follows Unreal's Water plugin and Crest -- Gerstner waves, depth-driven
//! absorption, Fresnel against a sky reflection, shoreline foam -- with two deliberate
//! departures.
//!
//! # Depth comes from the heightfield, not from a capture
//!
//! Unreal renders a top-down *Water Info* texture per Water Zone to learn how deep the
//! water is over the ground, because a shader cannot read the depth buffer it is
//! writing. That texture costs a pass, a resolution choice and a bounded region.
//!
//! None of it is needed here: the terrain heightfield is already a storage buffer, so
//! depth is `level - terrain_height` evaluated exactly wherever it is asked for. This
//! is the one place the renderer's existing design makes water simpler rather than
//! harder, and it is why there is no water-zone concept -- the whole world is the zone.
//!
//! # The geometry is the terrain's own LOD
//!
//! Unreal's Water Mesh is a separate CLOD quadtree. [`crate::cdlod`] already is one, so
//! the water surface is the same patch selection with a different height source. It
//! costs one instance buffer and no new machinery, and the water tessellates finely
//! near the camera for free -- which Gerstner waves need, since they displace vertices.
//!
//! # No refraction
//!
//! Refraction needs the scene colour behind the surface, which is the copy this avoids.
//! Absorption drives the blend alpha instead: shallows are transparent and the ground
//! shows through by ordinary blending, deep water goes opaque. Same physics, resolved
//! by the blender -- it simply cannot bend what it shows. A later pass with a scene
//! copy can replace the alpha term without touching anything else here.

use crate::camera::{Camera, CameraUniform};

use crate::cdlod::{self, Cdlod, PATCH_QUADS};
use crate::context::{DEPTH_FORMAT, SCENE_FORMAT};
use crate::frustum::{Frustum, FrustumUnion};
use crate::lighting::Lighting;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Vertex spacing the water surface aims for at the finest level, in metres.
///
/// Coarser than the terrain's 0.5 m: waves are metres long, and a vertex every half
/// metre spends its budget resolving a curve that is already smooth. Two metres still
/// leaves a 20 m wave with ten vertices along it.
const TARGET_SPACING_M: f32 = 2.0;

/// The most regions one world may hold.
///
/// Eight, matching the material palette's ceiling and for the same reason: the shader
/// walks them per fragment, so the count is a fixed-size uniform array rather than a
/// growable buffer. Eight lakes is more than a 4 km world wants and the loop stays
/// short enough not to show up in a frame time.
pub const MAX_REGIONS: usize = 8;

/// One rectangular body of water with its own level and its own waves.
///
/// A rectangle rather than a spline or a polygon, because it is a drag in the viewport
/// and there is nothing to edit afterwards but four numbers -- and because the shoreline
/// does not come from the shape at all. It comes from where the ground crosses the
/// level, exactly as the global surface does, so a rectangle over a lumpy basin already
/// produces an irregular shore. The rectangle only says *where to look*.
///
/// Fill only: placing one never reshapes the ground under it. That keeps placement
/// non-destructive and instant, at the cost of a region over ground that is entirely
/// above its level showing nothing -- which is the honest answer rather than a silent
/// terrain edit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WaterRegion {
    /// Minimum corner in world XZ.
    pub min: [f32; 2],
    /// Maximum corner in world XZ.
    pub max: [f32; 2],
    /// Surface height for this body, independent of the global level. This is what makes
    /// a hillside tarn possible above a valley lake.
    pub level_m: f32,
    pub wave_height_m: f32,
    pub wave_length_m: f32,
    /// Wave speed for *this* body. A sheltered pond and an open reservoir under the same
    /// wind do not move the same way, which is the whole point of these being per region.
    pub wave_speed: f32,
    pub wind_deg: f32,
}

impl WaterRegion {
    /// A region spanning two dragged corners, in either order.
    ///
    /// Normalised here rather than trusted, because a drag can finish up-left of where it
    /// started and an unsorted pair would make `contains` false everywhere.
    pub fn from_drag(a: [f32; 2], b: [f32; 2], level_m: f32) -> Self {
        Self {
            min: [a[0].min(b[0]), a[1].min(b[1])],
            max: [a[0].max(b[0]), a[1].max(b[1])],
            level_m,
            ..Default::default()
        }
    }

    pub fn contains(&self, x: f32, z: f32) -> bool {
        x >= self.min[0] && x <= self.max[0] && z >= self.min[1] && z <= self.max[1]
    }

    /// Width and depth in metres.
    pub fn size(&self) -> [f32; 2] {
        [self.max[0] - self.min[0], self.max[1] - self.min[1]]
    }

    /// Whether the drag produced something worth keeping.
    ///
    /// A stray click is a zero-area region, which would be invisible, unselectable and
    /// impossible to get rid of without hand-editing the file.
    pub fn is_usable(&self) -> bool {
        let s = self.size();
        s[0] >= MIN_REGION_M && s[1] >= MIN_REGION_M
    }

    pub fn wind_dir(&self) -> [f32; 2] {
        let r = self.wind_deg.to_radians();
        [r.cos(), r.sin()]
    }
}

impl Default for WaterRegion {
    fn default() -> Self {
        let w = WaterSettings::default();
        Self {
            min: [0.0, 0.0],
            max: [0.0, 0.0],
            level_m: w.level_m,
            // The global defaults, so a new region starts looking like the sea and is
            // tuned away from it rather than from nothing.
            wave_height_m: w.wave_height_m,
            wave_length_m: w.wave_length_m,
            wave_speed: w.wave_speed,
            wind_deg: w.wind_deg,
        }
    }
}

/// Shortest side a dragged region may have, in metres.
///
/// Below this it is a mis-click rather than a lake.
pub const MIN_REGION_M: f32 = 2.0;

/// Authored water settings for one world.
///
/// One global level rather than per-body regions, and that is a real limitation stated
/// plainly: it fills every basin the erosion solver dug to the same height, which is
/// what a sea does and is not what a hillside tarn does. Separate bodies need a region
/// list and a per-region level; the shader already takes level as a uniform, so that is
/// an addition rather than a rework.
// Not `Copy`: it owns a region list now. Cloned at the two places that need a snapshot,
// which is once a frame for the uniform and once on save.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterSettings {
    pub enabled: bool,
    /// Surface height in metres, in the same frame as the heightfield.
    pub level_m: f32,
    /// Colour of a thin film of water, before absorption takes hold.
    pub shallow: [f32; 3],
    /// Colour absorption converges on with depth.
    pub deep: [f32; 3],
    /// Absorption per metre. Larger goes opaque sooner.
    pub absorption: f32,
    /// Peak wave amplitude in metres, before the shallow-water fade.
    pub wave_height_m: f32,
    /// Length of the longest wave in the stack, in metres.
    pub wave_length_m: f32,
    /// Multiplier on the physical deep-water wave speed.
    pub wave_speed: f32,
    /// Wind heading in degrees, which is the direction the waves travel.
    pub wind_deg: f32,
    /// Width of the shoreline foam band, in metres of depth.
    pub foam_width_m: f32,
    /// Surface roughness, driving the specular lobe.
    pub roughness: f32,
    /// Bodies with their own level and waves, over and above the global surface.
    ///
    /// Defaulted so a `water.ron` written before regions existed still loads.
    #[serde(default)]
    pub regions: Vec<WaterRegion>,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            // Off. A world is landscape until someone decides otherwise, and a default
            // sea level would flood the valleys of every existing project on load.
            enabled: false,
            // The elevation a flat world starts at, so switching water on puts the
            // surface exactly at the ground and the level slider reads against
            // something rather than starting a kilometre up or down.
            level_m: terra_core::BASE_ELEVATION_M,
            // Coastal water: green-blue shallows over a deep blue. Linear, not sRGB --
            // everything downstream of here works in linear.
            shallow: [0.10, 0.30, 0.28],
            deep: [0.010, 0.045, 0.075],
            // Visibly clearing by a couple of metres and opaque by fifteen, which is
            // about right for lake water and reads legibly at editor distances.
            absorption: 0.16,
            wave_height_m: 0.35,
            wave_length_m: 22.0,
            wave_speed: 1.0,
            wind_deg: 35.0,
            foam_width_m: 1.2,
            roughness: 0.06,
            regions: Vec::new(),
        }
    }
}

impl WaterSettings {
    /// Wind as a unit vector in world XZ.
    ///
    /// `atan2(z, x)` convention, matching `Camera::yaw` and `Vehicle::heading`, so a
    /// heading means the same thing everywhere in the project.
    pub fn wind_dir(&self) -> [f32; 2] {
        let r = self.wind_deg.to_radians();
        [r.cos(), r.sin()]
    }

    /// Write to `edits/water.ron`, beside the environment.
    pub fn save(&self, paths: &terra_project::ProjectPaths) -> std::io::Result<()> {
        let path = paths.water();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(self, cfg)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, text)
    }

    /// Read it back, or `None` when a world has none.
    ///
    /// `None` rather than an error for a missing file: every world created before this
    /// existed simply has no water, and that must load rather than fail.
    pub fn load(paths: &terra_project::ProjectPaths) -> Option<Self> {
        let text = std::fs::read_to_string(paths.water()).ok()?;
        match ron::from_str(&text) {
            Ok(v) => Some(v),
            Err(e) => {
                log::warn!("ignoring unreadable {}: {e}", paths.water().display());
                None
            }
        }
    }
}

/// Mirrors `WaterUniform` in `water.wgsl`.
///
/// Every member is a `vec4`, which is the same discipline `EnvironmentUniform` follows
/// and for the same reason: WGSL rounds uniform members up to 16-byte alignment, so a
/// lone `f32` between two vectors inserts padding the Rust side does not, and the whole
/// block reads shifted from there on.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WaterUniform {
    shallow: [f32; 4],
    deep: [f32; 4],
    wave: [f32; 4],
    surface: [f32; 4],
    grid: [f32; 4],
    eye: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<WaterUniform>() == 96);

/// Mirrors `WaterRegion` in `water.wgsl`. Three `vec4`s, same discipline as above.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RegionUniform {
    bounds: [f32; 4],
    params: [f32; 4],
    wind: [f32; 4],
}

// 48 is a multiple of 16, which a uniform array's stride has to be.
const _: () = assert!(std::mem::size_of::<RegionUniform>() == 48);
const _: () = assert!(std::mem::size_of::<RegionUniform>().is_multiple_of(16));

pub struct Water {
    pipeline: wgpu::RenderPipeline,
    camera_ub: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    water_ub: wgpu::Buffer,
    region_ub: wgpu::Buffer,
    water_bg: wgpu::BindGroup,
    indices: wgpu::Buffer,
    index_count: u32,
    patch_buf: wgpu::Buffer,
    cdlod: Cdlod,
    patch_count: u32,
    extent_m: f32,
    height_res: u32,
}

impl Water {
    /// `heights` is the terrain's own height storage buffer, read for depth.
    ///
    /// Takes a device rather than a [`RenderContext`], because that is all it needs and
    /// a context cannot be built without a window -- which would make the pipeline
    /// untestable, and this pipeline has the most to get wrong: it is the renderer's
    /// only blended one and its bind groups span four layouts.
    pub fn new(
        device: &wgpu::Device,
        lighting: &Lighting,
        env: &crate::environment::EnvironmentGpu,
        heights: &wgpu::Buffer,
        extent_m: f32,
        height_res: u32,
    ) -> Self {
        let camera_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let water_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-uniform"),
            size: std::mem::size_of::<WaterUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let region_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-regions"),
            size: (MAX_REGIONS * std::mem::size_of::<RegionUniform>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cdlod = Cdlod::new(extent_m, TARGET_SPACING_M);
        let patch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-patches"),
            size: cdlod.buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let idx = cdlod::patch_indices();
        let index_count = idx.len() as u32;
        let indices = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("water-indices"),
                contents: bytemuck::cast_slice(&idx),
                usage: wgpu::BufferUsages::INDEX,
            },
        );

        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-camera-bgl"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-camera-bg"),
            layout: &cam_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_ub.as_entire_binding(),
            }],
        });

        let water_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, wgpu::ShaderStages::VERTEX),
                uniform_entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT),
            ],
        });
        let water_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-bg"),
            layout: &water_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: water_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: heights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: patch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: region_ub.as_entire_binding() },
            ],
        });

        // Composed the way the terrain composes its own: the common modules first, so
        // `atmosphere` and `cdlod_vertex_xz` are the same code the sky and the ground
        // use rather than a second copy that can drift.
        let source = format!(
            "{}\n{}\n{}\n{}\n{}",
            include_str!("../../../assets/shaders/common/noise.wgsl"),
            include_str!("../../../assets/shaders/common/camera.wgsl"),
            include_str!("../../../assets/shaders/common/lighting.wgsl"),
            include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
            include_str!("../../../assets/shaders/common/cdlod.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{source}\n{}", include_str!("../../../assets/shaders/render/water.wgsl"))
                    .into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water-layout"),
            // Camera 0, lighting 1, env 2, water 3. The order is forced by
            // `atmosphere.wgsl` owning group 2, not chosen.
            bind_group_layouts: &[
                Some(&cam_bgl),
                Some(&lighting.layout),
                Some(&env.layout),
                Some(&water_bgl),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    // The first blended pipeline in this renderer. Straight source-alpha
                    // over: the alpha the shader returns is the absorption, so shallow
                    // water lets the ground it is drawn over show through and deep water
                    // covers it. This is what stands in for a refraction fetch.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        // Alpha accumulates rather than being replaced, because the
                        // scene's alpha channel means "how much sky is behind this" for
                        // the god-ray march, and water is not sky.
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Both faces. The camera goes under the surface -- a vehicle drives into
                // a lake -- and a back-face-culled sheet vanishes from below, which reads
                // as the water having disappeared rather than as being under it.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // Tested but not written. Water must be hidden by ground in front of it,
                // and must not occlude anything drawn after it -- nor itself, since the
                // wave displacement makes a patch's own triangles overlap in depth.
                depth_write_enabled: Some(false),
                // Reversed-Z: nearer is greater.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_ub,
            camera_bg,
            water_ub,
            region_ub,
            water_bg,
            indices,
            index_count,
            patch_buf,
            cdlod,
            patch_count: 0,
            extent_m,
            height_res,
        }
    }

    /// Pick patches and push this frame's uniforms. Call before the scene pass.
    ///
    /// `time` drives the waves and is the same clock the clouds advect on, so a windy
    /// sky and a choppy surface stay in step.
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        settings: &WaterSettings,
        camera: &Camera,
        aspect: f32,
        time: f32,
    ) {
        // Something to draw means either the global surface or at least one region. A
        // world with water switched off but a lake placed still has a lake.
        if !settings.enabled && settings.regions.is_empty() {
            self.patch_count = 0;
            return;
        }
        queue.write_buffer(&self.camera_ub, 0, bytemuck::bytes_of(&camera.uniform(aspect)));

        // The patch selection's height range spans every body's surface plus its waves,
        // rather than the terrain's own span: a range taken from the ground would morph
        // patches against a slab the water does not occupy. Regions are included because
        // a tarn far above the sea widens the slab the LOD has to cover.
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        if settings.enabled {
            lo = lo.min(settings.level_m - settings.wave_height_m);
            hi = hi.max(settings.level_m + settings.wave_height_m);
        }
        for r in settings.regions.iter().take(MAX_REGIONS) {
            lo = lo.min(r.level_m - r.wave_height_m);
            hi = hi.max(r.level_m + r.wave_height_m);
        }
        if lo > hi {
            self.patch_count = 0;
            return;
        }
        // Frustum and quadtree culled, like the terrain. Water has no shadow pass of its
        // own -- it neither casts nor receives into the cascades -- so the light union is
        // empty and only the camera set is used.
        let frustum = Frustum::new(&(camera.projection(aspect) * camera.look_at()));
        self.cdlod.select_culled(
            camera.pos,
            (lo, hi),
            self.extent_m,
            Some(&frustum),
            &FrustumUnion::default(),
        );
        let patches = self.cdlod.patches();
        self.patch_count = patches.len() as u32;
        queue.write_buffer(&self.patch_buf, 0, bytemuck::cast_slice(patches));

        let wind = settings.wind_dir();
        let u = WaterUniform {
            shallow: [
                settings.shallow[0],
                settings.shallow[1],
                settings.shallow[2],
                settings.level_m,
            ],
            deep: [settings.deep[0], settings.deep[1], settings.deep[2], settings.absorption],
            wave: [settings.wave_height_m, settings.wave_length_m, settings.wave_speed, time],
            surface: [wind[0], wind[1], settings.foam_width_m, settings.roughness],
            grid: [
                self.extent_m,
                self.height_res as f32,
                PATCH_QUADS as f32,
                // How many regions the shader should walk.
                settings.regions.len().min(MAX_REGIONS) as f32,
            ],
            eye: [
                camera.pos.x,
                camera.pos.y,
                camera.pos.z,
                // Whether the *global* surface is on, which is separate from whether
                // there is anything to draw at all. Carried in the eye's unused w so the
                // uniform does not need a seventh vector for one flag.
                if settings.enabled { 1.0 } else { 0.0 },
            ],
        };
        queue.write_buffer(&self.water_ub, 0, bytemuck::bytes_of(&u));

        // Regions, padded to the full array. Slots past the count are never read, but a
        // uniform buffer has to be written whole or the tail keeps a previous frame's
        // rectangles -- which would matter the moment one is deleted.
        let mut packed = [RegionUniform::default(); MAX_REGIONS];
        for (slot, r) in packed.iter_mut().zip(settings.regions.iter().take(MAX_REGIONS)) {
            let wind = r.wind_dir();
            *slot = RegionUniform {
                bounds: [r.min[0], r.min[1], r.max[0], r.max[1]],
                params: [r.level_m, r.wave_height_m, r.wave_length_m, r.wave_speed],
                wind: [wind[0], wind[1], 0.0, 0.0],
            };
        }
        queue.write_buffer(&self.region_ub, 0, bytemuck::cast_slice(&packed));
    }

    /// Draw the surface. Must come after everything opaque in the same pass.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        env: &crate::environment::EnvironmentGpu,
    ) {
        if self.patch_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &lighting.bind_group, &[]);
        pass.set_bind_group(2, &env.bind_group, &[]);
        pass.set_bind_group(3, &self.water_bg, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.patch_count);
    }

    /// Triangles the surface submitted this frame, for the performance overlay.
    pub fn triangle_count(&self) -> u32 {
        self.patch_count * (self.index_count / 3)
    }

    /// Whether a world position is under the surface.
    ///
    /// On the CPU because gameplay asks: buoyancy, a drowning check, whether a wheel is
    /// in water. Waves are ignored deliberately -- a query that flickers with a passing
    /// crest is worse than one that answers about the mean surface.
    pub fn submerged(settings: &WaterSettings, p: Vec3) -> bool {
        settings.enabled && p.y < settings.level_m
    }

    /// Depth of water over ground of height `terrain_h`, in metres. Zero on dry land.
    ///
    /// The same expression the shader evaluates, kept here so CPU and GPU cannot
    /// disagree about where the shoreline is.
    pub fn depth_over(settings: &WaterSettings, terrain_h: f32) -> f32 {
        if !settings.enabled {
            return 0.0;
        }
        (settings.level_m - terrain_h).max(0.0)
    }

    /// Surface height at a world XZ, or `None` where there is no water.
    ///
    /// The CPU twin of the shader's `body_at`, and deliberately the same rule: the first
    /// containing region wins, then the global surface if it is on. Gameplay asking
    /// "am I in water" has to get the same answer the pixel did.
    pub fn level_at(settings: &WaterSettings, x: f32, z: f32) -> Option<f32> {
        for r in settings.regions.iter().take(MAX_REGIONS) {
            if r.contains(x, z) {
                return Some(r.level_m);
            }
        }
        settings.enabled.then_some(settings.level_m)
    }

    /// Depth of water over ground of height `terrain_h` at a world XZ, honouring regions.
    pub fn depth_at(settings: &WaterSettings, x: f32, z: f32, terrain_h: f32) -> f32 {
        Self::level_at(settings, x, z).map_or(0.0, |l| (l - terrain_h).max(0.0))
    }

    /// Whether a world position is under whichever body covers it.
    pub fn submerged_at(settings: &WaterSettings, p: Vec3) -> bool {
        Self::level_at(settings, p.x, p.z).is_some_and(|l| p.y < l)
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_is_off_until_asked_for() {
        // A default sea level would flood the valleys of every project that predates
        // this the moment it loaded.
        assert!(!WaterSettings::default().enabled);
    }

    #[test]
    fn the_default_level_is_the_ground_a_flat_world_starts_at() {
        // So switching water on puts the surface at the ground and the slider reads
        // against something, rather than starting a kilometre above or below it.
        assert_eq!(WaterSettings::default().level_m, terra_core::BASE_ELEVATION_M);
    }

    #[test]
    fn depth_is_zero_on_dry_land_and_positive_below_the_level() {
        let w = WaterSettings { enabled: true, level_m: 100.0, ..Default::default() };
        assert_eq!(Water::depth_over(&w, 120.0), 0.0, "ground above the level is dry");
        assert_eq!(Water::depth_over(&w, 100.0), 0.0, "exactly at the level is dry");
        assert_eq!(Water::depth_over(&w, 90.0), 10.0);
    }

    #[test]
    fn disabled_water_has_no_depth_anywhere() {
        let w = WaterSettings { enabled: false, level_m: 100.0, ..Default::default() };
        assert_eq!(Water::depth_over(&w, -500.0), 0.0);
        assert!(!Water::submerged(&w, Vec3::new(0.0, -500.0, 0.0)));
    }

    #[test]
    fn submersion_follows_the_level() {
        let w = WaterSettings { enabled: true, level_m: 50.0, ..Default::default() };
        assert!(Water::submerged(&w, Vec3::new(0.0, 49.0, 0.0)));
        assert!(!Water::submerged(&w, Vec3::new(0.0, 51.0, 0.0)));
    }

    #[test]
    fn wind_uses_the_same_heading_convention_as_the_camera() {
        // `atan2(z, x)`, so 0 degrees is +X and 90 is +Z. A different convention here
        // would make the waves travel somewhere other than where the number says.
        let w = WaterSettings { wind_deg: 0.0, ..Default::default() };
        let d = w.wind_dir();
        assert!((d[0] - 1.0).abs() < 1e-5 && d[1].abs() < 1e-5, "{d:?}");

        let w = WaterSettings { wind_deg: 90.0, ..Default::default() };
        let d = w.wind_dir();
        assert!(d[0].abs() < 1e-5 && (d[1] - 1.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn wind_is_always_a_unit_vector() {
        for deg in [-720.0f32, -37.0, 0.0, 123.0, 359.0, 1080.0] {
            let d = WaterSettings { wind_deg: deg, ..Default::default() }.wind_dir();
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "at {deg} deg the wind was {len} long");
        }
    }

    #[test]
    fn the_defaults_are_a_plausible_body_of_water() {
        let w = WaterSettings::default();
        assert!(w.absorption > 0.0, "zero absorption is glass, not water");
        assert!(w.wave_length_m > w.wave_height_m * 4.0, "waves steeper than this break");
        assert!(w.roughness > 0.0, "a zero-roughness specular lobe is a division by zero");
        // Deep water must be darker than shallow, or absorption runs the wrong way and
        // the middle of a lake glows.
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(sum(w.deep) < sum(w.shallow), "deep water is not darker than shallow");
    }

    #[test]
    fn settings_round_trip_through_ron() {
        // Saved beside the environment, and hand-editable for the same reason.
        let want = WaterSettings {
            enabled: true,
            level_m: 312.5,
            wind_deg: -20.0,
            absorption: 0.4,
            ..Default::default()
        };
        let text =
            ron::ser::to_string_pretty(&want, ron::ser::PrettyConfig::new().struct_names(true))
                .unwrap();
        let got: WaterSettings = ron::from_str(&text).unwrap();
        assert_eq!(got, want);
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    fn region(min: [f32; 2], max: [f32; 2], level: f32) -> WaterRegion {
        WaterRegion { min, max, level_m: level, ..Default::default() }
    }

    #[test]
    fn a_drag_normalises_its_corners() {
        // A drag can finish up-left of where it started, and an unsorted pair makes
        // `contains` false everywhere -- an invisible region nobody can select.
        let a = WaterRegion::from_drag([100.0, 200.0], [-50.0, -80.0], 10.0);
        assert_eq!(a.min, [-50.0, -80.0]);
        assert_eq!(a.max, [100.0, 200.0]);
        assert!(a.contains(0.0, 0.0));
    }

    #[test]
    fn a_stray_click_is_not_a_lake() {
        // Zero area would be invisible, unselectable and impossible to remove without
        // hand-editing the file.
        assert!(!WaterRegion::from_drag([0.0, 0.0], [0.0, 0.0], 10.0).is_usable());
        assert!(!WaterRegion::from_drag([0.0, 0.0], [0.5, 40.0], 10.0).is_usable());
        assert!(WaterRegion::from_drag([0.0, 0.0], [40.0, 40.0], 10.0).is_usable());
    }

    #[test]
    fn a_region_beats_the_global_surface() {
        // The point of regions: a tarn at 400 m over a sea at 100 m.
        let s = WaterSettings {
            enabled: true,
            level_m: 100.0,
            regions: vec![region([0.0, 0.0], [50.0, 50.0], 400.0)],
            ..Default::default()
        };
        assert_eq!(Water::level_at(&s, 25.0, 25.0), Some(400.0), "inside the region");
        assert_eq!(Water::level_at(&s, 500.0, 500.0), Some(100.0), "outside it");
    }

    #[test]
    fn the_first_containing_region_wins() {
        // Overlaps resolve by placement order rather than by blending, which would need a
        // signed distance per region and still leave a seam.
        let s = WaterSettings {
            enabled: false,
            regions: vec![
                region([0.0, 0.0], [100.0, 100.0], 10.0),
                region([50.0, 50.0], [150.0, 150.0], 20.0),
            ],
            ..Default::default()
        };
        assert_eq!(Water::level_at(&s, 75.0, 75.0), Some(10.0), "the earlier region wins");
        assert_eq!(Water::level_at(&s, 120.0, 120.0), Some(20.0));
    }

    #[test]
    fn a_region_works_with_the_global_surface_switched_off() {
        // A lake in an otherwise dry world, which is the common case: someone wants one
        // body of water, not a flooded map.
        let s = WaterSettings {
            enabled: false,
            regions: vec![region([0.0, 0.0], [50.0, 50.0], 30.0)],
            ..Default::default()
        };
        assert_eq!(Water::level_at(&s, 10.0, 10.0), Some(30.0));
        assert_eq!(Water::level_at(&s, 900.0, 900.0), None, "no water outside it");
        assert_eq!(Water::depth_at(&s, 900.0, 900.0, -1000.0), 0.0);
    }

    #[test]
    fn depth_and_submersion_follow_the_region_not_the_global_level() {
        let s = WaterSettings {
            enabled: true,
            level_m: 100.0,
            regions: vec![region([0.0, 0.0], [50.0, 50.0], 400.0)],
            ..Default::default()
        };
        // Ground at 380 m is 20 m under the tarn and far above the sea.
        assert_eq!(Water::depth_at(&s, 10.0, 10.0, 380.0), 20.0);
        assert_eq!(Water::depth_at(&s, 900.0, 900.0, 380.0), 0.0);
        assert!(Water::submerged_at(&s, Vec3::new(10.0, 390.0, 10.0)));
        assert!(!Water::submerged_at(&s, Vec3::new(900.0, 390.0, 900.0)));
    }

    #[test]
    fn a_region_keeps_its_own_wave_speed() {
        // The reason these are per region rather than global: a sheltered pond and an
        // open reservoir under the same wind do not move the same way.
        let mut r = region([0.0, 0.0], [10.0, 10.0], 5.0);
        r.wave_speed = 0.2;
        let s = WaterSettings { regions: vec![r], ..Default::default() };
        assert_eq!(s.regions[0].wave_speed, 0.2);
        assert_eq!(WaterSettings::default().wave_speed, 1.0, "the global default is untouched");
    }

    #[test]
    fn a_new_region_starts_from_the_global_look() {
        // So it is tuned away from the sea rather than from nothing.
        let d = WaterRegion::default();
        let g = WaterSettings::default();
        assert_eq!(d.wave_height_m, g.wave_height_m);
        assert_eq!(d.wave_length_m, g.wave_length_m);
        assert_eq!(d.wave_speed, g.wave_speed);
    }

    #[test]
    fn regions_survive_a_round_trip_and_an_older_file_still_loads() {
        let want = WaterSettings {
            enabled: false,
            regions: vec![region([-10.0, -20.0], [30.0, 40.0], 77.5)],
            ..Default::default()
        };
        let text =
            ron::ser::to_string_pretty(&want, ron::ser::PrettyConfig::new().struct_names(true))
                .unwrap();
        assert_eq!(ron::from_str::<WaterSettings>(&text).unwrap(), want);

        // A `water.ron` written before regions existed has no such field, and `serde`'s
        // default is what lets it load rather than fail.
        let older = ron::ser::to_string_pretty(
            &WaterSettings::default(),
            ron::ser::PrettyConfig::new().struct_names(true),
        )
        .unwrap();
        let stripped: String =
            older.lines().filter(|l| !l.contains("regions")).collect::<Vec<_>>().join("\n");
        let got: WaterSettings = ron::from_str(&stripped).expect("a pre-regions file must load");
        assert!(got.regions.is_empty());
    }

    #[test]
    fn only_the_first_eight_regions_reach_the_shader() {
        // The shader walks a fixed-size array, so anything past the ceiling has to be
        // ignored consistently rather than wrapping onto slot zero.
        let mut s = WaterSettings { enabled: false, ..Default::default() };
        for i in 0..MAX_REGIONS + 4 {
            s.regions.push(region([i as f32 * 100.0, 0.0], [i as f32 * 100.0 + 50.0, 50.0], 1.0));
        }
        // The ninth region's rectangle must not be found.
        let x = (MAX_REGIONS as f32) * 100.0 + 10.0;
        assert_eq!(Water::level_at(&s, x, 10.0), None, "a region past the ceiling was used");
    }
}

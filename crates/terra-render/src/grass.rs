//! Dense grass.
//!
//! Not a species in the scatter palette. Scatter bakes a world-sized instance
//! buffer on the CPU, which is right for 10^5 trees and impossible for 10^6
//! blades. Grass is generated on the GPU every frame into rings that follow the
//! camera, and nothing about it is stored.
//!
//! One blade per instance, following the approach Sucker Punch described for
//! *Ghost of Tsushima*. The earlier version instanced a clump of eight blades
//! as a single rigid mesh, which is cheaper to place but forfeits everything
//! downstream: a clump cannot be culled, level-of-detailed or thinned per
//! blade, so the whole tuft is drawn at full cost the moment any part of it is
//! visible. Per blade, culling and LOD both become exact, and the vertex shader
//! can evaluate the blade's shape from almost no data -- a vertex here carries
//! only how far up the blade it sits and which edge it is.
//!
//! Placement comes from the splat map, so grass grows where the grass material
//! was painted and stops where a path was painted over it. There is no second
//! mask to keep in agreement with the first.

use crate::camera::Camera;
use crate::context::{DEPTH_FORMAT, RenderContext, SCENE_FORMAT};
use crate::hiz::HiZ;
use crate::lighting::Lighting;
use crate::terrain::Terrain;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Segments up a blade, per level of detail.
///
/// The near band is a small fraction of the ground but most of the screen,
/// which is why it can afford curvature the far band cannot. Two segments is a
/// bent quad, which is all that survives at twenty metres; eight is a smooth
/// arc, which is what the eye is actually looking at when a blade fills a
/// hundred pixels.
const LOD_SEGMENTS: [u32; 3] = [8, 4, 2];

/// Blades one level of detail can hold. Kept in step with `LOD_CAPACITY` in
/// `grass_gen.wgsl`; the test below fails if they drift apart.
const LOD_CAPACITY: u32 = 260_000;

/// Concentric placement rings. Each doubles its spacing and its reach, so four
/// of them cover sixty-four times the area of one for a third more threads.
///
/// Four rather than three because the near field is where density is actually
/// seen: adding a ring outward buys reach, adding one inward buys a fourfold
/// finer grid under the camera, which is what separates a sward from a set of
/// separate blades with ground showing between them.
const RINGS: u32 = 4;

/// Cells along one side of a ring. The dispatch is `RINGS * SIDE^2` threads, so
/// this is the single knob that sets what placement costs.
const RING_SIDE_MAX: u32 = 448;

/// Bytes per instance: two `vec4`.
const INSTANCE_SIZE: u64 = 32;

/// Vertices, indices, and each level's `(first_index, index_count, base_vertex)`.
type BladeMeshes = (Vec<Vertex>, Vec<u32>, [(u32, u32, i32); 3]);

/// What kind of grass this is.
///
/// Not just a height. These differ in how upright they stand, how much their
/// lengths vary, how wide a blade is and how bluntly it ends -- and none of
/// those follows from the others. Getting only the height right gives short
/// meadow, which reads as scrub, or tall lawn, which reads as a carpet on
/// stilts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrassStyle {
    Lawn,
    Field,
    Meadow,
}

impl GrassStyle {
    pub const ALL: [GrassStyle; 3] = [GrassStyle::Lawn, GrassStyle::Field, GrassStyle::Meadow];

    pub fn label(self) -> &'static str {
        match self {
            GrassStyle::Lawn => "Lawn",
            GrassStyle::Field => "Field",
            GrassStyle::Meadow => "Meadow",
        }
    }

    /// `(height, blades per m2 in the near field, draw distance)`
    pub fn suggested(self) -> (f32, f32, f32) {
        self.defaults()
    }

    fn defaults(self) -> (f32, f32, f32) {
        match self {
            // Short blades need far more of them to close the ground, and are
            // invisible past twenty metres anyway.
            GrassStyle::Lawn => (0.075, 5200.0, 20.0),
            GrassStyle::Field => (0.26, 2600.0, 34.0),
            GrassStyle::Meadow => (0.58, 1100.0, 44.0),
        }
    }

    fn shape(self) -> Shape {
        match self {
            GrassStyle::Lawn => Shape {
                lean: (0.04, 0.20),
                // Cut, not grown to a point. This is the clearest single tell
                // that a lawn has been mown.
                tip: 0.45,
                half_width: 0.0022,
                curve: 0.16,
                variation: 0.45,
            },
            // Wild but not overgrown: leaning enough to catch light along its
            // length, varied enough that no two neighbours agree, and still
            // short enough to read as ground rather than as undergrowth.
            GrassStyle::Field => Shape {
                // Arcs over rather than standing up. A field of straight blades
                // reads as a crop; the curve is what makes it look grown.
                lean: (0.28, 0.85),
                tip: 1.0,
                half_width: 0.0058,
                curve: 0.26,
                variation: 0.95,
            },
            GrassStyle::Meadow => Shape {
                lean: (0.35, 1.05),
                tip: 1.0,
                half_width: 0.0068,
                curve: 0.32,
                variation: 1.0,
            },
        }
    }
}

/// Per-style blade geometry. All of it lives in the uniform now -- the mesh is
/// just a strip of height fractions, so changing the style costs an upload
/// rather than a buffer rebuild.
#[derive(Clone, Copy)]
struct Shape {
    /// Range the per-blade lean is drawn from, as a fraction of blade length.
    lean: (f32, f32),
    /// How much of the width is lost at the tip. 1 tapers to a point.
    tip: f32,
    /// Half-width at the base, in metres. Absolute rather than a fraction of
    /// height: a grass blade is a few millimetres across whether it is mown or
    /// waist-high, and scaling width with length makes a lawn blade a hair and
    /// a meadow blade a ribbon.
    half_width: f32,
    /// How far the curve's control point leads the tip. Higher arcs the blade
    /// over instead of bending it straight back.
    curve: f32,
    /// Scale on the per-blade and per-field colour variation.
    variation: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct GrassSettings {
    pub enabled: bool,
    pub style: GrassStyle,
    /// Blades per square metre in the near field.
    pub density: f32,
    pub height_m: f32,
    /// Metres at which blades are fully gone.
    pub draw_distance: f32,
    /// Fraction of the draw distance where the dissolve begins.
    pub fade_start: f32,
    pub wind_strength: f32,
    pub wind_speed: f32,
}

impl Default for GrassSettings {
    fn default() -> Self {
        let style = GrassStyle::Field;
        let (height_m, density, draw_distance) = style.defaults();
        Self {
            enabled: true,
            style,
            density,
            height_m,
            draw_distance,
            // Also the boundary of the coarsest level of detail, so the only
            // blades running the discarding shader are the ones actually
            // dissolving. See `draw`.
            fade_start: 0.45,
            // Barely there. A field that visibly sways all the time reads as a
            // flag rather than as grass.
            wind_strength: 0.06,
            wind_speed: 0.7,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrassUniform {
    grid: [f32; 4],
    eye: [f32; 4],
    blade: [f32; 4],
    thinning: [f32; 4],
    ground: [f32; 4],
    world: [f32; 4],
    shape: [f32; 4],
    planes: [[f32; 4]; 6],
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    /// x: height fraction along the blade. y: which edge, -1 or +1.
    ///
    /// That is the whole vertex. Position, normal, tangent and width are all
    /// derived in the vertex shader from the blade's curve, which is cheaper to
    /// evaluate than to fetch at this instance count.
    vert: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

pub struct Grass {
    pub settings: GrassSettings,
    /// Linear albedo the blades converge on as they dissolve.
    pub ground_color: Vec3,
    /// Which splat channel carries the grass weight.
    pub layer: u32,

    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// `(first_index, index_count, base_vertex)` per level of detail.
    lods: [(u32, u32, i32); 3],
    /// All three levels' instances, back to back in one allocation.
    blades: wgpu::Buffer,
    /// Three `DrawArgs`, one per level.
    args: wgpu::Buffer,

    gen_bgl: wgpu::BindGroupLayout,
    gen_bind_group: Option<wgpu::BindGroup>,
    gen_pipeline: wgpu::ComputePipeline,

    uniform_bg: wgpu::BindGroup,
    solid_pipeline: wgpu::RenderPipeline,
    faded_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,

    /// Cells per side of one ring this frame, and the innermost spacing.
    side: u32,
    spacing: f32,
}

impl Grass {
    pub fn new(
        ctx: &RenderContext,
        lighting: &Lighting,
        hiz: &HiZ,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let device = &ctx.device;
        let settings = GrassSettings::default();

        let (verts, idx, lods) = blade_meshes();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass-verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass-indices"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass-uniform"),
            size: std::mem::size_of::<GrassUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blades = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass-blades"),
            size: LOD_CAPACITY as u64 * 3 * INSTANCE_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass-args"),
            size: std::mem::size_of::<DrawArgs>() as u64 * 3,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buf = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let splat = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let gen_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grass-gen-bgl"),
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
                buf(1, true),
                splat(2),
                splat(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                buf(5, false),
                buf(6, false),
            ],
        });

        let gen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass-gen"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/render/grass_gen.wgsl").into(),
            ),
        });
        let gen_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grass-gen-layout"),
            bind_group_layouts: &[Some(&gen_bgl), Some(hiz.cull_layout())],
            immediate_size: 0,
        });
        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("grass-gen-pipeline"),
            layout: Some(&gen_layout),
            module: &gen_shader,
            entry_point: Some("place"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grass-uniform-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass-uniform-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../../../assets/shaders/common/noise.wgsl"),
                    include_str!("../../../assets/shaders/common/lighting.wgsl"),
                    include_str!("../../../assets/shaders/render/grass.wgsl"),
                )
                .into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grass-layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&uniform_bgl), Some(&lighting.layout)],
            immediate_size: 0,
        });

        let buffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2],
            },
            wgpu::VertexBufferLayout {
                array_stride: INSTANCE_SIZE,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4],
            },
        ];
        let mut desc = wgpu::RenderPipelineDescriptor {
            label: Some("grass-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(buffers[0].clone()), Some(buffers[1].clone())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Blades are ribbons: both sides are seen constantly.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        };
        let solid_pipeline = device.create_render_pipeline(&desc);
        desc.label = Some("grass-faded-pipeline");
        if let Some(f) = desc.fragment.as_mut() {
            f.entry_point = Some("fs_fade");
        }
        let faded_pipeline = device.create_render_pipeline(&desc);

        // Grass casts only into the near cascade; see `draw_shadow`.
        desc.label = Some("grass-shadow-pipeline");
        desc.fragment = None;
        desc.vertex.entry_point = Some("vs_shadow");
        // No lighting group: the depth-only pass has no fragment stage, and
        // binding it would name the shadow map as a sampled texture in the same
        // pass that writes it as a depth attachment.
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grass-shadow-layout"),
            bind_group_layouts: &[Some(&lighting.cascade_layout), Some(&uniform_bgl)],
            immediate_size: 0,
        });
        desc.layout = Some(&shadow_layout);
        let shadow_pipeline = device.create_render_pipeline(&desc);

        Self {
            settings,
            ground_color: Vec3::new(0.105, 0.150, 0.062),
            layer: 0,
            uniform,
            vertices,
            indices,
            lods,
            blades,
            args,
            gen_bgl,
            gen_bind_group: None,
            gen_pipeline,
            uniform_bg,
            solid_pipeline,
            faded_pipeline,
            shadow_pipeline,
            side: 0,
            spacing: 0.0,
        }
    }

    /// Bind to a world's terrain. Called whenever a world is opened, because
    /// the heightfield and splat map are the terrain's, not ours.
    pub fn attach(&mut self, device: &wgpu::Device, terrain: &Terrain, sampler: &wgpu::Sampler) {
        let views = terrain.splat_views();
        self.gen_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass-gen-bg"),
            layout: &self.gen_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: terrain.height_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry { binding: 5, resource: self.blades.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.args.as_entire_binding() },
            ],
        }));
    }

    pub fn detach(&mut self) {
        self.gen_bind_group = None;
    }

    /// Placement threads this frame -- what the pass actually costs, as opposed
    /// to how many blades survive it.
    pub fn slot_count(&self) -> u32 {
        self.side * self.side * RINGS
    }

    /// Upload this frame's state and place the visible blades.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cam: &Camera,
        aspect: f32,
        viewport_h: f32,
        terrain: &Terrain,
        hiz: &HiZ,
        time: f32,
    ) {
        if !self.settings.enabled {
            self.side = 0;
            return;
        }
        let Some(bg) = self.gen_bind_group.as_ref() else { return };
        let s = self.settings;

        // The outermost ring spans `side * 2^(RINGS-1) * spacing`, so half of
        // that has to reach the draw distance. Where the requested density
        // would need more cells than the budget allows, the spacing opens up
        // instead: thinner grass everywhere beats a full-density disc with a
        // visible square edge where the grid runs out.
        let reach = (1u32 << (RINGS - 1)) as f32;
        let mut spacing = 1.0 / s.density.max(1.0).sqrt();
        let side = ((s.draw_distance * 2.0 / (spacing * reach)).ceil() as u32)
            .clamp(64, RING_SIDE_MAX);
        spacing = spacing.max(s.draw_distance * 2.0 / (side as f32 * reach));
        self.side = side;
        self.spacing = spacing;

        // Radius held at full density, derived rather than dialled. Each ring
        // takes over where the thinning has brought the one inside it down to
        // its own coarseness, which puts the hand-off at `full * 2^ring` -- so
        // this radius is not free to choose: too large and a ring is asked to
        // cover ground that falls outside its own square, which shows up as a
        // bare band exactly one ring wide.
        let full = side as f32 * spacing * 0.25;

        // Height in pixels of a one-metre object one metre away.
        let px_scale = viewport_h / (2.0 * (cam.fov_y * 0.5).tan());

        let vp = cam.projection(aspect) * cam.look_at();
        let m = vp.transpose();
        let planes = [
            m.w_axis + m.x_axis,
            m.w_axis - m.x_axis,
            m.w_axis + m.y_axis,
            m.w_axis - m.y_axis,
            m.w_axis - m.z_axis,
            m.w_axis,
        ];
        let shape = s.style.shape();
        let fade_start = s.draw_distance * s.fade_start;

        let u = GrassUniform {
            grid: [cam.pos.x, cam.pos.z, spacing, side as f32],
            eye: cam.pos.extend(s.draw_distance).to_array(),
            blade: [s.height_m, fade_start, s.wind_strength, s.wind_speed],
            thinning: [
                full,
                px_scale,
                shape.lean.0,
                shape.lean.1,
            ],
            ground: self.ground_color.extend(terrain.mesh_resolution() as f32).to_array(),
            world: [terrain.resolution() as f32, terrain.extent_m(), time, self.layer as f32],
            shape: [shape.half_width, shape.curve, shape.tip, shape.variation],
            planes: planes.map(|p| p.to_array()),
            view_proj: vp.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        let reset: Vec<DrawArgs> = self
            .lods
            .iter()
            .map(|&(first_index, index_count, base_vertex)| DrawArgs {
                index_count,
                instance_count: 0,
                first_index,
                base_vertex,
                first_instance: 0,
            })
            .collect();
        queue.write_buffer(&self.args, 0, bytemuck::cast_slice(&reset));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("grass-place"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.gen_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.set_bind_group(1, hiz.cull_bind_group(), &[]);
        pass.dispatch_workgroups(self.slot_count().div_ceil(64), 1, 1);
    }

    fn instances(&self, lod: usize) -> wgpu::BufferSlice<'_> {
        let start = lod as u64 * LOD_CAPACITY as u64 * INSTANCE_SIZE;
        self.blades.slice(start..start + LOD_CAPACITY as u64 * INSTANCE_SIZE)
    }

    fn args_offset(lod: usize) -> u64 {
        lod as u64 * std::mem::size_of::<DrawArgs>() as u64
    }

    fn live(&self) -> bool {
        self.settings.enabled && self.gen_bind_group.is_some() && self.side > 0
    }

    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera_bg: &wgpu::BindGroup,
        lighting: &Lighting,
    ) {
        if !self.live() {
            return;
        }
        pass.set_bind_group(0, camera_bg, &[]);
        pass.set_bind_group(1, &self.uniform_bg, &[]);
        pass.set_bind_group(2, &lighting.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);

        // Near to far. The two near levels are nearer by construction, so
        // drawing them first fills the depth buffer the dissolving band then
        // tests against -- and neither of them discards, so the tile GPU can
        // reject occluded fragments before shading them, which is most of the
        // cost of a dense carpet.
        pass.set_pipeline(&self.solid_pipeline);
        for lod in 0..2 {
            pass.set_vertex_buffer(1, self.instances(lod));
            pass.draw_indexed_indirect(&self.args, Self::args_offset(lod));
        }

        // The far level is exactly the dissolving band, which is why the
        // discarding shader only ever runs on blades that are actually fading.
        pass.set_pipeline(&self.faded_pipeline);
        pass.set_vertex_buffer(1, self.instances(2));
        pass.draw_indexed_indirect(&self.args, Self::args_offset(2));
    }

    /// Cast into one cascade.
    ///
    /// Only the nearest. Grass in a cascade covering hundreds of metres is a
    /// texel of noise per blade, and the cost is a second full draw of the
    /// densest geometry in the scene.
    pub fn draw_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        cascade: usize,
    ) {
        if !self.live() || cascade != 0 {
            return;
        }
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &lighting.cascade_bind_group, &[Lighting::cascade_offset(cascade)]);
        pass.set_bind_group(1, &self.uniform_bg, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        // Only the near levels cast. The far one is the dissolving band, where
        // a blade is under a shadow texel anyway.
        for lod in 0..2 {
            pass.set_vertex_buffer(1, self.instances(lod));
            pass.draw_indexed_indirect(&self.args, Self::args_offset(lod));
        }
    }
}

/// Three strips of decreasing subdivision, packed into one pair of buffers.
///
/// A vertex is nothing but its height fraction and its edge sign; the shape
/// comes from the uniform. That means a style change is an upload rather than a
/// buffer rebuild, and the three levels differ only in how finely they sample
/// the same curve.
fn blade_meshes() -> BladeMeshes {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    let mut lods = [(0u32, 0u32, 0i32); 3];

    for (lod, &segments) in LOD_SEGMENTS.iter().enumerate() {
        let base_vertex = verts.len() as i32;
        let first_index = idx.len() as u32;
        for s in 0..=segments {
            let t = s as f32 / segments as f32;
            verts.push(Vertex { vert: [t, -1.0] });
            verts.push(Vertex { vert: [t, 1.0] });
        }
        for s in 0..segments {
            let a = s * 2;
            idx.extend([a, a + 1, a + 2, a + 1, a + 3, a + 2]);
        }
        lods[lod] = (first_index, idx.len() as u32 - first_index, base_vertex);
    }
    (verts, idx, lods)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEN_SRC: &str = include_str!("../../../assets/shaders/render/grass_gen.wgsl");
    const DRAW_SRC: &str = include_str!("../../../assets/shaders/render/grass.wgsl");

    #[test]
    fn each_level_is_a_well_formed_strip() {
        let (v, i, lods) = blade_meshes();
        for (lod, &segments) in LOD_SEGMENTS.iter().enumerate() {
            let (first, count, base) = lods[lod];
            assert_eq!(count, segments * 6, "level {lod} index count");
            let range = &i[first as usize..(first + count) as usize];
            // Indices are relative to `base_vertex`, so they address exactly
            // this level's own vertices and no others.
            let span = (segments + 1) * 2;
            assert!(range.iter().all(|k| *k < span), "level {lod} indexes past its own strip");
            assert!((base as usize) + span as usize <= v.len());
        }
    }

    #[test]
    fn levels_get_cheaper_with_distance() {
        // The point of the ladder. If a far level were not strictly cheaper it
        // would be costing three pipelines' worth of complexity for nothing.
        for w in LOD_SEGMENTS.windows(2) {
            assert!(w[0] > w[1], "levels should coarsen with distance: {LOD_SEGMENTS:?}");
        }
    }

    #[test]
    fn a_vertex_is_only_its_height_and_its_edge() {
        let (v, _, _) = blade_meshes();
        assert_eq!(std::mem::size_of::<Vertex>(), 8);
        assert!(v.iter().all(|x| (0.0..=1.0).contains(&x.vert[0])));
        assert!(v.iter().all(|x| x.vert[1] == -1.0 || x.vert[1] == 1.0));
        // Paired edges, so the shader's `half_width * edge` is symmetric about
        // the spine. Baking width into a position is what previously made a
        // widened blade move sideways instead of thickening.
        for pair in v.chunks(2) {
            assert_eq!(pair[0].vert[0], pair[1].vert[0]);
            assert_eq!(pair[0].vert[1], -pair[1].vert[1]);
        }
    }

    #[test]
    fn the_blade_runs_root_to_tip() {
        // The shader bends, shades and occludes by the height fraction, so it
        // has to span the full range.
        let (v, _, _) = blade_meshes();
        assert!(v.iter().any(|x| x.vert[0] == 0.0));
        assert!(v.iter().any(|x| x.vert[0] == 1.0));
    }

    #[test]
    fn instance_capacity_matches_the_shader() {
        // The compute pass indexes `lod * LOD_CAPACITY`, so a mismatch writes
        // one level's blades over another's rather than failing validation.
        let want = format!("const LOD_CAPACITY: u32 = {LOD_CAPACITY}u;");
        assert!(GEN_SRC.contains(&want), "grass_gen.wgsl should declare {want}");
        assert!(GEN_SRC.contains(&format!("const RINGS: u32 = {RINGS}u;")));
    }

    #[test]
    fn uniform_matches_the_shader_block() {
        // Both shaders declare the same struct by hand, and a mismatch here is
        // silent: fields shift and the grass simply comes out wrong.
        for (name, src) in [("grass_gen", GEN_SRC), ("grass", DRAW_SRC)] {
            let block = src
                .split_once("struct GrassU {")
                .and_then(|(_, rest)| rest.split_once("};"))
                .map(|(b, _)| b)
                .unwrap_or_else(|| panic!("{name}.wgsl has no GrassU"));
            let mut bytes = 0usize;
            for line in block.lines().map(str::trim) {
                if line.starts_with("//") {
                    continue;
                }
                let Some((_, ty)) = line.split_once(':') else { continue };
                bytes += match ty.trim_end_matches(',').trim() {
                    "vec4f" => 16,
                    "mat4x4f" => 64,
                    "array<vec4f, 6>" => 96,
                    other => panic!("{name}.wgsl: unhandled field type {other}"),
                };
            }
            assert_eq!(
                bytes,
                std::mem::size_of::<GrassUniform>(),
                "{name}.wgsl GrassU is {bytes} bytes, Rust is {}",
                std::mem::size_of::<GrassUniform>()
            );
        }
    }

    #[test]
    fn instance_stride_matches_the_vertex_layout() {
        // Two vec4 per blade. The compute pass writes this struct as storage
        // and the vertex stage reads it as attributes, so the two views of it
        // have to agree byte for byte.
        assert_eq!(INSTANCE_SIZE, 32);
        assert!(GEN_SRC.contains("pos_scale: vec4f"));
        assert!(GEN_SRC.contains("params: vec4f"));
        assert!(DRAW_SRC.contains("@location(1) pos_scale: vec4f"));
        assert!(DRAW_SRC.contains("@location(2) params: vec4f"));
    }

    #[test]
    fn styles_trade_height_against_density() {
        // Short grass needs far more blades to close the ground, and cannot be
        // seen from as far away. A style that got this backwards would either
        // show bare soil or spend its whole budget below one pixel.
        let mut last = f32::MAX;
        for style in [GrassStyle::Lawn, GrassStyle::Field, GrassStyle::Meadow] {
            let (h, d, dist) = style.defaults();
            assert!(d < last, "{} should be sparser than the style before it", style.label());
            last = d;
            // Blades per metre of height, near enough constant across styles:
            // that is what makes them all read as full rather than as scrub.
            assert!(h * h * d > 20.0, "{} would show bare ground", style.label());
            assert!(dist > h * 40.0, "{} is drawn too short a distance", style.label());
        }
    }

    #[test]
    fn a_lawn_is_short_even_and_upright_and_bluntly_cut() {
        let lawn = GrassStyle::Lawn.shape();
        let field = GrassStyle::Field.shape();
        // Mown, so it stands up rather than leaning.
        assert!(lawn.lean.1 < field.lean.0, "a lawn should stand straighter than a field");
        // Cut, not grown to a point.
        assert!(lawn.tip < 0.6, "a mown tip should keep width, got {}", lawn.tip);
        assert!(field.tip > 0.9, "a wild blade should taper to a point");
        // Cut to one level, so its lengths barely differ.
        assert!(lawn.variation < field.variation);
    }

    #[test]
    fn blades_stay_far_taller_than_they_are_wide() {
        // Real proportions. A blade fat enough to read as a strip up close is a
        // paddle, and a field of them looks like a succulent rather than grass.
        for style in GrassStyle::ALL {
            let (height, _, _) = style.defaults();
            let ratio = height / (style.shape().half_width * 2.0);
            assert!(ratio > 15.0, "{} is {ratio}:1 tall to wide", style.label());
        }
    }

    #[test]
    fn the_default_is_not_a_lawn() {
        // A lawn is a carpet: uniform, cropped and obviously placed. The
        // default should be ground that looks grown.
        assert_eq!(GrassSettings::default().style, GrassStyle::Field);
    }

    #[test]
    fn the_coarsest_level_is_exactly_the_dissolving_band() {
        // `fade_start` is both where the dither begins and where the compute
        // pass switches to level 2, which is what lets the two near levels use
        // a shader with no `discard` in it and keep early-Z.
        assert!(GEN_SRC.contains("dist < g.blade.y"), "level 2 should start at the fade distance");
        assert!(DRAW_SRC.contains("fn fs_fade"));
    }
}

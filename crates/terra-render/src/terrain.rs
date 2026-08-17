//! Terrain heightfield: GPU buffers, draw pipeline, and the sculpt brush.
//!
//! The CPU copy in [`Terrain::heights`] is authoritative. Sculpting edits it and
//! uploads only the touched rows, which keeps raycasting and saving correct
//! without any GPU readback. A brush covers at most a few thousand texels, so
//! this costs microseconds -- compute is for whole-map work like erosion.

use crate::camera::{Camera, CameraUniform};
use crate::cdlod::{self, Cdlod, PATCH_QUADS};
use crate::context::{DEPTH_FORMAT, RenderContext};
use crate::lighting::Lighting;
use crate::material::{MAX_LAYERS, Materials};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use terra_core::WorldSize;
use wgpu::util::DeviceExt;

/// Vertex spacing the finest CDLOD level aims for, in metres.
///
/// This is the number that decides whether a material can displace geometry at
/// all. Materials tile every few metres, so a vertex every half metre puts roughly
/// seven vertices across one repeat -- enough for a height map to read as bumps
/// rather than as the whole quad lurching. The uniform 512-square grid this
/// replaced had 7.81 m between vertices on a 4 km world, two full repeats inside
/// one quad.
///
/// It costs almost nothing to ask for, because CDLOD only reaches this spacing in
/// the patches nearest the camera: the measured selection is 232 patches and 0.48 M
/// triangles against the old grid's 0.52 M.
const CDLOD_TARGET_SPACING_M: f32 = 0.5;

/// Resolution of the painted layer weights.
///
/// Independent of, and much coarser than, the heightfield: painting is a
/// large-scale act -- the smallest brush is 8 m across -- and a splat map at
/// heightfield resolution would cost 8 bytes a texel for detail no brush can
/// place.
const SPLAT_RES: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SculptMode {
    Raise,
    Lower,
    Smooth,
    Flatten,
    /// Build material up toward the brush plane, never cutting into ground that
    /// already stands above it.
    Clay,
    /// Drag the surface along the stroke. Shifts heights rather than adding to
    /// them, so existing detail travels instead of being smeared flat.
    Move,
    /// Displace by a noise pattern -- the built-in basis or an uploaded map.
    Noise,
    /// Pull heights toward the brush centre, sharpening a rise into a ridge.
    Pinch,
}

impl SculptMode {
    pub const ALL: [SculptMode; 8] = [
        SculptMode::Clay,
        SculptMode::Raise,
        SculptMode::Lower,
        SculptMode::Move,
        SculptMode::Flatten,
        SculptMode::Smooth,
        SculptMode::Noise,
        SculptMode::Pinch,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SculptMode::Raise => "Raise",
            SculptMode::Lower => "Lower",
            SculptMode::Smooth => "Smooth",
            SculptMode::Flatten => "Flatten",
            SculptMode::Clay => "Clay",
            SculptMode::Move => "Move",
            SculptMode::Noise => "Noise",
            SculptMode::Pinch => "Pinch",
        }
    }

    /// Whether this mode reads the noise pattern.
    pub fn uses_noise(self) -> bool {
        matches!(self, SculptMode::Noise)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainUniform {
    world_extent: f32,
    height_res: u32,
    /// Quads per side of one CDLOD patch. See `cdlod::PATCH_QUADS`.
    patch_quads: u32,
    brush_radius: f32,
    brush_center: [f32; 2],
    brush_active: f32,
    /// Viewport visualization mode, as `ViewMode::shader_index`.
    ///
    /// Sits in what used to be padding. The slot exists because `layer_roles` is
    /// an array of vec4 and needs a 16-byte offset, which 44 bytes of preceding
    /// fields would not give it -- so this costs nothing.
    view_mode: u32,
    /// How many palette slots actually hold a material.
    layer_count: u32,
    /// Which slot the grass pass grows from, so the ground can darken under it.
    grass_layer: u32,
    _pad0: [u32; 2],
    /// Camera position for the CDLOD morph: `xy` is the eye's world XZ, `z` its
    /// vertical distance to the terrain's height slab, `w` unused.
    ///
    /// Here rather than read from the camera uniform because the shadow pass binds
    /// group 0 to a cascade's light matrix and has no camera at all. That is not a
    /// workaround, it is a requirement: if the depth-only caster morphed
    /// differently from the shaded surface, every level boundary would grow a band
    /// of shadow acne. One uniform, bound in both passes, makes disagreeing
    /// impossible.
    morph_eye: [f32; 4],
    /// Automatic role per layer, or `ROLE_NONE`. Packed as two vec4s because a
    /// `u32` array in a uniform is padded to 16 bytes a element anyway.
    layer_roles: [[u32; 4]; 2],
}

const _: () = assert!(std::mem::size_of::<TerrainUniform>() == 96);
// `morph_eye` is a vec4 in WGSL and must land on a 16-byte boundary, which is what
// `_pad0` buys; `layer_roles` then follows at 64.
const _: () = assert!(std::mem::offset_of!(TerrainUniform, morph_eye) == 48);
const _: () = assert!(std::mem::offset_of!(TerrainUniform, layer_roles) == 64);

/// Pre-brush copy of the region a Smooth pass reads, so the filter never
/// observes its own writes.
struct Scratch {
    x0: i32,
    x1: i32,
    z0: i32,
    z1: i32,
    w: usize,
    buf: Vec<f32>,
}

impl Scratch {
    fn get(&self, x: i32, z: i32) -> f32 {
        let ix = (x.clamp(self.x0, self.x1) - self.x0) as usize;
        let iz = (z.clamp(self.z0, self.z1) - self.z0) as usize;
        self.buf[iz * self.w + ix]
    }
}

pub struct Terrain {
    /// Authoritative heightfield in meters, `res * res`, row-major.
    pub heights: Vec<f32>,
    res: u32,
    extent_m: f32,

    /// Painted layer weights, `SPLAT_RES^2 * MAX_LAYERS`, layer-minor.
    ///
    /// All-zero at a texel means "never painted here", which the shader reads
    /// as a request for the automatic slope- and erosion-driven weights rather
    /// than as an instruction to paint nothing.
    splat: Vec<u8>,
    /// Cached: whether `splat` holds anything.
    ///
    /// The UI asks this every frame. Deriving it by scanning is 8 MB of reads
    /// per frame for one bool, which does not show up as a stutter -- it shows
    /// up as a couple of milliseconds of CPU that look like they belong to
    /// something else.
    splat_painted: bool,
    splat_tex: [wgpu::Texture; 2],
    /// Kept so the grass pass can bind the same weights the terrain shades by.
    splat_views: [wgpu::TextureView; 2],

    height_buf: wgpu::Buffer,
    flow_buf: wgpu::Buffer,
    deposit_buf: wgpu::Buffer,
    road_buf: wgpu::Buffer,
    rut_buf: wgpu::Buffer,
    terrain_ub: wgpu::Buffer,
    camera_ub: wgpu::Buffer,

    /// Quadtree LOD selection, rebuilt every frame from the camera.
    cdlod: Cdlod,
    /// The selected patches, as instance data for the vertex shader.
    patch_buf: wgpu::Buffer,
    /// Instances in `patch_buf`, i.e. how many patches the last selection chose.
    patch_count: u32,
    /// Min and max of `heights`, for the altitude term in LOD selection.
    ///
    /// Cached rather than scanned: selection runs every frame and the heightfield is
    /// up to 16 M floats. Sculpting widens it from the touched window, which can
    /// leave it wider than the true range after a Lower stroke -- harmless, because
    /// a wider slab only makes selection slightly more conservative.
    height_range: (f32, f32),

    index_buf: wgpu::Buffer,
    index_count: u32,
    /// Filled faces off, for the Wireframe view mode.
    wire_pipeline: wgpu::RenderPipeline,
    /// Grid edges as a line list. Used only on the fallback path; on the
    /// `POLYGON_MODE_LINE` path the triangle buffer is drawn instead.
    wire_index_buf: wgpu::Buffer,
    wire_index_count: u32,
    /// Which of the two wireframe paths this pipeline was built for.
    wire_uses_polygon_line: bool,

    camera_bgl: wgpu::BindGroupLayout,
    camera_bg: wgpu::BindGroup,
    terrain_bg: wgpu::BindGroup,
    /// Shared with every other terrain in the session; the handle is cheap.
    material_bg: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,

    brush: TerrainUniform,
}

impl Terrain {
    pub fn new(
        ctx: &RenderContext,
        size: WorldSize,
        materials: &Materials,
        lighting: &Lighting,
        clouds: &crate::clouds::Clouds,
    ) -> Self {
        let device = &ctx.device;
        let res = size.tier0_res();
        let extent_m = size.extent_m() as f32;

        // A new world starts flat. Generation is a toolbox action, not
        // something that fires at creation.
        let heights = vec![0.0f32; (res * res) as usize];

        let height_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("heightfield"),
            contents: bytemuck::cast_slice(&heights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Masks start neutral: no flow anywhere, deposition exactly balanced.
        let mask_buf = |label, fill: f32| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&vec![fill; (res * res) as usize]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let flow_buf = mask_buf("flow-mask", 0.0);
        let deposit_buf = mask_buf("deposition-mask", 0.5);
        let road_buf = mask_buf("road-mask", 0.0);
        let rut_buf = mask_buf("rut-mask", 0.0);

        // Two RGBA8 textures rather than an eight-slice array: four weights
        // per fetch means the fragment shader reads the whole palette in two
        // samples instead of eight.
        let splat = vec![0u8; (SPLAT_RES * SPLAT_RES * MAX_LAYERS) as usize];
        let splat_tex = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if i == 0 { "splat-0-3" } else { "splat-4-7" }),
                size: wgpu::Extent3d {
                    width: SPLAT_RES,
                    height: SPLAT_RES,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let splat_views: [wgpu::TextureView; 2] =
            std::array::from_fn(|i| splat_tex[i].create_view(&Default::default()));
        let splat_views_kept: [wgpu::TextureView; 2] =
            std::array::from_fn(|i| splat_tex[i].create_view(&Default::default()));
        // Clamped, not repeating: the splat map covers the world exactly once,
        // and wrapping would smear the far edge onto the near one.
        let splat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("splat-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let brush = TerrainUniform {
            world_extent: extent_m,
            height_res: res,
            patch_quads: PATCH_QUADS,
            brush_radius: 0.0,
            brush_center: [0.0, 0.0],
            brush_active: 0.0,
            view_mode: crate::ViewMode::Lit.shader_index(),
            layer_count: materials.count().min(MAX_LAYERS),
            grass_layer: materials
                .layers
                .iter()
                .position(|l| l.role == crate::material::GRASS)
                .unwrap_or(0) as u32,
            _pad0: [0; 2],
            morph_eye: [0.0; 4],
            layer_roles: pack_roles(materials),
        };

        let terrain_ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-uniform"),
            contents: bytemuck::bytes_of(&brush),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One patch's worth of indices, shared by every instance. The shadow pass
        // draws the same buffer: selection is distance-based and covers the whole
        // world, so the patch set is already everything that could cast.
        let indices = cdlod::patch_indices();
        let index_count = indices.len() as u32;
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-patch-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Sized for the first selection's worth of patches and grown on demand;
        // `select` tracks the high-water mark so a camera move does not reallocate.
        let mut cdlod = Cdlod::new(extent_m, CDLOD_TARGET_SPACING_M);
        let patches = cdlod.select(Vec3::ZERO, (0.0, 0.0), extent_m).to_vec();
        let patch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdlod-patches"),
            size: cdlod.buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seeded here so the first frame draws even if it renders before any camera
        // upload -- an empty instance range would be a blank viewport, not a stall.
        ctx.queue.write_buffer(&patch_buf, 0, bytemuck::cast_slice(&patches));
        let patch_count = patches.len() as u32;

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
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

        let terrain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Erosion by-products, read in the fragment stage for splatting.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Painted layer weights.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // CDLOD patches, read by `instance_index`. Vertex-only: the fragment
                // stage receives the world position it needs as an interpolant.
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_ub.as_entire_binding(),
            }],
        });

        let terrain_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-bg"),
            layout: &terrain_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: terrain_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: height_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flow_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: deposit_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: road_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rut_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&splat_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&splat_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&splat_sampler),
                },
                wgpu::BindGroupEntry { binding: 9, resource: patch_buf.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            // WGSL has no `#include`; the shared chunks are prepended the same
            // way the generation passes compose theirs.
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}\n{}\n{}\n{}",
                    include_str!("../../../assets/shaders/common/noise.wgsl"),
                    include_str!("../../../assets/shaders/common/lighting.wgsl"),
                    include_str!("../../../assets/shaders/common/cdlod.wgsl"),
                    include_str!("../../../assets/shaders/common/grid.wgsl"),
                    include_str!("../../../assets/shaders/common/brush.wgsl"),
                    include_str!("../../../assets/shaders/render/terrain.wgsl"),
                )
                .into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-layout"),
            bind_group_layouts: &[
                Some(&camera_bgl),
                Some(&terrain_bgl),
                Some(&materials.layout),
                Some(&lighting.layout),
                Some(&clouds.shadow_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-pipeline"),
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
                    format: crate::context::SCENE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                // Reversed-Z: nearer fragments have LARGER depth.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- wireframe ---
        //
        // Two paths, because `PolygonMode::Line` is not core WebGPU. When the
        // adapter has it, the same triangle buffer is drawn with filled faces
        // turned off, which gives triangle edges including the diagonals. When it
        // does not, a line-list buffer over the grid edges is drawn instead --
        // exact, and no optional feature. See `cdlod::grid_wire_indices` for why this
        // is not the usual barycentric trick.
        let line_mode = ctx.supports_polygon_line();
        let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-wireframe"),
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
                    format: crate::context::SCENE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: if line_mode {
                    wgpu::PrimitiveTopology::TriangleList
                } else {
                    wgpu::PrimitiveTopology::LineList
                },
                polygon_mode: if line_mode {
                    wgpu::PolygonMode::Line
                } else {
                    wgpu::PolygonMode::Fill
                },
                // No culling: a wireframe is meant to show the far side of the
                // surface as well, and back-face culling hides exactly the edges
                // that reveal a fold.
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
        });

        let wire_indices = cdlod::patch_wire_indices();
        let wire_index_count = wire_indices.len() as u32;
        let wire_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-wire-indices"),
            contents: bytemuck::cast_slice(&wire_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Depth-only pass for the shadow cascades. Group 0 is a cascade's light
        // matrix rather than the camera; the shader is the same module.
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-shadow-layout"),
            bind_group_layouts: &[Some(&lighting.cascade_layout), Some(&terrain_bgl)],
            immediate_size: 0,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-shadow-pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Front faces cast: shifting the shadow caster to the far side
                // of the geometry hides most acne without a large depth bias.
                cull_mode: Some(wgpu::Face::Front),
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
        });

        Self {
            heights,
            res,
            extent_m,
            splat,
            splat_painted: false,
            splat_tex,
            splat_views: splat_views_kept,
            height_buf,
            flow_buf,
            deposit_buf,
            road_buf,
            rut_buf,
            terrain_ub,
            camera_ub,
            cdlod,
            patch_buf,
            patch_count,
            height_range: (0.0, 0.0),
            index_buf,
            index_count,
            wire_pipeline,
            wire_index_buf,
            wire_index_count,
            wire_uses_polygon_line: line_mode,
            camera_bgl,
            camera_bg,
            terrain_bg,
            material_bg: materials.bind_group.clone(),
            pipeline,
            shadow_pipeline,
            brush,
        }
    }

    /// Replace the whole heightfield, e.g. from the generator. Uploads the
    /// full buffer; at 4096^2 that is 67 MB, which is a one-off on a
    /// regenerate, not a per-frame cost.
    pub fn set_heights(&mut self, queue: &wgpu::Queue, heights: Vec<f32>) {
        assert_eq!(heights.len(), self.heights.len(), "heightfield size mismatch");
        self.heights = heights;
        self.height_range = height_range(&self.heights);
        queue.write_buffer(&self.height_buf, 0, bytemuck::cast_slice(&self.heights));
    }

    /// Upload the erosion by-product masks. Both are `0..=1`, `res * res`.
    pub fn set_masks(&self, queue: &wgpu::Queue, flow: &[f32], deposition: &[f32]) {
        let n = self.heights.len();
        if flow.len() == n {
            queue.write_buffer(&self.flow_buf, 0, bytemuck::cast_slice(flow));
        }
        if deposition.len() == n {
            queue.write_buffer(&self.deposit_buf, 0, bytemuck::cast_slice(deposition));
        }
    }

    /// Upload the road surface masks. Both `0..=1`, `res * res`.
    pub fn set_road_masks(&self, queue: &wgpu::Queue, road: &[f32], rut: &[f32]) {
        let n = self.heights.len();
        if road.len() == n {
            queue.write_buffer(&self.road_buf, 0, bytemuck::cast_slice(road));
        }
        if rut.len() == n {
            queue.write_buffer(&self.rut_buf, 0, bytemuck::cast_slice(rut));
        }
    }

    /// Heightfield storage buffer, for passes that place things on the surface.
    /// The camera bind group layout, so passes that draw in this world can
    /// share it rather than declaring a structurally identical copy.
    pub fn camera_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bgl
    }

    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bg
    }

    pub fn height_buffer(&self) -> &wgpu::Buffer {
        &self.height_buf
    }

    /// Painted layer weights, four per view.
    pub fn splat_views(&self) -> &[wgpu::TextureView; 2] {
        &self.splat_views
    }

    pub fn extent_m(&self) -> f32 {
        self.extent_m
    }

    pub fn resolution(&self) -> u32 {
        self.res
    }

    /// Triangles the last selection submits. Varies with the camera now, which is
    /// the point -- the stats panel is showing an adaptive mesh, not a fixed grid.
    pub fn triangle_count(&self) -> u32 {
        self.cdlod.triangle_count()
    }

    /// Patches the last selection chose, and the LOD levels in the tree.
    pub fn lod_stats(&self) -> (u32, u32) {
        (self.patch_count, self.cdlod.levels())
    }

    /// Upload the camera and reselect the LOD patches for it.
    ///
    /// The two belong together: the patch set is a pure function of the camera, and
    /// splitting them would let a frame draw one camera's patches through another
    /// camera's matrix -- which looks like the terrain tearing along level
    /// boundaries as you move.
    pub fn upload_camera(&mut self, queue: &wgpu::Queue, cam: &Camera, aspect: f32) {
        queue.write_buffer(&self.camera_ub, 0, bytemuck::bytes_of(&cam.uniform(aspect)));

        let patches = self.cdlod.select(cam.pos, self.height_range, self.extent_m);
        self.patch_count = patches.len() as u32;
        queue.write_buffer(&self.patch_buf, 0, bytemuck::cast_slice(patches));

        // The shader morphs from the same eye and the same slab gap the selection
        // above used, so the two cannot disagree about where a level ends.
        let gap = cdlod::vertical_gap(cam.pos.y, self.height_range);
        self.brush.morph_eye = [cam.pos.x, cam.pos.z, gap, 0.0];
        queue.write_buffer(&self.terrain_ub, 0, bytemuck::bytes_of(&self.brush));
    }

    pub fn set_brush(&mut self, queue: &wgpu::Queue, center: Option<Vec2>, radius: f32) {
        self.brush.brush_radius = radius;
        match center {
            Some(c) => {
                self.brush.brush_center = c.into();
                self.brush.brush_active = 1.0;
            }
            None => self.brush.brush_active = 0.0,
        }
        queue.write_buffer(&self.terrain_ub, 0, bytemuck::bytes_of(&self.brush));
    }

    /// Height in meters at a world XZ position, bilinearly filtered.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let n = self.res as f32;
        let u = (x / self.extent_m + 0.5) * (n - 1.0);
        let v = (z / self.extent_m + 0.5) * (n - 1.0);
        let (x0, z0) = (u.floor(), v.floor());
        let (fx, fz) = (u - x0, v - z0);
        let (x0, z0) = (x0 as i32, z0 as i32);

        let t = |ix: i32, iz: i32| -> f32 {
            let ix = ix.clamp(0, self.res as i32 - 1) as usize;
            let iz = iz.clamp(0, self.res as i32 - 1) as usize;
            self.heights[iz * self.res as usize + ix]
        };

        let a = t(x0, z0) + (t(x0 + 1, z0) - t(x0, z0)) * fx;
        let b = t(x0, z0 + 1) + (t(x0 + 1, z0 + 1) - t(x0, z0 + 1)) * fx;
        a + (b - a) * fz
    }

    /// March a ray against the heightfield. Returns the first hit in world
    /// space.
    ///
    /// Fixed-step march plus a binary refine. Coarse enough to stay cheap at
    /// 16 km, fine enough that the brush lands where the cursor points; a
    /// proper hierarchical march is only worth it once terrain is much steeper
    /// than a sculpt session produces.
    /// Surface normal at a world position, from central differences on the CPU
    /// heightfield. Scatter needs it to reject steep ground and to lean
    /// instances with the slope.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let step = self.extent_m / (self.res - 1) as f32;
        let l = self.height_at(x - step, z);
        let r = self.height_at(x + step, z);
        let d = self.height_at(x, z - step);
        let u = self.height_at(x, z + step);
        Vec3::new(l - r, 2.0 * step, d - u).normalize_or(Vec3::Y)
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<Vec3> {
        let half = self.extent_m * 0.5;
        let step = (self.extent_m / self.res as f32).max(1.0);
        let max_dist = self.extent_m * 2.0;

        let mut t = 0.0f32;
        let mut prev = origin;
        while t < max_dist {
            t += step;
            let p = origin + dir * t;
            if p.x < -half || p.x > half || p.z < -half || p.z > half {
                if p.y < -1000.0 {
                    return None;
                }
                prev = p;
                continue;
            }
            if p.y <= self.height_at(p.x, p.z) {
                // Refine between the last miss and this hit.
                let (mut lo, mut hi) = (prev, p);
                for _ in 0..12 {
                    let mid = (lo + hi) * 0.5;
                    if mid.y <= self.height_at(mid.x, mid.z) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(hi);
            }
            prev = p;
        }
        None
    }

    /// Apply one brush dab to the live field, then upload only the rows it
    /// touched.
    ///
    /// Swap in a rebuilt palette.
    ///
    /// Importing a texture rebuilds [`Materials`], which means a new bind group
    /// and a new layer count. Without this the editor would need a restart to
    /// see anything it had just imported, which for a content browser is the
    /// same as the import not working.
    ///
    /// The bind group *layout* is unchanged, so the pipelines stay valid and
    /// only the group and three uniform fields move.
    pub fn set_materials(&mut self, queue: &wgpu::Queue, materials: &Materials) {
        self.material_bg = materials.bind_group.clone();
        self.brush.layer_count = materials.count().min(MAX_LAYERS);
        self.brush.grass_layer =
            materials.layers.iter().position(|l| l.role == crate::material::GRASS).unwrap_or(0)
                as u32;
        self.brush.layer_roles = pack_roles(materials);
        queue.write_buffer(&self.terrain_ub, 0, bytemuck::bytes_of(&self.brush));
    }

    /// Callers keeping a separate base layer (see the road system) should apply
    /// [`apply_brush`] to that layer with the same arguments, so the edit
    /// survives a road rebuild.
    pub fn sculpt(
        &mut self,
        queue: &wgpu::Queue,
        center: Vec2,
        radius: f32,
        strength: f32,
        op: &BrushOp<'_>,
    ) {
        let Some((x0, x1, z0, z1)) =
            apply_brush(&mut self.heights, self.res, self.extent_m, center, radius, strength, op)
        else {
            return;
        };
        let n = self.res as i32;
        // One write per touched row; rows are contiguous in memory.
        let row_len = (x1 - x0 + 1) as usize;
        for z in z0..=z1 {
            let start = (z * n + x0) as usize;
            let offset = (start * std::mem::size_of::<f32>()) as u64;
            let row = &self.heights[start..start + row_len];
            // Widen the cached height slab from the rows we already have in hand.
            // Only ever widened, never narrowed: recovering the true range after a
            // Lower stroke would mean rescanning the whole field mid-drag, and a slab
            // wider than the terrain only makes LOD selection more conservative.
            for h in row {
                self.height_range.0 = self.height_range.0.min(*h);
                self.height_range.1 = self.height_range.1.max(*h);
            }
            queue.write_buffer(&self.height_buf, offset, bytemuck::cast_slice(row));
        }
    }

    // --- material painting ---

    /// Painted weights, as saved with the world.
    pub fn splat(&self) -> &[u8] {
        &self.splat
    }

    pub fn splat_res() -> u32 {
        SPLAT_RES
    }

    /// Restore painted weights loaded from disk. A mismatched length is
    /// ignored rather than fatal -- a world saved before painting existed, or
    /// with a different resolution, simply comes back unpainted.
    pub fn set_splat(&mut self, queue: &wgpu::Queue, splat: Vec<u8>) {
        if splat.len() != self.splat.len() {
            if !splat.is_empty() {
                log::warn!("splat map is {} bytes, expected {}", splat.len(), self.splat.len());
            }
            return;
        }
        self.splat_painted = splat.iter().any(|&v| v != 0);
        self.splat = splat;
        self.upload_splat(queue, 0, 0, SPLAT_RES, SPLAT_RES);
    }

    /// Paint `layer` under the brush.
    ///
    /// The weights behave the way a terrain painter is expected to: the target
    /// layer rises toward full and the others give way in proportion, so the
    /// set always sums to at most one and no layer can be starved by painting
    /// its neighbour.
    pub fn paint(
        &mut self,
        queue: &wgpu::Queue,
        center: Vec2,
        radius_m: f32,
        strength: f32,
        layer: u32,
    ) {
        if layer >= MAX_LAYERS {
            return;
        }
        let Some((x0, x1, z0, z1)) = self.splat_window(center, radius_m) else {
            return;
        };
        let n = MAX_LAYERS as usize;

        for z in z0..=z1 {
            for x in x0..=x1 {
                let wx = (x as f32 / (SPLAT_RES - 1) as f32 - 0.5) * self.extent_m;
                let wz = (z as f32 / (SPLAT_RES - 1) as f32 - 0.5) * self.extent_m;
                let d = ((wx - center.x).powi(2) + (wz - center.y).powi(2)).sqrt();
                if d > radius_m {
                    continue;
                }
                // Soft-shouldered falloff. A linear one leaves a visible cone
                // edge wherever two strokes overlap.
                let t = (d / radius_m.max(1e-3)).clamp(0.0, 1.0);
                let fall = (1.0 - t * t).powi(2);
                let amount = (strength * fall).clamp(0.0, 1.0);
                if amount <= 0.0 {
                    continue;
                }

                let base = ((z * SPLAT_RES + x) as usize) * n;
                let cur = self.splat[base + layer as usize] as f32 / 255.0;
                let target = (cur + amount).min(1.0);
                let rest: f32 = 1.0 - target;

                let others: f32 = (0..n)
                    .filter(|i| *i != layer as usize)
                    .map(|i| self.splat[base + i] as f32 / 255.0)
                    .sum();
                if others > 1e-4 {
                    let k = (rest / others).min(1.0);
                    for i in 0..n {
                        if i != layer as usize {
                            let v = self.splat[base + i] as f32 / 255.0 * k;
                            self.splat[base + i] = (v * 255.0 + 0.5) as u8;
                        }
                    }
                }
                self.splat[base + layer as usize] = (target * 255.0 + 0.5) as u8;
                self.splat_painted = true;
            }
        }
        self.upload_splat(queue, x0, z0, x1 - x0 + 1, z1 - z0 + 1);
    }

    /// Lift painting under the brush, fading each texel back toward the
    /// automatic weights rather than toward an empty surface.
    pub fn erase(&mut self, queue: &wgpu::Queue, center: Vec2, radius_m: f32, strength: f32) {
        let Some((x0, x1, z0, z1)) = self.splat_window(center, radius_m) else {
            return;
        };
        let n = MAX_LAYERS as usize;
        for z in z0..=z1 {
            for x in x0..=x1 {
                let wx = (x as f32 / (SPLAT_RES - 1) as f32 - 0.5) * self.extent_m;
                let wz = (z as f32 / (SPLAT_RES - 1) as f32 - 0.5) * self.extent_m;
                let d = ((wx - center.x).powi(2) + (wz - center.y).powi(2)).sqrt();
                if d > radius_m {
                    continue;
                }
                let t = (d / radius_m.max(1e-3)).clamp(0.0, 1.0);
                let k = 1.0 - (strength * (1.0 - t * t).powi(2)).clamp(0.0, 1.0);
                let base = ((z * SPLAT_RES + x) as usize) * n;
                for i in 0..n {
                    self.splat[base + i] = (self.splat[base + i] as f32 * k) as u8;
                }
            }
        }
        self.upload_splat(queue, x0, z0, x1 - x0 + 1, z1 - z0 + 1);
    }

    /// Cover the whole world with one layer. The "set the base coat" action
    /// every terrain tool has, and the fastest way out of a painting mistake.
    pub fn fill(&mut self, queue: &wgpu::Queue, layer: u32) {
        if layer >= MAX_LAYERS {
            return;
        }
        let n = MAX_LAYERS as usize;
        for texel in self.splat.chunks_exact_mut(n) {
            texel.fill(0);
            texel[layer as usize] = 255;
        }
        self.splat_painted = true;
        self.upload_splat(queue, 0, 0, SPLAT_RES, SPLAT_RES);
    }

    /// Discard all painting and hand the surface back to the automatic weights.
    pub fn clear_paint(&mut self, queue: &wgpu::Queue) {
        self.splat.fill(0);
        self.splat_painted = false;
        self.upload_splat(queue, 0, 0, SPLAT_RES, SPLAT_RES);
    }

    /// True if anything has been painted. Lets the UI say which mode the
    /// surface is in rather than leaving it a guess.
    pub fn is_painted(&self) -> bool {
        self.splat_painted
    }

    /// Splat-space bounding box of a brush, clamped to the map.
    fn splat_window(&self, center: Vec2, radius_m: f32) -> Option<(u32, u32, u32, u32)> {
        let to_texel = |v: f32| (v / self.extent_m + 0.5) * (SPLAT_RES - 1) as f32;
        let last = (SPLAT_RES - 1) as f32;
        let x0 = to_texel(center.x - radius_m).floor().clamp(0.0, last) as u32;
        let x1 = to_texel(center.x + radius_m).ceil().clamp(0.0, last) as u32;
        let z0 = to_texel(center.y - radius_m).floor().clamp(0.0, last) as u32;
        let z1 = to_texel(center.y + radius_m).ceil().clamp(0.0, last) as u32;
        (x1 >= x0 && z1 >= z0).then_some((x0, x1, z0, z1))
    }

    /// Push a rectangle of weights to both halves of the palette.
    fn upload_splat(&self, queue: &wgpu::Queue, x: u32, z: u32, w: u32, h: u32) {
        let n = MAX_LAYERS as usize;
        for half in 0..2usize {
            let mut rows = Vec::with_capacity((w * h * 4) as usize);
            for row in 0..h {
                for col in 0..w {
                    let base = (((z + row) * SPLAT_RES + x + col) as usize) * n;
                    for c in 0..4 {
                        rows.push(self.splat[base + half * 4 + c]);
                    }
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.splat_tex[half],
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y: z, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &rows,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
    }

    /// Depth-only draw into one shadow cascade.
    pub fn draw_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        cascade: usize,
    ) {
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &lighting.cascade_bind_group, &[Lighting::cascade_offset(cascade)]);
        pass.set_bind_group(1, &self.terrain_bg, &[]);
        // The same patches the colour pass draws, morphed from the same camera. A
        // caster that disagreed with the shaded surface would put a band of acne
        // along every level boundary.
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.patch_count);
    }

    /// Which view mode subsequent draws use.
    ///
    /// Uploaded rather than passed per draw: the shader branches on it, and a
    /// mode change is a keypress rather than a per-frame value.
    pub fn set_view_mode(&mut self, queue: &wgpu::Queue, mode: crate::ViewMode) {
        if self.brush.view_mode == mode.shader_index() {
            return;
        }
        self.brush.view_mode = mode.shader_index();
        queue.write_buffer(&self.terrain_ub, 0, bytemuck::bytes_of(&self.brush));
    }

    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        clouds: &crate::clouds::Clouds,
        mode: crate::ViewMode,
    ) {
        if mode.is_wireframe() {
            pass.set_pipeline(&self.wire_pipeline);
        } else {
            pass.set_pipeline(&self.pipeline);
        }
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &self.terrain_bg, &[]);
        pass.set_bind_group(2, &self.material_bg, &[]);
        pass.set_bind_group(3, &lighting.bind_group, &[]);
        pass.set_bind_group(4, &clouds.shadow_bind_group, &[]);

        // On the fallback path the wireframe is a different buffer with a
        // different length, so the index buffer has to follow the mode and not
        // just the pipeline.
        if mode.is_wireframe() && !self.wire_uses_polygon_line {
            pass.set_index_buffer(self.wire_index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.wire_index_count, 0, 0..self.patch_count);
        } else {
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..self.patch_count);
        }
    }
}

/// Roles laid out the way the uniform block expects them.
fn pack_roles(materials: &Materials) -> [[u32; 4]; 2] {
    let mut out = [[crate::material::ROLE_NONE; 4]; 2];
    for (i, layer) in materials.layers.iter().take(MAX_LAYERS as usize).enumerate() {
        out[i / 4][i % 4] = layer.role;
    }
    out
}

/// Bilinear height at a world position, on a bare slice.
fn sample_bilinear(heights: &[f32], res: u32, extent_m: f32, x: f32, z: f32) -> f32 {
    let n = res as i32;
    let fx = ((x / extent_m + 0.5) * (res - 1) as f32).clamp(0.0, (res - 1) as f32);
    let fz = ((z / extent_m + 0.5) * (res - 1) as f32).clamp(0.0, (res - 1) as f32);
    let (x0, z0) = (fx.floor() as i32, fz.floor() as i32);
    let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
    let at = |ix: i32, iz: i32| -> f32 {
        heights[(iz.clamp(0, n - 1) * n + ix.clamp(0, n - 1)) as usize]
    };
    let top = at(x0, z0) + (at(x0 + 1, z0) - at(x0, z0)) * tx;
    let bot = at(x0, z0 + 1) + (at(x0 + 1, z0 + 1) - at(x0, z0 + 1)) * tx;
    top + (bot - top) * tz
}

/// Apply one brush dab to a heightfield in place.
///
/// Returns the inclusive texel rectangle it touched, or `None` if the brush
/// missed the field entirely. Free-standing so the same edit can be applied to
/// both the rendered field and a base layer underneath it.
#[allow(clippy::too_many_arguments)]
/// Everything one brush dab needs beyond its geometry.
///
/// A struct rather than three more positional arguments: `apply_brush` already
/// took seven, and `(.., strength, drag, noise, mode)` at a call site is a line
/// nobody can read without going to look at the signature.
pub struct BrushOp<'a> {
    pub mode: SculptMode,
    /// Cursor travel this dab, in world metres. Only [`SculptMode::Move`] reads
    /// it; zero is a no-op for that mode.
    pub drag: Vec2,
    /// Pattern for [`SculptMode::Noise`]. `None` makes that mode a no-op rather
    /// than an error, so a caller with no pattern loaded is still valid.
    pub noise: Option<&'a terra_voxel::NoiseField>,
}

impl BrushOp<'_> {
    pub fn new(mode: SculptMode) -> Self {
        Self { mode, drag: Vec2::ZERO, noise: None }
    }
}

pub fn apply_brush(
    heights: &mut [f32],
    res: u32,
    extent_m: f32,
    center: Vec2,
    radius: f32,
    strength: f32,
    op: &BrushOp<'_>,
) -> Option<(i32, i32, i32, i32)> {
    let mode = op.mode;
    // Noise with no pattern loaded, or a Move with no travel, would otherwise
    // walk the whole brush rectangle writing zeros.
    if (mode == SculptMode::Noise && op.noise.is_none())
        || (mode == SculptMode::Move && op.drag.length_squared() < 1e-12)
    {
        return None;
    }
    let n = res as i32;
    let to_texel = |w: f32| ((w / extent_m + 0.5) * (res - 1) as f32).round() as i32;
    let r_texels = (radius / extent_m * (res - 1) as f32).ceil() as i32 + 1;

    let cx = to_texel(center.x);
    let cz = to_texel(center.y);
    let x0 = (cx - r_texels).clamp(0, n - 1);
    let x1 = (cx + r_texels).clamp(0, n - 1);
    let z0 = (cz - r_texels).clamp(0, n - 1);
    let z1 = (cz + r_texels).clamp(0, n - 1);
    if x0 > x1 || z0 > z1 {
        return None;
    }

    // Drag converted to whole texels: the scratch snapshot is indexed by texel,
    // so a sub-texel drag rounds to zero and the mode correctly does nothing
    // until the cursor has actually moved a texel.
    let texels_per_m = (res - 1) as f32 / extent_m;
    let drag_texels = {
        let d = op.drag * texels_per_m;
        (d.x.round() as i32, d.y.round() as i32)
    };
    // Sampled in 3D so the pattern does not slide when the ground moves under
    // it; the Y coordinate is the current height.
    let noise_at = |wx: f32, wy: f32, wz: f32| -> f32 {
        op.noise.map_or(0.0, |n| n.sample(glam::Vec3::new(wx, wy, wz), glam::Vec3::Y))
    };

    // Smooth reads neighbours, so it must not observe its own writes.
    //
    // Copy ONLY the brush rectangle plus a one-texel apron. Cloning the
    // whole heightfield here allocated and freed the entire field on every
    // dab -- 67 MB per frame at the largest world size, which is enough
    // allocator churn to put the machine into memory pressure.
    let scratch =
        matches!(mode, SculptMode::Smooth | SculptMode::Move | SculptMode::Pinch).then(|| {
            let sx = (x0 - 1).max(0);
            let sxe = (x1 + 1).min(n - 1);
            let sz = (z0 - 1).max(0);
            let sze = (z1 + 1).min(n - 1);
            let w = (sxe - sx + 1) as usize;
            let mut buf = Vec::with_capacity(w * (sze - sz + 1) as usize);
            for z in sz..=sze {
                let start = (z * n + sx) as usize;
                buf.extend_from_slice(&heights[start..start + w]);
            }
            Scratch { x0: sx, x1: sxe, z0: sz, z1: sze, w, buf }
        });

    // Flatten targets the height under the brush centre, sampled once so a
    // held stroke converges instead of chasing itself.
    let target = sample_bilinear(heights, res, extent_m, center.x, center.y);

    let texel_m = extent_m / (res - 1) as f32;
    for z in z0..=z1 {
        for x in x0..=x1 {
            let wx = (x as f32 / (res - 1) as f32 - 0.5) * extent_m;
            let wz = (z as f32 / (res - 1) as f32 - 0.5) * extent_m;
            let d = Vec2::new(wx - center.x, wz - center.y).length();
            if d > radius {
                continue;
            }
            // Squared smoothstep -- a linear falloff leaves a visible cone
            // tip at the brush centre when strokes overlap.
            let f = 1.0 - smoothstep(0.0, radius, d);
            let w = f * f;
            let i = (z * n + x) as usize;

            match mode {
                SculptMode::Raise => heights[i] += w * strength,
                SculptMode::Lower => heights[i] -= w * strength,
                SculptMode::Flatten => {
                    let a = (w * strength / texel_m.max(1.0)).clamp(0.0, 1.0);
                    heights[i] += (target - heights[i]) * a;
                }
                SculptMode::Smooth => {
                    let s = scratch.as_ref().unwrap();
                    let avg =
                        (s.get(x - 1, z) + s.get(x + 1, z) + s.get(x, z - 1) + s.get(x, z + 1))
                            * 0.25;
                    let a = (w * strength / texel_m.max(1.0)).clamp(0.0, 1.0);
                    heights[i] += (avg - heights[i]) * a;
                }
                // Raise toward the brush plane, never past it and never below
                // the existing ground. That "never cuts" property is what makes
                // repeated strokes build a shape instead of inflating one.
                SculptMode::Clay => {
                    let a = (w * strength / texel_m.max(1.0)).clamp(0.0, 1.0);
                    if target > heights[i] {
                        heights[i] += (target - heights[i]) * a;
                    }
                }
                // Sample the pre-stroke field one texel back along the drag, so
                // the surface travels with its detail rather than being averaged.
                SculptMode::Move => {
                    let s = scratch.as_ref().unwrap();
                    let (dx, dz) = drag_texels;
                    let src = s.get(x - dx, z - dz);
                    let a = (w * strength / texel_m.max(1.0)).clamp(0.0, 1.0);
                    heights[i] += (src - heights[i]) * a;
                }
                // Displacement, not a target: noise adds relief rather than
                // converging on a height, so a held stroke keeps roughening
                // instead of settling.
                SculptMode::Noise => {
                    let v = noise_at(wx, heights[i], wz);
                    heights[i] += w * strength * v;
                }
                // Pull heights inward toward the brush centre. Sampling further
                // out drags the outer profile in, which narrows a rise into a
                // ridge; the clamp stops the sample crossing the centre.
                SculptMode::Pinch => {
                    let s = scratch.as_ref().unwrap();
                    let (ox, oz) = (x - cx, z - cz);
                    let len = ((ox * ox + oz * oz) as f32).sqrt();
                    if len >= 1.0 {
                        let step = 1.0f32.min(len);
                        let sx = x + ((ox as f32 / len) * step).round() as i32;
                        let sz = z + ((oz as f32 / len) * step).round() as i32;
                        let outer = s.get(sx, sz);
                        let a = (w * strength / texel_m.max(1.0)).clamp(0.0, 1.0);
                        heights[i] += (outer - heights[i]) * a;
                    }
                }
            }
        }
    }

    Some((x0, x1, z0, z1))
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Min and max of a heightfield, for the altitude term in LOD selection.
fn height_range(heights: &[f32]) -> (f32, f32) {
    heights.iter().fold((f32::MAX, f32::MIN), |(lo, hi), h| (lo.min(*h), hi.max(*h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader block is 96 bytes. Inserting a scalar in the wrong place
    /// pushes `brush_center` off its 8-byte alignment and the two sides
    /// silently disagree -- which shows up as a validation error the moment a
    /// world is open, and never before.
    #[test]
    fn uniform_matches_the_shader_block() {
        assert_eq!(std::mem::size_of::<TerrainUniform>(), 96);
    }

    #[test]
    fn scratch_clamps_to_its_own_window() {
        // Rows 2..=4, columns 1..=3 of some larger field.
        let s = Scratch {
            x0: 1,
            x1: 3,
            z0: 2,
            z1: 4,
            w: 3,
            buf: vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0],
        };
        assert_eq!(s.get(1, 2), 10.0);
        assert_eq!(s.get(3, 4), 32.0);
        // Neighbour lookups that fall outside the copied window clamp to its
        // edge rather than indexing out of bounds.
        assert_eq!(s.get(-99, 2), 10.0);
        assert_eq!(s.get(3, 99), 32.0);
    }

    #[test]
    fn smoothstep_is_clamped_and_monotonic() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!(smoothstep(0.0, 1.0, 0.25) < smoothstep(0.0, 1.0, 0.75));
    }
}

#[cfg(test)]
mod brush_mode_tests {
    use super::*;

    const RES: u32 = 96;
    const EXTENT: f32 = 200.0;

    fn flat() -> Vec<f32> {
        vec![100.0f32; (RES * RES) as usize]
    }

    fn at(h: &[f32], wx: f32, wz: f32) -> f32 {
        let t = |w: f32| {
            (((w / EXTENT + 0.5) * (RES - 1) as f32).round() as i32).clamp(0, RES as i32 - 1)
        };
        h[(t(wz) * RES as i32 + t(wx)) as usize]
    }

    fn dab(h: &mut [f32], op: &BrushOp<'_>, at_xz: Vec2, radius: f32, strength: f32) {
        apply_brush(h, RES, EXTENT, at_xz, radius, strength, op);
    }

    #[test]
    fn clay_builds_up_but_never_cuts_down() {
        // The property that separates Clay from Raise: it converges on the brush
        // plane instead of adding without limit, and leaves higher ground alone.
        let mut h = flat();
        // Put a mound inside the brush that already stands above the plane.
        for (i, v) in h.iter_mut().enumerate() {
            let x = (i as u32 % RES) as i32;
            let z = (i as u32 / RES) as i32;
            if (x - 48).abs() < 3 && (z - 48).abs() < 3 {
                *v = 130.0;
            }
        }
        let before_mound = at(&h, 0.0, 0.0);

        let mut op = BrushOp::new(SculptMode::Clay);
        op.drag = Vec2::ZERO;
        for _ in 0..40 {
            dab(&mut h, &op, Vec2::ZERO, 40.0, 4.0);
        }

        let flat_ground = at(&h, 18.0, 0.0);
        assert!(flat_ground > 100.5, "clay should fill toward the plane, got {flat_ground}");
        assert!(
            at(&h, 0.0, 0.0) <= before_mound + 0.01,
            "clay must not raise ground already above the plane: {} -> {}",
            before_mound,
            at(&h, 0.0, 0.0)
        );
    }

    #[test]
    fn move_shifts_the_surface_along_the_drag() {
        // A step in the terrain, dragged in +X: the step must travel, so the
        // height just past its original edge rises.
        let mut h = flat();
        for (i, v) in h.iter_mut().enumerate() {
            if (i as u32 % RES) < 48 {
                *v = 140.0;
            }
        }
        let probe = 6.0f32;
        let before = at(&h, probe, 0.0);

        let mut op = BrushOp::new(SculptMode::Move);
        op.drag = Vec2::new(6.0, 0.0);
        for _ in 0..12 {
            dab(&mut h, &op, Vec2::ZERO, 40.0, 6.0);
        }
        let after = at(&h, probe, 0.0);
        assert!(after > before + 2.0, "the step did not travel: {before} -> {after}");
    }

    #[test]
    fn move_with_no_drag_is_a_no_op() {
        // Otherwise a held Move brush that is not being dragged would slowly
        // blur the terrain, which is Smooth's job and not what was asked for.
        let mut h = flat();
        h[(48 * RES + 48) as usize] = 150.0;
        let before = h.clone();
        let op = BrushOp::new(SculptMode::Move);
        dab(&mut h, &op, Vec2::ZERO, 40.0, 4.0);
        assert_eq!(h, before);
    }

    #[test]
    fn noise_roughens_and_needs_a_pattern() {
        let spread = |h: &[f32]| {
            let s: Vec<f32> = (-8..8).map(|i| at(h, i as f32 * 2.0, 0.0)).collect();
            let mean = s.iter().sum::<f32>() / s.len() as f32;
            (s.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / s.len() as f32).sqrt()
        };

        // With no pattern the mode must do nothing rather than panic.
        let mut none = flat();
        let before = none.clone();
        dab(&mut none, &BrushOp::new(SculptMode::Noise), Vec2::ZERO, 40.0, 4.0);
        assert_eq!(none, before, "noise with no pattern must be a no-op");

        let field = terra_voxel::NoiseField::procedural(7, 4, false, 14.0);
        let mut h = flat();
        let mut op = BrushOp::new(SculptMode::Noise);
        op.noise = Some(&field);
        dab(&mut h, &op, Vec2::ZERO, 40.0, 6.0);
        assert!(spread(&h) > 0.5, "noise did not roughen: spread {}", spread(&h));
    }

    #[test]
    fn noise_is_a_displacement_not_a_target() {
        // A held stroke must keep adding relief rather than converging, or the
        // amplitude slider would have no effect past the first dab.
        let field = terra_voxel::NoiseField::procedural(3, 3, false, 14.0);
        let mut op = BrushOp::new(SculptMode::Noise);
        op.noise = Some(&field);

        let extent_of = |dabs: u32| {
            let mut h = flat();
            for _ in 0..dabs {
                dab(&mut h, &op, Vec2::ZERO, 40.0, 3.0);
            }
            let s: Vec<f32> = (-8..8).map(|i| at(&h, i as f32 * 2.0, 0.0)).collect();
            s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                - s.iter().cloned().fold(f32::INFINITY, f32::min)
        };
        assert!(extent_of(8) > extent_of(1) * 1.5, "{} vs {}", extent_of(8), extent_of(1));
    }

    #[test]
    fn pinch_sharpens_a_rise() {
        // Pinch pulls the outer profile inward, so a broad dome narrows: the
        // centre keeps its height while the flanks drop.
        let mut h = flat();
        for (i, v) in h.iter_mut().enumerate() {
            let x = (i as u32 % RES) as f32;
            let z = (i as u32 / RES) as f32;
            let r = ((x - 47.5).powi(2) + (z - 47.5).powi(2)).sqrt();
            *v = 100.0 + (30.0 - r).max(0.0);
        }
        let sharpness = |h: &[f32]| at(h, 0.0, 0.0) - at(h, 24.0, 0.0);
        let before = sharpness(&h);

        let op = BrushOp::new(SculptMode::Pinch);
        for _ in 0..30 {
            dab(&mut h, &op, Vec2::ZERO, 50.0, 4.0);
        }
        assert!(sharpness(&h) > before, "pinch did not sharpen: {before} -> {}", sharpness(&h));
        assert!(h.iter().all(|v| v.is_finite()), "pinch produced non-finite heights");
    }

    #[test]
    fn every_mode_leaves_the_field_finite_and_bounded() {
        // A blanket guard: none of the eight may produce NaN or run away, which
        // is what a bad falloff or a division by a zero radius would do.
        let field = terra_voxel::NoiseField::default();
        for m in SculptMode::ALL {
            let mut h = flat();
            let mut op = BrushOp::new(m);
            op.drag = Vec2::new(3.0, -2.0);
            op.noise = Some(&field);
            for _ in 0..25 {
                dab(&mut h, &op, Vec2::new(5.0, -5.0), 35.0, 5.0);
            }
            assert!(h.iter().all(|v| v.is_finite()), "{} produced NaN", m.label());
            let lo = h.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!((-500.0..3000.0).contains(&lo), "{} lo {lo}", m.label());
            assert!((-500.0..3000.0).contains(&hi), "{} hi {hi}", m.label());
        }
    }
}

#[cfg(test)]
mod wireframe_tests {
    use super::*;

    #[test]
    fn the_terrain_uniform_keeps_its_vectors_aligned() {
        // The same trap that made `LayerParams` render the terrain red: WGSL
        // aligns vectors, `repr(C)` does not, and the two disagree silently. A
        // `vec2` needs 8 and the `vec4` array needs 16.
        assert_eq!(std::mem::offset_of!(TerrainUniform, brush_center) % 8, 0);
        // `morph_eye` is a vec4 and `layer_roles` an array of them, so both need 16.
        assert_eq!(std::mem::offset_of!(TerrainUniform, morph_eye) % 16, 0);
        assert_eq!(std::mem::offset_of!(TerrainUniform, layer_roles) % 16, 0);
        assert_eq!(std::mem::offset_of!(TerrainUniform, layer_roles), 64);
        assert_eq!(std::mem::size_of::<TerrainUniform>(), 96);
    }

    #[test]
    fn the_view_mode_lands_in_the_uniform() {
        // The shader branches on this, and the slot it uses was padding, so a
        // mismatch would read as the wrong mode rather than as an error.
        assert_eq!(std::mem::size_of::<TerrainUniform>(), 96);
        for m in crate::ViewMode::ALL {
            let u = TerrainUniform { view_mode: m.shader_index(), ..bytemuck::Zeroable::zeroed() };
            let words: &[u32; 24] = bytemuck::cast_ref(&u);
            // Offset 7: after extent, height_res, patch_quads, brush_radius,
            // brush_center (two), brush_active.
            assert_eq!(words[7], m.shader_index(), "{} landed in the wrong slot", m.label());
        }
    }
}

#[cfg(test)]
mod aliasing_tests {
    /// How many heightfield texels land in one pixel at a given view distance.
    ///
    /// The quantity that decides whether the terrain shimmers: once it exceeds
    /// one, the normal is being sampled finer than the pixel can show, and the
    /// temporal jitter puts each frame's sample somewhere different.
    fn texels_per_pixel(dist_m: f32, world_extent_m: f32, height_res: u32, viewport_h: f32) -> f32 {
        let fov_y = 60f32.to_radians();
        let pixel_m = 2.0 * dist_m * (fov_y * 0.5).tan() / viewport_h;
        let texel_m = world_extent_m / (height_res - 1) as f32;
        pixel_m / texel_m
    }

    #[test]
    fn zooming_out_undersamples_the_heightfield_badly() {
        // Why zooming out shook. A 4 km world at 1024 texels is a 3.9 m texel;
        // from 40 km one pixel spans 51 m, so thirteen texels fall inside it.
        let close = texels_per_pixel(900.0, 4000.0, 1024, 900.0);
        let far = texels_per_pixel(40_000.0, 4000.0, 1024, 900.0);
        assert!(close < 1.0, "up close a pixel should be finer than a texel, got {close}");
        assert!(far > 8.0, "far out should be badly undersampled, got {far}");
    }

    /// The fragment shader's fade, restated: full detail while a pixel is finer
    /// than a texel, gone by the time it spans six.
    fn detail_fade(oversample: f32) -> f32 {
        let t = ((oversample - 1.0) / 5.0).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    #[test]
    fn detail_is_kept_up_close_and_dropped_when_it_cannot_be_resolved() {
        assert_eq!(detail_fade(0.5), 1.0, "a pixel finer than a texel keeps full detail");
        assert_eq!(detail_fade(8.0), 0.0, "detail finer than the pixel must go");
        // Monotonic, so there is no distance at which detail comes back.
        let mut prev = 1.0;
        for i in 0..=80 {
            let f = detail_fade(i as f32 * 0.1);
            assert!(f <= prev + 1e-6, "detail rose with distance at {}", i as f32 * 0.1);
            prev = f;
        }
    }

    #[test]
    fn losing_the_bump_gains_roughness() {
        // The trade: the surface loses a bump it could not resolve and gains the
        // blur that bump would have averaged to. Dropping one without the other
        // leaves a mirror-smooth distant surface that sparkles just as badly.
        let rough_at = |oversample: f32| {
            let base = 0.4f32;
            let fade = detail_fade(oversample);
            (base + (1.0 - base) * ((1.0 - fade) * 0.75)).clamp(0.03, 1.0)
        };
        assert!((rough_at(0.5) - 0.4).abs() < 1e-6, "up close the material is unchanged");
        assert!(rough_at(8.0) > 0.7, "far out it must be much rougher, got {}", rough_at(8.0));
        assert!(rough_at(8.0) <= 1.0);
    }
}

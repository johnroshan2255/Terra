//! Terrain heightfield: GPU buffers, draw pipeline, and the sculpt brush.
//!
//! The CPU copy in [`Terrain::heights`] is authoritative. Sculpting edits it and
//! uploads only the touched rows, which keeps raycasting and saving correct
//! without any GPU readback. A brush covers at most a few thousand texels, so
//! this costs microseconds -- compute is for whole-map work like erosion.

use crate::camera::{Camera, CameraUniform};
use crate::context::{DEPTH_FORMAT, RenderContext};
use crate::lighting::Lighting;
use crate::material::{MAX_LAYERS, Materials};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use terra_core::WorldSize;
use wgpu::util::DeviceExt;

/// Grid quads per side. Independent of heightfield resolution: this is a
/// uniform grid placeholder for CDLOD, and 512 keeps the vertex count at ~263k
/// (0.5 M triangles), comfortably inside the 1.2 ms terrain budget.
const GRID_RES: u32 = 512;

/// Grid the terrain casts shadows from.
///
/// The same as the render grid. A coarser caster was tried -- 96 rather than
/// 512, which is thirty times less vertex work across three cascades -- and
/// measured no faster on this GPU, twice, interleaved. Depth-only vertex
/// throughput is simply not what this frame is spending its time on, and a
/// coarse caster costs silhouette accuracy for nothing.
const SHADOW_GRID_RES: u32 = GRID_RES;

/// Resolution of the painted layer weights.
///
/// Independent of, and much coarser than, the heightfield: painting is a
/// large-scale act -- the smallest brush is 8 m across -- and a splat map at
/// heightfield resolution would cost 8 bytes a texel for detail no brush can
/// place.
const SPLAT_RES: u32 = 1024;

/// Metres per material tile repeat.
///
/// Sets how much texture there is per metre of ground, and therefore how close
/// you can get before it turns to blur: a 1k tile over seven metres is under
/// seven millimetres per texel, while a screen pixel at arm's length covers
/// about two, so the near ground was magnified three times over and read as a
/// smear. Halving the repeat halves that. The cost is that the tile itself
/// repeats twice as often, which the macro wobble below and the grass standing
/// on top of it both hide.
const MATERIAL_SCALE_M: f32 = 3.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SculptMode {
    Raise,
    Lower,
    Smooth,
    Flatten,
}

impl SculptMode {
    pub const ALL: [SculptMode; 4] =
        [SculptMode::Raise, SculptMode::Lower, SculptMode::Smooth, SculptMode::Flatten];

    pub fn label(self) -> &'static str {
        match self {
            SculptMode::Raise => "Raise",
            SculptMode::Lower => "Lower",
            SculptMode::Smooth => "Smooth",
            SculptMode::Flatten => "Flatten",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainUniform {
    world_extent: f32,
    height_res: u32,
    grid_res: u32,
    brush_radius: f32,
    brush_center: [f32; 2],
    brush_active: f32,
    /// Metres covered by one repeat of a material tile. Smaller tiles show more
    /// grain up close and repeat more visibly at distance; this is the dial.
    mat_scale_m: f32,
    /// How many palette slots actually hold a material.
    layer_count: u32,
    /// Which slot the grass pass grows from, so the ground can darken under it.
    grass_layer: u32,
    /// Kept here rather than beside `grid_res`: a `u32` inserted there pushes
    /// `brush_center` off its 8-byte alignment, and the block silently grows to
    /// 96 bytes on the shader side while staying 84 on this one.
    shadow_grid_res: u32,
    _pad: u32,
    /// Automatic role per layer, or `ROLE_NONE`. Packed as two vec4s because a
    /// `u32` array in a uniform is padded to 16 bytes a element anyway.
    layer_roles: [[u32; 4]; 2],
}

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
    index_buf: wgpu::Buffer,
    index_count: u32,
    shadow_index_buf: wgpu::Buffer,
    shadow_index_count: u32,

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
            grid_res: GRID_RES,
            brush_radius: 0.0,
            brush_center: [0.0, 0.0],
            brush_active: 0.0,
            mat_scale_m: MATERIAL_SCALE_M,
            layer_count: materials.count().min(MAX_LAYERS),
            shadow_grid_res: SHADOW_GRID_RES,
            grass_layer: materials
                .layers
                .iter()
                .position(|l| l.role == crate::material::GRASS)
                .unwrap_or(0) as u32,
            _pad: 0,
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

        let (indices, index_count) = build_indices(GRID_RES);
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let (shadow_indices, shadow_index_count) = build_indices(SHADOW_GRID_RES);
        let shadow_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-shadow-indices"),
            contents: bytemuck::cast_slice(&shadow_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

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
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            // WGSL has no `#include`; the shared chunks are prepended the same
            // way the generation passes compose theirs.
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../../../assets/shaders/common/noise.wgsl"),
                    include_str!("../../../assets/shaders/common/lighting.wgsl"),
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
            index_buf,
            index_count,
            shadow_index_buf,
            shadow_index_count,
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

    /// Cells per side of the drawn mesh, which is coarser than the heightfield.
    /// Anything that has to sit *on* the visible ground -- grass especially --
    /// has to interpolate at this resolution, not the heightfield's, or it ends
    /// up under a surface that bridges over the detail it was placed in.
    pub fn mesh_resolution(&self) -> u32 {
        GRID_RES
    }

    pub fn resolution(&self) -> u32 {
        self.res
    }

    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }

    pub fn upload_camera(&self, queue: &wgpu::Queue, cam: &Camera, aspect: f32) {
        queue.write_buffer(&self.camera_ub, 0, bytemuck::bytes_of(&cam.uniform(aspect)));
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
    /// Callers keeping a separate base layer (see the road system) should apply
    /// [`apply_brush`] to that layer with the same arguments, so the edit
    /// survives a road rebuild.
    pub fn sculpt(
        &mut self,
        queue: &wgpu::Queue,
        center: Vec2,
        radius: f32,
        strength: f32,
        mode: SculptMode,
    ) {
        let Some((x0, x1, z0, z1)) =
            apply_brush(&mut self.heights, self.res, self.extent_m, center, radius, strength, mode)
        else {
            return;
        };
        let n = self.res as i32;
        // One write per touched row; rows are contiguous in memory.
        let row_len = (x1 - x0 + 1) as usize;
        for z in z0..=z1 {
            let start = (z * n + x0) as usize;
            let offset = (start * std::mem::size_of::<f32>()) as u64;
            queue.write_buffer(
                &self.height_buf,
                offset,
                bytemuck::cast_slice(&self.heights[start..start + row_len]),
            );
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
        pass.set_index_buffer(self.shadow_index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.shadow_index_count, 0, 0..1);
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, lighting: &Lighting) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &self.terrain_bg, &[]);
        pass.set_bind_group(2, &self.material_bg, &[]);
        pass.set_bind_group(3, &lighting.bind_group, &[]);
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
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
pub fn apply_brush(
    heights: &mut [f32],
    res: u32,
    extent_m: f32,
    center: Vec2,
    radius: f32,
    strength: f32,
    mode: SculptMode,
) -> Option<(i32, i32, i32, i32)> {
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

    // Smooth reads neighbours, so it must not observe its own writes.
    //
    // Copy ONLY the brush rectangle plus a one-texel apron. Cloning the
    // whole heightfield here allocated and freed the entire field on every
    // dab -- 67 MB per frame at the largest world size, which is enough
    // allocator churn to put the machine into memory pressure.
    let scratch = matches!(mode, SculptMode::Smooth).then(|| {
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
            }
        }
    }

    Some((x0, x1, z0, z1))
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Triangle-list indices for an `n x n` quad grid with `n + 1` verts per side.
/// Counter-clockwise when viewed from above, matching `cull_mode: Back`.
fn build_indices(n: u32) -> (Vec<u32>, u32) {
    let verts = n + 1;
    let mut idx = Vec::with_capacity((n * n * 6) as usize);
    for z in 0..n {
        for x in 0..n {
            let a = z * verts + x;
            let b = a + 1;
            let c = a + verts;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let count = idx.len() as u32;
    (idx, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader block is 80 bytes. Inserting a scalar in the wrong place
    /// pushes `brush_center` off its 8-byte alignment and the two sides
    /// silently disagree -- which shows up as a validation error the moment a
    /// world is open, and never before.
    #[test]
    fn uniform_matches_the_shader_block() {
        assert_eq!(std::mem::size_of::<TerrainUniform>(), 80);
    }

    #[test]
    fn index_buffer_covers_every_quad() {
        let (idx, count) = build_indices(4);
        assert_eq!(count, 4 * 4 * 6);
        assert_eq!(idx.iter().copied().max().unwrap(), 4 * 5 + 4);
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

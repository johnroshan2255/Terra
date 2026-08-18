//! Instanced solid meshes: the player's vehicle body and its four wheels.
//!
//! The geometry comes from the vehicle model on disk, split into a body and four wheels by
//! `terra_assets::VehicleRig`, so the thing drawn here is the same thing the collider was
//! measured from. `box_mesh` and `cylinder_mesh` remain only as the placeholder used when
//! that model cannot be read.
//!
//! That replacement was the plan from the start: the generators existed because a driveable
//! vehicle needs correct physics before it needs a nice model, and building the asset
//! pipeline first would have delayed the part that had to be right. The pipeline here did
//! not have to change to accept the real mesh, which is what that bet was for.

use crate::camera::{Camera, CameraUniform};
use crate::context::{DEPTH_FORMAT, RenderContext};
use crate::lighting::Lighting;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// One drawn instance, in **32 bytes**.
///
/// This used to be a 4x4 matrix plus an RGBA float colour: 80 bytes. The matrix
/// is the wasteful part, because every instance this renderer draws -- scattered
/// foliage, hand-placed props, the vehicle's chassis and wheels -- is a *rigid*
/// transform with *uniform* scale. Sixteen floats to say what a quaternion, a
/// scalar and a position say in twenty-two bytes.
///
/// Shrinking it is what makes per-LOD instance buffers affordable: three 32-byte
/// output buffers plus a 32-byte source total less memory than the single 80-byte
/// source and 80-byte output buffer they replace.
///
/// ```text
///  0..12   pos    3 x f32
/// 12..20   rot    4 x i16, snorm quaternion
/// 20..22   scale  f16, uniform
/// 22..24   pad
/// 24..28   color  4 x u8, unorm
/// 28..32   seed   u32
/// ```
///
/// The quaternion is a full one rather than a packed yaw, because instances are
/// not yaw-only: `Rules::align_to_normal` tilts an instance into the ground's
/// normal and `random_pitch_deg` adds more on top, so two axes of rotation are
/// live. A yaw-only species simply stores a quaternion whose x and z are zero.
///
/// `color` occupies what would otherwise be padding. It is needed per instance
/// rather than per draw: the Select tool highlights one prop by brightening its
/// colour, and the vehicle's parts are separate instances. Eight bits a channel
/// is ample for a tint that multiplies an albedo map.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub pos: [f32; 3],
    pub rot: [i16; 4],
    pub scale: u16,
    pub _pad: u16,
    pub color: [u8; 4],
    pub seed: u32,
}

const _: () = assert!(std::mem::size_of::<Instance>() == 32);
// Field offsets are load-bearing: the vertex attributes below and the packed
// reads in `scatter_cull.wgsl` both address this by byte offset.
const _: () = assert!(std::mem::offset_of!(Instance, pos) == 0);
const _: () = assert!(std::mem::offset_of!(Instance, rot) == 12);
const _: () = assert!(std::mem::offset_of!(Instance, scale) == 20);
const _: () = assert!(std::mem::offset_of!(Instance, color) == 24);
const _: () = assert!(std::mem::offset_of!(Instance, seed) == 28);

impl Instance {
    /// Build from a rigid, uniformly-scaled transform.
    ///
    /// Decomposing here rather than at every call site keeps the callers reading
    /// as they did. A non-uniform scale cannot be represented and is *not*
    /// silently averaged -- see [`Self::from_parts`].
    pub fn new(model: Mat4, color: Vec3) -> Self {
        let (scale, rot, pos) = model.to_scale_rotation_translation();
        Self::from_parts(pos, rot, scale.x, color, 0)
    }

    /// Build from the parts directly, which is what the scatter generator has
    /// anyway -- it composes a matrix only to have it taken apart again.
    pub fn from_parts(pos: Vec3, rot: glam::Quat, scale: f32, color: Vec3, seed: u32) -> Self {
        // i16 snorm gives ~3e-5 per component, which is far finer than the
        // orientation of a scattered rock needs, and the decode is a free
        // hardware conversion via `Snorm16x4`.
        let q = rot.normalize();
        let enc = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        Self {
            pos: pos.to_array(),
            rot: [enc(q.x), enc(q.y), enc(q.z), enc(q.w)],
            scale: half::f16::from_f32(scale).to_bits(),
            _pad: 0,
            color: [c(color.x), c(color.y), c(color.z), 255],
            seed,
        }
    }

    /// The instance's uniform scale, as stored. For tests and the debug readout.
    pub fn scale_f32(&self) -> f32 {
        half::f16::from_bits(self.scale).to_f32()
    }

    /// The stored rotation, decoded the way the vertex shader decodes it.
    pub fn rot_quat(&self) -> glam::Quat {
        let d = |v: i16| v as f32 / 32767.0;
        glam::Quat::from_xyzw(d(self.rot[0]), d(self.rot[1]), d(self.rot[2]), d(self.rot[3]))
            .normalize()
    }
}

/// Instance-step vertex attributes for [`Instance`], by byte offset.
///
/// `scale` is read as `Float16x2` covering bytes 20..24 -- the second component
/// is the padding and is ignored. There is no single-component f16 vertex format,
/// and pairing it with the pad costs nothing.
const INSTANCE_ATTRS: &[wgpu::VertexAttribute] = &[
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm16x4, offset: 12, shader_location: 4 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float16x2, offset: 20, shader_location: 5 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 24, shader_location: 6 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 28, shader_location: 7 },
];

/// Vertices, indices and the material bindings for one shape.
pub struct Mesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: wgpu::BindGroup,
    /// Leaf cards need both faces; solid geometry does not and should not pay
    /// for the doubled fill.
    double_sided: bool,
}

impl Mesh {
    pub fn index_count(&self) -> u32 {
        self.index_count
    }
}

/// A unit box spanning `-half..=half`, with flat per-face normals.
pub fn box_mesh(half: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let (x, y, z) = (half[0], half[1], half[2]);
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]),
        ([0.0, 0.0, -1.0], [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]]),
        ([1.0, 0.0, 0.0], [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]]),
        ([-1.0, 0.0, 0.0], [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]]),
        ([0.0, 1.0, 0.0], [[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]]),
        ([0.0, -1.0, 0.0], [[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]]),
    ];

    let mut verts = Vec::with_capacity(24);
    let mut idx = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = verts.len() as u32;
        for c in corners {
            verts.push(Vertex { position: c, normal, uv: [0.0, 0.0] });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, idx)
}

/// A cylinder lying along X -- the wheel axle direction -- so a wheel needs no
/// extra rotation to be mounted.
pub fn cylinder_mesh(radius: f32, half_width: f32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();

    // Tread.
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        let n = [0.0, c, s];
        verts.push(Vertex {
            position: [-half_width, c * radius, s * radius],
            normal: n,
            uv: [0.0, 0.0],
        });
        verts.push(Vertex {
            position: [half_width, c * radius, s * radius],
            normal: n,
            uv: [0.0, 0.0],
        });
    }
    for i in 0..segments {
        let b = i * 2;
        idx.extend_from_slice(&[b, b + 1, b + 3, b, b + 3, b + 2]);
    }

    // Two caps, each a fan around its own centre.
    for (sign, normal) in [(-1.0f32, [-1.0, 0.0, 0.0]), (1.0, [1.0, 0.0, 0.0])] {
        let centre = verts.len() as u32;
        verts.push(Vertex { position: [half_width * sign, 0.0, 0.0], normal, uv: [0.0, 0.0] });
        for i in 0..=segments {
            let a = i as f32 / segments as f32 * std::f32::consts::TAU;
            let (s, c) = a.sin_cos();
            verts.push(Vertex {
                position: [half_width * sign, c * radius, s * radius],
                normal,
                uv: [0.0, 0.0],
            });
        }
        for i in 0..segments {
            let (a, b) = (centre + 1 + i, centre + 2 + i);
            // Wind each cap the opposite way, or one of them faces inward and
            // vanishes under back-face culling.
            if sign < 0.0 {
                idx.extend_from_slice(&[centre, b, a]);
            } else {
                idx.extend_from_slice(&[centre, a, b]);
            }
        }
    }
    (verts, idx)
}

/// Draws solid meshes with per-instance transforms.
pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_ub: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    /// Same pipeline with back-face culling off, for cut-out foliage.
    pipeline_double: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    material_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    instances: wgpu::Buffer,
    capacity: usize,
    /// The vehicle body, one entry per part so each keeps its own material. Every part is
    /// drawn at the same chassis transform.
    pub chassis: Vec<Mesh>,
    /// One mesh per corner, front-left, front-right, rear-left, rear-right.
    ///
    /// Four rather than one reused, because the left and right wheels are mirrored copies.
    /// Reusing one would need a negative scale on the far side, which inverts triangle
    /// winding and lets back-face culling eat the visible faces -- the tyre turns inside
    /// out on one side of the vehicle.
    pub wheels: [Mesh; 4],
}

/// Instances uploaded per frame. One chassis plus four wheels, with room spare.
const MAX_INSTANCES: usize = 64;

impl MeshRenderer {
    /// `rig` is the vehicle's split mesh. `None` falls back to primitives, so a missing or
    /// unreadable vehicle model leaves the editor runnable instead of refusing to start --
    /// a box on four cylinders is obviously a placeholder, which is the right way to fail.
    pub fn new(
        ctx: &RenderContext,
        rig: Option<&terra_assets::VehicleRig>,
        lighting: &Lighting,
    ) -> Self {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let camera_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh-bgl"),
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
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh-camera-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_ub.as_entire_binding(),
            }],
        });

        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh-material-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 4,
            ..Default::default()
        });

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-instances"),
            size: (MAX_INSTANCES * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../../../assets/shaders/common/lighting.wgsl"),
                    include_str!("../../../assets/shaders/render/mesh.wgsl"),
                )
                .into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-layout"),
            bind_group_layouts: &[Some(&bgl), Some(&material_bgl), Some(&lighting.layout)],
            immediate_size: 0,
        });

        let mut descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("mesh-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3, 1 => Float32x3, 2 => Float32x2
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        // Spelled out rather than built with `vertex_attr_array!`,
                        // because these are not uniformly-sized fields packed in
                        // order -- the offsets are the ones asserted on `Instance`,
                        // and the formats do the decoding for free: `Snorm16x4`
                        // arrives as -1..1 floats and `Unorm8x4` as 0..1.
                        attributes: INSTANCE_ATTRS,
                    }),
                ],
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
                // Reversed-Z, same as the terrain.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        };
        let pipeline = device.create_render_pipeline(&descriptor);

        // Identical but without back-face culling, for cut-out foliage.
        descriptor.label = Some("mesh-pipeline-double");
        descriptor.primitive.cull_mode = None;
        let pipeline_double = device.create_render_pipeline(&descriptor);

        // Depth-only, alpha-tested. Group 0 is a cascade matrix.
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-shadow-layout"),
            bind_group_layouts: &[Some(&lighting.cascade_layout), Some(&material_bgl)],
            immediate_size: 0,
        });
        let mut sd = descriptor.clone();
        sd.label = Some("mesh-shadow-pipeline");
        sd.layout = Some(&shadow_layout);
        sd.vertex.entry_point = Some("vs_shadow");
        sd.fragment = Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_shadow"),
            targets: &[],
            compilation_options: Default::default(),
        });
        // Cut-out foliage casts from both faces, and the terrain's front-face
        // trick does not apply to a leaf card with no inside.
        sd.primitive.cull_mode = None;
        let shadow_pipeline = device.create_render_pipeline(&sd);

        // Placeholder dimensions, used only when there is no vehicle mesh to load.
        const FALLBACK_HALF: [f32; 3] = [1.3, 0.8, 2.6];
        const FALLBACK_WHEEL: f32 = 0.57;

        let (chassis, wheels): (Vec<Mesh>, [Mesh; 4]) = match rig {
            Some(r) => {
                let body = r
                    .body
                    .iter()
                    .map(|p| upload(device, queue, &material_bgl, &sampler, "vehicle-body", p))
                    .collect();
                let wheels = std::array::from_fn(|i| {
                    upload(device, queue, &material_bgl, &sampler, "vehicle-wheel", &r.wheels[i])
                });
                (body, wheels)
            }
            None => {
                log::warn!("no vehicle mesh: drawing a placeholder box on four cylinders");
                let (cv, ci) = box_mesh(FALLBACK_HALF);
                let (wv, wi) = cylinder_mesh(FALLBACK_WHEEL, 0.22, 20);
                let chassis = vec![build_mesh(
                    device,
                    queue,
                    &material_bgl,
                    &sampler,
                    "chassis",
                    &cv,
                    &ci,
                    None,
                    None,
                    false,
                )];
                let wheels = std::array::from_fn(|_| {
                    build_mesh(
                        device,
                        queue,
                        &material_bgl,
                        &sampler,
                        "wheel",
                        &wv,
                        &wi,
                        None,
                        None,
                        false,
                    )
                });
                (chassis, wheels)
            }
        };

        Self {
            pipeline,
            pipeline_double,
            shadow_pipeline,
            material_bgl,
            sampler,
            camera_ub,
            camera_bg,
            instances,
            capacity: MAX_INSTANCES,
            chassis,
            wheels,
        }
    }

    /// Upload an imported or generated mesh, with its material.
    pub fn upload_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &terra_assets::MeshData,
    ) -> Mesh {
        let verts: Vec<Vertex> = (0..data.positions.len())
            .map(|i| Vertex {
                position: data.positions[i],
                normal: *data.normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]),
                uv: *data.uvs.get(i).unwrap_or(&[0.0, 0.0]),
            })
            .collect();
        build_mesh(
            device,
            queue,
            &self.material_bgl,
            &self.sampler,
            "imported",
            &verts,
            &data.indices,
            data.albedo.as_ref(),
            data.alpha_cutoff,
            data.double_sided,
        )
    }

    pub fn upload_camera(&self, queue: &wgpu::Queue, cam: &Camera, aspect: f32) {
        queue.write_buffer(&self.camera_ub, 0, bytemuck::bytes_of(&cam.uniform(aspect)));
    }

    /// Upload instances. Returns how many were accepted.
    pub fn upload_instances(&self, queue: &wgpu::Queue, instances: &[Instance]) -> u32 {
        let n = instances.len().min(self.capacity);
        if n > 0 {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances[..n]));
        }
        n as u32
    }

    /// Draw with a GPU-written instance count.
    ///
    /// The count lives in `args` and is never read back, which is the whole
    /// point of culling on the GPU -- asking how many survived would cost a
    /// sync every frame.
    /// Depth-only instanced draw into a shadow cascade.
    // Eight, because a shadow draw needs a cascade as well as the LOD's own args
    // offset. Grouping them into a struct would be a struct built at every call
    // site to carry two integers.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_shadow_indirect(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        cascade: usize,
        mesh: &Mesh,
        instances: &wgpu::Buffer,
        args: &wgpu::Buffer,
        args_offset: u64,
    ) {
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &lighting.cascade_bind_group, &[Lighting::cascade_offset(cascade)]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(args, args_offset);
    }

    /// Depth-only instanced draw from a caller-owned buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_shadow_instanced(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        cascade: usize,
        mesh: &Mesh,
        instances: &wgpu::Buffer,
        offset: u32,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let stride = std::mem::size_of::<Instance>() as u64;
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &lighting.cascade_bind_group, &[Lighting::cascade_offset(cascade)]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(offset as u64 * stride..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..count);
    }

    pub fn draw_indirect(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        mesh: &Mesh,
        instances: &wgpu::Buffer,
        args: &wgpu::Buffer,
        args_offset: u64,
    ) {
        pass.set_pipeline(if mesh.double_sided { &self.pipeline_double } else { &self.pipeline });
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_bind_group(2, &lighting.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(args, args_offset);
    }

    /// Draw from a caller-owned instance buffer.
    ///
    /// Scatter keeps one static buffer per species -- the transforms do not
    /// change between frames -- so it has nothing to gain from the shared
    /// per-frame buffer the vehicle uses.
    pub fn draw_instanced(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        mesh: &Mesh,
        instances: &wgpu::Buffer,
        offset: u32,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let stride = std::mem::size_of::<Instance>() as u64;
        pass.set_pipeline(if mesh.double_sided { &self.pipeline_double } else { &self.pipeline });
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_bind_group(2, &lighting.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(offset as u64 * stride..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..count);
    }

    /// Draw `count` instances of `mesh`, reading from `offset` in the instance
    /// buffer.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        mesh: &Mesh,
        offset: u32,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let stride = std::mem::size_of::<Instance>() as u64;
        pass.set_pipeline(if mesh.double_sided { &self.pipeline_double } else { &self.pipeline });
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_bind_group(2, &lighting.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, self.instances.slice(offset as u64 * stride..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..count);
    }
}

/// Upload a `MeshData`, for callers that do not yet have a `MeshRenderer`.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    label: &str,
    data: &terra_assets::MeshData,
) -> Mesh {
    let verts: Vec<Vertex> = (0..data.positions.len())
        .map(|i| Vertex {
            position: data.positions[i],
            normal: *data.normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]),
            uv: *data.uvs.get(i).unwrap_or(&[0.0, 0.0]),
        })
        .collect();
    build_mesh(
        device,
        queue,
        material_bgl,
        sampler,
        label,
        &verts,
        &data.indices,
        data.albedo.as_ref(),
        data.alpha_cutoff,
        data.double_sided,
    )
}

/// Upload geometry plus its material bindings.
///
/// Meshes with no map get a 1x1 white texture rather than a second pipeline: the shader
/// multiplies by it, so untextured geometry is unaffected and there is one code path.
#[allow(clippy::too_many_arguments)]
fn build_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    label: &str,
    verts: &[Vertex],
    idx: &[u32],
    albedo: Option<&terra_assets::mesh::Texture>,
    alpha_cutoff: Option<f32>,
    double_sided: bool,
) -> Mesh {
    let white = terra_assets::mesh::Texture { width: 1, height: 1, rgba: vec![255; 4] };
    let tex = albedo.unwrap_or(&white);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: tex.width.max(1),
            height: tex.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &tex.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex.width.max(1) * 4),
            rows_per_image: Some(tex.height.max(1)),
        },
        wgpu::Extent3d {
            width: tex.width.max(1),
            height: tex.height.max(1),
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());

    // Negative disables the test, so opaque geometry never branches on it.
    let cutoff = [alpha_cutoff.unwrap_or(-1.0), 0.0, 0.0, 0.0];
    let cutoff_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-alpha-cutoff"),
        contents: bytemuck::cast_slice(&cutoff),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let material = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: cutoff_buf.as_entire_binding() },
        ],
    });

    Mesh {
        vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(idx),
            usage: wgpu::BufferUsages::INDEX,
        }),
        index_count: idx.len() as u32,
        material,
        double_sided,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_has_six_flat_faces() {
        let (v, i) = box_mesh([1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 24, "each face needs its own vertices for flat normals");
        assert_eq!(i.len(), 36);
        // Every index must address a real vertex.
        assert!(i.iter().all(|k| (*k as usize) < v.len()));
    }

    #[test]
    fn box_corners_span_the_requested_half_extents() {
        let (v, _) = box_mesh([1.0, 2.0, 3.0]);
        let mx = v.iter().map(|a| a.position[0].abs()).fold(0.0f32, f32::max);
        let my = v.iter().map(|a| a.position[1].abs()).fold(0.0f32, f32::max);
        let mz = v.iter().map(|a| a.position[2].abs()).fold(0.0f32, f32::max);
        assert_eq!((mx, my, mz), (1.0, 2.0, 3.0));
    }

    #[test]
    fn cylinder_is_closed_and_axis_aligned_to_x() {
        let (v, i) = cylinder_mesh(0.5, 0.2, 12);
        assert!(i.iter().all(|k| (*k as usize) < v.len()));
        // Every vertex sits within the requested radius on the YZ plane.
        for a in &v {
            let r = (a.position[1].powi(2) + a.position[2].powi(2)).sqrt();
            assert!(r <= 0.5 + 1e-4, "vertex outside the wheel radius: {r}");
            assert!(a.position[0].abs() <= 0.2 + 1e-4, "wheel wider than requested");
        }
        // Tread plus two caps.
        assert_eq!(i.len(), (12 * 6) + (12 * 3 * 2));
    }

    #[test]
    fn instance_matches_the_shader_layout() {
        // 32 bytes, and every vertex attribute has to land inside it at the
        // offset it declares. This was 80 -- a full 4x4 matrix plus an rgba float
        // colour -- and the size is the whole point of the record: three per-LOD
        // instance buffers at 32 bytes cost less than one source plus one output
        // buffer did at 80.
        assert_eq!(std::mem::size_of::<Instance>(), 32);
        for a in INSTANCE_ATTRS {
            let end = a.offset + a.format.size();
            assert!(
                end <= 32,
                "attribute at location {} runs to {end}, past the 32-byte record",
                a.shader_location
            );
        }
        // The locations the shader declares, in order, with no gaps or repeats.
        let locs: Vec<u32> = INSTANCE_ATTRS.iter().map(|a| a.shader_location).collect();
        assert_eq!(locs, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn a_rigid_transform_survives_the_round_trip() {
        // The record stores a quaternion and a scalar instead of a matrix, so the
        // question is whether i16 snorm and f16 give back what went in. Anything
        // worse than a millimetre at these sizes would show as instances jittering
        // against the ground they were placed on.
        let rot = glam::Quat::from_euler(glam::EulerRot::YXZ, 0.9, 0.3, -0.4);
        let pos = Vec3::new(1234.5, 67.25, -890.75);
        let inst = Instance::from_parts(pos, rot, 3.75, Vec3::new(0.5, 0.25, 1.0), 7);

        assert_eq!(Vec3::from(inst.pos), pos, "position is stored as full f32");
        assert!((inst.scale_f32() - 3.75).abs() < 1e-3, "scale came back {}", inst.scale_f32());
        // Quaternions q and -q are the same rotation, so compare by what they do.
        let v = Vec3::new(0.3, 0.9, -0.2);
        let err = (rot * v - inst.rot_quat() * v).length();
        assert!(err < 1e-3, "rotation error {err} is visible at instance scale");
        assert_eq!(inst.seed, 7);
    }

    #[test]
    fn colour_survives_at_eight_bits() {
        // It multiplies an albedo map and carries the Select tool's highlight, so
        // it only has to be visually right, not exact.
        let inst = Instance::from_parts(Vec3::ZERO, glam::Quat::IDENTITY, 1.0, Vec3::ONE, 0);
        assert_eq!(inst.color, [255, 255, 255, 255], "white must stay white");
        let half = Instance::from_parts(Vec3::ZERO, glam::Quat::IDENTITY, 1.0, Vec3::splat(0.5), 0);
        assert!((half.color[0] as f32 / 255.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn new_decomposes_a_matrix_the_way_the_shader_recomposes_it() {
        // `Instance::new` is what props and the vehicle still call. Both build a
        // rigid, uniformly-scaled matrix, so decomposing and rebuilding has to be
        // the identity to within the encoding.
        let rot = glam::Quat::from_rotation_y(1.1);
        let m =
            Mat4::from_scale_rotation_translation(Vec3::splat(2.5), rot, Vec3::new(5.0, 6.0, 7.0));
        let inst = Instance::new(m, Vec3::ONE);

        let v = Vec3::new(1.0, 2.0, 3.0);
        let want = m.transform_point3(v);
        let got = Vec3::from(inst.pos) + inst.rot_quat() * (v * inst.scale_f32());
        assert!((want - got).length() < 1e-2, "want {want}, got {got}");
    }

    #[test]
    fn three_lod_buffers_cost_less_than_the_old_pair() {
        // The acceptance test for the shrink. The old layout needed a source and
        // an output buffer at 80 bytes each; the new one needs a source and three
        // per-LOD outputs at 32.
        const INSTANCES: usize = 35_000;
        let before = INSTANCES * 80 * 2;
        let after = INSTANCES * std::mem::size_of::<Instance>() * 4;
        assert!(after < before, "per-species instance memory went up: {before} -> {after} bytes");
    }
}

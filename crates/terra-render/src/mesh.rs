//! Instanced solid meshes: the car chassis and its wheels.
//!
//! Geometry is generated in code -- a box and a cylinder -- rather than loaded.
//! A driveable car needs correct physics, not a nice model, and building the
//! asset pipeline first would delay the part that actually has to be right.
//! glTF import replaces the generators later without touching this pipeline.

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl Instance {
    pub fn new(model: Mat4, color: Vec3) -> Self {
        Self { model: model.to_cols_array_2d(), color: color.extend(1.0).into() }
    }
}

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
    pub chassis: Mesh,
    pub wheel: Mesh,
}

/// Instances uploaded per frame. One chassis plus four wheels, with room spare.
const MAX_INSTANCES: usize = 64;

impl MeshRenderer {
    pub fn new(
        ctx: &RenderContext,
        chassis_half: [f32; 3],
        wheel_radius: f32,
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
                        attributes: &wgpu::vertex_attr_array![
                            3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4,
                            7 => Float32x4
                        ],
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

        let (cv, ci) = box_mesh(chassis_half);
        let (wv, wi) = cylinder_mesh(wheel_radius, 0.16, 20);
        let chassis = build_mesh(
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
        );
        let wheel = build_mesh(
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
        );

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
            wheel,
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
    pub fn draw_shadow_indirect(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        lighting: &Lighting,
        cascade: usize,
        mesh: &Mesh,
        instances: &wgpu::Buffer,
        args: &wgpu::Buffer,
    ) {
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &lighting.cascade_bind_group, &[Lighting::cascade_offset(cascade)]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(args, 0);
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
    ) {
        pass.set_pipeline(if mesh.double_sided { &self.pipeline_double } else { &self.pipeline });
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &mesh.material, &[]);
        pass.set_bind_group(2, &lighting.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(args, 0);
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

/// Upload geometry plus its material bindings.
///
/// Meshes with no map get a 1x1 white texture rather than a second pipeline:
/// the shader multiplies by it, so untextured geometry is unaffected and there
/// is one code path.
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
        // 4x4 matrix plus an rgba colour, as the vertex attributes declare.
        assert_eq!(std::mem::size_of::<Instance>(), 16 * 4 + 4 * 4);
    }
}

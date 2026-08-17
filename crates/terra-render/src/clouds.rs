//! The half-resolution, temporally-accumulated cloud pass.
//!
//! The march lives in `atmosphere.wgsl`; this owns the targets, the history
//! ping-pong and the reprojection matrix that make it affordable. See the header
//! of `clouds.wgsl` for why both halves are needed.
//!
//! Two targets rather than one, alternating: a pass cannot sample the texture it
//! is writing, so this frame reads last frame's and writes the other. The sky
//! pass then samples whichever was just written.

use crate::camera::Camera;
use crate::context::RenderContext;
use crate::environment::EnvironmentGpu;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// Fraction of the surface the clouds are marched at, per axis.
///
/// Half, so a quarter of the pixels. Quarter-res was tried and the layer's edges
/// break up into visible blocks against the sky gradient -- the upsample cannot
/// invent the silhouette, and the silhouette is what reads as cloud.
pub const SCALE: u32 = 2;

/// Side length of the cloud shadow map, in texels.
///
/// 512 over an 8 km region is about 16 m a texel. Cloud shadow edges are soft by
/// the time they reach the ground, so this is already finer than the signal.
pub const SHADOW_RES: u32 = 512;

/// Ground area the shadow map covers, in metres.
///
/// Centred on the camera and snapped to texel multiples. A region large enough
/// for the whole world would be 16 km at 31 m a texel, which is coarser than a
/// cloud edge; following the camera keeps the resolution where it is looked at.
pub const SHADOW_EXTENT_M: f32 = 8000.0;

/// `Rgba16Float`: rgb is scattered radiance in linear HDR, well outside 0..1 for
/// a sunlit edge, and a is transmittance. An 8-bit target clips the lining and
/// bands the transmittance ramp, which shows as a hard edge on every cloud.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowRegion {
    /// xy centre in world XZ, z side length in metres, w texels per side.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Reproject {
    prev_view_proj: [[f32; 4]; 4],
    /// x frame index, y history valid, zw target size.
    params: [f32; 4],
}

pub struct Clouds {
    pipeline: wgpu::RenderPipeline,
    camera_ub: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    reproject_ub: wgpu::Buffer,
    /// Layout for the pass's own history binding.
    history_layout: wgpu::BindGroupLayout,
    /// Layout the sky uses to read the result.
    pub sample_layout: wgpu::BindGroupLayout,
    views: [wgpu::TextureView; 2],
    /// Reads target `1 - i` while writing `i`.
    history_bgs: [wgpu::BindGroup; 2],
    /// Lets the sky sample target `i`.
    sample_bgs: [wgpu::BindGroup; 2],
    sampler: wgpu::Sampler,
    /// Which target was written most recently.
    current: usize,
    frame: u32,
    prev_view_proj: Mat4,
    /// False until one frame has been written, so the first frame does not blend
    /// against an uninitialized texture.
    has_history: bool,
    size: (u32, u32),

    // --- ground shadow ---
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_region_ub: wgpu::Buffer,
    shadow_region_bg: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    /// Layout consumers bind to read the shadow map. The terrain uses it.
    pub shadow_layout: wgpu::BindGroupLayout,
    pub shadow_bind_group: wgpu::BindGroup,
    shadow_center: glam::Vec2,
}

fn target_size(ctx: &RenderContext) -> (u32, u32) {
    ((ctx.config.width / SCALE).max(1), (ctx.config.height / SCALE).max(1))
}

fn build(
    device: &wgpu::Device,
    history_layout: &wgpu::BindGroupLayout,
    sample_layout: &wgpu::BindGroupLayout,
    reproject_ub: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    size: (u32, u32),
) -> ([wgpu::TextureView; 2], [wgpu::BindGroup; 2], [wgpu::BindGroup; 2]) {
    let make = |label: &str| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let views = [make("clouds-a"), make("clouds-b")];

    let history = |read: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-history"),
            layout: history_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: reproject_ub.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[read]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    };
    // Index by the target being *written*, so entry i reads the other one.
    let history_bgs = [history(1), history(0)];

    let sample = |i: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-sample"),
            layout: sample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[i]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    };
    let sample_bgs = [sample(0), sample(1)];

    (views, history_bgs, sample_bgs)
}

impl Clouds {
    pub fn new(ctx: &RenderContext, env: &EnvironmentGpu) -> Self {
        let device = &ctx.device;

        let camera_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-camera"),
            size: std::mem::size_of::<crate::camera::CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clouds-camera-bgl"),
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
            label: Some("clouds-camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_ub.as_entire_binding(),
            }],
        });

        let reproject_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-reproject"),
            size: std::mem::size_of::<Reproject>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let history_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clouds-history-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            ],
        });

        let sample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clouds-sample-bgl"),
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
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clouds-sampler"),
            // Clamp: a reprojected lookup that lands outside is rejected in the
            // shader, and wrapping would fetch the opposite edge of the sky.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
                    include_str!("../../../assets/shaders/render/clouds.wgsl"),
                )
                .into(),
            ),
        });

        // `atmosphere.wgsl` pins the environment to group 2, so the layout has to
        // leave that slot for it; group 3 is unused here and gets no entry.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-layout"),
            bind_group_layouts: &[Some(&camera_bgl), Some(&history_layout), Some(&env.layout)],
            ..Default::default()
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- ground shadow ---
        let shadow_region_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud-shadow-region"),
            size: std::mem::size_of::<ShadowRegion>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let region_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud-shadow-region-bgl"),
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
        let shadow_region_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud-shadow-region-bg"),
            layout: &region_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_region_ub.as_entire_binding(),
            }],
        });

        let shadow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud-shadow"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../../../assets/shaders/common/atmosphere.wgsl"),
                    include_str!("../../../assets/shaders/render/cloud_shadow.wgsl"),
                )
                .into(),
            ),
        });
        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud-shadow-layout"),
            // Group 1 unused; group 2 is where `atmosphere.wgsl` pins the
            // environment.
            bind_group_layouts: &[Some(&region_bgl), None, Some(&env.layout)],
            ..Default::default()
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud-shadow"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &shadow_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // One channel: transmittance. R16 rather than R8 because the
                    // ramp between lit and shadowed is wide and shallow, and 8
                    // bits bands it into visible steps.
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud-shadow"),
            size: wgpu::Extent3d {
                width: SHADOW_RES,
                height: SHADOW_RES,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());

        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud-shadow-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            ],
        });
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud-shadow-bg"),
            layout: &shadow_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shadow_region_ub.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let size = target_size(ctx);
        let (views, history_bgs, sample_bgs) =
            build(device, &history_layout, &sample_layout, &reproject_ub, &sampler, size);

        Self {
            pipeline,
            camera_ub,
            camera_bg,
            reproject_ub,
            history_layout,
            sample_layout,
            views,
            history_bgs,
            sample_bgs,
            sampler,
            current: 0,
            frame: 0,
            prev_view_proj: Mat4::IDENTITY,
            has_history: false,
            size,
            shadow_pipeline,
            shadow_region_ub,
            shadow_region_bg,
            shadow_view,
            shadow_layout,
            shadow_bind_group,
            shadow_center: glam::Vec2::ZERO,
        }
    }

    pub fn resize(&mut self, ctx: &RenderContext) {
        let size = target_size(ctx);
        if size == self.size {
            return;
        }
        let (views, history_bgs, sample_bgs) = build(
            &ctx.device,
            &self.history_layout,
            &self.sample_layout,
            &self.reproject_ub,
            &self.sampler,
            size,
        );
        self.views = views;
        self.history_bgs = history_bgs;
        self.sample_bgs = sample_bgs;
        self.size = size;
        // The old history is a different shape; blending against it would smear.
        self.has_history = false;
    }

    /// Drop the accumulated history.
    ///
    /// Needed wherever the picture changes discontinuously -- opening a world,
    /// or jumping the time of day -- because four frames of blending toward the
    /// new state reads as a slow wipe.
    pub fn invalidate(&mut self) {
        self.has_history = false;
    }

    /// Bind group the sky uses to sample the most recent result.
    pub fn sample_bind_group(&self) -> &wgpu::BindGroup {
        &self.sample_bgs[self.current]
    }

    /// Render one frame of clouds. Call before the scene pass.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        env: &EnvironmentGpu,
        cam: &Camera,
        aspect: f32,
    ) {
        // Alternate targets, so this frame never samples what it is writing.
        let write = 1 - self.current;

        // Unjittered, deliberately.
        //
        // The camera handed in carries the TAA sub-pixel offset, which changes
        // every frame. Rendering the cloud buffer with it and then sampling that
        // buffer from the *also* jittered sky pass shifted the whole layer by a
        // fraction of a pixel each frame -- which is what the shaking was. The
        // sky's own jitter plus the TAA resolve already antialias the result.
        let mut steady = cam.clone();
        steady.jitter = glam::Vec2::ZERO;
        queue.write_buffer(&self.camera_ub, 0, bytemuck::bytes_of(&steady.uniform(aspect)));
        queue.write_buffer(
            &self.reproject_ub,
            0,
            bytemuck::bytes_of(&Reproject {
                prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
                params: [
                    self.frame as f32,
                    if self.has_history { 1.0 } else { 0.0 },
                    self.size.0 as f32,
                    self.size.1 as f32,
                ],
            }),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clouds"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[write],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Every pixel is written, so the clear is only for the
                        // first frame after a resize.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_bind_group(1, &self.history_bgs[write], &[]);
            pass.set_bind_group(2, &env.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.current = write;
        self.frame = self.frame.wrapping_add(1);
        // Must match the matrix the pass was rendered with, or every reprojected
        // lookup is off by the jitter and the history never lines up.
        self.prev_view_proj = steady.projection(aspect) * steady.look_at();
        self.has_history = true;
    }

    /// Rebuild the ground shadow map around `camera_xz`.
    ///
    /// Must run before anything that samples it -- the terrain pass and the
    /// shadow cascades both do.
    pub fn render_shadow(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        env: &EnvironmentGpu,
        camera_xz: glam::Vec2,
    ) {
        // Snap the centre to whole texels. Without this the map slides
        // continuously under the terrain and every cloud shadow edge crawls as
        // the camera moves, which is far more visible than the shadow being a few
        // metres out of place.
        let texel = SHADOW_EXTENT_M / SHADOW_RES as f32;
        self.shadow_center = (camera_xz / texel).floor() * texel;

        queue.write_buffer(
            &self.shadow_region_ub,
            0,
            bytemuck::bytes_of(&ShadowRegion {
                params: [
                    self.shadow_center.x,
                    self.shadow_center.y,
                    SHADOW_EXTENT_M,
                    SHADOW_RES as f32,
                ],
            }),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud-shadow"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.shadow_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &self.shadow_region_bg, &[]);
        pass.set_bind_group(2, &env.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Pixels marched per frame, for the performance overlay.
    pub fn marched_pixels(&self) -> u32 {
        self.size.0 * self.size.1
    }
}

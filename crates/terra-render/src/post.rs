//! Post-processing: god rays, exposure and the final encode.
//!
//! The scene renders into a linear HDR target; this resolves it to the
//! swapchain. That indirection is what makes the rays possible at all -- they
//! need the sun's real brightness, and an 8-bit buffer has already clipped it
//! to white before the effect can read it.

use crate::camera::Camera;
use crate::context::{RenderContext, SCENE_FORMAT};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// The march runs at this fraction of the screen. Light shafts carry no
/// high-frequency detail, so full resolution buys nothing an upsample loses.
const RAY_SCALE: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    sun_uv: [f32; 2],
    strength: f32,
    /// Named `enabled` rather than `active`: the latter is a reserved WGSL
    /// keyword and the shader will not parse with it.
    enabled: f32,
    exposure: f32,
    density: f32,
    decay: f32,
    tone_mapper: u32,
    contrast: f32,
    saturation: f32,
    white_balance_k: f32,
    _pad: f32,
}

const _: () = assert!(std::mem::size_of::<PostUniform>() == 48);

pub struct Post {
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    scene_bgl: wgpu::BindGroupLayout,
    rays_bgl: wgpu::BindGroupLayout,
    source_bgl: wgpu::BindGroupLayout,
    downsample_pipeline: wgpu::RenderPipeline,
    rays_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,

    /// Half-resolution scene, the march's actual source.
    source_view: wgpu::TextureView,
    source_bgs: Vec<wgpu::BindGroup>,
    raw_bg: wgpu::BindGroup,
    ray_view: wgpu::TextureView,
    /// One per temporal-resolve parity: the source alternates every frame.
    scene_bgs: Vec<wgpu::BindGroup>,
    ray_bg: wgpu::BindGroup,
    size: (u32, u32),
}

impl Post {
    pub fn new(ctx: &RenderContext) -> Self {
        let device = &ctx.device;

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post-uniform"),
            size: std::mem::size_of::<PostUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post-sampler"),
            // Clamped: the march walks off the edge of the screen toward a sun
            // that may be outside it, and wrapping would fold the far side of
            // the image into the shaft.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-scene-bgl"),
            entries: &[
                texture_entry(0),
                sampler_entry(1),
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
        let rays_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-rays-bgl"),
            entries: &[texture_entry(0), sampler_entry(1)],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/render/post.wgsl").into(),
            ),
        });

        let source_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-source-bgl"),
            entries: &[texture_entry(0)],
        });
        let downsample_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-downsample-layout"),
            bind_group_layouts: &[Some(&scene_bgl)],
            immediate_size: 0,
        });
        let downsample_pipeline =
            fullscreen(device, &shader, &downsample_layout, "fs_downsample", SCENE_FORMAT);
        let rays_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-rays-layout"),
            bind_group_layouts: &[Some(&scene_bgl), None, Some(&source_bgl)],
            immediate_size: 0,
        });
        let rays_pipeline = fullscreen(device, &shader, &rays_layout, "fs_rays", SCENE_FORMAT);

        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-resolve-layout"),
            bind_group_layouts: &[Some(&scene_bgl), Some(&rays_bgl)],
            immediate_size: 0,
        });
        let resolve_pipeline =
            fullscreen(device, &shader, &resolve_layout, "fs_resolve", ctx.config.format);

        let size = (ctx.config.width, ctx.config.height);
        let t = build_targets(
            device,
            &scene_bgl,
            &rays_bgl,
            &sampler,
            &uniform,
            &source_bgl,
            [&ctx.scene_view, &ctx.scene_view],
            &ctx.scene_view,
            size,
        );

        Self {
            uniform,
            sampler,
            scene_bgl,
            rays_bgl,
            downsample_pipeline,
            rays_pipeline,
            resolve_pipeline,
            ray_view: t.ray_view,
            scene_bgs: t.scene_bgs,
            ray_bg: t.ray_bg,
            source_view: t.source_view,
            raw_bg: t.raw_bg,
            source_bgs: t.source_bgs,
            source_bgl,
            size,
        }
    }

    /// Rebuild the bind groups when the surface, and therefore the scene view,
    /// has been recreated.
    /// Point the post chain at `source`, the texture holding the resolved
    /// scene. Called on resize and whenever the TAA buffers are rebuilt.
    pub fn rebind(&mut self, ctx: &RenderContext, sources: [&wgpu::TextureView; 2]) {
        let size = (ctx.config.width, ctx.config.height);
        let t = build_targets(
            &ctx.device,
            &self.scene_bgl,
            &self.rays_bgl,
            &self.sampler,
            &self.uniform,
            &self.source_bgl,
            sources,
            &ctx.scene_view,
            size,
        );
        self.ray_view = t.ray_view;
        self.scene_bgs = t.scene_bgs;
        self.ray_bg = t.ray_bg;
        self.source_view = t.source_view;
        self.raw_bg = t.raw_bg;
        self.source_bgs = t.source_bgs;
        self.size = size;
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Project the sun and upload the effect's parameters.
    ///
    /// Returns whether the rays will draw anything, so the caller can skip the
    /// march entirely rather than dispatching one that accumulates zero.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        queue: &wgpu::Queue,
        cam: &Camera,
        aspect: f32,
        sun_direction: Vec3,
        daylight: f32,
        strength: f32,
        tone: &crate::environment::ToneMapping,
        fog_active: bool,
    ) -> bool {
        // The sun is directional, so project a point far along it rather than
        // a position.
        let world = cam.pos + sun_direction * 10_000.0;
        let clip = (cam.projection(aspect) * cam.look_at()) * world.extend(1.0);

        // Behind the camera the projection wraps and the shaft would sweep the
        // wrong way across the screen.
        let in_front = clip.w > 0.0;
        let ndc = if in_front { clip.truncate() / clip.w } else { glam::Vec3::ZERO };
        let sun_uv = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];

        // Fade out as the sun leaves the frame, or shafts pop on at the edge.
        let margin = (sun_uv[0].clamp(0.0, 1.0) - sun_uv[0]).abs()
            + (sun_uv[1].clamp(0.0, 1.0) - sun_uv[1]).abs();
        let on_screen = (1.0 - margin * 2.5).clamp(0.0, 1.0);
        let active = in_front && on_screen > 0.0 && daylight > 0.01 && strength > 0.0;

        // With volumetrics running, the shafts are already there and they are
        // physically derived. Adding a screen-space approximation of the same
        // scattering on top counts it twice, and the presets made that worse at
        // higher quality -- more fog *and* more fake shafts. What the froxel
        // grid genuinely cannot provide is the sun disc itself, which sits past
        // its far plane, so this reduces to a tight glare around it.
        let (spread, gain) = if fog_active { (0.35, 0.30) } else { (0.85, 1.0) };

        let u = PostUniform {
            sun_uv,
            strength: strength * on_screen * daylight * gain,
            enabled: if active { 1.0 } else { 0.0 },
            exposure: 2f32.powf(tone.exposure_ev),
            density: spread,
            decay: if fog_active { 0.90 } else { 0.965 },
            tone_mapper: tone.mapper.index(),
            contrast: tone.contrast,
            saturation: tone.saturation,
            white_balance_k: tone.white_balance_k,
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        active
    }

    /// March the shafts into the half-resolution target.
    pub fn render_rays(&self, encoder: &mut wgpu::CommandEncoder, source: usize) {
        // Downsample the raw scene first, so the march reads a filtered source.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("god-rays-downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.source_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &self.raw_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        let _ = source;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("god-rays"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ray_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.rays_pipeline);
        pass.set_bind_group(0, &self.raw_bg, &[]);
        pass.set_bind_group(2, &self.source_bgs[0], &[]);
        pass.draw(0..3, 0..1);
    }

    /// Composite, expose and encode onto the swapchain.
    pub fn resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        source: usize,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post-resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Every pixel is written, so there is nothing to preserve.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &self.scene_bgs[source & 1], &[]);
        pass.set_bind_group(1, &self.ray_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_targets(
    device: &wgpu::Device,
    scene_bgl: &wgpu::BindGroupLayout,
    rays_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    source_bgl: &wgpu::BindGroupLayout,
    sources: [&wgpu::TextureView; 2],
    raw_scene: &wgpu::TextureView,
    size: (u32, u32),
) -> Targets {
    let ray_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("god-rays"),
            size: wgpu::Extent3d {
                width: (size.0 / RAY_SCALE).max(1),
                height: (size.1 / RAY_SCALE).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    let scene_bgs = sources
        .iter()
        .map(|source| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post-scene-bg"),
                layout: scene_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
                ],
            })
        })
        .collect();
    let ray_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post-rays-bg"),
        layout: rays_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ray_view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });

    // The march reads the *raw* scene, before the temporal resolve. Its
    // occlusion mask is the alpha channel, and temporal blending turns that
    // into a fractional value at every moving silhouette -- shafts would leak
    // through geometry that should block them for several frames.
    let source_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("post-source"),
            size: wgpu::Extent3d {
                width: (size.0 / RAY_SCALE).max(1),
                height: (size.1 / RAY_SCALE).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    let raw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post-raw-bg"),
        layout: scene_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(raw_scene),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
        ],
    });
    let source_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post-source-bg"),
        layout: source_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&source_view),
        }],
    });

    Targets { ray_view, scene_bgs, ray_bg, source_view, raw_bg, source_bgs: vec![source_bg] }
}

/// Everything `build_targets` produces, so the signature stays readable.
struct Targets {
    ray_view: wgpu::TextureView,
    scene_bgs: Vec<wgpu::BindGroup>,
    ray_bg: wgpu::BindGroup,
    source_view: wgpu::TextureView,
    raw_bg: wgpu::BindGroup,
    source_bgs: Vec<wgpu::BindGroup>,
}

fn fullscreen(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

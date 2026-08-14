//! Temporal anti-aliasing.
//!
//! Two history buffers, ping-ponged: each frame reads the one written last and
//! writes the other. The output is what the post pass consumes, so god rays and
//! the tonemap see the resolved image rather than the raw one.
//!
//! The jitter sequence is Halton(2,3). It is low-discrepancy -- successive
//! offsets spread out rather than clustering -- so eight frames cover a pixel
//! evenly, where a random sequence would leave gaps and double up.

use crate::context::{RenderContext, SCENE_FORMAT};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};

/// Frames before the jitter pattern repeats. Longer converges finer but takes
/// longer to settle after a cut, and anything past about eight is invisible.
const JITTER_PERIOD: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaUniform {
    prev_view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

pub struct Taa {
    pub enabled: bool,
    /// How much of the accumulated history to keep. Higher is smoother and
    /// ghosts more.
    pub history_weight: f32,

    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,

    /// Ping-ponged: `views[write]` is this frame's output.
    views: [wgpu::TextureView; 2],
    /// One bind group per parity, each reading the *other* buffer as history.
    bind_groups: Vec<wgpu::BindGroup>,
    write: usize,

    prev_view_proj: Mat4,
    frame: u32,
    size: (u32, u32),
    /// True until a full frame has been accumulated, so the first frame after
    /// a resize or a world change does not blend against uninitialised memory.
    reset: bool,
}

impl Taa {
    pub fn new(ctx: &RenderContext) -> Self {
        let device = &ctx.device;

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa-uniform"),
            size: std::mem::size_of::<TaaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tex = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa-bgl"),
            entries: &[
                tex(0, wgpu::TextureSampleType::Float { filterable: true }),
                tex(1, wgpu::TextureSampleType::Float { filterable: true }),
                tex(2, wgpu::TextureSampleType::Depth),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/render/taa.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa-pipeline"),
            layout: Some(&pipeline_layout),
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
        });

        let size = (ctx.config.width, ctx.config.height);
        let (views, bind_groups) = build(device, &layout, &sampler, &uniform, ctx, size);

        Self {
            enabled: true,
            history_weight: 0.88,
            uniform,
            sampler,
            layout,
            pipeline,
            views,
            bind_groups,
            write: 0,
            prev_view_proj: Mat4::IDENTITY,
            frame: 0,
            size,
            reset: true,
        }
    }

    pub fn resize(&mut self, ctx: &RenderContext) {
        let size = (ctx.config.width, ctx.config.height);
        let (views, bind_groups) =
            build(&ctx.device, &self.layout, &self.sampler, &self.uniform, ctx, size);
        self.views = views;
        self.bind_groups = bind_groups;
        self.size = size;
        self.reset = true;
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Throw the accumulated history away. Anything that changes the whole
    /// image -- opening a world, teleporting the camera -- would otherwise blend
    /// the old scene into the new one for several frames.
    pub fn invalidate(&mut self) {
        self.reset = true;
    }

    /// This frame's sub-pixel offset, in NDC. Zero when disabled, or the image
    /// would simply wobble.
    pub fn jitter(&self, width: u32, height: u32) -> Vec2 {
        if !self.enabled {
            return Vec2::ZERO;
        }
        let i = self.frame % JITTER_PERIOD + 1;
        // Halton(2,3), centred on the pixel.
        let x = halton(i, 2) - 0.5;
        let y = halton(i, 3) - 0.5;
        // NDC spans 2 units across the viewport, hence the doubling.
        Vec2::new(x * 2.0 / width.max(1) as f32, y * 2.0 / height.max(1) as f32)
    }

    /// Both history buffers, so the post chain can bind one per parity.
    pub fn outputs(&self) -> [&wgpu::TextureView; 2] {
        [&self.views[0], &self.views[1]]
    }

    /// Which buffer this frame's resolve wrote.
    pub fn output_index(&self) -> usize {
        self.write
    }

    /// Resolve `ctx.scene_view` against the history.
    pub fn resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        jitter: Vec2,
    ) {
        self.write ^= 1;

        let u = TaaUniform {
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            params: [
                jitter.x,
                jitter.y,
                if self.reset { 0.0 } else { self.history_weight },
                if self.enabled { 1.0 } else { 0.0 },
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("taa"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.views[self.write],
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[self.write], &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        self.prev_view_proj = view_proj;
        self.frame = self.frame.wrapping_add(1);
        self.reset = false;
    }
}

/// Radical-inverse in `base`. Low-discrepancy: successive values spread out
/// rather than clustering, which is what makes eight frames cover a pixel
/// evenly.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

fn build(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    ctx: &RenderContext,
    size: (u32, u32),
) -> ([wgpu::TextureView; 2], Vec<wgpu::BindGroup>) {
    let views: [wgpu::TextureView; 2] = std::array::from_fn(|i| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("taa-history"),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCENE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some(if i == 0 { "taa-a" } else { "taa-b" }),
                ..Default::default()
            })
    });

    // Two bind groups: writing buffer 0 reads buffer 1 as history, and back.
    let bind_groups = (0..2)
        .map(|write| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa-bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&ctx.scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[write ^ 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&ctx.depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
                ],
            })
        })
        .collect();

    (views, bind_groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_covers_the_pixel_evenly() {
        // The property that matters is coverage, not spacing: over one period
        // every eighth of the pixel must receive a sample, or the jitter leaves
        // parts of it never sampled and those parts never resolve.
        for base in [2u32, 3] {
            let v: Vec<f32> = (1..=JITTER_PERIOD).map(|i| halton(i, base)).collect();
            assert!(v.iter().all(|x| (0.0..1.0).contains(x)), "base {base}: {v:?}");
            for bin in 0..JITTER_PERIOD {
                let lo = bin as f32 / JITTER_PERIOD as f32;
                let hi = (bin + 1) as f32 / JITTER_PERIOD as f32;
                assert!(
                    v.iter().any(|x| *x >= lo && *x < hi),
                    "base {base}: nothing in {lo}..{hi} of {v:?}"
                );
            }
        }
    }

    #[test]
    fn halton_base_three_differs_from_base_two() {
        // The two axes must not be correlated, or the jitter walks a diagonal
        // and never covers the pixel.
        for i in 1..=8 {
            assert!((halton(i, 2) - halton(i, 3)).abs() > 1e-3, "axis {i} matches");
        }
    }
}

//! Hierarchical depth pyramid, for occlusion culling.
//!
//! Phase B of `docs/culling.md` listed this alongside GPU instance culling and
//! it was the piece left undone. Grass is where it pays: a ridge hides an
//! enormous amount of it, and every hidden blade otherwise runs a vertex shader
//! and shades fragments that are immediately thrown away.
//!
//! Built from the previous frame's depth, which is what makes it free of any
//! read-back or stall. The cost is one frame of latency, handled by a margin in
//! the test rather than by trying to be exact.

use crate::context::RenderContext;

/// Single channel float: the pyramid stores depth, not colour.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

pub struct HiZ {
    /// One view per level, as render targets.
    levels: Vec<wgpu::TextureView>,
    /// Feeds level zero from the depth buffer.
    copy_bg: wgpu::BindGroup,
    /// Feeds each level from the one above it.
    reduce_bgs: Vec<wgpu::BindGroup>,
    /// The whole chain plus its sampler, as a group a culling pass can bind
    /// directly. Owned here so a window resize rebuilds it in one place rather
    /// than leaving a stale view in every pass that reads the pyramid.
    cull_bg: wgpu::BindGroup,
    cull_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    // The copy reads the depth buffer and the reduce reads the level above, and
    // they need separate layouts rather than one with both: a single layout
    // would have the copy pass binding level zero as a sampled texture in the
    // same pass that writes it as a colour target.
    copy_bgl: wgpu::BindGroupLayout,
    reduce_bgl: wgpu::BindGroupLayout,
    copy_pipeline: wgpu::RenderPipeline,
    reduce_pipeline: wgpu::RenderPipeline,
    size: (u32, u32),
}

impl HiZ {
    pub fn new(ctx: &RenderContext) -> Self {
        let device = &ctx.device;

        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz-copy-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz-reduce-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    // Unfilterable: the reduce reads exact texels, and a
                    // filtered average of depths means nothing.
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz-cull-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/render/hiz.wgsl").into(),
            ),
        });
        let make = |entry: &str, bgl: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
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
        };
        let copy_pipeline = make("fs_copy", &copy_bgl);
        let reduce_pipeline = make("fs_reduce", &reduce_bgl);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hiz-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let seed = placeholder(device);
        let mut me = Self {
            levels: Vec::new(),
            copy_bg: copy_group(device, &copy_bgl, &ctx.depth_view),
            reduce_bgs: Vec::new(),
            cull_bg: cull_group(device, &cull_bgl, &seed, &sampler),
            cull_bgl,
            sampler,
            copy_bgl,
            reduce_bgl,
            copy_pipeline,
            reduce_pipeline,
            size: (0, 0),
        };
        me.resize(ctx);
        me
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Layout a culling pass declares to read the pyramid.
    pub fn cull_layout(&self) -> &wgpu::BindGroupLayout {
        &self.cull_bgl
    }

    pub fn cull_bind_group(&self) -> &wgpu::BindGroup {
        &self.cull_bg
    }

    pub fn resize(&mut self, ctx: &RenderContext) {
        let device = &ctx.device;
        // Half resolution. An occlusion test does not need per-pixel depth, and
        // halving it takes three quarters off both the cost and the memory.
        let (w, h) = ((ctx.config.width / 2).max(1), (ctx.config.height / 2).max(1));
        let mips = (w.max(h) as f32).log2().floor() as u32 + 1;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.levels = (0..mips)
            .map(|m| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hiz-level"),
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        self.reduce_bgs = (1..mips as usize)
            .map(|m| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz-reduce-bg"),
                    layout: &self.reduce_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.levels[m - 1]),
                    }],
                })
            })
            .collect();
        self.copy_bg = copy_group(device, &self.copy_bgl, &ctx.depth_view);
        self.cull_bg = cull_group(device, &self.cull_bgl, &texture, &self.sampler);
        self.size = (w, h);
    }

    /// Rebuild the pyramid from the depth buffer as it currently stands.
    pub fn build(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = |view: &wgpu::TextureView,
                        pipeline: &wgpu::RenderPipeline,
                        bg: &wgpu::BindGroup,
                        label: &str| {
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Every texel is written, so there is nothing to load.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, bg, &[]);
            p.draw(0..3, 0..1);
        };

        pass(&self.levels[0], &self.copy_pipeline, &self.copy_bg, "hiz-copy");
        for (i, bg) in self.reduce_bgs.iter().enumerate() {
            pass(&self.levels[i + 1], &self.reduce_pipeline, bg, "hiz-reduce");
        }
    }
}

fn copy_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hiz-copy-bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(depth),
        }],
    })
}

fn cull_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let view = texture
        .create_view(&wgpu::TextureViewDescriptor { label: Some("hiz-all"), ..Default::default() });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hiz-cull-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

/// A one-texel stand-in, so the cull group is bindable before the first resize.
fn placeholder(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hiz-placeholder"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

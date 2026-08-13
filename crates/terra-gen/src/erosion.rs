//! GPU hydraulic erosion: host side of the six-pass pipe-model solver.
//!
//! This is the one stage that genuinely needs the GPU. A 1024^2 field run for
//! 2000 iterations is ~12 billion cell updates; on the CPU that is minutes,
//! on an M4 it is a couple of seconds.
//!
//! Shader: `assets/shaders/gen/erosion.wgsl`

use bytemuck::{Pod, Zeroable};
use terra_project::params::ErosionParams;

/// Iterations per submit. Long enough that queue overhead is irrelevant, short
/// enough to report progress and keep any single command buffer modest.
const CHUNK: u32 = 100;

const PASSES: [&str; 6] =
    ["rain", "flux_pass", "water_update", "erode_deposit", "advect", "evaporate"];

/// Two percentiles of `data`, estimated from a stride sample.
///
/// Sorting 16.7 M floats at 4096^2 would cost about a second for a number that
/// only needs to be approximately right; every 17th element is plenty and keeps
/// this in the low milliseconds.
fn percentiles(data: &[f32], a: f32, b: f32) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let stride = (data.len() / 200_000).max(1);
    let mut sample: Vec<f32> = data.iter().copied().step_by(stride).collect();
    sample.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |p: f32| sample[((sample.len() as f32 * p) as usize).min(sample.len() - 1)];
    (pick(a), pick(b))
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    res: u32,
    dt: f32,
    rain_rate: f32,
    evaporation: f32,
    capacity: f32,
    dissolve: f32,
    deposit: f32,
    min_slope: f32,
    pipe_area: f32,
    gravity: f32,
    cell_size: f32,
    _pad: f32,
}

/// What a run produces. The height field is the terrain; `flow` is the drainage
/// network the solver traced while producing it, which costs nothing extra and
/// is the material mask everything downstream wants.
pub struct ErosionResult {
    pub height: Vec<f32>,
    /// Accumulated discharge per cell, raw. Heavy-tailed -- normalize before use.
    pub flow: Vec<f32>,
}

pub struct Erosion {
    res: u32,
    height: wgpu::Buffer,
    flow: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    pipelines: Vec<wgpu::ComputePipeline>,
    /// Kept alive for the lifetime of the bind groups.
    _params: wgpu::Buffer,
}

impl Erosion {
    /// Allocate solver state for a `res x res` field. `cell_size_m` is the
    /// ground distance between texels, which sets the physical scale of the
    /// gradients the solver sees.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: u32,
        cell_size_m: f32,
        p: &ErosionParams,
    ) -> Self {
        let n = (res as u64) * (res as u64);

        let uniforms = Uniforms {
            res,
            dt: p.dt,
            rain_rate: p.rain_rate,
            evaporation: p.evaporation,
            capacity: p.capacity,
            dissolve: p.dissolve_rate,
            deposit: p.deposit_rate,
            min_slope: p.min_slope,
            pipe_area: p.pipe_area,
            gravity: p.gravity,
            cell_size: cell_size_m,
            _pad: 0.0,
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("erosion-params"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Buffers are zero-initialized by wgpu, which is exactly the starting
        // state we want for water, sediment, flux and velocity.
        let field = |label, bytes_per_texel: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: n * bytes_per_texel,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };

        let height =
            field("erosion-height", 4, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let water = field("erosion-water", 4, wgpu::BufferUsages::empty());
        let flux = field("erosion-flux", 16, wgpu::BufferUsages::empty());
        let vel = field("erosion-velocity", 8, wgpu::BufferUsages::empty());
        let sed_a = field("erosion-sediment-a", 4, wgpu::BufferUsages::empty());
        let sed_b = field("erosion-sediment-b", 4, wgpu::BufferUsages::empty());
        let flow = field("erosion-flow", 4, wgpu::BufferUsages::COPY_SRC);

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("erosion-readback"),
            // Height and flow, back to back in one mapping.
            size: n * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion-bgl"),
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
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                storage(5),
                storage(6),
                storage(7),
            ],
        });

        // Two bind groups differing only in which sediment buffer is source
        // and which is destination. Advection samples neighbours, so it cannot
        // read and write the same buffer; the host alternates per iteration.
        let make_bg = |label, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: height.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: water.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: flux.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: flow.as_entire_binding() },
                ],
            })
        };
        let bind_groups =
            [make_bg("erosion-bg-a", &sed_a, &sed_b), make_bg("erosion-bg-b", &sed_b, &sed_a)];

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("erosion"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::EROSION.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("erosion-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = PASSES
            .iter()
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();

        // Parameters are fixed for the whole run.
        queue.write_buffer(&params, 0, bytemuck::bytes_of(&uniforms));

        Self { res, height, flow, staging, bind_groups, pipelines, _params: params }
    }

    /// Run the solver and return the eroded heightfield in meters.
    ///
    /// `progress` is called with 0..1 between chunks. This blocks until the GPU
    /// finishes -- generation is an explicit user action, not a per-frame cost.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        heights: &[f32],
        iterations: u32,
        mut progress: impl FnMut(f32),
    ) -> ErosionResult {
        queue.write_buffer(&self.height, 0, bytemuck::cast_slice(heights));

        let groups = self.res.div_ceil(8);
        let mut done = 0u32;

        while done < iterations {
            let batch = CHUNK.min(iterations - done);
            let mut encoder = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("erode") });

            for i in 0..batch {
                let bg = &self.bind_groups[((done + i) % 2) as usize];
                for pipeline in &self.pipelines {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups, groups, 1);
                }
            }
            queue.submit(Some(encoder.finish()));
            done += batch;
            progress(done as f32 / iterations as f32);
        }

        self.readback(device, queue, heights.len())
    }

    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue, len: usize) -> ErosionResult {
        let bytes = (len * 4) as u64;
        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("erode-read") });
        encoder.copy_buffer_to_buffer(&self.height, 0, &self.staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.flow, 0, &self.staging, bytes, bytes);
        queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        match rx.recv() {
            Ok(Ok(())) => {}
            other => {
                log::error!("erosion readback failed: {other:?}");
                return ErosionResult { height: Vec::new(), flow: Vec::new() };
            }
        }
        let out = {
            let Ok(view) = self.staging.slice(..).get_mapped_range() else {
                self.staging.unmap();
                return ErosionResult { height: Vec::new(), flow: Vec::new() };
            };
            let all: &[f32] = bytemuck::cast_slice(&view);
            ErosionResult { height: all[..len].to_vec(), flow: all[len..len * 2].to_vec() }
        };
        self.staging.unmap();
        out
    }

    /// Turn accumulated stream power into a 0..1 channel mask.
    ///
    /// A percentile stretch rather than a log or a linear normalize. Both of
    /// those answer "how much power passed here", which is not the question --
    /// a material mask needs "is this cell part of the drainage network", and
    /// channels are by definition the top few percent of any landscape.
    /// Mapping the 88th percentile to 0 and the 99.8th to 1 gives that
    /// directly, and is stable across worlds with very different total rainfall.
    pub fn normalize_flow(flow: &[f32]) -> Vec<f32> {
        let (lo, hi) = percentiles(flow, 0.88, 0.998);
        if hi <= lo {
            return vec![0.0; flow.len()];
        }
        let inv = 1.0 / (hi - lo);
        flow.iter().map(|f| ((f - lo) * inv).clamp(0.0, 1.0)).collect()
    }

    /// Net height change, remapped to 0..1 with 0.5 meaning "unchanged".
    /// Below 0.5 is scoured bedrock, above is deposited sediment.
    pub fn deposition_map(before: &[f32], after: &[f32]) -> Vec<f32> {
        let peak =
            before.iter().zip(after).map(|(a, b)| (b - a).abs()).fold(0.0f32, f32::max).max(1e-4);
        before
            .iter()
            .zip(after)
            .map(|(a, b)| (((b - a) / peak) * 0.5 + 0.5).clamp(0.0, 1.0))
            .collect()
    }
}

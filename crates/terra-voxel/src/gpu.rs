//! The compute-shader side of Surface Nets.
//!
//! Three dispatches per chunk, and the result stays on the GPU: vertices,
//! indices and a filled-in `draw_indexed_indirect` argument block, ready to be
//! bound and drawn without the CPU ever seeing a triangle.
//!
//! [`Extractor::readback`] pulls the mesh back to system memory anyway, but
//! only for tests and for building physics colliders. It is deliberately not
//! on the per-frame path: a readback stalls until the queue drains, and doing
//! that for every chunk a camera walks past would cost more than the
//! extraction saved.

use crate::lod::DrawIndexedIndirect;
use crate::surface_nets::{Mesh, SampleGrid};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// WGSL source for the extraction passes. `include_str!` so a moved shader is
/// a build error, matching how `terra-gen` embeds its own.
pub const SURFACE_NETS: &str = include_str!("../../../assets/shaders/voxel/surface_nets.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dim: u32,
    voxel: f32,
    max_vertices: u32,
    max_indices: u32,
    origin: [f32; 3],
    _pad: f32,
}

/// A vertex as the shader writes it. Padded to 32 bytes so both `vec3`s land
/// on the 16-byte alignment WGSL requires in storage; a tighter 24-byte layout
/// reads back misaligned and the normals come out as garbage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub _p0: f32,
    pub normal: [f32; 3],
    pub _p1: f32,
}

const _: () = assert!(std::mem::size_of::<GpuVertex>() == 32);

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Counters {
    vertex_count: u32,
    index_count: u32,
    vertex_overflow: u32,
    index_overflow: u32,
}

/// How much room a chunk's output is given.
#[derive(Debug, Clone, Copy)]
pub struct Capacity {
    pub max_vertices: u32,
    pub max_indices: u32,
}

impl Capacity {
    /// The true worst case for `dim` cells.
    ///
    /// Vertices are hard-bounded at one per cell by the algorithm itself.
    /// Indices are bounded by every cell contributing one quad; a real surface
    /// is closer to `dim^2`, but a fully convoluted field can approach this
    /// and a chunk that overflows silently loses geometry.
    pub fn worst_case(dim: u32) -> Self {
        let cells = dim * dim * dim;
        Self { max_vertices: cells, max_indices: cells * 6 }
    }

    /// Sized for a surface that behaves like a surface. Roughly 8x less memory
    /// than the worst case; the overflow flags catch anything that exceeds it.
    pub fn typical(dim: u32) -> Self {
        let face = dim * dim;
        Self { max_vertices: face * 4, max_indices: face * 24 }
    }

    fn vertex_bytes(&self) -> u64 {
        self.max_vertices as u64 * std::mem::size_of::<GpuVertex>() as u64
    }

    fn index_bytes(&self) -> u64 {
        self.max_indices as u64 * 4
    }
}

/// Per-chunk GPU buffers. Reused across extractions of the same size -- a
/// chunk being re-sculpted is re-extracted many times a second, and
/// reallocating a megabyte per stroke is the kind of thing that shows up as a
/// hitch rather than as a frame-time number.
pub struct ChunkBuffers {
    pub dim: u32,
    pub capacity: Capacity,
    samples: wgpu::Buffer,
    /// Scratch: the cell-to-vertex map the first pass writes and the second
    /// reads. Never touched from the CPU, but owned here so its lifetime
    /// matches the bind group that references it.
    #[allow(dead_code)]
    cell_vertex: wgpu::Buffer,
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    counters: wgpu::Buffer,
    /// Bindable directly as `INDIRECT` -- this is the buffer a render pass
    /// hands to `draw_indexed_indirect`.
    pub args: wgpu::Buffer,
    staging: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ChunkBuffers {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        dim: u32,
        capacity: Capacity,
    ) -> Self {
        let n = (dim + 1) as u64;
        let samples = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-samples"),
            size: n * n * n * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_vertex = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-cell-index"),
            size: (dim as u64).pow(3) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-vertices"),
            size: capacity.vertex_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-indices"),
            size: capacity.index_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-counters"),
            size: std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-draw-args"),
            size: std::mem::size_of::<DrawIndexedIndirect>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // One staging buffer for the whole readback, so a mesh comes back in a
        // single map rather than three.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-staging"),
            size: capacity.vertex_bytes()
                + capacity.index_bytes()
                + std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-extract"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: samples.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_vertex.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vertices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: counters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: args.as_entire_binding() },
            ],
        });

        Self {
            dim,
            capacity,
            samples,
            cell_vertex,
            vertices,
            indices,
            counters,
            args,
            staging,
            params_buffer: params,
            bind_group,
        }
    }

    fn params(&self) -> &wgpu::Buffer {
        &self.params_buffer
    }
}

/// How an extraction turned out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractStats {
    pub vertex_count: u32,
    pub index_count: u32,
    /// True when the chunk needed more room than it was given. The geometry
    /// that did fit is still valid, but the chunk has holes.
    pub overflowed: bool,
}

/// Compiled pipelines, shared by every chunk.
pub struct Extractor {
    place: wgpu::ComputePipeline,
    quads: wgpu::ComputePipeline,
    write_args: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

impl Extractor {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface-nets"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_NETS.into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-extract"),
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
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-extract"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });

        let stage = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            place: stage("place_vertices", "sn-place"),
            quads: stage("emit_quads", "sn-quads"),
            write_args: stage("write_args", "sn-args"),
            layout,
        }
    }

    pub fn buffers(&self, device: &wgpu::Device, dim: u32, capacity: Capacity) -> ChunkBuffers {
        ChunkBuffers::new(device, &self.layout, dim, capacity)
    }

    /// Upload a sampled block and run the three passes.
    ///
    /// The counters are zeroed from the CPU rather than by a clear kernel: it
    /// is 16 bytes, `write_buffer` folds into the same submission, and a
    /// fourth dispatch to zero four integers is not worth its own barrier.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &ChunkBuffers,
        grid: &SampleGrid,
    ) {
        assert_eq!(grid.dim, buffers.dim, "grid and buffers disagree on chunk size");

        queue.write_buffer(&buffers.samples, 0, bytemuck::cast_slice(&grid.values));
        queue.write_buffer(&buffers.counters, 0, bytemuck::bytes_of(&Counters::default()));
        queue.write_buffer(
            buffers.params(),
            0,
            bytemuck::bytes_of(&Params {
                dim: grid.dim,
                voxel: grid.voxel,
                max_vertices: buffers.capacity.max_vertices,
                max_indices: buffers.capacity.max_indices,
                origin: grid.origin.to_array(),
                _pad: 0.0,
            }),
        );

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sn-extract") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface-nets"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &buffers.bind_group, &[]);

            // Cells: dim^3 threads at 4x4x4 per workgroup.
            let cells = grid.dim.div_ceil(4);
            pass.set_pipeline(&self.place);
            pass.dispatch_workgroups(cells, cells, cells);

            // Lattice points: (dim + 1)^3. Separate dispatch, not just a
            // second entry point in the same one -- the quad pass reads the
            // cell-to-vertex map the first pass writes, and only a dispatch
            // boundary guarantees those writes are visible.
            let lattice = (grid.dim + 1).div_ceil(4);
            pass.set_pipeline(&self.quads);
            pass.dispatch_workgroups(lattice, lattice, lattice);

            pass.set_pipeline(&self.write_args);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Run the passes and copy the mesh back to system memory.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &ChunkBuffers,
        grid: &SampleGrid,
    ) -> (Mesh, ExtractStats) {
        self.dispatch(device, queue, buffers, grid);

        let vbytes = buffers.capacity.vertex_bytes();
        let ibytes = buffers.capacity.index_bytes();
        let cbytes = std::mem::size_of::<Counters>() as u64;

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sn-read") });
        encoder.copy_buffer_to_buffer(&buffers.vertices, 0, &buffers.staging, 0, vbytes);
        encoder.copy_buffer_to_buffer(&buffers.indices, 0, &buffers.staging, vbytes, ibytes);
        encoder.copy_buffer_to_buffer(
            &buffers.counters,
            0,
            &buffers.staging,
            vbytes + ibytes,
            cbytes,
        );
        queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        buffers.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let empty =
            (Mesh::default(), ExtractStats { vertex_count: 0, index_count: 0, overflowed: false });
        match rx.recv() {
            Ok(Ok(())) => {}
            other => {
                log::error!("surface nets readback failed: {other:?}");
                return empty;
            }
        }

        let out = {
            let Ok(view) = buffers.staging.slice(..).get_mapped_range() else {
                buffers.staging.unmap();
                return empty;
            };
            let counters: Counters =
                *bytemuck::from_bytes(&view[(vbytes + ibytes) as usize..][..cbytes as usize]);

            // Clamp to capacity. On overflow the atomic counters keep counting
            // past the buffer, so the raw values are the *demand*, not what
            // was written.
            let vcount = counters.vertex_count.min(buffers.capacity.max_vertices) as usize;
            let icount = counters.index_count.min(buffers.capacity.max_indices) as usize;

            let verts: &[GpuVertex] = bytemuck::cast_slice(&view[..vbytes as usize]);
            let idx: &[u32] = bytemuck::cast_slice(&view[vbytes as usize..][..ibytes as usize]);

            let mesh = Mesh {
                positions: verts[..vcount].iter().map(|v| Vec3::from(v.position)).collect(),
                normals: verts[..vcount].iter().map(|v| Vec3::from(v.normal)).collect(),
                indices: idx[..icount].to_vec(),
            };
            let stats = ExtractStats {
                vertex_count: counters.vertex_count,
                index_count: counters.index_count,
                overflowed: counters.vertex_overflow != 0 || counters.index_overflow != 0,
            };
            (mesh, stats)
        };
        buffers.staging.unmap();
        out
    }
}

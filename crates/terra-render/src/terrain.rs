//! Terrain heightfield: GPU buffers, draw pipeline, and the sculpt brush.
//!
//! The CPU copy in [`Terrain::heights`] is authoritative. Sculpting edits it and
//! uploads only the touched rows, which keeps raycasting and saving correct
//! without any GPU readback. A brush covers at most a few thousand texels, so
//! this costs microseconds -- compute is for whole-map work like erosion.

use crate::camera::{Camera, CameraUniform};
use crate::context::{DEPTH_FORMAT, RenderContext};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use terra_core::WorldSize;
use wgpu::util::DeviceExt;

/// Grid quads per side. Independent of heightfield resolution: this is a
/// uniform grid placeholder for CDLOD, and 512 keeps the vertex count at ~263k
/// (0.5 M triangles), comfortably inside the 1.2 ms terrain budget.
const GRID_RES: u32 = 512;

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
    _pad: f32,
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

    height_buf: wgpu::Buffer,
    flow_buf: wgpu::Buffer,
    deposit_buf: wgpu::Buffer,
    road_buf: wgpu::Buffer,
    rut_buf: wgpu::Buffer,
    terrain_ub: wgpu::Buffer,
    camera_ub: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,

    camera_bg: wgpu::BindGroup,
    terrain_bg: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

    brush: TerrainUniform,
}

impl Terrain {
    pub fn new(ctx: &RenderContext, size: WorldSize) -> Self {
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

        let brush = TerrainUniform {
            world_extent: extent_m,
            height_res: res,
            grid_res: GRID_RES,
            brush_radius: 0.0,
            brush_center: [0.0, 0.0],
            brush_active: 0.0,
            _pad: 0.0,
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
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/render/terrain.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-layout"),
            bind_group_layouts: &[Some(&camera_bgl), Some(&terrain_bgl)],
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
                    format: ctx.config.format,
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

        Self {
            heights,
            res,
            extent_m,
            height_buf,
            flow_buf,
            deposit_buf,
            road_buf,
            rut_buf,
            terrain_ub,
            camera_ub,
            index_buf,
            index_count,
            camera_bg,
            terrain_bg,
            pipeline,
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

    pub fn extent_m(&self) -> f32 {
        self.extent_m
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

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &self.terrain_bg, &[]);
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
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

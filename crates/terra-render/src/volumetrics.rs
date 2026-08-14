//! Volumetric fog on a froxel grid.
//!
//! A camera-aligned 3D texture: x and y are screen space, z is distance from
//! the camera distributed exponentially so the first few metres get as many
//! cells as the last few hundred. Two compute passes fill it -- density and
//! in-scattered light per cell, then a front-to-back march integrating both --
//! and every shading pass samples the result by distance.
//!
//! This replaces the analytic `1 - exp(-d)` fog the passes used to apply. That
//! fog had no way to know whether the sun reached a given point, so it lit the
//! air inside a shadow exactly as brightly as the air beside it. Testing each
//! cell against the shadow map is what turns fog into shafts.

use crate::camera::{Camera, CameraUniform};
use crate::lighting::Lighting;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Grid resolution. Screen-space x and y are coarse on purpose -- fog has no
/// high-frequency detail, and the cost is linear in cell count.
pub const FROXELS: [u32; 3] = [160, 90, 64];

/// Format of both grids. Half float: in-scattering is HDR, and the whole point
/// of the buffer is that a sunlit cell can be far brighter than a shadowed one.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Debug, Clone, Copy)]
pub struct FogSettings {
    pub enabled: bool,
    /// Uniform haze, per metre.
    pub density: f32,
    /// Extra density that pools in low ground.
    pub mist_strength: f32,
    /// Height above `mist_base` at which the mist has thinned to 1/e.
    pub mist_falloff: f32,
    pub mist_base: f32,
    /// Forward scattering. 0 is isotropic; toward 1 the medium throws light
    /// forward, which is what makes looking toward the sun through fog bright.
    pub anisotropy: f32,
    /// Colour of the medium.
    pub albedo: Vec3,
    /// Furthest the grid reaches. Beyond it, fog stops accumulating.
    pub distance: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            // Clear air by default. These are extinction per metre, and the
            // scale is unforgiving: 0.004 leaves 16% of a 450 m view, which
            // erases the sky gradient and the sun disc along with it. Terrain
            // sits at 256 m, so a mist base below that put valley mist across
            // the entire world rather than in the valleys.
            density: 0.00025,
            mist_strength: 0.0008,
            mist_falloff: 0.02,
            mist_base: 280.0,
            anisotropy: 0.72,
            albedo: Vec3::new(0.92, 0.95, 1.0),
            distance: 450.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FogUniform {
    range: [f32; 4],
    mist: [f32; 4],
    medium: [f32; 4],
    screen: [f32; 4],
}

/// The two grids, created before anything that binds them.
///
/// Split out because the light state has to name the result grid, and the fog
/// pipelines have to name the light state -- one of the two has to exist first,
/// and it is the textures.
pub struct FroxelGrids {
    injected: wgpu::TextureView,
    scattered: wgpu::TextureView,
}

impl FroxelGrids {
    pub fn new(device: &wgpu::Device) -> Self {
        let grid = |label, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: FROXELS[0],
                        height: FROXELS[1],
                        depth_or_array_layers: FROXELS[2],
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: FORMAT,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some(label),
                    dimension: Some(wgpu::TextureViewDimension::D3),
                    ..Default::default()
                })
        };
        Self {
            injected: grid(
                "froxel-injected",
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            scattered: grid(
                "froxel-scattered",
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
        }
    }

    /// The grid the shading passes sample.
    pub fn scattered(&self) -> &wgpu::TextureView {
        &self.scattered
    }
}

pub struct Volumetrics {
    pub settings: FogSettings,

    fog_uniform: wgpu::Buffer,
    camera_uniform: wgpu::Buffer,
    /// Written by `inject`, read by `accumulate`. Held only so the views it
    /// backs stay alive for as long as the bind groups referencing them.
    _injected: wgpu::TextureView,
    /// The result every shading pass samples.
    scattered: wgpu::TextureView,

    grid_bind_group: wgpu::BindGroup,
    accum_grid_bind_group: wgpu::BindGroup,
    accum_bind_group: wgpu::BindGroup,
    inject_pipeline: wgpu::ComputePipeline,
    accumulate_pipeline: wgpu::ComputePipeline,
}

impl Volumetrics {
    pub fn new(device: &wgpu::Device, lighting: &Lighting, grids: FroxelGrids) -> Self {
        let settings = FogSettings::default();
        let FroxelGrids { injected, scattered } = grids;

        let fog_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-uniform"),
            size: std::mem::size_of::<FogUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding, access| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access,
                format: FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
        let grid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-grid-bgl"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(2, wgpu::StorageTextureAccess::WriteOnly),
            ],
        });
        // The march reads the injected grid, so its group 1 must not also name
        // that grid as a write target -- binding it both ways inside one
        // dispatch is a usage conflict, however unused the write binding is.
        let accum_grid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-accum-grid-bgl"),
            entries: &[uniform_entry(0)],
        });
        let accum_grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-accum-grid-bg"),
            layout: &accum_grid_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fog_uniform.as_entire_binding(),
            }],
        });

        let accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-accum-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(1, wgpu::StorageTextureAccess::WriteOnly),
            ],
        });
        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-grid-bg"),
            layout: &grid_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fog_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&injected),
                },
            ],
        });
        let accum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-accum-bg"),
            layout: &accum_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&injected),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scattered),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetrics"),
            source: wgpu::ShaderSource::Wgsl(
                [
                    include_str!("../../../assets/shaders/common/noise.wgsl"),
                    include_str!("../../../assets/shaders/common/camera.wgsl"),
                    include_str!("../../../assets/shaders/common/lighting.wgsl"),
                    include_str!("../../../assets/shaders/render/volumetrics.wgsl"),
                    include_str!("../../../assets/shaders/render/volumetrics_gen.wgsl"),
                ]
                .join("\n")
                .into(),
            ),
        });

        // The compute passes bind the light *without* the froxel grid, because
        // they are writing it. Naming a texture as both a sampled resource and
        // a storage target in one dispatch is a usage conflict, not a subtle
        // performance question.
        let pipeline = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let inject_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog-inject-layout"),
            bind_group_layouts: &[Some(&lighting.compute_layout), Some(&grid_bgl)],
            immediate_size: 0,
        });
        // The march needs no light state: everything it integrates was already
        // written into the grid.
        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog-accum-layout"),
            bind_group_layouts: &[None, Some(&accum_grid_bgl), Some(&accum_bgl)],
            immediate_size: 0,
        });

        Self {
            settings,
            fog_uniform,
            camera_uniform,
            _injected: injected,
            scattered,
            grid_bind_group,
            accum_grid_bind_group,
            accum_bind_group,
            inject_pipeline: pipeline("inject", &inject_layout),
            accumulate_pipeline: pipeline("accumulate", &accum_layout),
        }
    }

    /// The grid the shading passes sample.
    pub fn scattered_view(&self) -> &wgpu::TextureView {
        &self.scattered
    }

    /// Near plane of the grid. Shared with the shading passes so both map a
    /// distance to the same slice.
    pub fn near(&self) -> f32 {
        0.5
    }

    pub fn build(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        lighting: &Lighting,
        cam: &Camera,
        aspect: f32,
        time: f32,
    ) {
        if !self.settings.enabled {
            return;
        }
        queue.write_buffer(&self.camera_uniform, 0, bytemuck::bytes_of(&cam.uniform(aspect)));

        let s = &self.settings;
        let u = FogUniform {
            range: [self.near(), s.distance, FROXELS[2] as f32, s.density],
            mist: [s.mist_falloff, s.mist_base, s.mist_strength, time],
            medium: s.albedo.extend(s.anisotropy).to_array(),
            // Sky contribution to in-scattering. At 0.35 the ambient term was
            // twice the sun's, so the fog was mostly flat grey rather than
            // light from a direction.
            screen: [0.0, 0.0, 0.12, 0.0],
        };
        queue.write_buffer(&self.fog_uniform, 0, bytemuck::bytes_of(&u));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fog-inject"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inject_pipeline);
            pass.set_bind_group(0, &lighting.compute_bind_group, &[]);
            pass.set_bind_group(1, &self.grid_bind_group, &[]);
            pass.dispatch_workgroups(FROXELS[0].div_ceil(8), FROXELS[1].div_ceil(8), FROXELS[2]);
        }

        // A separate pass, not just a second dispatch: the march reads what
        // injection wrote, and a write followed by a read of the same texture
        // inside one pass is a hazard rather than an ordering.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fog-accumulate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.set_bind_group(1, &self.accum_grid_bind_group, &[]);
            pass.set_bind_group(2, &self.accum_bind_group, &[]);
            // One thread per screen cell: the march is sequential along z, so
            // the depth axis is the loop rather than the dispatch.
            pass.dispatch_workgroups(FROXELS[0].div_ceil(8), FROXELS[1].div_ceil(8), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `froxel_distance` / `froxel_slice` in the shader.
    fn dist(slice: f32, near: f32, far: f32, n: f32) -> f32 {
        near * (far / near).powf(slice / n)
    }
    fn slice(d: f32, near: f32, far: f32, n: f32) -> f32 {
        n * (d.max(near) / near).ln() / (far / near).ln()
    }

    #[test]
    fn slice_distribution_round_trips() {
        let (near, far, n) = (0.5, 700.0, 64.0);
        for s in [0.0, 1.0, 17.0, 63.0, 64.0] {
            let d = dist(s, near, far, n);
            assert!((slice(d, near, far, n) - s).abs() < 1e-3, "slice {s} -> {d}");
        }
    }

    #[test]
    fn near_slices_are_finer_than_far_ones() {
        // The reason for an exponential distribution: a linear grid spends the
        // same resolution on the last hundred metres as on the first one, and
        // the first one is where fog is actually looked at.
        let (near, far, n) = (0.5, 700.0, 64.0);
        let first = dist(1.0, near, far, n) - dist(0.0, near, far, n);
        let last = dist(64.0, near, far, n) - dist(63.0, near, far, n);
        assert!(last > first * 50.0, "first {first} m, last {last} m");
        assert!(first < 0.2, "near slice should be under 20 cm, got {first}");
    }

    #[test]
    fn grid_reaches_the_configured_distance() {
        let s = FogSettings::default();
        let d = dist(64.0, 0.5, s.distance, 64.0);
        assert!((d - s.distance).abs() < 0.01);
    }
}

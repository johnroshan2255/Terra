//! Sun, sky and cascaded shadow maps.
//!
//! One uniform drives every shading pass, so the sun cannot disagree with the
//! sky it is drawn in -- which is exactly what a hardcoded `SUN` constant in
//! three shaders was doing before.
//!
//! Shadows are cascaded: the camera's depth range is split, each slice gets an
//! orthographic light frustum fitted to it, and the whole set lives in one
//! depth texture array. This is Phase C of `docs/culling.md`, and the note
//! there is the important one -- *cascade fitting is the bigger win, not
//! culling more objects out of a badly-fit one*. So the fit is snapped to the
//! shadow map's own texel grid, which is what stops the shadow edges crawling
//! as the camera moves.

use crate::camera::Camera;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Cascade count. Three covers the near-to-mid range that matters; a fourth
/// costs a full extra shadow render for ground nobody looks at.
pub const CASCADES: usize = 3;

/// Shadow map resolution presets, exposed in the graphics settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {
    Off,
    Low,
    Medium,
    High,
}

impl ShadowQuality {
    pub const ALL: [ShadowQuality; 4] =
        [ShadowQuality::Off, ShadowQuality::Low, ShadowQuality::Medium, ShadowQuality::High];

    pub fn label(self) -> &'static str {
        match self {
            ShadowQuality::Off => "Off",
            ShadowQuality::Low => "Low",
            ShadowQuality::Medium => "Medium",
            ShadowQuality::High => "High",
        }
    }

    pub fn resolution(self) -> u32 {
        match self {
            ShadowQuality::Off => 256,
            ShadowQuality::Low => 1024,
            ShadowQuality::Medium => 2048,
            ShadowQuality::High => 4096,
        }
    }

    pub fn enabled(self) -> bool {
        !matches!(self, ShadowQuality::Off)
    }
}

/// One dial that moves everything that costs frame time.
///
/// Presets rather than a slider: the settings interact, and a player who turns
/// shadows to Ultra and fog to nothing has not chosen a quality level, they
/// have chosen a strange-looking one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Quality {
    pub const ALL: [Quality; 4] = [Quality::Low, Quality::Medium, Quality::High, Quality::Ultra];

    pub fn label(self) -> &'static str {
        match self {
            Quality::Low => "Low",
            Quality::Medium => "Medium",
            Quality::High => "High",
            Quality::Ultra => "Ultra",
        }
    }

    /// `(shadows, shadow distance, god rays, temporal AA)`
    pub fn sky(self) -> (ShadowQuality, f32, f32, bool) {
        match self {
            Quality::Low => (ShadowQuality::Off, 200.0, 0.0, false),
            Quality::Medium => (ShadowQuality::Low, 320.0, 0.35, true),
            Quality::High => (ShadowQuality::Medium, 420.0, 0.55, true),
            Quality::Ultra => (ShadowQuality::High, 700.0, 0.75, true),
        }
    }

    /// `(fog on, froxel distance)`
    pub fn fog(self) -> (bool, f32) {
        match self {
            Quality::Low => (false, 400.0),
            Quality::Medium => (true, 450.0),
            Quality::High => (true, 700.0),
            Quality::Ultra => (true, 1100.0),
        }
    }
}

/// Which sun a viewport shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightMode {
    /// The world's own time of day, advancing if the cycle is running.
    Scene,
    /// A fixed neutral sun. Editing under a moving sun means the ground you
    /// are painting changes colour while you paint it, and a world authored at
    /// dusk looks nothing like the same world at noon.
    Fixed,
}

/// The hour the fixed editor sun sits at. High enough to be neutral white and
/// to light both sides of a ridge, low enough to still cast a readable shadow.
pub const EDITOR_HOUR: f32 = 10.5;

/// What the artist controls.
#[derive(Debug, Clone, Copy)]
pub struct SkySettings {
    /// Hours, 0..24. 12 is noon.
    pub time_of_day: f32,
    /// Hours per real second when the cycle is running.
    pub day_speed: f32,
    pub cycle_running: bool,
    /// How far shadows are drawn, in metres.
    pub shadow_distance: f32,
    pub shadow_quality: ShadowQuality,
    /// Strength of the sun-facing scattering in the sky.
    pub haze: f32,
    pub exposure: f32,
    /// Strength of the god-ray shafts. 0 disables the pass entirely.
    pub god_rays: f32,
    /// Show the world's time of day in the editor viewport rather than the
    /// fixed sun. Off by default: the time is authored for play, and previewing
    /// it while building fights the work.
    pub editor_preview: bool,
    /// Temporal anti-aliasing. Also what resolves the scatter dissolve: without
    /// it the fade band is a stipple rather than a fade.
    pub temporal_aa: bool,
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            time_of_day: 9.5,
            day_speed: 0.25,
            cycle_running: false,
            // Defaults are the Medium preset. Measured on an M4 at 1600x900:
            // Low 1.85 ms, Medium 2.39, High 3.87, Ultra 6.13 -- and the frame
            // budget is 5. Medium leaves room for everything that is not the
            // renderer; High is one click away.
            shadow_distance: 320.0,
            shadow_quality: ShadowQuality::Low,
            haze: 1.0,
            exposure: 1.0,
            god_rays: 0.35,
            editor_preview: false,
            temporal_aa: true,
        }
    }
}

/// Sun state derived from the time of day.
#[derive(Debug, Clone, Copy)]
pub struct Sun {
    /// Unit vector pointing *toward* the sun.
    pub direction: Vec3,
    pub color: Vec3,
    /// Above the horizon, 0 at night.
    pub daylight: f32,
    /// True when the moon is the key light rather than the sun.
    pub night: bool,
}

impl Sun {
    /// Sun position for an hour of the day.
    ///
    /// Rises in +X at 06:00, overhead at 12:00, sets in -X at 18:00, with a
    /// northward tilt so it never passes exactly through the zenith -- a sun
    /// straight overhead flattens every slope at noon.
    pub fn at(hour: f32) -> Self {
        let phase = (hour - 6.0) / 12.0 * std::f32::consts::PI;
        let solar = Vec3::new(phase.cos(), phase.sin(), 0.35).normalize();

        // Below the horizon the moon takes over: the opposite arc, cold and
        // dim. Without it the world goes to pure black, which reads as broken.
        let night = solar.y < 0.0;
        let direction = if night { -solar } else { solar };
        let elevation = direction.y.max(0.0);

        // Warm and weak near the horizon, white and strong overhead. This is
        // the whole of "golden hour" and it costs one mix.
        let warmth = 1.0 - (elevation / 0.35).clamp(0.0, 1.0);
        let day_color = Vec3::new(1.0, 0.96, 0.90).lerp(Vec3::new(1.0, 0.52, 0.26), warmth);
        let night_color = Vec3::new(0.42, 0.52, 0.78);

        // Fades through twilight rather than switching, so dusk is a period
        // and not an event.
        let daylight = smoothstep(-0.12, 0.10, solar.y);

        Self { direction, color: if night { night_color } else { day_color }, daylight, night }
    }

    /// Key light intensity, already accounting for night.
    pub fn intensity(&self) -> f32 {
        if self.night { 0.16 } else { 0.25 + 1.55 * self.daylight }
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Mirrors `Light` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightUniform {
    /// Toward the sun; w is daylight, 0 at night.
    sun_direction: [f32; 4],
    /// rgb radiance, w intensity.
    sun_color: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    /// rgb ambient, w exposure.
    ambient: [f32; 4],
    cascade_view_proj: [[[f32; 4]; 4]; CASCADES],
    /// Far distance of each cascade, in view space.
    cascade_split: [f32; 4],
    /// x = shadows on, y = texel world size, z = haze, w = night factor.
    params: [f32; 4],
    /// x = fog near, y = fog far, z = fog on, w = slices.
    fog: [f32; 4],
    /// xy = 1/viewport.
    fog_screen: [f32; 4],
}

/// A cascade's light matrix, written per shadow draw.
///
/// Laid out exactly like `CameraUniform`, because that is what the shadow
/// entry points read: they share a module with the shading passes, and WGSL
/// cannot declare two different types at one group and binding. Only
/// `view_proj` is used; the rest exists so the binding size matches what the
/// shader was compiled against.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CascadeUniform {
    pub view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
}

/// Uniform slot stride. Dynamic offsets must respect the device's alignment,
/// and 256 satisfies every backend without querying.
const SLOT: u64 = 256;

/// Light values taken from the Environment Light Mixer, replacing the ones this
/// module would otherwise derive from its own tables.
#[derive(Debug, Clone, Copy)]
struct EnvOverride {
    sun_color: Vec3,
    zenith: Vec3,
    horizon: Vec3,
    ground: Vec3,
}

pub struct Lighting {
    pub settings: SkySettings,
    pub sun: Sun,
    /// `None` until the mixer has pushed its values, so the old derivation is the
    /// fallback rather than a hard dependency.
    env_override: Option<EnvOverride>,

    uniform: wgpu::Buffer,
    /// One `CascadeUniform` per cascade, addressed by dynamic offset.
    cascade_ub: wgpu::Buffer,

    shadow_map: wgpu::Texture,
    /// One view per cascade, for the depth attachments.
    pub cascade_views: Vec<wgpu::TextureView>,
    resolution: u32,

    /// Sampled by the shading passes: uniform + shadow array + comparison
    /// sampler.
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,

    /// The same light state without the fog grid, for the passes that write it.
    /// Binding a texture as sampled and as a storage target in one dispatch is
    /// a usage conflict.
    pub compute_layout: wgpu::BindGroupLayout,
    pub compute_bind_group: wgpu::BindGroup,

    /// Bound by the shadow passes: the current cascade's matrix.
    pub cascade_layout: wgpu::BindGroupLayout,
    pub cascade_bind_group: wgpu::BindGroup,
}

impl Lighting {
    pub fn new(device: &wgpu::Device, settings: SkySettings, fog: &wgpu::TextureView) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-uniform"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cascade_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade-uniform"),
            size: SLOT * CASCADES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light-bgl"),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // Comparison sampling does the depth test in the sampler
                    // and filters the *results*, which is what makes hardware
                    // PCF cheaper than four manual fetches.
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let cascade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cascade-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cascade_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cascade-bg"),
            layout: &cascade_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &cascade_ub,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<CascadeUniform>() as u64),
                }),
            }],
        });

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light-compute-bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let resolution = settings.shadow_quality.resolution();
        let (shadow_map, cascade_views, bind_group, compute_bind_group) =
            build_shadow_map(device, &layout, &compute_layout, &uniform, resolution, fog);

        Self {
            settings,
            sun: Sun::at(settings.time_of_day),
            env_override: None,
            uniform,
            cascade_ub,
            shadow_map,
            cascade_views,
            resolution,
            layout,
            bind_group,
            compute_layout,
            compute_bind_group,
            cascade_layout,
            cascade_bind_group,
        }
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Rebuild the shadow map if the quality setting changed.
    pub fn apply_quality(&mut self, device: &wgpu::Device, fog: &wgpu::TextureView) {
        let wanted = self.settings.shadow_quality.resolution();
        if wanted == self.resolution {
            return;
        }
        self.resolution = wanted;
        let (map, views, bg, compute_bg) = build_shadow_map(
            device,
            &self.layout,
            &self.compute_layout,
            &self.uniform,
            wanted,
            fog,
        );
        self.shadow_map = map;
        self.cascade_views = views;
        self.bind_group = bg;
        self.compute_bind_group = compute_bg;
    }

    /// Advance the clock and recompute the sun.
    ///
    /// Only `Scene` advances time. A cycle that keeps running behind the menu
    /// and the editor means the world's authored hour drifts while nobody is
    /// looking at it.
    pub fn update(&mut self, dt: f32, mode: LightMode) {
        match mode {
            LightMode::Scene => {
                if self.settings.cycle_running {
                    self.settings.time_of_day =
                        (self.settings.time_of_day + self.settings.day_speed * dt).rem_euclid(24.0);
                }
                self.sun = Sun::at(self.settings.time_of_day);
            }
            LightMode::Fixed => self.sun = Sun::at(EDITOR_HOUR),
        }
    }

    /// Fit the cascades to the camera and upload everything the shaders read.
    #[allow(clippy::too_many_arguments)]
    /// Override the derived sun colour and ambient with the mixer's.
    ///
    /// Without this the sky shader and the terrain disagree about the time of
    /// day: the sky computes real scattering from the atmosphere coefficients
    /// while the terrain was lit from a separate table of hardcoded day, dusk and
    /// night colours. At noon they roughly agree and at dusk they do not, which
    /// is what made running the clock look wrong -- an orange sky over ground lit
    /// as though it were midday.
    ///
    /// Called before [`Self::upload`]; the sun *direction* still comes from the
    /// clock, which both already shared.
    pub fn set_environment(&mut self, env: &crate::environment::Environment) {
        let (zenith, horizon, ground) = env.ambient_tints();
        self.env_override =
            Some(EnvOverride { sun_color: env.sun.radiance(), zenith, horizon, ground });
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        cam: &Camera,
        aspect: f32,
        fog: [f32; 4],
        viewport: [f32; 2],
    ) {
        let splits = self.splits();
        let mut cascade_view_proj = [[[0.0f32; 4]; 4]; CASCADES];

        for (i, window) in splits.windows(2).enumerate() {
            let m = self.fit_cascade(cam, aspect, window[0], window[1]);
            cascade_view_proj[i] = m.to_cols_array_2d();
            queue.write_buffer(
                &self.cascade_ub,
                SLOT * i as u64,
                bytemuck::bytes_of(&CascadeUniform {
                    view_proj: cascade_view_proj[i],
                    inv_view_proj: [[0.0; 4]; 4],
                    eye: [0.0; 4],
                }),
            );
        }

        let sun = &self.sun;
        let elevation = sun.direction.y.max(0.0);
        // Sky colours track the sun rather than being constants, so dusk is
        // orange at the horizon and the zenith stays deep.
        let day_zenith = Vec3::new(0.055, 0.115, 0.255);
        let night_zenith = Vec3::new(0.010, 0.016, 0.040);
        let day_horizon = Vec3::new(0.64, 0.66, 0.66);
        let dusk_horizon = Vec3::new(0.62, 0.30, 0.16);
        let night_horizon = Vec3::new(0.045, 0.055, 0.090);

        let dusk = 1.0 - smoothstep(0.02, 0.30, elevation);
        let horizon = day_horizon.lerp(dusk_horizon, dusk).lerp(night_horizon, 1.0 - sun.daylight);
        let zenith = day_zenith.lerp(night_zenith, 1.0 - sun.daylight);
        let (zenith, horizon, ambient) = match self.env_override {
            // From the mixer, so the ground is lit by the same model the sky is
            // drawn with. The ground bounce is folded into the ambient term
            // because the terrain has no separate downward-facing lookup.
            Some(o) => (o.zenith, o.horizon, o.zenith * 0.55 + o.horizon * 0.25 + o.ground * 0.20),
            None => {
                (zenith, horizon, (zenith * 0.55 + horizon * 0.25) * (0.25 + 0.75 * sun.daylight))
            }
        };
        let sun_color = self.env_override.map_or(sun.color, |o| o.sun_color);

        let u = LightUniform {
            sun_direction: sun.direction.extend(sun.daylight).to_array(),
            sun_color: (sun_color * sun.intensity()).extend(sun.intensity()).to_array(),
            sky_zenith: zenith.extend(0.0).to_array(),
            sky_horizon: horizon.extend(0.0).to_array(),
            ambient: ambient.extend(self.settings.exposure).to_array(),
            cascade_view_proj,
            cascade_split: [splits[1], splits[2], splits[3], 0.0],
            params: [
                if self.settings.shadow_quality.enabled() { 1.0 } else { 0.0 },
                1.0 / self.resolution as f32,
                self.settings.haze,
                if sun.night { 1.0 } else { 0.0 },
            ],
            fog,
            fog_screen: [1.0 / viewport[0].max(1.0), 1.0 / viewport[1].max(1.0), 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    /// Cascade boundaries in view-space distance.
    ///
    /// Weighted toward logarithmic: uniform splits waste the near cascade on
    /// ground that is a few pixels tall, and pure logarithmic starves the far
    /// one.
    fn splits(&self) -> [f32; CASCADES + 1] {
        Self::splits_for(&self.settings)
    }

    pub fn splits_for(settings: &SkySettings) -> [f32; CASCADES + 1] {
        let near = 1.0;
        let far = settings.shadow_distance.max(20.0);
        let mut out = [0.0; CASCADES + 1];
        for (i, slot) in out.iter_mut().enumerate() {
            let t = i as f32 / CASCADES as f32;
            let log = near * (far / near).powf(t);
            let uniform = near + (far - near) * t;
            *slot = log * 0.75 + uniform * 0.25;
        }
        out
    }

    /// Orthographic light matrix covering one slice of the camera frustum.
    fn fit_cascade(&self, cam: &Camera, aspect: f32, near: f32, far: f32) -> Mat4 {
        let forward = cam.forward();
        let right = cam.right();
        let up = right.cross(forward).normalize();

        let tan_half = (cam.fov_y * 0.5).tan();
        let mut corners = [Vec3::ZERO; 8];
        for (i, d) in [near, far].iter().enumerate() {
            let h = tan_half * d;
            let w = h * aspect;
            let centre = cam.pos + forward * *d;
            corners[i * 4] = centre + up * h + right * w;
            corners[i * 4 + 1] = centre + up * h - right * w;
            corners[i * 4 + 2] = centre - up * h + right * w;
            corners[i * 4 + 3] = centre - up * h - right * w;
        }

        // A sphere around the slice, so the fit does not change size as the
        // camera turns. A box fitted to the corners grows and shrinks with
        // rotation, and the shadows visibly swim.
        let centre = corners.iter().copied().sum::<Vec3>() / 8.0;
        let radius = corners.iter().map(|c| c.distance(centre)).fold(0.0, f32::max).max(1.0);

        let light_dir = self.sun.direction;
        // Any up vector works except one parallel to the light.
        let up_ref = if light_dir.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
        let eye = centre + light_dir * (radius + 200.0);
        let view = look_at(eye, centre, up_ref);
        let proj = ortho_reversed(radius, radius * 2.0 + 400.0);
        let view_proj = proj * view;

        // Snap the projected centre to whole shadow texels. Without this the
        // shadow edges crawl along every surface as the camera moves, which is
        // far more visible than the aliasing it comes from. The snap has to
        // happen in clip space -- the texel grid is the map's, not the world's.
        let clip = view_proj * centre.extend(1.0);
        let half = self.resolution as f32 * 0.5;
        let texels = clip.truncate().truncate() / clip.w * half;
        let offset = (texels.round() - texels) / half;

        Mat4::from_translation(Vec3::new(offset.x, offset.y, 0.0)) * view_proj
    }

    /// Dynamic offset for a cascade's uniform slot.
    pub fn cascade_offset(index: usize) -> u32 {
        (SLOT * index as u64) as u32
    }
}

/// Right-handed look-at, built by hand for the same reason `Camera::look_at`
/// is: glam has deprecated and moved these between minor versions.
fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        glam::Vec4::new(s.x, u.x, -f.x, 0.0),
        glam::Vec4::new(s.y, u.y, -f.y, 0.0),
        glam::Vec4::new(s.z, u.z, -f.z, 0.0),
        glam::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Orthographic projection with **reversed** depth: near maps to 1, far to 0.
///
/// This has to match the scene, because the shadow pipeline clears to 0.0 and
/// compares `Greater`, and so does the comparison sampler. A conventional
/// near-to-zero ortho here would store the *farthest* caster in each texel and
/// every shadow would be inside out -- which looks like a bias problem and is
/// not one.
fn ortho_reversed(radius: f32, depth: f32) -> Mat4 {
    let inv = 1.0 / radius;
    let k = 1.0 / depth;
    Mat4::from_cols(
        glam::Vec4::new(inv, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, inv, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, k, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
    )
}

#[allow(clippy::type_complexity)]
fn build_shadow_map(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    compute_layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    resolution: u32,
    fog: &wgpu::TextureView,
) -> (wgpu::Texture, Vec<wgpu::TextureView>, wgpu::BindGroup, wgpu::BindGroup) {
    let shadow_map = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow-map"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: CASCADES as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::context::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let cascade_views = (0..CASCADES)
        .map(|i| {
            shadow_map.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow-cascade"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();

    let array_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
        label: Some("shadow-array"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::Greater),
        ..Default::default()
    });

    let fog_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&array_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(fog) },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&fog_sampler),
            },
        ],
    });

    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light-compute-bg"),
        layout: compute_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&array_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    (shadow_map, cascade_views, bind_group, compute_bind_group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_rises_peaks_and_sets() {
        assert!(Sun::at(6.0).direction.y.abs() < 0.05, "level at sunrise");
        assert!(Sun::at(12.0).direction.y > 0.9, "high at noon");
        assert!(Sun::at(18.0).direction.y.abs() < 0.05, "level at sunset");
        // Sunrise in the east, sunset in the west.
        assert!(Sun::at(7.0).direction.x > 0.0);
        assert!(Sun::at(17.0).direction.x < 0.0);
    }

    #[test]
    fn night_keeps_a_key_light_above_the_horizon() {
        // A light pointing into the ground lights nothing, and the world goes
        // black rather than dark.
        for hour in [0.0, 1.0, 22.0, 23.5] {
            let s = Sun::at(hour);
            assert!(s.night, "{hour} should be night");
            assert!(s.direction.y > 0.0, "{hour}: moon must be above the horizon");
            assert!(s.intensity() > 0.0);
        }
    }

    #[test]
    fn daylight_fades_through_twilight_rather_than_switching() {
        let dawn = Sun::at(5.6).daylight;
        let day = Sun::at(9.0).daylight;
        let midnight = Sun::at(0.0).daylight;
        assert_eq!(midnight, 0.0);
        assert!(day > 0.99);
        assert!(dawn > 0.0 && dawn < 1.0, "twilight must be partial, got {dawn}");
    }

    #[test]
    fn low_sun_is_warm_and_high_sun_is_not() {
        let dawn = Sun::at(6.6).color;
        let noon = Sun::at(12.0).color;
        assert!(dawn.x - dawn.z > 0.4, "low sun should be strongly warm: {dawn:?}");
        assert!((noon.x - noon.z).abs() < 0.15, "noon should be near white: {noon:?}");
    }

    /// The shadow entry points share a module with the shading passes, so the
    /// cascade uniform has to match what those declare at group 0. A shorter
    /// binding is a validation error the moment a world is open -- and never
    /// before, which is how it survived being added.
    #[test]
    fn cascade_uniform_matches_the_camera_layout() {
        assert_eq!(
            std::mem::size_of::<CascadeUniform>(),
            std::mem::size_of::<crate::camera::CameraUniform>(),
        );
    }

    #[test]
    fn shadow_projection_is_reversed_z() {
        // Near must map to 1 and far to 0, matching the scene's reversed-Z.
        // Getting this backwards stores the farthest caster per texel, and
        // every shadow comes out inverted.
        let p = ortho_reversed(50.0, 200.0);
        let near = p * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        let far = p * glam::Vec4::new(0.0, 0.0, -200.0, 1.0);
        assert!((near.z / near.w - 1.0).abs() < 1e-5, "near should be 1, got {}", near.z / near.w);
        assert!((far.z / far.w).abs() < 1e-5, "far should be 0, got {}", far.z / far.w);
    }

    #[test]
    fn shadow_projection_keeps_the_slice_in_range() {
        let p = ortho_reversed(50.0, 200.0);
        for (x, y) in [(-50.0, -50.0), (50.0, 50.0), (0.0, 0.0)] {
            let c = p * glam::Vec4::new(x, y, -100.0, 1.0);
            assert!(c.x.abs() <= 1.0 + 1e-5 && c.y.abs() <= 1.0 + 1e-5, "{c:?} outside clip");
        }
    }

    #[test]
    fn look_at_faces_the_target() {
        let eye = Vec3::new(10.0, 40.0, -5.0);
        let target = Vec3::ZERO;
        let m = look_at(eye, target, Vec3::Y);
        let seen = m * target.extend(1.0);
        // The target must land on the view axis, in front of the eye.
        assert!(seen.x.abs() < 1e-4 && seen.y.abs() < 1e-4, "{seen:?}");
        assert!(seen.z < 0.0, "target must be in front: {seen:?}");
    }

    #[test]
    fn cascade_splits_increase_and_reach_the_distance() {
        let l = SkySettings { shadow_distance: 400.0, ..Default::default() };
        let splits = Lighting::splits_for(&l);
        for w in splits.windows(2) {
            assert!(w[1] > w[0], "splits must increase: {splits:?}");
        }
        assert!((splits[CASCADES] - 400.0).abs() < 1.0);
        // Near cascade must be much tighter than the far one, or the detail
        // near the camera is wasted.
        assert!(splits[1] - splits[0] < (splits[CASCADES] - splits[CASCADES - 1]) * 0.5);
    }
}

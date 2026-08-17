//! The Environment Light Mixer: one struct for everything that lights the world.
//!
//! Sun, sky, ambient bounce, fog, clouds and tone mapping were six settings
//! blocks in four places -- `SkySettings` carried the sun *and* the shadows *and*
//! the exposure, `FogSettings` carried the fog, quality presets reached into
//! both, and the UI had a section per struct with no ordering between them. The
//! practical failure was that they interact: raising fog density without
//! touching exposure darkens the whole frame, and a user adjusting one at a time
//! could not see why.
//!
//! So this is deliberately one struct, in the order light physically arrives:
//!
//! ```text
//! Sun          the directional source, or the moon at night
//! Atmosphere   Rayleigh + Mie scattering, which makes the sky blue and the horizon pale
//! Sky light    the hemisphere bounce that fills shadow -- ambient, but directional
//! Fog          exponential height fog, and the god rays that come from marching it
//! Clouds       volumetric layer between the sun and the ground
//! Tone map     the transfer from radiance to pixels
//! ```
//!
//! # Physical units, and where they stop
//!
//! Scattering coefficients are per metre and the Rayleigh values are the real
//! ones for air at sea level, because those are what make a sky read as sky
//! rather than as a blue gradient. Everything downstream of the atmosphere --
//! cloud coverage, god ray intensity, tone map contrast -- is an artistic dial
//! with no physical claim, and is documented as such rather than dressed up in
//! units it does not respect.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Rayleigh scattering coefficients for air at sea level, per metre, for
/// wavelengths around 680/550/440 nm.
///
/// These specific numbers are why the sky is blue and the horizon is pale: blue
/// scatters roughly 5.5x as strongly as red, so a long path through air loses
/// its blue to the sky and arrives reddened. Replacing them with a hand-picked
/// gradient is the single most common way a sky ends up looking like a
/// screensaver.
pub const RAYLEIGH_SEA_LEVEL: Vec3 = Vec3::new(5.802e-6, 13.558e-6, 33.1e-6);

/// Mie scattering for haze and aerosol, per metre. Nearly wavelength-neutral,
/// which is why haze greys a view rather than tinting it.
pub const MIE_SEA_LEVEL: f32 = 3.996e-6;

/// Ozone absorption, per metre. Small, and the reason a clear zenith trends
/// toward violet at dusk rather than simply darkening.
pub const OZONE_ABSORPTION: Vec3 = Vec3::new(0.650e-6, 1.881e-6, 0.085e-6);

// ---------------------------------------------------------------------------
// Sub-settings
// ---------------------------------------------------------------------------

/// The directional source. One light that is the sun by day and the moon by
/// night, rather than two that have to be kept from both being on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SunLight {
    /// Degrees above the horizon. Negative is below, at which point the moon
    /// takes over as the key light.
    pub pitch_deg: f32,
    /// Compass bearing in degrees.
    pub yaw_deg: f32,
    /// Radiance multiplier. 1 is the calibrated daylight value.
    pub intensity: f32,
    /// Tint applied on top of the physical colour, for grading.
    pub tint: Vec3,
    /// Angular diameter in degrees. The real sun is about 0.53; larger softens
    /// every shadow edge in the scene at once, which is the cheapest way to
    /// make an overcast day read as overcast.
    pub angular_diameter_deg: f32,
    /// Fraction of the sun's intensity the moon gets. The real ratio is about
    /// 1/400,000, which is unusable: a night lit that faithfully is black.
    pub moon_intensity: f32,
    pub casts_shadows: bool,
}

impl Default for SunLight {
    fn default() -> Self {
        Self {
            // Unreal's Quick Create default, and a good one: high enough to
            // light both sides of a ridge, low enough to cast a readable shadow.
            pitch_deg: -45.0,
            yaw_deg: 135.0,
            intensity: 1.0,
            tint: Vec3::ONE,
            angular_diameter_deg: 0.53,
            moon_intensity: 0.02,
            casts_shadows: true,
        }
    }
}

impl SunLight {
    /// Unit vector pointing *toward* the light.
    ///
    /// Pitch is measured from the horizon, so a pitch of -45 degrees is the sun
    /// 45 degrees *up* in the sky. The sign convention follows Unreal's
    /// directional light, where the rotation describes the direction light
    /// travels and the vector toward the source is its negation.
    pub fn direction(&self) -> Vec3 {
        let pitch = -self.pitch_deg.to_radians();
        let yaw = self.yaw_deg.to_radians();
        Vec3::new(pitch.cos() * yaw.cos(), pitch.sin(), pitch.cos() * yaw.sin()).normalize()
    }

    /// Above the horizon, 0 at or below it.
    pub fn daylight(&self) -> f32 {
        self.direction().y.max(0.0)
    }

    pub fn is_night(&self) -> bool {
        self.direction().y <= 0.0
    }

    /// The key light's direction, which flips to the moon's after dusk.
    pub fn key_direction(&self) -> Vec3 {
        let d = self.direction();
        if d.y <= 0.0 { -d } else { d }
    }

    /// Radiance of the key light, physical colour times tint.
    ///
    /// Warm and weak near the horizon, white and strong overhead: the low sun is
    /// seen through a long air path that has scattered its blue away, which is
    /// the same Rayleigh fact the sky colour comes from.
    pub fn radiance(&self) -> Vec3 {
        let d = self.key_direction();
        let elevation = d.y.clamp(0.0, 1.0);
        if self.is_night() {
            // Moonlight is sunlight off a grey rock, so it is neutral in
            // reality; the blue is a convention the eye reads as night.
            return Vec3::new(0.55, 0.68, 1.0) * self.moon_intensity * self.intensity;
        }
        let warm = Vec3::new(1.0, 0.62, 0.32);
        let white = Vec3::new(1.0, 0.98, 0.95);
        let t = elevation.powf(0.45);
        (warm + (white - warm) * t) * self.intensity * self.tint
    }
}

/// Rayleigh and Mie scattering. What makes the sky a sky.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SkyAtmosphere {
    pub enabled: bool,
    /// Multiplier on [`RAYLEIGH_SEA_LEVEL`]. 1 is Earth.
    pub rayleigh_scale: f32,
    /// Multiplier on [`MIE_SEA_LEVEL`]. This is the haze dial: raising it greys
    /// the horizon and brightens the region around the sun.
    pub mie_scale: f32,
    /// Mie forward scattering, 0 isotropic to ~0.95 sharply forward. This is
    /// what puts a bright halo around a low sun.
    pub mie_anisotropy: f32,
    /// Multiplier on [`OZONE_ABSORPTION`].
    pub ozone_scale: f32,
    /// Scale height of the air column, in metres: the height at which density
    /// has fallen to 1/e.
    pub rayleigh_height_m: f32,
    /// Scale height of the aerosol column, in metres. Much lower than the air's,
    /// which is why haze sits in the lower sky.
    pub mie_height_m: f32,
    /// Ground albedo, which bounces back up into the sky.
    pub ground_albedo: Vec3,
}

impl Default for SkyAtmosphere {
    fn default() -> Self {
        Self {
            enabled: true,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            mie_anisotropy: 0.76,
            ozone_scale: 1.0,
            rayleigh_height_m: 8000.0,
            mie_height_m: 1200.0,
            ground_albedo: Vec3::splat(0.1),
        }
    }
}

impl SkyAtmosphere {
    /// Effective Rayleigh coefficients, per metre.
    pub fn rayleigh(&self) -> Vec3 {
        RAYLEIGH_SEA_LEVEL * self.rayleigh_scale.max(0.0)
    }

    pub fn mie(&self) -> f32 {
        MIE_SEA_LEVEL * self.mie_scale.max(0.0)
    }

    pub fn ozone(&self) -> Vec3 {
        OZONE_ABSORPTION * self.ozone_scale.max(0.0)
    }
}

/// The hemisphere bounce that fills shadow.
///
/// "Ambient" in the sense that it has no single direction, but not a constant:
/// a zenith tint and a horizon tint, so a shadowed north face reads cool from
/// the sky and a shadowed underside reads warm from the ground. A single flat
/// ambient term is what makes shadowed geometry look pasted on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SkyLight {
    pub enabled: bool,
    pub intensity: f32,
    /// Sky colour looking straight up.
    pub zenith: Vec3,
    /// Sky colour at the horizon.
    pub horizon: Vec3,
    /// Bounce colour from below, which is the ground, not the sky.
    pub ground: Vec3,
    /// Follow the atmosphere rather than the explicit colours above. On by
    /// default: a sky light that does not track its own sky is the most common
    /// way an evening scene ends up lit like noon.
    pub capture_from_atmosphere: bool,
}

impl Default for SkyLight {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: 1.0,
            zenith: Vec3::new(0.24, 0.42, 0.78),
            horizon: Vec3::new(0.68, 0.76, 0.88),
            ground: Vec3::new(0.22, 0.19, 0.16),
            capture_from_atmosphere: true,
        }
    }
}

/// Exponential height fog, and the god rays that come from marching it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HeightFog {
    pub enabled: bool,
    /// Extinction at the reference height, per metre.
    pub density: f32,
    /// Height at which density has fallen to 1/e, in metres. Small values pool
    /// fog in valleys; large ones fill the whole air column.
    pub height_falloff_m: f32,
    /// Height the density is quoted at.
    pub base_height_m: f32,
    /// Forward scattering, 0 isotropic to near 1. What makes looking toward the
    /// sun through fog bright and away from it flat.
    pub anisotropy: f32,
    /// Colour of the medium.
    pub albedo: Vec3,
    /// Furthest the froxel grid reaches, in metres.
    pub distance_m: f32,
    /// Volumetric god ray strength. 0 skips the pass.
    pub god_rays: f32,
    /// Extra density that pools in low ground, independent of the exponential
    /// term. Valley mist is a separate phenomenon from air extinction and
    /// folding it into `density` made both unusable.
    pub mist_strength: f32,
}

impl Default for HeightFog {
    fn default() -> Self {
        Self {
            enabled: true,
            // The Quick Create value. Thin enough to leave a clear day clear,
            // thick enough that distance reads as distance.
            density: 0.002,
            height_falloff_m: 400.0,
            base_height_m: 280.0,
            anisotropy: 0.72,
            albedo: Vec3::new(0.72, 0.78, 0.86),
            distance_m: 450.0,
            god_rays: 0.35,
            mist_strength: 0.0008,
        }
    }
}

/// How many samples the cloud march is allowed.
///
/// A dial rather than a constant because the march is by far the most expensive
/// thing in the sky -- measured at 1280x720, Medium costs about 10 ms against
/// the sky's 2.4 -- and what it buys is edge detail that a distant layer cannot
/// show anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudQuality {
    /// Half the samples. Soft edges, and the only setting that fits a tight
    /// frame budget without a half-resolution pass.
    Low,
    Medium,
    High,
}

impl CloudQuality {
    pub const ALL: [CloudQuality; 3] =
        [CloudQuality::Low, CloudQuality::Medium, CloudQuality::High];

    pub fn label(self) -> &'static str {
        match self {
            CloudQuality::Low => "Low",
            CloudQuality::Medium => "Medium",
            CloudQuality::High => "High",
        }
    }

    /// Multiplier on the base step count.
    pub fn step_scale(self) -> f32 {
        match self {
            CloudQuality::Low => 0.5,
            CloudQuality::Medium => 1.0,
            CloudQuality::High => 1.75,
        }
    }
}

/// The volumetric cloud layer.
///
/// Settings and uniform only: **no cloud pass renders yet.** They are here so
/// the mixer is the single place environment state lives rather than growing a
/// second home later, and so the uniform layout is fixed before a shader
/// depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VolumetricClouds {
    pub enabled: bool,
    /// Fraction of sky covered, 0 clear to 1 overcast.
    pub coverage: f32,
    /// Bottom of the layer, in metres above sea level.
    pub base_m: f32,
    /// Layer thickness, in metres.
    pub thickness_m: f32,
    /// Extinction per metre inside a cloud.
    pub density: f32,
    /// Metres per repeat of the shape noise.
    pub feature_scale_m: f32,
    /// Wind, in metres per second, which advects the layer.
    pub wind: Vec3,
    pub quality: CloudQuality,
}

impl Default for VolumetricClouds {
    fn default() -> Self {
        Self {
            // Off by default: a cloud pass is the most expensive thing in a sky,
            // and nothing draws it yet.
            enabled: false,
            coverage: 0.45,
            base_m: 1500.0,
            thickness_m: 2800.0,
            density: 0.05,
            feature_scale_m: 12_000.0,
            wind: Vec3::new(6.0, 0.0, 2.0),
            quality: CloudQuality::Medium,
        }
    }
}

/// How radiance becomes pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToneMapper {
    /// Straight clamp. Useful only for checking whether something is blowing
    /// out; everything bright goes flat white.
    None,
    /// Reinhard. Never clips, but desaturates highlights badly.
    Reinhard,
    /// ACES filmic. The default, and what Unreal ships: highlights roll off
    /// while keeping their hue, so a sun disc stays yellow instead of turning
    /// into a white hole.
    Aces,
}

impl ToneMapper {
    pub const ALL: [ToneMapper; 3] = [ToneMapper::Aces, ToneMapper::Reinhard, ToneMapper::None];

    pub fn label(self) -> &'static str {
        match self {
            ToneMapper::None => "None",
            ToneMapper::Reinhard => "Reinhard",
            ToneMapper::Aces => "ACES",
        }
    }

    /// Index the shader switches on.
    pub fn index(self) -> u32 {
        match self {
            ToneMapper::None => 0,
            ToneMapper::Reinhard => 1,
            ToneMapper::Aces => 2,
        }
    }
}

/// Post-process transfer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToneMapping {
    pub mapper: ToneMapper,
    /// Stops of exposure compensation.
    pub exposure_ev: f32,
    pub contrast: f32,
    pub saturation: f32,
    /// White balance in Kelvin. 6500 is neutral.
    pub white_balance_k: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            mapper: ToneMapper::Aces,
            exposure_ev: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            white_balance_k: 6500.0,
        }
    }
}

// ---------------------------------------------------------------------------
// The mixer
// ---------------------------------------------------------------------------

/// Everything that lights the world, in one place.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Environment {
    pub sun: SunLight,
    pub atmosphere: SkyAtmosphere,
    pub sky_light: SkyLight,
    pub fog: HeightFog,
    pub clouds: VolumetricClouds,
    pub tone: ToneMapping,
    /// Hours, 0..24, when the day/night cycle is driving the sun.
    pub time_of_day: f32,
    /// Hours per real second while running.
    pub day_speed: f32,
    pub cycle_running: bool,
    /// Show the world's own time of day in the editor viewport rather than a
    /// fixed neutral sun. Off by default: the time is authored for play, and
    /// previewing it while building fights the work.
    pub editor_preview: bool,
}

impl Default for Environment {
    fn default() -> Self {
        Self::daylight()
    }
}

impl Environment {
    /// Write the mixer to `edits/environment.ron`.
    ///
    /// Saved with the world rather than held in memory, because everything in here
    /// is authored: a user who spends ten minutes finding the right dusk and fog
    /// has done real work, and losing it on close would make the panel a toy.
    ///
    /// RON with struct names, matching the other documents, so the file stays
    /// hand-editable and a diff is readable.
    pub fn save(&self, paths: &terra_project::ProjectPaths) -> std::io::Result<()> {
        let path = paths.environment();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(self, cfg)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, text)
    }

    /// Read it back, or `None` when there is nothing saved.
    ///
    /// `None` rather than an error for a missing file: a world created before this
    /// existed, or one never saved, simply has no authored environment and should
    /// open on [`Environment::daylight`]. A *corrupt* file is different -- it is
    /// logged and also treated as absent, because refusing to open a world over a
    /// bad lighting file would be a worse failure than losing the lighting.
    ///
    /// Every struct here carries `#[serde(default)]`, so a file written before a
    /// field existed still loads and that field takes its default. Without it,
    /// adding one setting would silently reset every world's environment.
    pub fn load(paths: &terra_project::ProjectPaths) -> Option<Self> {
        let path = paths.environment();
        let text = std::fs::read_to_string(&path).ok()?;
        match ron::from_str(&text) {
            Ok(env) => Some(env),
            Err(e) => {
                log::warn!("ignoring unreadable {}: {e}", path.display());
                None
            }
        }
    }

    /// Whether this differs from what was last written to disk in a way worth
    /// saving.
    ///
    /// Needed because the mixer panel edits the live `Environment` through `&mut`
    /// and there is no single place a widget could set a dirty flag -- and adding
    /// one to each widget is exactly how the mixer and its derived settings used to
    /// fall out of step.
    ///
    /// `time_of_day` is excluded while the cycle is running on both sides: it
    /// advances every frame by design, and counting that as an edit would leave the
    /// editor permanently dirty and rewrite the whole world on every exit. Scrubbing
    /// the time with the cycle stopped *is* an edit and does count, as does starting
    /// or stopping the cycle.
    ///
    /// The sun's pitch and yaw go with it, because [`Self::tick`] re-derives them
    /// from the clock: while the cycle runs they are outputs, not settings, and
    /// comparing two different instants would report an edit every frame. Both sides
    /// are re-synced to the same time rather than blanked, so a genuine edit to
    /// anything else about the sun -- intensity, tint, shadows -- still registers.
    pub fn differs_for_saving(&self, saved: &Self) -> bool {
        if self.cycle_running && saved.cycle_running {
            let (mut a, mut b) = (*self, *saved);
            a.time_of_day = 0.0;
            b.time_of_day = 0.0;
            a.sync_sun_to_clock();
            b.sync_sun_to_clock();
            a != b
        } else {
            self != saved
        }
    }

    /// Standard daylight. The Quick Create action.
    ///
    /// The point is that an empty project looks physically grounded
    /// immediately: sun at -45 degrees, real Rayleigh coefficients so the sky is
    /// the blue air actually is, thin exponential fog so distance reads as
    /// distance, and ACES so none of it clips. Every one of those is a default
    /// rather than a thing to discover.
    pub fn daylight() -> Self {
        Self {
            sun: SunLight::default(),
            atmosphere: SkyAtmosphere::default(),
            sky_light: SkyLight::default(),
            fog: HeightFog::default(),
            clouds: VolumetricClouds::default(),
            tone: ToneMapping::default(),
            time_of_day: 10.5,
            day_speed: 0.25,
            cycle_running: false,
            editor_preview: false,
        }
    }

    /// Reset to the daylight defaults, keeping nothing.
    ///
    /// The full reset, as opposed to [`Self::apply_preset`]: this is the button
    /// for "put everything back", so it deliberately does discard the toggles a
    /// preset would have kept.
    pub fn reset(&mut self) {
        *self = Self::daylight();
    }

    /// Adopt a preset's *look* without undoing what the user switched on.
    ///
    /// Assigning the preset wholesale is what the Quick Create buttons used to
    /// do, and it silently turned clouds back off: switch clouds on, click
    /// Daylight to fix the lighting, and the clouds vanish with no indication
    /// that the button did it.
    ///
    /// So the feature toggles are unioned rather than replaced -- a preset can
    /// turn something on, never off. Overcast still enables clouds because being
    /// overcast means having them; Daylight leaves them however they were. Cloud
    /// quality is preserved outright, since it is a cost choice about the
    /// machine and not part of any look.
    pub fn apply_preset(&mut self, preset: Self) {
        let clouds_on = self.clouds.enabled;
        let quality = self.clouds.quality;
        let atmosphere_on = self.atmosphere.enabled;
        let sky_light_on = self.sky_light.enabled;
        let fog_on = self.fog.enabled;

        *self = preset;

        self.clouds.enabled |= clouds_on;
        self.clouds.quality = quality;
        self.atmosphere.enabled |= atmosphere_on;
        self.sky_light.enabled |= sky_light_on;
        self.fog.enabled |= fog_on;
    }

    /// Overcast: sun buried, haze up, clouds on, fog thicker.
    ///
    /// A second preset rather than a slider, for the same reason quality is a
    /// preset: the settings interact, and half-overcast is not a look.
    pub fn overcast() -> Self {
        let mut e = Self::daylight();
        e.sun.intensity = 0.35;
        e.sun.angular_diameter_deg = 8.0;
        e.atmosphere.mie_scale = 6.0;
        e.fog.density = 0.008;
        e.fog.god_rays = 0.1;
        e.clouds.enabled = true;
        e.clouds.coverage = 0.85;
        e.sky_light.intensity = 1.4;
        e
    }

    /// Night: moon as key, cool ambient, no god rays worth the pass.
    pub fn night() -> Self {
        let mut e = Self::daylight();
        e.sun.pitch_deg = 20.0;
        e.time_of_day = 23.0;
        e.fog.god_rays = 0.0;
        e.sky_light.zenith = Vec3::new(0.04, 0.06, 0.14);
        e.sky_light.horizon = Vec3::new(0.08, 0.10, 0.18);
        e.tone.exposure_ev = 1.5;
        e
    }

    /// Advance the cycle, and keep the sun in step with the clock.
    pub fn tick(&mut self, dt: f32) {
        if !self.cycle_running {
            return;
        }
        self.time_of_day = (self.time_of_day + self.day_speed * dt).rem_euclid(24.0);
        self.sync_sun_to_clock();
    }

    /// Derive the sun's pitch from the time of day.
    ///
    /// Sunrise at 06:00, noon overhead, sunset at 18:00. The yaw sweeps with the
    /// clock too, or the shadows would all point the same way all day.
    pub fn sync_sun_to_clock(&mut self) {
        let phase = (self.time_of_day - 6.0) / 12.0 * std::f32::consts::PI;
        // Elevation peaks at 75 rather than 90 degrees: a sun straight overhead
        // flattens every slope at noon.
        self.sun.pitch_deg = -(phase.sin() * 75.0);
        self.sun.yaw_deg = 90.0 + (self.time_of_day - 6.0) / 12.0 * 180.0;
    }

    /// Ambient hemisphere tints, taking the atmosphere into account when the sky
    /// light is set to follow it.
    ///
    /// Returns `(zenith, horizon, ground)`.
    pub fn ambient_tints(&self) -> (Vec3, Vec3, Vec3) {
        if !self.sky_light.enabled {
            return (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        }
        let k = self.sky_light.intensity;
        if !self.sky_light.capture_from_atmosphere {
            return (
                self.sky_light.zenith * k,
                self.sky_light.horizon * k,
                self.sky_light.ground * k,
            );
        }
        // Derived from the scattering coefficients rather than authored: the
        // zenith is what Rayleigh leaves after a short path, the horizon what a
        // long path leaves once Mie has greyed it.
        let r = self.atmosphere.rayleigh();
        let unit = r / r.max_element().max(1e-12);
        let haze = (self.atmosphere.mie_scale / 6.0).clamp(0.0, 1.0);
        let level = self.sky_level();
        let zenith = unit * level * k;
        let horizon = (unit.lerp(Vec3::ONE, 0.55 + haze * 0.4)) * level * k;
        let ground = self.atmosphere.ground_albedo * horizon;
        (zenith, horizon, ground)
    }

    /// How much light the sky is giving, as a function of sun elevation.
    ///
    /// Deliberately continuous *through* the horizon. Clamping at elevation zero
    /// -- which is what `daylight()` does -- made dusk and midnight produce the
    /// identical ambient, so the ground dropped to full night the instant the sun
    /// set and stayed there. Civil twilight is bright for a good while after
    /// sunset, and that ramp is most of what makes an evening read as an evening.
    pub fn sky_level(&self) -> f32 {
        let elevation = self.sun.direction().y;
        let day = elevation.max(0.0);
        // Reaches zero around twelve degrees below the horizon, which is roughly
        // where nautical twilight ends.
        let twilight = ((elevation + 0.21) / 0.21).clamp(0.0, 1.0);
        // The floor is moonlight. Not zero: a scene lit to nothing is a black
        // screen, and the eye reads a dim blue night as night perfectly well.
        (day + 0.12 * twilight).max(0.015)
    }

    /// Linear exposure multiplier from the EV compensation.
    pub fn exposure(&self) -> f32 {
        2f32.powf(self.tone.exposure_ev)
    }

    /// Pack for the GPU, with the animation clock at zero.
    pub fn uniform(&self) -> EnvironmentUniform {
        self.uniform_with_time(0.0)
    }

    /// Pack for the GPU. `time` is seconds since start and drives wind.
    pub fn uniform_with_time(&self, time: f32) -> EnvironmentUniform {
        let (zenith, horizon, ground) = self.ambient_tints();
        let key = self.sun.key_direction();
        let radiance = self.sun.radiance();
        let r = self.atmosphere.rayleigh();
        let o = self.atmosphere.ozone();

        EnvironmentUniform {
            sun_direction: [key.x, key.y, key.z, self.sun.daylight()],
            sun_radiance: [radiance.x, radiance.y, radiance.z, self.sun.angular_diameter_deg],
            rayleigh: [r.x, r.y, r.z, self.atmosphere.rayleigh_height_m],
            mie: [
                self.atmosphere.mie(),
                self.atmosphere.mie_height_m,
                self.atmosphere.mie_anisotropy,
                if self.atmosphere.enabled { 1.0 } else { 0.0 },
            ],
            ozone: [o.x, o.y, o.z, 0.0],
            ambient_zenith: [zenith.x, zenith.y, zenith.z, self.sky_light.intensity],
            ambient_horizon: [horizon.x, horizon.y, horizon.z, 0.0],
            ambient_ground: [ground.x, ground.y, ground.z, 0.0],
            fog_params: [
                if self.fog.enabled { self.fog.density } else { 0.0 },
                self.fog.height_falloff_m,
                self.fog.base_height_m,
                self.fog.anisotropy,
            ],
            fog_albedo: [
                self.fog.albedo.x,
                self.fog.albedo.y,
                self.fog.albedo.z,
                self.fog.distance_m,
            ],
            fog_extra: [self.fog.god_rays, self.fog.mist_strength, 0.0, 0.0],
            cloud_params: [
                if self.clouds.enabled { self.clouds.coverage } else { 0.0 },
                self.clouds.base_m,
                self.clouds.thickness_m,
                self.clouds.density,
            ],
            cloud_wind: [
                self.clouds.wind.x,
                self.clouds.wind.y,
                self.clouds.wind.z,
                self.clouds.feature_scale_m,
            ],
            tone: [
                self.exposure(),
                self.tone.contrast,
                self.tone.saturation,
                self.tone.white_balance_k,
            ],
            flags: [
                self.tone.mapper.index(),
                u32::from(self.sun.casts_shadows),
                u32::from(self.clouds.enabled),
                u32::from(self.sky_light.enabled),
            ],
            frame: [time, self.clouds.quality.step_scale(), 0.0, 0.0],
        }
    }
}

/// The uniform buffer and bind group the sky and cloud passes read.
pub struct EnvironmentGpu {
    buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl EnvironmentGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("environment"),
            size: std::mem::size_of::<EnvironmentUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("environment-bgl"),
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("environment-bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        Self { buffer, layout, bind_group }
    }

    /// Push the current state. Called once a frame, before any pass reads it.
    pub fn upload(&self, queue: &wgpu::Queue, env: &Environment, time: f32) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&env.uniform_with_time(time)));
    }
}

// ---------------------------------------------------------------------------
// GPU layout
// ---------------------------------------------------------------------------

/// The packed block the shaders read.
///
/// Every field is a `vec4` or a `vec4u`. That is not tidiness: WGSL rounds
/// uniform struct members up to 16-byte alignment, so a lone `f32` between two
/// vectors silently inserts 12 bytes of padding on the shader side that Rust
/// does not, and the whole block reads shifted from that point on. Packing four
/// related scalars into each vector makes the two layouts agree by construction
/// and is why the `assert!` below can be a real check rather than a hope.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct EnvironmentUniform {
    /// `xyz` toward the key light, `w` daylight (0 at night).
    pub sun_direction: [f32; 4],
    /// `rgb` radiance, `w` angular diameter in degrees.
    pub sun_radiance: [f32; 4],
    /// `rgb` Rayleigh scattering per metre, `w` air scale height in metres.
    pub rayleigh: [f32; 4],
    /// `x` Mie per metre, `y` aerosol scale height, `z` anisotropy, `w` enabled.
    pub mie: [f32; 4],
    /// `rgb` ozone absorption per metre.
    pub ozone: [f32; 4],
    /// `rgb` zenith tint, `w` sky light intensity.
    pub ambient_zenith: [f32; 4],
    /// `rgb` horizon tint.
    pub ambient_horizon: [f32; 4],
    /// `rgb` bounce from the ground.
    pub ambient_ground: [f32; 4],
    /// `x` density per metre, `y` height falloff, `z` base height, `w` anisotropy.
    pub fog_params: [f32; 4],
    /// `rgb` medium albedo, `w` grid distance in metres.
    pub fog_albedo: [f32; 4],
    /// `x` god ray strength, `y` valley mist strength.
    pub fog_extra: [f32; 4],
    /// `x` coverage, `y` base height, `z` thickness, `w` density.
    pub cloud_params: [f32; 4],
    /// `xyz` wind in m/s, `w` feature scale in metres.
    pub cloud_wind: [f32; 4],
    /// `x` linear exposure, `y` contrast, `z` saturation, `w` white balance K.
    pub tone: [f32; 4],
    /// `x` tone mapper index, `y` shadows, `z` clouds, `w` sky light.
    pub flags: [u32; 4],
    /// `x` seconds since start (wind advection), `y` cloud step scale. `zw` reserved.
    ///
    /// Per-frame rather than per-setting, which is why it is filled by
    /// [`EnvironmentGpu::upload`] rather than by the settings struct: a cloud
    /// layer that does not advect looks painted on.
    pub frame: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<EnvironmentUniform>() == 256);
const _: () = assert!(std::mem::size_of::<EnvironmentUniform>().is_multiple_of(16));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_create_matches_the_documented_defaults() {
        // The four values the Quick Create action promises. If any drifts, an
        // empty project stops looking physically grounded on arrival.
        let e = Environment::daylight();
        assert_eq!(e.sun.pitch_deg, -45.0, "sun pitch");
        assert_eq!(e.fog.density, 0.002, "fog density");
        assert_eq!(e.tone.mapper, ToneMapper::Aces, "tonemapper");
        assert_eq!(e.atmosphere.rayleigh(), RAYLEIGH_SEA_LEVEL, "Rayleigh must be the real values");
        assert!(e.atmosphere.enabled && e.fog.enabled && e.sky_light.enabled);
    }

    #[test]
    fn rayleigh_makes_the_sky_blue() {
        // Not decoration: blue must scatter several times more strongly than
        // red, or the sky is a gradient someone picked.
        let r = RAYLEIGH_SEA_LEVEL;
        assert!(r.z > r.x * 4.0, "blue {} should dominate red {}", r.z, r.x);
        assert!(r.y > r.x, "green should exceed red");
    }

    #[test]
    fn negative_pitch_puts_the_sun_up() {
        // The sign convention is the easiest thing here to get backwards, and
        // getting it backwards lights the world from underground.
        let e = Environment::daylight();
        let d = e.sun.direction();
        assert!(d.y > 0.6, "pitch -45 should be well above the horizon, got y = {}", d.y);
        assert!(!e.sun.is_night());
        assert!((d.length() - 1.0).abs() < 1e-5, "direction must be normalized");
    }

    #[test]
    fn positive_pitch_is_night_and_the_moon_takes_over() {
        let mut e = Environment::daylight();
        e.sun.pitch_deg = 30.0;
        assert!(e.sun.is_night());
        assert_eq!(e.sun.daylight(), 0.0);
        // The key light must flip to the opposite side, or a night scene is lit
        // from below the ground.
        assert!(e.sun.key_direction().y > 0.0, "the moon must be above the horizon");
        let r = e.sun.radiance();
        assert!(r.z > r.x, "moonlight should read cool");
        assert!(r.length() > 0.0, "a night lit to zero is a black screen");
    }

    #[test]
    fn a_low_sun_is_warmer_than_a_high_one() {
        // Same Rayleigh fact as the blue sky: a long air path reddens the light.
        let mut low = Environment::daylight();
        low.sun.pitch_deg = -3.0;
        let mut high = Environment::daylight();
        high.sun.pitch_deg = -85.0;

        let ratio = |v: Vec3| v.x / v.z.max(1e-6);
        assert!(
            ratio(low.sun.radiance()) > ratio(high.sun.radiance()) * 1.5,
            "low sun R/B {} should exceed high sun {}",
            ratio(low.sun.radiance()),
            ratio(high.sun.radiance())
        );
    }

    #[test]
    fn the_clock_drives_the_sun() {
        let mut e = Environment::daylight();
        e.time_of_day = 12.0;
        e.sync_sun_to_clock();
        let noon = e.sun.direction().y;

        e.time_of_day = 6.5;
        e.sync_sun_to_clock();
        let dawn = e.sun.direction().y;

        assert!(noon > dawn, "noon ({noon}) should be higher than dawn ({dawn})");
        assert!(dawn > 0.0, "06:30 should still be daylight");

        e.time_of_day = 23.0;
        e.sync_sun_to_clock();
        assert!(e.sun.is_night(), "23:00 must be night");
    }

    #[test]
    fn the_cycle_only_advances_when_running() {
        let mut e = Environment::daylight();
        let before = e.time_of_day;
        e.tick(1.0);
        assert_eq!(e.time_of_day, before, "a stopped cycle must not move");

        e.cycle_running = true;
        e.tick(1.0);
        assert!((e.time_of_day - before - e.day_speed).abs() < 1e-5);
    }

    #[test]
    fn the_clock_wraps_at_midnight() {
        let mut e = Environment::daylight();
        e.cycle_running = true;
        e.time_of_day = 23.9;
        e.day_speed = 1.0;
        e.tick(0.5);
        assert!((0.0..24.0).contains(&e.time_of_day), "got {}", e.time_of_day);
        assert!(e.time_of_day < 1.0, "should have wrapped past midnight, got {}", e.time_of_day);
    }

    #[test]
    fn ambient_follows_the_atmosphere_when_asked_to() {
        let mut e = Environment::daylight();
        assert!(e.sky_light.capture_from_atmosphere);
        let (zenith, horizon, _) = e.ambient_tints();
        // Derived from Rayleigh, so the zenith must be the bluer of the two.
        let blueness = |v: Vec3| v.z / (v.x + v.y + v.z).max(1e-6);
        assert!(
            blueness(zenith) > blueness(horizon),
            "zenith {zenith} should be bluer than horizon {horizon}"
        );

        // Pinned to the authored colours instead, it must use them verbatim.
        e.sky_light.capture_from_atmosphere = false;
        e.sky_light.intensity = 1.0;
        let (z, h, g) = e.ambient_tints();
        assert_eq!(z, e.sky_light.zenith);
        assert_eq!(h, e.sky_light.horizon);
        assert_eq!(g, e.sky_light.ground);
    }

    #[test]
    fn a_disabled_sky_light_contributes_nothing() {
        let mut e = Environment::daylight();
        e.sky_light.enabled = false;
        assert_eq!(e.ambient_tints(), (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO));
        assert_eq!(e.uniform().flags[3], 0);
    }

    #[test]
    fn haze_greys_the_horizon() {
        // Raising Mie must desaturate the horizon tint, which is the whole
        // visible effect of the haze slider.
        let sat = |v: Vec3| {
            let m = v.max_element().max(1e-6);
            1.0 - v.min_element() / m
        };
        let mut clear = Environment::daylight();
        clear.atmosphere.mie_scale = 0.2;
        let mut hazy = Environment::daylight();
        hazy.atmosphere.mie_scale = 6.0;

        let (_, ch, _) = clear.ambient_tints();
        let (_, hh, _) = hazy.ambient_tints();
        assert!(
            sat(hh) < sat(ch),
            "hazy horizon {} should be greyer than clear {}",
            sat(hh),
            sat(ch)
        );
    }

    #[test]
    fn exposure_is_stops() {
        let mut e = Environment::daylight();
        assert_eq!(e.exposure(), 1.0, "0 EV must be unity");
        e.tone.exposure_ev = 1.0;
        assert!((e.exposure() - 2.0).abs() < 1e-6, "one stop must double");
        e.tone.exposure_ev = -2.0;
        assert!((e.exposure() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn disabling_fog_zeroes_its_density_in_the_uniform() {
        // The shader branches on density, so the toggle has to reach it. Leaving
        // the authored value in place with a separate flag is how a disabled
        // effect keeps rendering.
        let mut e = Environment::daylight();
        e.fog.enabled = false;
        assert_eq!(e.uniform().fog_params[0], 0.0);
        e.fog.enabled = true;
        assert_eq!(e.uniform().fog_params[0], e.fog.density);
    }

    #[test]
    fn disabled_clouds_report_no_coverage() {
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.coverage = 0.8;
        assert_eq!(e.uniform().cloud_params[0], 0.8);
        e.clouds.enabled = false;
        assert_eq!(e.uniform().cloud_params[0], 0.0);
    }

    #[test]
    fn the_uniform_is_all_vec4s_and_the_right_size() {
        // 16 vectors of 16 bytes. The assertion in the module guards the total;
        // this one guards the reason it is that total, so a future field added
        // as a bare f32 fails here with an explanation rather than at the
        // size check with a number.
        assert_eq!(std::mem::size_of::<EnvironmentUniform>(), 16 * 16);
        assert_eq!(std::mem::align_of::<EnvironmentUniform>(), 4);
        // Round-trips as plain bytes, which is what `write_buffer` does.
        let e = Environment::daylight();
        let u = e.uniform();
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 256);
        assert_eq!(*bytemuck::from_bytes::<EnvironmentUniform>(bytes), u);
    }

    #[test]
    fn the_uniform_never_carries_nan() {
        // A NaN here poisons every shading pass at once and shows as black or
        // white pixels with no obvious cause. The extremes of every slider are
        // the place it would come from.
        let mut cases =
            vec![Environment::daylight(), Environment::overcast(), Environment::night()];
        let mut zeroed = Environment::daylight();
        zeroed.atmosphere.rayleigh_scale = 0.0;
        zeroed.atmosphere.mie_scale = 0.0;
        zeroed.atmosphere.ozone_scale = 0.0;
        zeroed.sky_light.intensity = 0.0;
        zeroed.sun.intensity = 0.0;
        zeroed.fog.density = 0.0;
        cases.push(zeroed);

        for (i, e) in cases.iter().enumerate() {
            let u = e.uniform();
            let floats: &[f32] = bytemuck::cast_slice(&bytemuck::bytes_of(&u)[..224]);
            for (j, v) in floats.iter().enumerate() {
                assert!(v.is_finite(), "case {i} field {j} is {v}");
            }
        }
    }

    #[test]
    fn presets_are_distinct_and_all_reset_to_daylight() {
        assert_ne!(Environment::daylight(), Environment::overcast());
        assert_ne!(Environment::daylight(), Environment::night());
        for mut e in [Environment::overcast(), Environment::night()] {
            e.reset();
            assert_eq!(e, Environment::daylight(), "reset must return to the Quick Create state");
        }
    }

    #[test]
    fn overcast_is_hazier_and_dimmer_than_daylight() {
        let d = Environment::daylight();
        let o = Environment::overcast();
        assert!(o.atmosphere.mie_scale > d.atmosphere.mie_scale, "overcast must be hazier");
        assert!(o.sun.intensity < d.sun.intensity, "overcast must dim the sun");
        assert!(o.fog.density > d.fog.density);
        assert!(o.clouds.enabled);
        assert!(o.sun.angular_diameter_deg > d.sun.angular_diameter_deg, "softer shadows");
    }

    #[test]
    fn tone_mapper_indices_are_stable() {
        // The shader switches on these, so they are part of the ABI.
        assert_eq!(ToneMapper::None.index(), 0);
        assert_eq!(ToneMapper::Reinhard.index(), 1);
        assert_eq!(ToneMapper::Aces.index(), 2);
        for m in ToneMapper::ALL {
            assert!(!m.label().is_empty());
        }
        assert_eq!(ToneMapper::ALL[0], ToneMapper::Aces, "ACES should be offered first");
    }
}

// ---------------------------------------------------------------------------
// Bridge to the existing passes
// ---------------------------------------------------------------------------

impl Environment {
    /// Push this state into the sky and fog settings the render passes already
    /// read.
    ///
    /// The mixer is the single source of truth; `SkySettings` and `FogSettings`
    /// become derived state that nothing else writes. Doing it this way rather
    /// than re-plumbing `sky.wgsl`, `volumetrics.wgsl` and `post.wgsl` onto
    /// [`EnvironmentUniform`] means the panel drives the picture today, and the
    /// shaders can migrate one at a time behind an unchanged UI.
    ///
    /// Shadow resolution, shadow distance and temporal AA are deliberately *not*
    /// touched: they are quality settings that happen to live in the same struct,
    /// they cost frame time rather than changing the look, and the mixer has no
    /// business overwriting what the user set in Quality.
    pub fn apply_to(
        &self,
        sky: &mut crate::lighting::SkySettings,
        fog: &mut crate::volumetrics::FogSettings,
    ) {
        sky.time_of_day = self.time_of_day;
        sky.day_speed = self.day_speed;
        sky.cycle_running = self.cycle_running;
        sky.editor_preview = self.editor_preview;
        // `SkySettings::haze` is a 0..2-ish artistic multiplier where the mixer
        // carries a physical Mie scale, so it is a ratio against Earth rather
        // than the coefficient itself.
        sky.haze = self.atmosphere.mie_scale;
        // Exposure is *not* set here any more: the post pass reads
        // `ToneMapping` directly, and `SkySettings::exposure` only feeds the
        // ambient term in the lighting uniform. Writing both was two paths to
        // one number.
        sky.exposure = self.exposure();
        sky.god_rays = if self.fog.enabled { self.fog.god_rays } else { 0.0 };

        fog.enabled = self.fog.enabled;
        fog.density = self.fog.density;
        fog.mist_strength = self.fog.mist_strength;
        fog.mist_falloff = self.fog.height_falloff_m;
        fog.mist_base = self.fog.base_height_m;
        fog.anisotropy = self.fog.anisotropy;
        fog.albedo = self.fog.albedo;
        fog.distance = self.fog.distance_m;
    }
}

#[cfg(test)]
mod bridge_tests {
    use super::*;

    fn derived(e: &Environment) -> (crate::lighting::SkySettings, crate::volumetrics::FogSettings) {
        let mut sky = crate::lighting::SkySettings::default();
        let mut fog = crate::volumetrics::FogSettings::default();
        e.apply_to(&mut sky, &mut fog);
        (sky, fog)
    }

    #[test]
    fn the_mixer_drives_the_fog_pass() {
        let mut e = Environment::daylight();
        e.fog.density = 0.0123;
        e.fog.height_falloff_m = 777.0;
        e.fog.anisotropy = 0.41;
        let (_, fog) = derived(&e);
        assert_eq!(fog.density, 0.0123);
        assert_eq!(fog.mist_falloff, 777.0);
        assert_eq!(fog.anisotropy, 0.41);
        assert!(fog.enabled);
    }

    #[test]
    fn disabling_fog_also_disables_god_rays() {
        // God rays are the fog pass being marched toward the sun. Leaving them
        // on with the fog off costs a pass that can only produce nothing.
        let mut e = Environment::daylight();
        e.fog.god_rays = 0.9;
        e.fog.enabled = false;
        let (sky, fog) = derived(&e);
        assert!(!fog.enabled);
        assert_eq!(sky.god_rays, 0.0, "god rays must follow the fog toggle");
    }

    #[test]
    fn exposure_reaches_the_post_chain_as_a_multiplier() {
        let mut e = Environment::daylight();
        e.tone.exposure_ev = 2.0;
        let (sky, _) = derived(&e);
        assert!((sky.exposure - 4.0).abs() < 1e-5, "two stops is 4x, got {}", sky.exposure);
    }

    #[test]
    fn quality_settings_are_left_alone() {
        // The mixer must not clobber what the Quality preset set: these cost
        // frame time rather than changing the look, and they are not its business.
        let mut sky = crate::lighting::SkySettings {
            shadow_quality: crate::lighting::ShadowQuality::High,
            shadow_distance: 999.0,
            temporal_aa: false,
            ..Default::default()
        };
        let mut fog = crate::volumetrics::FogSettings::default();

        Environment::overcast().apply_to(&mut sky, &mut fog);

        assert_eq!(sky.shadow_quality, crate::lighting::ShadowQuality::High);
        assert_eq!(sky.shadow_distance, 999.0);
        assert!(!sky.temporal_aa);
    }

    #[test]
    fn the_clock_carries_across() {
        let mut e = Environment::daylight();
        e.time_of_day = 17.25;
        e.cycle_running = true;
        e.editor_preview = true;
        let (sky, _) = derived(&e);
        assert_eq!(sky.time_of_day, 17.25);
        assert!(sky.cycle_running);
        assert!(sky.editor_preview);
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn a_preset_does_not_switch_clouds_back_off() {
        // Reported: switch clouds on, click Daylight to fix the lighting, and the
        // clouds vanish. Assigning the preset wholesale is what did it.
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.coverage = 0.8;
        e.apply_preset(Environment::daylight());
        assert!(e.clouds.enabled, "Daylight turned the user's clouds off");
    }

    #[test]
    fn a_preset_still_changes_the_look() {
        // The preservation must not make presets inert.
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.apply_preset(Environment::night());
        assert!(e.sun.is_night(), "Night did not move the sun");
        assert!(e.clouds.enabled, "and it should still have left the clouds alone");
    }

    #[test]
    fn a_preset_may_turn_something_on() {
        // Overcast means having clouds, so it is allowed to enable them.
        let mut e = Environment::daylight();
        assert!(!e.clouds.enabled);
        e.apply_preset(Environment::overcast());
        assert!(e.clouds.enabled, "Overcast should bring clouds with it");
    }

    #[test]
    fn cloud_quality_is_a_cost_choice_and_survives_every_preset() {
        for preset in [Environment::daylight(), Environment::overcast(), Environment::night()] {
            let mut e = Environment::daylight();
            e.clouds.quality = CloudQuality::Low;
            e.apply_preset(preset);
            assert_eq!(e.clouds.quality, CloudQuality::Low, "a preset changed the quality dial");
        }
    }

    #[test]
    fn reset_is_the_one_that_discards_everything() {
        // The distinction between the preset buttons and Reset environment.
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.quality = CloudQuality::High;
        e.reset();
        assert!(!e.clouds.enabled, "Reset should clear the toggles a preset keeps");
        assert_eq!(e, Environment::daylight());
    }

    #[test]
    fn preserved_toggles_do_not_leak_other_settings() {
        // Only the enable flags and the quality dial are carried across; a preset
        // must not inherit the coverage or density the user had set.
        let mut e = Environment::daylight();
        e.clouds.enabled = true;
        e.clouds.coverage = 0.11;
        e.fog.density = 0.04;
        e.apply_preset(Environment::daylight());
        assert_eq!(e.clouds.coverage, Environment::daylight().clouds.coverage);
        assert_eq!(e.fog.density, Environment::daylight().fog.density);
    }
}

#[cfg(test)]
mod time_of_day_tests {
    use super::*;

    /// Warmth as the red-to-blue ratio.
    fn warmth(v: Vec3) -> f32 {
        v.x / v.z.max(1e-6)
    }

    #[test]
    fn the_ambient_follows_the_sun_down() {
        // The reason the terrain now takes its light from here: it was lit from a
        // separate table of day/dusk/night colours while the sky was computing
        // real scattering, so at dusk the ground stayed lit as though it were
        // midday under an orange sky.
        let brightness = |hour: f32| {
            let mut e = Environment::daylight();
            e.time_of_day = hour;
            e.sync_sun_to_clock();
            let (z, h, _) = e.ambient_tints();
            (z + h).length()
        };
        let noon = brightness(12.0);
        let dusk = brightness(18.2);
        let night = brightness(23.0);

        assert!(noon > dusk, "noon ({noon}) should be brighter than dusk ({dusk})");
        assert!(dusk > night, "dusk ({dusk}) should be brighter than night ({night})");
    }

    #[test]
    fn the_key_light_warms_toward_dusk() {
        let at = |hour: f32| {
            let mut e = Environment::daylight();
            e.time_of_day = hour;
            e.sync_sun_to_clock();
            e.sun.radiance()
        };
        assert!(
            warmth(at(17.5)) > warmth(at(12.0)) * 1.3,
            "a low sun should be visibly warmer: {} against {}",
            warmth(at(17.5)),
            warmth(at(12.0))
        );
    }

    #[test]
    fn the_sun_sweeps_rather_than_only_rising() {
        // Yaw has to move with the clock as well as pitch, or every shadow in the
        // scene points the same way all day and the time of day reads as a
        // brightness dial rather than a time.
        let yaw = |hour: f32| {
            let mut e = Environment::daylight();
            e.time_of_day = hour;
            e.sync_sun_to_clock();
            e.sun.yaw_deg
        };
        assert!((yaw(16.0) - yaw(8.0)).abs() > 45.0, "the sun barely moved across the sky");
    }

    #[test]
    fn every_hour_produces_a_finite_uniform() {
        // The clock is a slider, so every value on it has to be safe -- including
        // the moment the sun crosses the horizon, where the key light flips.
        for i in 0..=240 {
            let mut e = Environment::daylight();
            e.time_of_day = i as f32 * 0.1;
            e.sync_sun_to_clock();
            let u = e.uniform();
            let floats: &[f32] = bytemuck::cast_slice(&bytemuck::bytes_of(&u)[..224]);
            assert!(
                floats.iter().all(|v| v.is_finite()),
                "hour {:.1} produced a non-finite uniform",
                e.time_of_day
            );
        }
    }
}

#[cfg(test)]
mod twilight_tests {
    use super::*;

    #[test]
    fn the_sky_level_falls_monotonically_all_the_way_to_night() {
        // The bug: `daylight()` clamps to zero below the horizon, so dusk and
        // midnight gave the identical ambient and the ground snapped to full night
        // the instant the sun set.
        let level = |hour: f32| {
            let mut e = Environment::daylight();
            e.time_of_day = hour;
            e.sync_sun_to_clock();
            e.sky_level()
        };
        let samples: Vec<f32> = (0..=40).map(|i| level(12.0 + i as f32 * 0.3)).collect();
        for w in samples.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "sky level rose across the afternoon: {w:?}");
        }
        assert!(level(12.0) > level(18.5), "noon must be brighter than dusk");
        assert!(level(18.5) > level(21.0), "dusk must be brighter than late evening");
        assert!(level(21.0) >= level(24.0), "and it must keep falling into the night");
    }

    #[test]
    fn twilight_is_a_ramp_not_a_cliff() {
        // A few degrees below the horizon should still be appreciably lit, or
        // sunset happens in one frame.
        let mut just_set = Environment::daylight();
        just_set.sun.pitch_deg = 3.0;
        let mut deep_night = Environment::daylight();
        deep_night.sun.pitch_deg = 60.0;

        assert!(just_set.sun.is_night(), "pitch +3 is below the horizon");
        assert!(
            just_set.sky_level() > deep_night.sky_level() * 3.0,
            "just after sunset ({}) should be far brighter than deep night ({})",
            just_set.sky_level(),
            deep_night.sky_level()
        );
    }

    #[test]
    fn night_is_dim_but_never_black() {
        let mut e = Environment::daylight();
        e.sun.pitch_deg = 80.0;
        assert!(e.sky_level() > 0.0, "a scene lit to zero is a black screen");
        assert!(e.sky_level() < 0.05, "night should be dim: {}", e.sky_level());
    }

    // --- persistence ---
    //
    // What the user actually asked for: configure fog, close the world, come back
    // and find it. Every one of these is a way that can silently fail.

    fn scratch(name: &str) -> terra_project::ProjectPaths {
        let dir = std::env::temp_dir().join(format!("terra-env-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = terra_project::ProjectPaths::new(dir);
        // The real tree, so the test writes to the path the editor writes to rather
        // than to one the test invented.
        paths.scaffold().unwrap();
        paths
    }

    /// An environment with every section moved off its default, so a field dropped
    /// in serialization shows up as a difference rather than coincidentally matching.
    fn authored() -> Environment {
        let mut e = Environment::daylight();
        e.sun.pitch_deg = -8.5;
        e.sun.yaw_deg = 133.0;
        e.sun.intensity = 4.25;
        e.sun.tint = Vec3::new(1.0, 0.72, 0.48);
        e.sun.casts_shadows = false;
        e.atmosphere.mie_scale = 3.1;
        e.atmosphere.ground_albedo = Vec3::splat(0.21);
        e.sky_light.intensity = 0.66;
        e.sky_light.horizon = Vec3::new(0.4, 0.3, 0.25);
        e.fog.enabled = true;
        e.fog.density = 0.0175;
        e.fog.height_falloff_m = 240.0;
        e.fog.albedo = Vec3::new(0.9, 0.85, 0.8);
        e.fog.god_rays = 0.8;
        e.clouds.enabled = true;
        e.clouds.coverage = 0.62;
        e.clouds.base_m = 1750.0;
        e.clouds.wind = Vec3::new(9.0, 0.0, -3.0);
        e.clouds.quality = CloudQuality::High;
        e.tone.mapper = ToneMapper::Reinhard;
        e.tone.exposure_ev = -1.25;
        e.tone.white_balance_k = 5200.0;
        e.time_of_day = 19.4;
        e.day_speed = 1.75;
        e.cycle_running = true;
        e.editor_preview = true;
        e
    }

    #[test]
    fn an_authored_environment_survives_a_round_trip() {
        let paths = scratch("round-trip");
        let want = authored();
        want.save(&paths).expect("save");
        let got = Environment::load(&paths).expect("load");
        // Exact, field for field. `PartialEq` covers every section, so a setting
        // that failed to serialize cannot hide behind the ones that did.
        assert_eq!(got, want);
    }

    #[test]
    fn a_world_with_no_saved_environment_opens_on_daylight() {
        // The path a world created before this existed takes, and the one a never
        // saved world takes. It must not be an error, and it must not be black.
        let paths = scratch("absent");
        assert_eq!(Environment::load(&paths), None);
        let fallback = Environment::load(&paths).unwrap_or_default();
        assert_eq!(fallback, Environment::daylight());
    }

    #[test]
    fn a_corrupt_environment_file_is_ignored_rather_than_fatal() {
        // Refusing to open a world over a bad lighting file would be a worse failure
        // than losing the lighting.
        let paths = scratch("corrupt");
        std::fs::write(paths.environment(), "this is not ron {{{").unwrap();
        assert_eq!(Environment::load(&paths), None);
    }

    #[test]
    fn a_file_missing_a_whole_section_still_loads() {
        // The forward-compatibility promise of `#[serde(default)]`: adding a setting
        // must not reset every existing world's environment. Simulated by writing a
        // file that predates one -- here, all of `clouds` and `tone`.
        let paths = scratch("partial");
        std::fs::write(
            paths.environment(),
            "(sun: (pitch_deg: -12.0, yaw_deg: 40.0), fog: (density: 0.009), time_of_day: 6.25)",
        )
        .unwrap();
        let got = Environment::load(&paths).expect("a partial file must still load");
        // The fields that were present.
        assert_eq!(got.sun.pitch_deg, -12.0);
        assert_eq!(got.sun.yaw_deg, 40.0);
        assert_eq!(got.fog.density, 0.009);
        assert_eq!(got.time_of_day, 6.25);
        // And the sections that were absent fall back rather than zeroing, which
        // would be a black sky and no tone mapping.
        assert_eq!(got.clouds, VolumetricClouds::default());
        assert_eq!(got.tone, ToneMapping::default());
        assert_eq!(got.sun.intensity, SunLight::default().intensity, "unlisted field zeroed");
    }

    #[test]
    fn saving_overwrites_rather_than_appending() {
        // Two saves in a row must leave one readable document. A write that appended
        // would parse as garbage on the second load.
        let paths = scratch("overwrite");
        authored().save(&paths).unwrap();
        let mut second = Environment::daylight();
        second.fog.density = 0.033;
        second.save(&paths).unwrap();
        assert_eq!(Environment::load(&paths), Some(second));
    }

    #[test]
    fn every_edit_marks_the_world_unsaved() {
        // `exit_editor` only saves a dirty world, so anything this misses is a
        // setting the user loses on close.
        let saved = Environment::daylight();
        assert!(!saved.differs_for_saving(&saved), "an untouched environment is not dirty");

        let mut e = saved;
        e.fog.density = 0.02;
        assert!(e.differs_for_saving(&saved), "fog");

        let mut e = saved;
        e.clouds.enabled = !saved.clouds.enabled;
        assert!(e.differs_for_saving(&saved), "clouds");

        let mut e = saved;
        e.sun.yaw_deg += 30.0;
        assert!(e.differs_for_saving(&saved), "sun");

        let mut e = saved;
        e.tone.exposure_ev -= 1.0;
        assert!(e.differs_for_saving(&saved), "tone mapping");

        // Scrubbing the time with the cycle stopped is an edit.
        let mut e = saved;
        e.time_of_day = 3.0;
        assert!(!e.cycle_running);
        assert!(e.differs_for_saving(&saved), "a manual time scrub");

        // Starting the cycle is an edit too.
        let mut e = saved;
        e.cycle_running = true;
        assert!(e.differs_for_saving(&saved), "starting the day cycle");
    }

    #[test]
    fn a_running_day_cycle_does_not_hold_the_world_permanently_dirty() {
        // The reason `differs_for_saving` exists rather than a plain `!=`: with the
        // cycle running the time advances every frame, and treating that as an edit
        // would rewrite the entire world on every exit.
        let mut saved = Environment::daylight();
        saved.cycle_running = true;
        let mut e = saved;
        for _ in 0..600 {
            e.tick(1.0 / 60.0);
        }
        assert_ne!(e.time_of_day, saved.time_of_day, "the cycle should have advanced");
        // `tick` re-derives the sun angles from the clock, so those moved too. Both
        // are outputs of the cycle, and neither is an edit.
        assert_ne!(e.sun.pitch_deg, saved.sun.pitch_deg, "the sun should have moved");
        assert!(!e.differs_for_saving(&saved), "a running cycle must not count as an edit");

        // But a real edit made while the cycle runs still registers.
        let mut edited = e;
        edited.fog.density += 0.01;
        assert!(edited.differs_for_saving(&saved), "fog edited during a running cycle");
    }
}

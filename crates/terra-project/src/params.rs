//! Terrain generation parameters, as serialized into `world.ron`.
//!
//! These live here rather than in `terra-gen` so that the GPU crate can depend
//! on the project crate without a cycle: the parameters are data, the pipelines
//! that consume them are not.
//!
//! Defaults are tuned to produce a plausible mountain valley on the first run,
//! so a new project shows real terrain before the user touches a slider.

use serde::{Deserialize, Serialize};

/// Ridged multifractal base heightfield.
///
/// The multifractal weighting is what separates this from `1 - abs(fbm)`: each
/// octave is scaled by the previous octave's value, so ridges reinforce ridges
/// and flat ground stays flat. Without it you get uniform crumpled foil.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RmfParams {
    /// Octave count. Above ~10 the extra detail falls below one texel.
    pub octaves: u32,
    /// Frequency multiplier per octave. 2.0 is standard; 1.9-2.1 avoids
    /// harmonics lining up into visible grids.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
    /// Ridge offset. Higher values give broader, flatter-topped ridges.
    pub offset: f32,
    /// How strongly each octave is weighted by the previous one. 0 disables the
    /// multifractal behaviour and degenerates to plain ridged noise.
    pub sharpness: f32,
    /// Size of the largest feature, in meters.
    pub feature_scale_m: f32,
    /// Peak height in meters before erosion.
    pub amplitude_m: f32,
    /// Strength of the low-frequency domain warp, in meters. This is what makes
    /// ridgelines meander like real ranges instead of running in straight
    /// statistical lines. Set to 0 to disable.
    ///
    /// The warp displaces the *input coordinate*; it is not a second terrain
    /// layer, and it is evaluated with a smooth basis rather than a ridged one.
    /// Warping with a ridged field would crease the terrain, because ridged
    /// noise has a gradient discontinuity along every ridge.
    pub warp_strength_m: f32,
    /// Feature size of the warp field, in meters.
    pub warp_scale_m: f32,
}

impl Default for RmfParams {
    fn default() -> Self {
        Self {
            octaves: 8,
            lacunarity: 2.03,
            gain: 0.5,
            offset: 1.0,
            sharpness: 0.8,
            feature_scale_m: 4096.0,
            amplitude_m: 900.0,
            warp_strength_m: 380.0,
            warp_scale_m: 6144.0,
        }
    }
}

/// Grid ("pipe model") hydraulic erosion, after Mei et al. 2007.
///
/// A particle/droplet solver is the other option, but it needs scattered atomic
/// adds; WebGPU has no f32 atomics at all, and the contention is bad even
/// natively. The pipe model is a pure gather -- one thread per texel, no atomics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ErosionParams {
    /// Solver iterations. 500 for a preview, 2000-3000 for a final bake.
    pub iterations: u32,
    /// Integration step. Raising this past ~0.05 violates the CFL condition and
    /// the flux field diverges.
    pub dt: f32,
    /// Water added per iteration.
    pub rain_rate: f32,
    /// Fraction of standing water removed per iteration.
    pub evaporation: f32,
    /// Sediment capacity constant. The main "how carved is it" dial.
    pub capacity: f32,
    /// Rate at which terrain dissolves into suspension.
    pub dissolve_rate: f32,
    /// Rate at which suspended sediment settles out.
    pub deposit_rate: f32,
    /// Floor on `sin(tilt)` in the capacity term. Without this, flat ground has
    /// zero capacity, rivers stop cutting, and valley floors never form.
    pub min_slope: f32,
    /// Cross-sectional area of the virtual pipes between texels.
    pub pipe_area: f32,
    /// Gravity, in m/s^2.
    pub gravity: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            iterations: 2000,
            dt: 0.02,
            rain_rate: 0.012,
            evaporation: 0.015,
            capacity: 0.05,
            dissolve_rate: 0.5,
            deposit_rate: 1.0,
            min_slope: 0.05,
            pipe_area: 1.0,
            gravity: 9.81,
        }
    }
}

/// Thermal erosion -- material sliding down slopes steeper than the angle of
/// repose. Run a short pass before hydraulic to relax noise artifacts, and a
/// longer pass after to build the talus/scree aprons at cliff bases.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalParams {
    pub pre_iterations: u32,
    pub post_iterations: u32,
    /// Angle of repose in degrees. Loose rock sits around 33-37 degrees.
    pub talus_angle_deg: f32,
    /// Fraction of the excess moved per iteration.
    pub rate: f32,
}

impl Default for ThermalParams {
    fn default() -> Self {
        Self { pre_iterations: 50, post_iterations: 120, talus_angle_deg: 35.0, rate: 0.5 }
    }
}

/// Everything that feeds the generator, bundled for change detection.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct TerrainParams {
    pub rmf: RmfParams,
    pub erosion: ErosionParams,
    pub thermal: ThermalParams,
}

impl TerrainParams {
    /// Whether a change between two parameter sets invalidates baked tiles.
    /// Any generator change does, so this is a plain inequality today -- but it
    /// is a named method because render-only settings will land in `world.ron`
    /// later and must not trigger a multi-second re-bake.
    pub fn invalidates_cache(&self, other: &Self) -> bool {
        self != other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_numerically_stable() {
        let e = ErosionParams::default();
        assert!(e.dt <= 0.05, "dt above the CFL limit will diverge");
        assert!(e.min_slope > 0.0, "a zero slope floor stalls river cutting");
        assert!(e.evaporation > 0.0, "water must leave or the map floods");
    }

    #[test]
    fn multifractal_weighting_is_on_by_default() {
        assert!(RmfParams::default().sharpness > 0.0);
    }

    #[test]
    fn identical_params_do_not_invalidate() {
        let a = TerrainParams::default();
        assert!(!a.invalidates_cache(&TerrainParams::default()));
        let mut b = a;
        b.rmf.octaves += 1;
        assert!(a.invalidates_cache(&b));
    }
}

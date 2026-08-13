//! Shader sources, embedded at compile time and composed.
//!
//! WGSL has no `#include`, so every pass that needs the shared noise basis gets
//! it by string concatenation here. `include_str!` resolves at compile time, so
//! a moved or deleted shader is a build error rather than a runtime panic.

/// Shared noise basis: value noise, ridged multifractal, domain warp.
pub const NOISE: &str = include_str!("../../../assets/shaders/common/noise.wgsl");

/// Six-pass grid hydraulic erosion.
pub const EROSION: &str = include_str!("../../../assets/shaders/gen/erosion.wgsl");

/// Thermal erosion / angle of repose.
pub const THERMAL: &str = include_str!("../../../assets/shaders/gen/thermal.wgsl");

/// Tier-0 -> tier-1 bake.
pub const TILES: &str = include_str!("../../../assets/shaders/gen/tiles.wgsl");

/// Prepend the shared basis to a pass that samples noise.
pub fn with_noise(pass: &str) -> String {
    format!("{NOISE}\n{pass}")
}

pub fn tiles_source() -> String {
    with_noise(TILES)
}

/// Erosion and thermal are pure simulation over the heightfield and never
/// sample noise, so they are used as-is.
pub fn erosion_source() -> &'static str {
    EROSION
}

pub fn thermal_source() -> &'static str {
    THERMAL
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared basis must expose ridged multifractal. The base heightfield
    /// itself is generated on the CPU (see `heightfield.rs`); this basis is what
    /// the render-time detail layer uses, and it must match.
    #[test]
    fn shared_basis_offers_ridged_multifractal() {
        assert!(NOISE.contains("fn ridged_multifractal("));
    }

    /// No fBm function may exist in the shared basis. `warp_basis` is summed
    /// octaves too, but it produces a coordinate offset rather than height and
    /// is named so it cannot be mistaken for a terrain function.
    #[test]
    fn no_fbm_height_function_is_defined() {
        assert!(!NOISE.contains("fn fbm"), "fBm must not be a terrain basis");
        assert!(NOISE.contains("fn warp_basis("));
        assert!(NOISE.contains("fn warp_offset("));
    }

    /// Erosion is simulated, not approximated by shaping noise.
    #[test]
    fn erosion_is_a_simulation_not_a_noise_trick() {
        for entry in [
            "fn rain(",
            "fn flux_pass(",
            "fn water_update(",
            "fn erode_deposit(",
            "fn advect(",
            "fn evaporate(",
        ] {
            assert!(EROSION.contains(entry), "erosion pass {entry} is missing");
        }
        // The simulation passes operate on the heightfield and must not need
        // the noise basis; if they start sampling noise, something has been
        // faked rather than solved.
        assert!(!EROSION.contains("ridged_multifractal"));
        assert!(!EROSION.contains("noise2("));
    }

    /// Detail added after erosion must use the same basis as the base, or the
    /// boundary between eroded and non-eroded scales becomes visible.
    #[test]
    fn tier1_detail_uses_the_same_basis() {
        assert!(TILES.contains("ridged_multifractal"));
    }

    #[test]
    fn composition_puts_the_basis_first() {
        let src = tiles_source();
        let basis = src.find("fn ridged_multifractal(").expect("basis present");
        let pass = src.find("@compute").expect("entry point present");
        assert!(basis < pass, "WGSL has no forward declarations");
    }
}

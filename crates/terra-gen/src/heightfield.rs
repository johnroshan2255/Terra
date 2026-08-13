//! Ridged multifractal base heightfield.
//!
//! Runs on the CPU. At 1024^2 with 8 octaves this is ~30 ms across 10 cores and
//! under a second at 4096^2 -- a compute pass would add pipeline plumbing and a
//! readback to save nothing. The GPU is reserved for erosion, which iterates
//! thousands of times over the same field and genuinely needs it.
//!
//! Value noise here mirrors `assets/shaders/common/noise.wgsl` exactly, so the
//! render-time detail layer lines up with the base it sits on.

use rayon::prelude::*;
use terra_project::params::RmfParams;

/// Per-octave rotation. Breaks the axis alignment that makes stacked value
/// noise read as a grid.
const ROT: [f32; 4] = [0.8, 0.6, -0.6, 0.8];

fn rot(x: f32, y: f32) -> (f32, f32) {
    (ROT[0] * x + ROT[2] * y, ROT[1] * x + ROT[3] * y)
}

fn hash21(x: f32, y: f32) -> f32 {
    let mut hx = (x * 0.1031).fract();
    let mut hy = (y * 0.1030).fract();
    let d = hx * (hy + 33.33) + hy * (hx + 33.33);
    hx += d;
    hy += d;
    ((hx + hy) * hx).fract() * 2.0 - 1.0
}

/// Value noise with a quintic interpolant (C2, so shading has no facet seams).
fn noise2(x: f32, y: f32) -> f32 {
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (x - ix, y - iy);

    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    let a = hash21(ix, iy);
    let b = hash21(ix + 1.0, iy);
    let c = hash21(ix, iy + 1.0);
    let d = hash21(ix + 1.0, iy + 1.0);

    let k1 = b - a;
    let k2 = c - a;
    let k3 = a - b - c + d;
    a + k1 * ux + k2 * uy + k3 * ux * uy
}

/// Smooth summed-octave basis. Used **only** for the domain-warp offset, never
/// for height: ridged noise is C0 but not C1, and warping coordinates with a
/// creased field creases the terrain.
fn warp_basis(mut x: f32, mut y: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    for _ in 0..4 {
        sum += amp * noise2(x, y);
        norm += amp;
        amp *= 0.5;
        let (rx, ry) = rot(x, y);
        x = rx * 2.0;
        y = ry * 2.0;
    }
    sum / norm.max(1e-6)
}

/// Ridged multifractal.
///
/// `prev` is what makes this multifractal rather than plain ridged fBm: each
/// octave is scaled by the previous octave's value, so detail concentrates on
/// ridges and lowlands stay smooth.
pub fn ridged_multifractal(mut x: f32, mut y: f32, p: &RmfParams) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut prev = 1.0;
    let mut norm = 0.0;

    for _ in 0..p.octaves {
        let mut n = p.offset - noise2(x, y).abs();
        n *= n;
        n *= 1.0 + (prev - 1.0) * p.sharpness;

        sum += n * amp;
        norm += amp;
        prev = n.clamp(0.0, 1.0); // unclamped, the feedback term explodes

        amp *= p.gain;
        let (rx, ry) = rot(x, y);
        x = rx * p.lacunarity;
        y = ry * p.lacunarity;
    }
    sum / norm.max(1e-6)
}

/// Height in meters at a world position, warp included.
pub fn height_at(wx: f32, wz: f32, p: &RmfParams) -> f32 {
    let (mut x, mut y) = (wx / p.feature_scale_m, wz / p.feature_scale_m);

    if p.warp_strength_m > 0.0 {
        let s = p.warp_scale_m / p.feature_scale_m;
        let k = p.warp_strength_m / p.feature_scale_m;
        x += warp_basis(x / s, y / s) * k;
        y += warp_basis(x / s + 5.2, y / s + 1.3) * k;
    }

    ridged_multifractal(x, y, p) * p.amplitude_m
}

/// Generate a square heightfield in meters, row-major.
///
/// `extent_m` is the full world width, so texel spacing is
/// `extent_m / (res - 1)` and the field is centered on the origin.
pub fn generate(res: u32, extent_m: f32, p: &RmfParams) -> Vec<f32> {
    let n = res as usize;
    let step = extent_m / (res - 1) as f32;
    let half = extent_m * 0.5;

    let mut out = vec![0.0f32; n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(z, row)| {
        let wz = z as f32 * step - half;
        for (x, h) in row.iter_mut().enumerate() {
            *h = height_at(x as f32 * step - half, wz, p);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> RmfParams {
        RmfParams::default()
    }

    #[test]
    fn output_is_finite_and_sized() {
        let h = generate(64, 4096.0, &params());
        assert_eq!(h.len(), 64 * 64);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn terrain_has_relief() {
        let h = generate(128, 4096.0, &params());
        let lo = h.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // A flat result means the basis collapsed -- the failure mode that
        // silently produces an empty-looking world.
        assert!(hi - lo > 50.0, "range was only {}", hi - lo);
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate(32, 2048.0, &params());
        let b = generate(32, 2048.0, &params());
        assert_eq!(a, b, "same params must give the same terrain");
    }

    #[test]
    fn sharpness_changes_the_result() {
        let mut flat = params();
        flat.sharpness = 0.0;
        let a = generate(48, 4096.0, &params());
        let b = generate(48, 4096.0, &flat);
        assert_ne!(a, b, "multifractal weighting must actually do something");
    }

    #[test]
    fn ridges_stay_within_unit_range() {
        let p = params();
        for i in 0..500 {
            let v = ridged_multifractal(i as f32 * 0.37, i as f32 * 0.11, &p);
            assert!((0.0..=1.5).contains(&v), "octave feedback diverged: {v}");
        }
    }
}

//! Noise sources for the noise sculpt brush: a built-in procedural basis and
//! user-supplied greyscale images.
//!
//! # Convention: mid-grey is "leave it alone"
//!
//! Every source returns a **signed** value in `-1..=1`. For an uploaded image
//! that means 0.5 grey maps to 0.0, black to -1 and white to +1, which is the
//! same convention every displacement map in every DCC tool uses. Treating
//! black as zero instead would make an ordinary noise texture -- which averages
//! mid-grey -- inflate the whole brush area as a side effect of adding detail.
//!
//! # Why images are sampled triplanar
//!
//! An uploaded texture is 2D and the surface being sculpted is not. Projecting
//! down world XZ, the way a landscape tool does, stretches the pattern into
//! vertical streaks on any cliff or cave wall -- which is most of what this
//! crate exists to build.
//!
//! So images are blended across all three axis-aligned projections, weighted by
//! the surface normal. The weights come from the *stroke's* normal rather than
//! the per-voxel gradient: a dab covers a small patch of roughly one
//! orientation, and a per-voxel gradient would cost six extra field samples per
//! voxel to refine a choice the user cannot see.
//!
//! # Note on the no-fBm rule
//!
//! `terra-gen` forbids summed-octave noise as a *terrain basis*, because the
//! erosion solver is what is supposed to produce erosion features. That rule
//! does not apply here: this is a brush displacement the user asked for and
//! aimed by hand, not a generator pretending to be geology.

use glam::Vec3;

/// Default feature size, in metres. Roughly rock-surface roughness rather than
/// landscape scale -- the brush is for texture, not for terrain.
pub const DEFAULT_SCALE_M: f32 = 12.0;

/// A greyscale image uploaded by the user.
///
/// Stored as `0..=1` luminance. Decoding PNG or JPEG is deliberately *not* done
/// here: this crate has no image dependency, and the editor already links
/// `image` for its texture pipeline. Constructors take raw pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseImage {
    pub name: String,
    width: u32,
    height: u32,
    /// Row-major, `width * height` samples in `0..=1`.
    data: Vec<f32>,
}

/// Why an image could not be accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    Empty,
    /// Pixel count does not match the stated dimensions.
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageError::Empty => write!(f, "image has zero width or height"),
            ImageError::SizeMismatch { expected, actual } => {
                write!(f, "expected {expected} samples, got {actual}")
            }
        }
    }
}

impl std::error::Error for ImageError {}

impl NoiseImage {
    pub fn new(
        name: impl Into<String>,
        width: u32,
        height: u32,
        data: Vec<f32>,
    ) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::Empty);
        }
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(ImageError::SizeMismatch { expected, actual: data.len() });
        }
        Ok(Self { name: name.into(), width, height, data })
    }

    /// From 8-bit greyscale bytes.
    pub fn from_gray8(
        name: impl Into<String>,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<Self, ImageError> {
        let data = bytes.iter().map(|b| *b as f32 / 255.0).collect();
        Self::new(name, width, height, data)
    }

    /// From 8-bit RGBA, taking perceptual luminance.
    ///
    /// Rec. 709 weights rather than a flat average: a user who uploads a
    /// coloured map rather than a true greyscale one gets the brightness they
    /// see, and for an actually-grey image the two agree exactly.
    pub fn from_rgba8(
        name: impl Into<String>,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<Self, ImageError> {
        let expected = (width as usize) * (height as usize) * 4;
        if bytes.len() != expected {
            return Err(ImageError::SizeMismatch { expected, actual: bytes.len() });
        }
        let data = bytes
            .chunks_exact(4)
            .map(|p| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0)
            .collect();
        Self::new(name, width, height, data)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Mean luminance, used to warn that a map is not centred on mid-grey and
    /// will therefore bias the surface as well as roughen it.
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    fn texel(&self, x: i32, y: i32) -> f32 {
        // Wrap rather than clamp: a noise map is tiled across the brush, and
        // clamping would smear the edge row into a visible streak.
        let x = x.rem_euclid(self.width as i32) as usize;
        let y = y.rem_euclid(self.height as i32) as usize;
        self.data[y * self.width as usize + x]
    }

    /// Bilinear sample in tile units; `1.0` is one full repeat.
    fn sample_uv(&self, u: f32, v: f32) -> f32 {
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let a = self.texel(x0, y0);
        let b = self.texel(x0 + 1, y0);
        let c = self.texel(x0, y0 + 1);
        let d = self.texel(x0 + 1, y0 + 1);
        let top = a + (b - a) * tx;
        let bot = c + (d - c) * tx;
        top + (bot - top) * ty
    }
}

/// Where the noise comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum NoisePattern {
    /// The built-in basis. Always available, needs no upload, and is what the
    /// tool starts on.
    Procedural { seed: u32, octaves: u32, ridged: bool },
    /// A user-uploaded greyscale map.
    Image(NoiseImage),
}

impl Default for NoisePattern {
    fn default() -> Self {
        // Ridged by default: rock surfaces are creased, and plain value noise
        // reads as lumpy plaster.
        NoisePattern::Procedural { seed: 0x5EED, octaves: 4, ridged: true }
    }
}

impl NoisePattern {
    /// Label for the tool palette.
    pub fn label(&self) -> &str {
        match self {
            NoisePattern::Procedural { ridged: true, .. } => "Ridged (built-in)",
            NoisePattern::Procedural { .. } => "Billow (built-in)",
            NoisePattern::Image(i) => &i.name,
        }
    }

    pub fn is_uploaded(&self) -> bool {
        matches!(self, NoisePattern::Image(_))
    }
}

/// A noise pattern plus the scale it is applied at.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseField {
    pub pattern: NoisePattern,
    /// Feature size in metres: one repeat of the pattern.
    pub scale_m: f32,
}

impl Default for NoiseField {
    fn default() -> Self {
        Self { pattern: NoisePattern::default(), scale_m: DEFAULT_SCALE_M }
    }
}

impl NoiseField {
    pub fn procedural(seed: u32, octaves: u32, ridged: bool, scale_m: f32) -> Self {
        Self { pattern: NoisePattern::Procedural { seed, octaves, ridged }, scale_m }
    }

    pub fn image(img: NoiseImage, scale_m: f32) -> Self {
        Self { pattern: NoisePattern::Image(img), scale_m }
    }

    /// Signed displacement in `-1..=1`. Zero leaves the surface where it is.
    ///
    /// `normal` orients the triplanar blend for image patterns and is ignored
    /// by the procedural basis, which is natively 3D.
    pub fn sample(&self, p: Vec3, normal: Vec3) -> f32 {
        let scale = self.scale_m.max(1e-3);
        let q = p / scale;
        match &self.pattern {
            NoisePattern::Procedural { seed, octaves, ridged } => {
                procedural(q, *seed, (*octaves).clamp(1, 8), *ridged)
            }
            NoisePattern::Image(img) => {
                // Squared normal components as blend weights: the standard
                // triplanar weighting, sharp enough that a face aligned to an
                // axis reads almost purely from that one projection.
                let n = normal.normalize_or(Vec3::Y).abs();
                let w = n * n;
                let sum = w.x + w.y + w.z;
                let w = if sum > 1e-6 { w / sum } else { Vec3::Y };
                let along_x = img.sample_uv(q.z, q.y);
                let along_y = img.sample_uv(q.x, q.z);
                let along_z = img.sample_uv(q.x, q.y);
                let v = along_x * w.x + along_y * w.y + along_z * w.z;
                // Mid-grey to zero. See the module note.
                (v - 0.5) * 2.0
            }
        }
    }
}

/// Integer hash to a `0..=1` float. Cheap, well-mixed, and deterministic
/// across platforms -- which matters because a sculpt has to reproduce from a
/// saved seed.
fn hash3(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x1656_67b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 / 0x00ff_ffff as f32
}

/// Trilinear value noise with a smoothstep fade, in `0..=1`.
fn value_noise(p: Vec3, seed: u32) -> f32 {
    let b = p.floor();
    let f = p - b;
    // Smoothstep the interpolant, or the lattice shows as a visible grid of
    // creases where the linear segments meet.
    let s = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (x0, y0, z0) = (b.x as i32, b.y as i32, b.z as i32);

    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c = |dx, dy, dz| hash3(x0 + dx, y0 + dy, z0 + dz, seed);
    let x00 = mix(c(0, 0, 0), c(1, 0, 0), s.x);
    let x10 = mix(c(0, 1, 0), c(1, 1, 0), s.x);
    let x01 = mix(c(0, 0, 1), c(1, 0, 1), s.x);
    let x11 = mix(c(0, 1, 1), c(1, 1, 1), s.x);
    mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z)
}

/// Summed octaves, normalized to `-1..=1`.
fn procedural(p: Vec3, seed: u32, octaves: u32, ridged: bool) -> f32 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    for o in 0..octaves {
        // Both operations must wrap: `o * golden` overflows u32 from the third
        // octave onward, which panics in debug and silently differs in release.
        let v = value_noise(p * freq, seed.wrapping_add(o.wrapping_mul(0x9e37_79b9)));
        // Ridged: fold the octave about its midpoint so the maxima become
        // creases, then square to sharpen them. Billow: use it as-is.
        let shaped = if ridged {
            let r = 1.0 - (2.0 * v - 1.0).abs();
            r * r
        } else {
            v
        };
        sum += shaped * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm <= 0.0 {
        return 0.0;
    }
    ((sum / norm) - 0.5) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probes() -> Vec<Vec3> {
        let mut v = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                v.push(Vec3::new(i as f32 * 1.7 - 10.0, j as f32 * 2.3 - 8.0, i as f32 * 0.9));
            }
        }
        v
    }

    #[test]
    fn procedural_stays_in_signed_range() {
        for ridged in [true, false] {
            let f = NoiseField::procedural(7, 5, ridged, 10.0);
            for p in probes() {
                let v = f.sample(p, Vec3::Y);
                assert!((-1.0..=1.0).contains(&v), "ridged={ridged} gave {v} at {p}");
            }
        }
    }

    #[test]
    fn procedural_is_deterministic_for_a_seed() {
        // A saved sculpt has to reproduce, so the same seed must give the same
        // field on every run.
        let a = NoiseField::procedural(42, 4, true, 10.0);
        let b = NoiseField::procedural(42, 4, true, 10.0);
        for p in probes() {
            assert_eq!(a.sample(p, Vec3::Y), b.sample(p, Vec3::Y), "at {p}");
        }
    }

    #[test]
    fn different_seeds_give_different_fields() {
        let a = NoiseField::procedural(1, 4, true, 10.0);
        let b = NoiseField::procedural(2, 4, true, 10.0);
        let differing =
            probes().into_iter().filter(|p| a.sample(*p, Vec3::Y) != b.sample(*p, Vec3::Y)).count();
        assert!(differing > 100, "only {differing} of 144 probes differ");
    }

    #[test]
    fn procedural_actually_varies() {
        // A basis that returns a constant would pass every range check and do
        // nothing at all.
        let f = NoiseField::procedural(3, 4, false, 10.0);
        let vals: Vec<f32> = probes().iter().map(|p| f.sample(*p, Vec3::Y)).collect();
        let lo = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(hi - lo > 0.3, "range is only {}", hi - lo);
    }

    #[test]
    fn scale_controls_feature_size() {
        // Doubling the scale must halve how fast the field changes per metre.
        let step = 0.5;
        let roughness = |scale: f32| {
            let f = NoiseField::procedural(9, 3, false, scale);
            (0..200)
                .map(|i| {
                    let x = i as f32 * step;
                    (f.sample(Vec3::new(x + step, 0.0, 0.0), Vec3::Y)
                        - f.sample(Vec3::new(x, 0.0, 0.0), Vec3::Y))
                    .abs()
                })
                .sum::<f32>()
        };
        let fine = roughness(4.0);
        let coarse = roughness(32.0);
        assert!(coarse < fine * 0.5, "coarse {coarse} vs fine {fine}");
    }

    #[test]
    fn value_noise_has_no_visible_lattice_creases() {
        // Without the smoothstep fade the field is C0 but not C1, and the
        // lattice shows as a grid. Check the second difference stays bounded
        // across a lattice boundary.
        let mut worst = 0.0f32;
        for i in 0..400 {
            let x = 0.9 + i as f32 * 0.001;
            let a = value_noise(Vec3::new(x - 0.001, 0.3, 0.4), 5);
            let b = value_noise(Vec3::new(x, 0.3, 0.4), 5);
            let c = value_noise(Vec3::new(x + 0.001, 0.3, 0.4), 5);
            worst = worst.max((c - 2.0 * b + a).abs());
        }
        assert!(worst < 1e-3, "second difference {worst} suggests a crease at the lattice");
    }

    // --- uploaded images ---

    fn checker(w: u32, h: u32) -> NoiseImage {
        let data = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if (x + y) % 2 == 0 { 1.0 } else { 0.0 }
            })
            .collect();
        NoiseImage::new("checker", w, h, data).unwrap()
    }

    #[test]
    fn image_rejects_bad_input() {
        assert_eq!(NoiseImage::new("z", 0, 4, vec![]), Err(ImageError::Empty));
        assert_eq!(
            NoiseImage::new("s", 2, 2, vec![0.0; 3]),
            Err(ImageError::SizeMismatch { expected: 4, actual: 3 })
        );
        assert!(NoiseImage::from_rgba8("rgba", 2, 2, &[0; 15]).is_err());
    }

    #[test]
    fn gray8_maps_the_full_byte_range() {
        let img = NoiseImage::from_gray8("g", 3, 1, &[0, 128, 255]).unwrap();
        assert_eq!(img.texel(0, 0), 0.0);
        assert!((img.texel(1, 0) - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(img.texel(2, 0), 1.0);
    }

    #[test]
    fn rgba_luminance_matches_gray_for_a_grey_image() {
        // An actually-grey upload must give the same field either way, or the
        // same texture would sculpt differently depending on its file format.
        let levels = [0u8, 40, 130, 200, 255];
        let rgba: Vec<u8> = levels.iter().flat_map(|v| [*v, *v, *v, 255]).collect();
        let a = NoiseImage::from_rgba8("a", 5, 1, &rgba).unwrap();
        let b = NoiseImage::from_gray8("b", 5, 1, &levels).unwrap();
        for x in 0..5 {
            assert!((a.texel(x, 0) - b.texel(x, 0)).abs() < 1e-4, "at {x}");
        }
    }

    #[test]
    fn mid_grey_is_no_displacement() {
        // The convention that stops an ordinary noise map from inflating the
        // brush area as a side effect.
        let flat = NoiseImage::new("mid", 4, 4, vec![0.5; 16]).unwrap();
        let f = NoiseField::image(flat, 8.0);
        for p in probes() {
            assert!(f.sample(p, Vec3::Y).abs() < 1e-5, "at {p}");
        }
    }

    #[test]
    fn black_and_white_reach_both_extremes() {
        let black = NoiseField::image(NoiseImage::new("k", 2, 2, vec![0.0; 4]).unwrap(), 8.0);
        let white = NoiseField::image(NoiseImage::new("w", 2, 2, vec![1.0; 4]).unwrap(), 8.0);
        assert!((black.sample(Vec3::ZERO, Vec3::Y) + 1.0).abs() < 1e-5);
        assert!((white.sample(Vec3::ZERO, Vec3::Y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn image_sampling_stays_in_range_and_tiles() {
        let f = NoiseField::image(checker(8, 8), 5.0);
        for p in probes() {
            let v = f.sample(p, Vec3::new(0.3, 0.8, -0.5));
            assert!((-1.0..=1.0).contains(&v), "{v} at {p}");
        }
        // One full repeat apart must read identically -- that is what tiling
        // means, and a clamped sampler would fail it.
        let img = checker(8, 8);
        for (u, v) in [(0.1f32, 0.2f32), (0.63, 0.44)] {
            assert!((img.sample_uv(u, v) - img.sample_uv(u + 1.0, v)).abs() < 1e-5);
            assert!((img.sample_uv(u, v) - img.sample_uv(u, v + 3.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn triplanar_favours_the_axis_the_normal_points_along() {
        // A map that is bright on one projection and dark on another: the
        // blend must follow the normal, or vertical faces get streaks.
        let mut data = vec![0.0f32; 64];
        for (i, d) in data.iter_mut().enumerate() {
            // Bright where the u coordinate is in the left half.
            *d = if (i % 8) < 4 { 1.0 } else { 0.0 };
        }
        let img = NoiseImage::new("ramp", 8, 8, data).unwrap();
        let f = NoiseField::image(img, 10.0);

        // Brightness in this map depends only on the u coordinate, and the
        // three projections feed different world axes into u: the +Y plane
        // uses x, the +X plane uses z. So the probe has to straddle -- x in the
        // bright half, z in the dark half -- or every projection agrees and the
        // test passes without testing anything.
        let p = Vec3::new(3.0, 7.1, 8.0);
        let up = f.sample(p, Vec3::Y);
        let side = f.sample(p, Vec3::X);
        assert!(up > 0.5, "the +Y projection should read the bright half, got {up}");
        assert!(side < -0.5, "the +X projection should read the dark half, got {side}");
    }

    #[test]
    fn a_degenerate_normal_still_samples() {
        // `Move` and friends can hand over a zero normal on perfectly flat
        // field regions; it must not produce NaN.
        let f = NoiseField::image(checker(4, 4), 6.0);
        let v = f.sample(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO);
        assert!(v.is_finite(), "got {v}");
    }

    #[test]
    fn mean_reports_bias() {
        assert!((checker(8, 8).mean() - 0.5).abs() < 1e-6);
        assert!((NoiseImage::new("d", 4, 4, vec![0.25; 16]).unwrap().mean() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn default_pattern_is_built_in_and_needs_no_upload() {
        let f = NoiseField::default();
        assert!(!f.pattern.is_uploaded(), "the tool must work before anything is uploaded");
        assert!(f.sample(Vec3::new(1.0, 2.0, 3.0), Vec3::Y).is_finite());
    }
}

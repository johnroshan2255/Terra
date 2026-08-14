//! CPU-side square heightmap backed by `u16` texels.
//!
//! This is the in-memory form of every `.r16` on disk. `R16Unorm` is the
//! interchange format for the whole pipeline: it uploads to the GPU without
//! conversion, and World Machine, Gaea and Unity all import it, which keeps an
//! escape hatch open if the built-in generator is ever not enough.

use crate::error::{CoreError, Result};
use crate::units::HEIGHT_RANGE_M;
use rayon::prelude::*;

/// A square `u16` heightmap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeightGrid {
    res: u32,
    data: Vec<u16>,
}

impl HeightGrid {
    /// Allocate a flat grid at zero height.
    pub fn new(res: u32) -> Self {
        Self { res, data: vec![0; (res as usize) * (res as usize)] }
    }

    /// Wrap existing texels, verifying they form a square of `res` edge length.
    pub fn from_data(res: u32, data: Vec<u16>) -> Result<Self> {
        let expected = (res as usize) * (res as usize);
        if data.len() != expected {
            return Err(CoreError::GridSize { expected, actual: data.len() });
        }
        Ok(Self { res, data })
    }

    /// Decode little-endian `.r16` bytes.
    pub fn from_r16_bytes(res: u32, bytes: &[u8]) -> Result<Self> {
        let expected = (res as usize) * (res as usize) * 2;
        if bytes.len() != expected {
            return Err(CoreError::GridSize { expected, actual: bytes.len() });
        }
        let data = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Self { res, data })
    }

    /// Encode to little-endian `.r16` bytes.
    pub fn to_r16_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 2);
        for h in &self.data {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    }

    pub fn res(&self) -> u32 {
        self.res
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u16] {
        &mut self.data
    }

    /// Raw texel, clamped to the edge rather than wrapping. Clamping matters:
    /// wrapping would let the erosion solver drain water off one side of the
    /// map and back in on the other.
    pub fn get(&self, x: i32, y: i32) -> u16 {
        let x = x.clamp(0, self.res as i32 - 1) as usize;
        let y = y.clamp(0, self.res as i32 - 1) as usize;
        self.data[y * self.res as usize + x]
    }

    pub fn set(&mut self, x: u32, y: u32, v: u16) {
        debug_assert!(x < self.res && y < self.res);
        self.data[(y * self.res + x) as usize] = v;
    }

    /// Height in meters at a texel.
    pub fn height_m(&self, x: i32, y: i32) -> f32 {
        self.get(x, y) as f32 / u16::MAX as f32 * HEIGHT_RANGE_M
    }

    /// Build from heights in meters, quantizing to `u16` over
    /// `[0, HEIGHT_RANGE_M]`.
    ///
    /// Out-of-range values are clamped rather than wrapped: a sculpt stroke
    /// that digs below the floor should flatten out, not tunnel through and
    /// reappear as a mountain.
    pub fn from_meters(res: u32, meters: &[f32]) -> Result<Self> {
        let expected = (res as usize) * (res as usize);
        if meters.len() != expected {
            return Err(CoreError::GridSize { expected, actual: meters.len() });
        }
        let data = meters
            .par_iter()
            .map(|m| {
                let t = (m / HEIGHT_RANGE_M).clamp(0.0, 1.0);
                (t * u16::MAX as f32).round() as u16
            })
            .collect();
        Ok(Self { res, data })
    }

    /// Decode every texel to meters.
    pub fn to_meters(&self) -> Vec<f32> {
        // A 4096^2 world is 16.7 M texels; this runs on every world load and
        // save, and it is a pure per-element map with nothing to share.
        self.data.par_iter().map(|h| *h as f32 / u16::MAX as f32 * HEIGHT_RANGE_M).collect()
    }

    /// Build from values already in `0..=1`, e.g. a flow or deposition mask.
    /// Same `.r16` container as a heightmap, different meaning per texel.
    pub fn from_unit(res: u32, values: &[f32]) -> Result<Self> {
        let expected = (res as usize) * (res as usize);
        if values.len() != expected {
            return Err(CoreError::GridSize { expected, actual: values.len() });
        }
        let data = values
            .par_iter()
            .map(|v| (v.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16)
            .collect();
        Ok(Self { res, data })
    }

    /// Decode every texel to `0..=1`.
    pub fn to_unit(&self) -> Vec<f32> {
        self.data.par_iter().map(|v| *v as f32 / u16::MAX as f32).collect()
    }

    /// Lowest and highest texel, as raw values. Useful for the editor's
    /// histogram and for auto-framing the camera on load.
    pub fn range(&self) -> (u16, u16) {
        self.data.iter().fold((u16::MAX, 0), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r16_round_trips() {
        let mut g = HeightGrid::new(4);
        g.set(1, 2, 40_000);
        g.set(3, 3, u16::MAX);
        let bytes = g.to_r16_bytes();
        assert_eq!(HeightGrid::from_r16_bytes(4, &bytes).unwrap(), g);
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(HeightGrid::from_data(4, vec![0; 15]).is_err());
        assert!(HeightGrid::from_r16_bytes(4, &[0; 30]).is_err());
    }

    #[test]
    fn meters_round_trip_within_quantization_error() {
        let src = vec![0.0, 256.0, 1024.0, 2047.0];
        let g = HeightGrid::from_meters(2, &src).unwrap();
        for (a, b) in src.iter().zip(g.to_meters()) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn unit_masks_round_trip() {
        let src = vec![0.0, 0.25, 0.5, 1.0];
        let g = HeightGrid::from_unit(2, &src).unwrap();
        for (a, b) in src.iter().zip(g.to_unit()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn out_of_range_meters_clamp_rather_than_wrap() {
        let g = HeightGrid::from_meters(2, &[-500.0, 0.0, HEIGHT_RANGE_M * 2.0, 10.0]).unwrap();
        let m = g.to_meters();
        assert_eq!(m[0], 0.0);
        assert!((m[2] - HEIGHT_RANGE_M).abs() < 0.05);
    }

    #[test]
    fn sampling_clamps_instead_of_wrapping() {
        let mut g = HeightGrid::new(4);
        g.set(0, 0, 1234);
        assert_eq!(g.get(-5, -5), 1234);
        assert_eq!(g.get(99, 0), g.get(3, 0));
    }
}

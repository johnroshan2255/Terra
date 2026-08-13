//! Sculpt deltas -- the user's manual height edits.
//!
//! Stored separately from the generated heightfield so that re-running erosion
//! with new parameters never destroys hand work. Regeneration rebuilds the
//! tier-1 tile, then replays these deltas on top.
//!
//! Sparse because sculpting touches a tiny fraction of a 512x512 tile: a full
//! dense delta layer would be 512 KB per tile whether or not it was used.

use crate::error::{ProjectError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One edited texel: index within the tile, and the signed height change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    /// `y * TEXELS_PER_TILE + x`
    pub index: u32,
    /// Change in raw `u16` height units.
    pub dh: i16,
}

/// All manual edits for one tile.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SculptLayer {
    pub deltas: Vec<Delta>,
}

impl SculptLayer {
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Apply to a baked tile, saturating at the `u16` range so a deep dig or a
    /// tall pile clamps instead of wrapping around.
    pub fn apply(&self, texels: &mut [u16]) {
        for d in &self.deltas {
            if let Some(t) = texels.get_mut(d.index as usize) {
                *t = t.saturating_add_signed(d.dh);
            }
        }
    }

    /// Merge a brush stroke, accumulating where the user paints over an
    /// existing edit rather than replacing it.
    pub fn accumulate(&mut self, index: u32, dh: i16) {
        match self.deltas.iter_mut().find(|d| d.index == index) {
            Some(d) => d.dh = d.dh.saturating_add(dh),
            None => self.deltas.push(Delta { index, dh }),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let Ok(bytes) = std::fs::read(path) else {
            return Ok(Self::default()); // no edits yet is the common case
        };
        let raw = zstd::decode_all(&bytes[..])
            .map_err(|e| ProjectError::Io { path: path.to_path_buf(), source: e })?;
        let deltas = raw
            .chunks_exact(6)
            .map(|c| Delta {
                index: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                dh: i16::from_le_bytes([c[4], c[5]]),
            })
            .collect();
        Ok(Self { deltas })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut raw = Vec::with_capacity(self.deltas.len() * 6);
        for d in &self.deltas {
            raw.extend_from_slice(&d.index.to_le_bytes());
            raw.extend_from_slice(&d.dh.to_le_bytes());
        }
        let packed = zstd::encode_all(&raw[..], 3)
            .map_err(|e| ProjectError::Io { path: path.to_path_buf(), source: e })?;
        std::fs::write(path, packed)
            .map_err(|e| ProjectError::Io { path: path.to_path_buf(), source: e })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_saturates_at_the_extremes() {
        let mut texels = [0u16, u16::MAX];
        let layer =
            SculptLayer { deltas: vec![Delta { index: 0, dh: -500 }, Delta { index: 1, dh: 500 }] };
        layer.apply(&mut texels);
        assert_eq!(texels, [0, u16::MAX]);
    }

    #[test]
    fn repeated_strokes_accumulate() {
        let mut layer = SculptLayer::default();
        layer.accumulate(7, 100);
        layer.accumulate(7, 50);
        assert_eq!(layer.deltas.len(), 1);
        assert_eq!(layer.deltas[0].dh, 150);
    }

    #[test]
    fn round_trips_through_disk() {
        let path = std::env::temp_dir().join("terra-sculpt-test.delta");
        let layer = SculptLayer {
            deltas: vec![Delta { index: 1, dh: -20 }, Delta { index: 99, dh: 4000 }],
        };
        layer.save(&path).unwrap();
        assert_eq!(SculptLayer::load(&path).unwrap(), layer);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn missing_file_reads_as_empty() {
        let path = std::env::temp_dir().join("terra-no-such.delta");
        let _ = std::fs::remove_file(&path);
        assert!(SculptLayer::load(&path).unwrap().is_empty());
    }
}

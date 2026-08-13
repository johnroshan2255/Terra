//! Tile addressing.
//!
//! The world is origin-centered: a 4x4 world spans tile coords `-2..=1` on both
//! axes. Coordinates are signed even though the world cannot grow, because
//! centering on the origin keeps float precision best where the player spends
//! most of their time.

use crate::units::{TILE_SIZE_M, WorldSize};
use serde::{Deserialize, Serialize};

/// Integer address of one terrain tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileCoord {
    pub x: i32,
    pub z: i32,
}

impl TileCoord {
    pub const fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// Filename stem used under `world/cache/tiles/` and `world/edits/sculpt/`.
    /// Negative coordinates render as `-1`, which every target filesystem
    /// accepts.
    pub fn stem(self, prefix: &str) -> String {
        format!("{prefix}_{}_{}", self.x, self.z)
    }

    /// South-west corner of this tile in world meters.
    pub fn origin_m(self) -> (f32, f32) {
        (self.x as f32 * TILE_SIZE_M as f32, self.z as f32 * TILE_SIZE_M as f32)
    }

    /// Center of this tile in world meters.
    pub fn center_m(self) -> (f32, f32) {
        let (x, z) = self.origin_m();
        let h = TILE_SIZE_M as f32 * 0.5;
        (x + h, z + h)
    }
}

/// The fixed rectangular extent of a world's tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileBounds {
    /// Inclusive minimum corner.
    pub min: TileCoord,
    /// Inclusive maximum corner.
    pub max: TileCoord,
}

impl TileBounds {
    /// Origin-centered bounds for a world size.
    ///
    /// For an even tile count the world cannot be perfectly centered, so the
    /// extra tile goes to the negative side: a 4-wide world spans `-2..=1`.
    pub fn centered(size: WorldSize) -> Self {
        let n = size.tiles_per_side() as i32;
        let lo = -(n / 2);
        let hi = lo + n - 1;
        Self { min: TileCoord::new(lo, lo), max: TileCoord::new(hi, hi) }
    }

    pub fn contains(&self, c: TileCoord) -> bool {
        c.x >= self.min.x && c.x <= self.max.x && c.z >= self.min.z && c.z <= self.max.z
    }

    pub fn tiles_per_side(&self) -> u32 {
        (self.max.x - self.min.x + 1) as u32
    }

    /// Every tile in the world, row-major from the minimum corner.
    pub fn iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        (self.min.z..=self.max.z)
            .flat_map(move |z| (self.min.x..=self.max.x).map(move |x| TileCoord::new(x, z)))
    }

    /// Convert a world-space position in meters to the tile containing it.
    pub fn tile_at(&self, x_m: f32, z_m: f32) -> Option<TileCoord> {
        let t = TileCoord::new(
            (x_m / TILE_SIZE_M as f32).floor() as i32,
            (z_m / TILE_SIZE_M as f32).floor() as i32,
        );
        self.contains(t).then_some(t)
    }

    /// Normalized `[0, 1]` position within the world, for sampling tier-0.
    pub fn world_to_uv(&self, x_m: f32, z_m: f32) -> (f32, f32) {
        let span = self.tiles_per_side() as f32 * TILE_SIZE_M as f32;
        let ox = self.min.x as f32 * TILE_SIZE_M as f32;
        let oz = self.min.z as f32 * TILE_SIZE_M as f32;
        ((x_m - ox) / span, (z_m - oz) / span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_are_centered_and_complete() {
        for size in WorldSize::ALL {
            let b = TileBounds::centered(size);
            assert_eq!(b.iter().count() as u32, size.tile_count());
            assert_eq!(b.tiles_per_side(), size.tiles_per_side());
            assert!(b.contains(TileCoord::new(0, 0)), "origin must be inside {size:?}");
        }
    }

    #[test]
    fn medium_world_spans_minus_two_to_one() {
        let b = TileBounds::centered(WorldSize::Medium);
        assert_eq!(b.min, TileCoord::new(-2, -2));
        assert_eq!(b.max, TileCoord::new(1, 1));
    }

    #[test]
    fn negative_coords_make_legal_filenames() {
        assert_eq!(TileCoord::new(-2, 3).stem("h"), "h_-2_3");
    }

    #[test]
    fn uv_covers_unit_square() {
        let b = TileBounds::centered(WorldSize::Medium);
        let (x0, z0) = b.min.origin_m();
        assert_eq!(b.world_to_uv(x0, z0), (0.0, 0.0));
    }
}

//! World scale constants and the fixed set of world sizes.
//!
//! Two numbers are frozen for the lifetime of the format:
//!
//! * [`METERS_PER_TEXEL`] -- the resolution of a baked terrain tile
//! * [`TILE_SIZE_M`]      -- the footprint of one tile
//!
//! World size is chosen once at project creation and never changes. That lets
//! the tier-0 (eroded) heightmap be sized to fit the world exactly, instead of
//! being kept coarse to leave room for growth.

use serde::{Deserialize, Serialize};

/// Resolution of a baked tier-1 terrain tile. Frozen.
pub const METERS_PER_TEXEL: f32 = 2.0;

/// Footprint of one terrain tile, in meters. Frozen.
pub const TILE_SIZE_M: u32 = 1024;

/// Heightmap texels along one edge of a tile. Derived from the two constants
/// above; asserted at compile time so a careless edit cannot desync them.
pub const TEXELS_PER_TILE: u32 = 512;

const _: () = assert!(TEXELS_PER_TILE as f32 * METERS_PER_TEXEL == TILE_SIZE_M as f32);

/// Vertical range of the heightfield in meters. Heights are stored as `u16`
/// normalized over this range, giving ~3.1 cm of vertical precision.
pub const HEIGHT_RANGE_M: f32 = 2048.0;

/// Height a freshly created (flat) world sits at.
///
/// Not zero: storage is unsigned, so a world starting at the bottom of the
/// range could not be dug into at all. This leaves 256 m of headroom below the
/// starting surface for valleys and riverbeds.
pub const BASE_ELEVATION_M: f32 = 256.0;

const _: () = assert!(BASE_ELEVATION_M > 0.0 && BASE_ELEVATION_M < HEIGHT_RANGE_M);

/// The world sizes offered at project creation.
///
/// Tier-0 resolution is chosen so hydraulic erosion runs at ~4 m/texel for
/// every size. Erosion happens on a single image with no tiling, so there are
/// never seams to hide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldSize {
    /// 2 km across. 2x2 tiles. Prototypes and test levels.
    Small,
    /// 4 km across. 4x4 tiles. The recommended default.
    Medium,
    /// 8 km across. 8x8 tiles.
    Large,
    /// 16 km across. 16x16 tiles. Terrain is cheap at this scale; filling it
    /// with content is not.
    Huge,
}

impl WorldSize {
    pub const ALL: [WorldSize; 4] =
        [WorldSize::Small, WorldSize::Medium, WorldSize::Large, WorldSize::Huge];

    /// Tiles along one edge of the world.
    pub const fn tiles_per_side(self) -> u32 {
        match self {
            WorldSize::Small => 2,
            WorldSize::Medium => 4,
            WorldSize::Large => 8,
            WorldSize::Huge => 16,
        }
    }

    /// Total tile count.
    pub const fn tile_count(self) -> u32 {
        let n = self.tiles_per_side();
        n * n
    }

    /// Width of the world in meters.
    pub const fn extent_m(self) -> u32 {
        self.tiles_per_side() * TILE_SIZE_M
    }

    /// Edge length of the tier-0 heightmap, in texels. This is the image
    /// hydraulic erosion actually runs on.
    pub const fn tier0_res(self) -> u32 {
        match self {
            WorldSize::Small => 1024,
            WorldSize::Medium => 1024,
            WorldSize::Large => 2048,
            WorldSize::Huge => 4096,
        }
    }

    /// Ground distance covered by one tier-0 texel.
    pub fn tier0_meters_per_texel(self) -> f32 {
        self.extent_m() as f32 / self.tier0_res() as f32
    }

    /// Bytes of `R16Unorm` tier-0 data for one map layer.
    pub const fn tier0_bytes(self) -> u64 {
        let r = self.tier0_res() as u64;
        r * r * 2
    }

    /// Label for the creation UI.
    pub fn label(self) -> &'static str {
        match self {
            WorldSize::Small => "Small - 2 km",
            WorldSize::Medium => "Medium - 4 km",
            WorldSize::Large => "Large - 8 km",
            WorldSize::Huge => "Huge - 16 km",
        }
    }

    /// Reconstruct from the tile count stored in a manifest. Used on load to
    /// verify the manifest agrees with the files on disk.
    pub fn from_tiles_per_side(n: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.tiles_per_side() == n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erosion_resolution_stays_in_range() {
        // Every size must erode somewhere in the 2-8 m/texel band. Coarser and
        // drainage detail disappears; finer and bake time balloons.
        for size in WorldSize::ALL {
            let mpt = size.tier0_meters_per_texel();
            assert!((2.0..=8.0).contains(&mpt), "{size:?} erodes at {mpt} m/texel");
        }
    }

    #[test]
    fn round_trips_through_tile_count() {
        for size in WorldSize::ALL {
            assert_eq!(WorldSize::from_tiles_per_side(size.tiles_per_side()), Some(size));
        }
    }
}

//! Types shared by every other crate: world scale, tile addressing, and the
//! CPU-side heightmap. Deliberately has no GPU, filesystem, or UI dependency so
//! it stays cheap to compile and easy to unit test.

pub mod coords;
pub mod error;
pub mod grid;
pub mod units;
pub mod vehicle;

pub use coords::{TileBounds, TileCoord};
pub use error::{CoreError, Result};
pub use grid::HeightGrid;
pub use units::{
    BASE_ELEVATION_M, HEIGHT_RANGE_M, METERS_PER_TEXEL, TEXELS_PER_TILE, TILE_SIZE_M, WorldSize,
};
pub use vehicle::VehicleDims;

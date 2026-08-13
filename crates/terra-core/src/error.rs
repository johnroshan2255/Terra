//! Errors for the shared data types.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("heightmap size mismatch: expected {expected} elements, got {actual}")]
    GridSize { expected: usize, actual: usize },

    #[error("tile {x},{z} is outside the world bounds")]
    TileOutOfBounds { x: i32, z: i32 },

    #[error("{0} is not one of the supported world sizes")]
    UnknownWorldSize(u32),
}

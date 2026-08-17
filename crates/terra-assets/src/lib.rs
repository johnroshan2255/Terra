//! Shared asset library: meshes and textures usable by any project.
//!
//! Assets are referenced by stable `AssetId`, never by path. Paths break the
//! moment a user reorganizes a folder; an id plus a content hash survives it,
//! and lets ten projects share one copy of the same boulder.

pub mod db;
pub mod mesh;
pub mod rig;

pub use db::{AssetDb, AssetId};
pub use mesh::MeshData;
pub use rig::VehicleRig;

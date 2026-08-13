//! Project and world persistence: the folder layout, the manifests, the
//! library index, and the format-version guard.
//!
//! Nothing here touches the GPU. Save/load is testable and fast to compile,
//! which matters because it is the layer everything else is built on.

pub mod data;
pub mod edits;
pub mod error;
pub mod layout;
pub mod library;
pub mod params;
pub mod project;
pub mod roads;
pub mod version;
pub mod world;

pub use data::WorldData;
pub use edits::{Delta, SculptLayer};
pub use error::{ProjectError, Result};
pub use layout::ProjectPaths;
pub use library::{Library, ProjectEntry};
pub use params::{ErosionParams, RmfParams, TerrainParams, ThermalParams};
pub use project::{Project, ProjectManifest};
pub use roads::{Road, RoadNetwork, Surface};
pub use version::FORMAT_VERSION;
pub use world::WorldManifest;

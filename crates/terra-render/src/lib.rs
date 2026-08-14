//! Rendering: wgpu surface setup, camera, terrain, instanced scatter.
//!
//! Shared by the editor and the runtime so that what the designer sees is what
//! the player gets.

pub mod camera;
pub mod cdlod;
pub mod context;
pub mod grass;
pub mod instancing;
pub mod hiz;
pub mod lighting;
pub mod material;
pub mod mesh;
pub mod post;
pub mod scatter;
pub mod sky;
pub mod stats;
pub mod taa;
pub mod terrain;
pub mod texture_set;
pub mod volumetrics;

pub use camera::Camera;
pub use context::RenderContext;
pub use grass::{Grass, GrassSettings};
pub use lighting::{Lighting, ShadowQuality, SkySettings, Sun};
pub use material::Materials;
pub use mesh::{Instance, MeshRenderer};
pub use post::Post;
pub use scatter::{Rules, Scatter, Species};
pub use sky::Sky;
pub use stats::{FrameStats, GpuTimer, Ring};
pub use taa::Taa;
pub use terrain::{SculptMode, Terrain};
pub use volumetrics::{FogSettings, FroxelGrids, Volumetrics};

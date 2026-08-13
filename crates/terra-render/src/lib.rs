//! Rendering: wgpu surface setup, camera, terrain, instanced scatter.
//!
//! Shared by the editor and the runtime so that what the designer sees is what
//! the player gets.

pub mod camera;
pub mod cdlod;
pub mod context;
pub mod instancing;
pub mod mesh;
pub mod sky;
pub mod stats;
pub mod terrain;

pub use camera::Camera;
pub use context::RenderContext;
pub use mesh::{Instance, MeshRenderer};
pub use sky::Sky;
pub use stats::{FrameStats, GpuTimer, Ring};
pub use terrain::{SculptMode, Terrain};

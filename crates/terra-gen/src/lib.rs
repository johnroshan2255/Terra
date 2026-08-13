//! GPU terrain generation.
//!
//! The pipeline, in order:
//!
//! ```text
//! 1. heightfield   ridged multifractal + domain warp   -> tier-0 raw
//! 2. thermal       pre-pass, relaxes noise artifacts
//! 3. erosion       6-pass grid hydraulic solver        -> tier-0 carved
//! 4. thermal       post-pass, talus at cliff bases
//! 5. tiles         upsample + ridged detail            -> tier-1
//! ```
//!
//! Two invariants, enforced by tests in [`shaders`]:
//!
//! * Height always comes from ridged multifractal noise, never fBm.
//! * Erosion features always come from the step-3 simulation, never from a
//!   noise function shaped to imitate them.
//!
//! Parameters come from `terra_project::params`; this crate owns only the
//! pipelines.

pub mod erosion;
pub mod gpu;
pub mod heightfield;
pub mod road;
pub mod shaders;
pub mod thermal;
pub mod tiles;

pub use gpu::GpuContext;

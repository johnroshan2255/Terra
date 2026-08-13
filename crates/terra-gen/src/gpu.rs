//! Headless wgpu device used for baking.
//!
//! Separate from the renderer's device so terrain can be generated without a
//! window -- needed for CLI bakes and for tests.

// TODO: request_adapter / request_device with the limits erosion needs
//       (storage buffer bindings up to the tier-0 working set).
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

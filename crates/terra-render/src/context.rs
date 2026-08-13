//! Surface, device, queue and depth buffer.

use anyhow::{Context, Result};
use std::sync::Arc;
use winit::window::Window;

/// Depth format. `Depth32Float` rather than 24-bit because we use reversed-Z,
/// which spends its precision near the far plane where 24 bits is not enough at
/// 16 km view distances.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
    window: Arc<Window>,
    vsync: bool,
    supports_uncapped: bool,
    timestamps: bool,
}

impl RenderContext {
    pub async fn new(window: Arc<Window>, vsync: bool) -> Result<Self> {
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .context("no suitable GPU adapter")?;

        let info = adapter.get_info();
        log::info!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // Timestamp queries are what make the GPU column of the perf overlay
        // real rather than inferred. Optional: not every adapter has them.
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if !timestamps {
            log::warn!("adapter has no TIMESTAMP_QUERY; GPU timing unavailable");
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("terra-device"),
                required_features: if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .context("failed to create device")?;

        let caps = surface.get_capabilities(&adapter);
        // Deliberately a NON-sRGB format. egui emits colors that are already
        // gamma-encoded and warns against an sRGB target; the terrain shader
        // therefore applies the sRGB transfer function itself.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);

        // Uncapped presentation is what makes a 200 FPS (5 ms) target
        // measurable at all -- with Fifo the GPU idles at the 75 Hz refresh and
        // every frame reads as 13.3 ms regardless of how fast it really was.
        let supports_uncapped = caps.present_modes.contains(&wgpu::PresentMode::Immediate)
            || caps.present_modes.contains(&wgpu::PresentMode::Mailbox);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Srgb,
            width: w,
            height: h,
            present_mode: pick_present_mode(&caps, vsync),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = create_depth(&device, w, h);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            depth_view,
            window,
            vsync,
            supports_uncapped,
            timestamps,
        })
    }

    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    pub fn supports_uncapped(&self) -> bool {
        self.supports_uncapped
    }

    /// Whether GPU timestamp queries are available on this device.
    pub fn supports_timestamps(&self) -> bool {
        self.timestamps
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    /// Re-apply the current configuration. Used when the surface reports
    /// Outdated or Lost, where the size has not changed but the swapchain must
    /// be rebuilt.
    pub fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 || (w == self.config.width && h == self.config.height) {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, w, h);
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        if vsync == self.vsync {
            return;
        }
        self.vsync = vsync;
        self.config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else if self.supports_uncapped {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::AutoVsync
        };
        self.surface.configure(&self.device, &self.config);
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }
}

fn pick_present_mode(caps: &wgpu::SurfaceCapabilities, vsync: bool) -> wgpu::PresentMode {
    if vsync {
        return wgpu::PresentMode::AutoVsync;
    }
    for mode in [wgpu::PresentMode::Immediate, wgpu::PresentMode::Mailbox] {
        if caps.present_modes.contains(&mode) {
            return mode;
        }
    }
    wgpu::PresentMode::AutoVsync
}

fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

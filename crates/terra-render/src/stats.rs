//! Frame timing: rolling history and real GPU timestamps.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Number of frames kept for the graph. At 75 Hz this is ~2.7 s of history --
/// long enough to see a hitch, short enough that the graph still reacts.
pub const HISTORY: usize = 200;

/// Fixed-capacity rolling buffer of timings, oldest first when iterated.
#[derive(Clone)]
pub struct Ring {
    buf: Vec<f32>,
    head: usize,
    len: usize,
}

impl Ring {
    pub fn new(cap: usize) -> Self {
        Self { buf: vec![0.0; cap], head: 0, len: 0 }
    }

    pub fn push(&mut self, v: f32) {
        self.buf[self.head] = v;
        self.head = (self.head + 1) % self.buf.len();
        self.len = (self.len + 1).min(self.buf.len());
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Oldest sample first.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        let cap = self.buf.len();
        let start = (self.head + cap - self.len) % cap;
        (0..self.len).map(move |i| self.buf[(start + i) % cap])
    }

    pub fn last(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        self.buf[(self.head + self.buf.len() - 1) % self.buf.len()]
    }

    pub fn max(&self) -> f32 {
        self.iter().fold(0.0, f32::max)
    }

    pub fn avg(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        self.iter().sum::<f32>() / self.len as f32
    }

    /// Worst 1%, the number that actually correlates with visible stutter --
    /// an average hides a hitch every second entirely.
    pub fn p99(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let mut v: Vec<f32> = self.iter().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[((v.len() as f32 * 0.99) as usize).min(v.len() - 1)]
    }
}

/// Rolling history for every series the overlay draws.
pub struct FrameStats {
    pub frame: Ring,
    pub cpu: Ring,
    pub gpu: Ring,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self { frame: Ring::new(HISTORY), cpu: Ring::new(HISTORY), gpu: Ring::new(HISTORY) }
    }
}

impl FrameStats {
    pub fn fps(&self) -> f32 {
        let ms = self.frame.avg();
        if ms > 0.0 { 1000.0 / ms } else { 0.0 }
    }
}

/// GPU timing via timestamp queries.
///
/// Readback is asynchronous, so only one measurement is in flight at a time and
/// the result lands a frame or two later. That is fine for a HUD and avoids the
/// pipeline stall a synchronous read would cost.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    ready: Arc<AtomicBool>,
    in_flight: bool,
    last_ms: f32,
    reported: bool,
}

/// Timestamps per frame, as begin/end pairs: scene, then UI.
const QUERIES: u32 = 4;
const BYTES: u64 = QUERIES as u64 * 8;

impl GpuTimer {
    /// `None` when the adapter does not support timestamp queries.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, supported: bool) -> Option<Self> {
        if !supported {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERIES,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp-resolve"),
            size: BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp-readback"),
            size: BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve,
            readback,
            period_ns: queue.get_timestamp_period(),
            ready: Arc::new(AtomicBool::new(false)),
            in_flight: false,
            last_ms: 0.0,
            reported: false,
        })
    }

    /// Whether this frame should carry timestamp writes. False while a previous
    /// measurement is still being read back.
    pub fn arm(&self) -> bool {
        !self.in_flight
    }

    pub fn scene_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.arm().then_some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    pub fn ui_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.arm().then_some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(2),
            end_of_pass_write_index: Some(3),
        })
    }

    /// Queue resolution of this frame's timestamps. Call after the passes are
    /// encoded and before `finish`.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.arm() {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..QUERIES, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, BYTES);
    }

    /// Begin the asynchronous readback. Call after `queue.submit`.
    pub fn map(&mut self) {
        if self.in_flight {
            return;
        }
        self.in_flight = true;
        let ready = self.ready.clone();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
    }

    /// Collect a completed measurement, if one has landed. Call once per frame
    /// after polling the device.
    pub fn poll(&mut self) -> Option<f32> {
        if !self.ready.swap(false, Ordering::Acquire) {
            return None;
        }
        {
            // get_mapped_range is fallible in wgpu 30; a failure here means the
            // mapping raced, so drop the sample rather than reporting garbage.
            let Ok(view) = self.readback.slice(..).get_mapped_range() else {
                self.readback.unmap();
                self.in_flight = false;
                return None;
            };
            let ts: &[u64] = bytemuck::cast_slice(&view);

            // Sum each pass independently rather than taking last-minus-first.
            //
            // The Metal backend does not write the end-of-pass timestamp for
            // the final pass -- it comes back as 0 -- so a span from the first
            // to the last query saturates to zero and reports a GPU time of
            // 0.00 ms. Per-pair accumulation degrades to "scene only" instead,
            // which is both correct and the number that matters for the budget.
            let mut ticks = 0u64;
            let mut dropped = 0;
            for pair in ts.chunks_exact(2) {
                match (pair[0], pair[1]) {
                    (a, b) if b > a => ticks += b - a,
                    _ => dropped += 1,
                }
            }
            self.last_ms = ticks as f32 * self.period_ns / 1.0e6;

            if !self.reported {
                self.reported = true;
                if dropped > 0 {
                    log::info!(
                        "gpu timing: {dropped} of {} pass timers unwritten by this backend; \
                         reporting the rest ({:.3} ms)",
                        ts.len() / 2,
                        self.last_ms
                    );
                } else {
                    log::info!("gpu timing: {:.3} ms across {} passes", self.last_ms, ts.len() / 2);
                }
            }
        }
        self.readback.unmap();
        self.in_flight = false;
        Some(self.last_ms)
    }

    pub fn last_ms(&self) -> f32 {
        self.last_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_yields_oldest_first_and_wraps() {
        let mut r = Ring::new(3);
        for v in [1.0, 2.0, 3.0, 4.0] {
            r.push(v);
        }
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![2.0, 3.0, 4.0]);
        assert_eq!(r.last(), 4.0);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn empty_ring_is_safe() {
        let r = Ring::new(8);
        assert!(r.is_empty());
        assert_eq!(r.last(), 0.0);
        assert_eq!(r.avg(), 0.0);
        assert_eq!(r.max(), 0.0);
        assert_eq!(r.p99(), 0.0);
    }

    #[test]
    fn p99_tracks_the_hitch_that_the_average_hides() {
        let mut r = Ring::new(100);
        for _ in 0..99 {
            r.push(5.0);
        }
        r.push(80.0);
        assert!(r.avg() < 6.0, "average absorbs the spike");
        assert_eq!(r.p99(), 80.0, "p99 must surface it");
    }

    #[test]
    fn fps_derives_from_average_frame_time() {
        let mut s = FrameStats::default();
        for _ in 0..10 {
            s.frame.push(20.0);
        }
        assert!((s.fps() - 50.0).abs() < 0.01);
    }
}

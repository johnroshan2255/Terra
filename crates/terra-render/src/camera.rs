//! Fly camera with a reversed-Z, infinite-far projection.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub eye: [f32; 4],
}

#[derive(Clone)]
pub struct Camera {
    pub pos: Vec3,
    /// Radians, right-handed about +Y.
    pub yaw: f32,
    /// Radians, clamped just short of straight up/down to avoid gimbal flip.
    pub pitch: f32,
    pub fov_y: f32,
    pub znear: f32,
    pub speed: f32,
    /// Sub-pixel offset in NDC, set by the temporal resolve. Applied inside
    /// `projection`, so every pass that builds a matrix from this camera --
    /// scene, culling, reprojection -- agrees about where the samples landed.
    pub jitter: glam::Vec2,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: Vec3::new(0.0, 420.0, 900.0),
            yaw: std::f32::consts::FRAC_PI_2 * 3.0,
            pitch: -0.32,
            fov_y: 60f32.to_radians(),
            znear: 0.25,
            speed: 120.0,
            jitter: glam::Vec2::ZERO,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Right-handed look-to, built by hand rather than via a glam helper --
    /// glam has deprecated and moved those between minor versions.
    pub fn look_at(&self) -> Mat4 {
        let f = self.forward();
        let s = f.cross(Vec3::Y).normalize();
        let u = s.cross(f);
        Mat4::from_cols(
            Vec4::new(s.x, u.x, -f.x, 0.0),
            Vec4::new(s.y, u.y, -f.y, 0.0),
            Vec4::new(s.z, u.z, -f.z, 0.0),
            Vec4::new(-s.dot(self.pos), -u.dot(self.pos), f.dot(self.pos), 1.0),
        )
    }

    /// Reversed-Z with an infinite far plane.
    ///
    /// Depth 1.0 is the near plane and 0.0 is infinity, so the compare function
    /// is `Greater` and the buffer clears to 0.0. This puts float precision
    /// where the terrain actually is; a conventional projection z-fights badly
    /// on distant ridgelines at 8-16 km.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        Mat4::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(self.jitter.x, self.jitter.y, 0.0, -1.0),
            Vec4::new(0.0, 0.0, self.znear, 0.0),
        )
    }

    pub fn uniform(&self, aspect: f32) -> CameraUniform {
        let vp = self.projection(aspect) * self.look_at();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            eye: self.pos.extend(1.0).into(),
        }
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        const LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        self.yaw += dx;
        self.pitch = (self.pitch - dy).clamp(-LIMIT, LIMIT);
    }

    /// Camera-space up. Matches the basis [`Self::look_at`] builds, so panning
    /// moves along the same axes the image is framed by.
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    /// World distance one pixel of drag covers at `dist` metres.
    ///
    /// Derived from the projection rather than picked by feel: it is what makes
    /// the ground stay under the cursor while dragging, at any altitude and any
    /// field of view. A constant here is the reason so many hand-rolled cameras
    /// pan too fast up high and too slow at ground level.
    pub fn pixel_scale(&self, dist: f32, viewport_h: f32) -> f32 {
        2.0 * dist * (self.fov_y * 0.5).tan() / viewport_h.max(1.0)
    }

    /// Drag the view. `dx`/`dy` are pixels; the scene follows the cursor, so
    /// the camera moves against the drag.
    pub fn pan(&mut self, dx: f32, dy: f32, scale: f32) {
        self.pos += (-self.right() * dx + self.up() * dy) * scale;
    }

    /// Move along the view direction. Positive is toward what you are looking
    /// at.
    pub fn dolly(&mut self, metres: f32) {
        self.pos += self.forward() * metres;
    }

    /// Aim at a point without moving.
    pub fn look_toward(&mut self, target: Vec3) {
        let d = target - self.pos;
        if d.length_squared() < 1e-6 {
            return;
        }
        let d = d.normalize();
        self.yaw = d.z.atan2(d.x);
        self.pitch = d.y.clamp(-1.0, 1.0).asin();
    }

    /// Swing around `pivot`, keeping it framed and keeping the distance to it.
    ///
    /// Distinct from [`Self::rotate`], which turns the camera on the spot: to
    /// look *at* a piece of terrain from another side you want the camera to
    /// travel around it, which is what every DCC tool means by orbit and what
    /// turning in place can never do.
    pub fn orbit(&mut self, pivot: Vec3, dyaw: f32, dpitch: f32) {
        const LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.02;
        let offset = self.pos - pivot;
        let radius = offset.length();
        if radius < 1e-3 {
            return;
        }
        let azimuth = offset.z.atan2(offset.x) + dyaw;
        let elevation = ((offset.y / radius).clamp(-1.0, 1.0).asin() + dpitch).clamp(-LIMIT, LIMIT);
        let (sa, ca) = azimuth.sin_cos();
        let (se, ce) = elevation.sin_cos();
        self.pos = pivot + Vec3::new(ca * ce, se, sa * ce) * radius;
        self.look_toward(pivot);
    }

    /// `dir` is (right, up, forward) in camera space, each in -1..=1.
    pub fn translate(&mut self, dir: Vec3, dt: f32, boost: bool) {
        if dir == Vec3::ZERO {
            return;
        }
        let speed = self.speed * if boost { 6.0 } else { 1.0 };
        let delta = self.right() * dir.x + Vec3::Y * dir.y + self.forward() * dir.z;
        self.pos += delta.normalize_or_zero() * speed * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversed_z_maps_near_to_one_and_far_toward_zero() {
        let cam = Camera { pos: Vec3::ZERO, znear: 0.1, ..Default::default() };
        let p = cam.projection(16.0 / 9.0);

        let near = p * Vec4::new(0.0, 0.0, -0.1, 1.0);
        assert!((near.z / near.w - 1.0).abs() < 1e-4, "near plane must be depth 1");

        let far = p * Vec4::new(0.0, 0.0, -100_000.0, 1.0);
        assert!(far.z / far.w < 1e-4, "distance must approach depth 0");
    }

    #[test]
    fn panning_moves_the_scene_with_the_cursor() {
        // Dragging right must carry the view right, which means the camera
        // itself travels left. Getting this backwards is the single most
        // common way a pan control feels wrong.
        let mut cam = Camera { pos: Vec3::ZERO, yaw: 0.0, pitch: 0.0, ..Default::default() };
        let right = cam.right();
        cam.pan(10.0, 0.0, 1.0);
        assert!(cam.pos.dot(right) < 0.0, "dragging right must move the camera left");

        let mut cam = Camera { pos: Vec3::ZERO, yaw: 0.0, pitch: 0.0, ..Default::default() };
        cam.pan(0.0, 10.0, 1.0);
        assert!(cam.pos.dot(cam.up()) > 0.0, "dragging down must move the camera up");
    }

    #[test]
    fn pan_scale_tracks_distance_and_field_of_view() {
        let cam = Camera { fov_y: 60f32.to_radians(), ..Default::default() };
        let near = cam.pixel_scale(100.0, 900.0);
        let far = cam.pixel_scale(1000.0, 900.0);
        // Ten times the distance is ten times the world span per pixel, which
        // is what keeps the drag feeling identical at any altitude.
        assert!((far / near - 10.0).abs() < 1e-3, "{near} -> {far}");

        let wide = Camera { fov_y: 90f32.to_radians(), ..Default::default() };
        assert!(wide.pixel_scale(100.0, 900.0) > near, "a wider view covers more per pixel");
    }

    #[test]
    fn dolly_moves_along_the_view_direction() {
        let mut cam = Camera { pos: Vec3::ZERO, ..Default::default() };
        let f = cam.forward();
        cam.dolly(5.0);
        assert!((cam.pos - f * 5.0).length() < 1e-4);
        cam.dolly(-5.0);
        assert!(cam.pos.length() < 1e-4, "zooming out must undo zooming in");
    }

    #[test]
    fn orbit_circles_the_pivot_without_leaving_it() {
        let pivot = Vec3::new(10.0, 0.0, 0.0);
        let mut cam = Camera { pos: Vec3::new(10.0, 0.0, 100.0), ..Default::default() };
        let before = (cam.pos - pivot).length();

        cam.orbit(pivot, 0.7, 0.2);
        let after = (cam.pos - pivot).length();
        assert!((before - after).abs() < 1e-3, "orbit must preserve distance: {before} -> {after}");
        assert!(cam.pos.distance(Vec3::new(10.0, 0.0, 100.0)) > 1.0, "the camera must move");

        // And it must still be looking at what it is circling.
        let to_pivot = (pivot - cam.pos).normalize();
        assert!(cam.forward().dot(to_pivot) > 0.999, "orbit must keep the pivot framed");
    }

    #[test]
    fn orbit_cannot_tip_over_the_pole() {
        let pivot = Vec3::ZERO;
        let mut cam = Camera { pos: Vec3::new(0.0, 0.0, 50.0), ..Default::default() };
        for _ in 0..200 {
            cam.orbit(pivot, 0.0, 1.0);
        }
        assert!(cam.pos.is_finite());
        assert!((cam.pos - pivot).length() > 49.0, "distance must survive the clamp");
    }

    #[test]
    fn pitch_cannot_flip_over_the_pole() {
        let mut cam = Camera::default();
        for _ in 0..200 {
            cam.rotate(0.0, 1.0);
        }
        assert!(cam.pitch > -std::f32::consts::FRAC_PI_2);
        assert!(cam.forward().is_finite());
    }
}

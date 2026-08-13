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
            Vec4::new(0.0, 0.0, 0.0, -1.0),
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
    fn pitch_cannot_flip_over_the_pole() {
        let mut cam = Camera::default();
        for _ in 0..200 {
            cam.rotate(0.0, 1.0);
        }
        assert!(cam.pitch > -std::f32::consts::FRAC_PI_2);
        assert!(cam.forward().is_finite());
    }
}

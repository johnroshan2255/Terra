//! Frustum planes, and the tests every culling pass shares.
//!
//! Extracted from `scatter.rs`, which had it privately, because the terrain and the
//! water need the same planes and a second copy of the plane extraction is the last
//! thing this should have. Getting the extraction subtly wrong is invisible until
//! something disappears at an angle nobody tried.
//!
//! # Conservative, always
//!
//! Both tests may return `true` for something outside the frustum. Neither may ever
//! return `false` for something inside it. A cull that is too eager pops geometry out of
//! view, which is a bug a user sees immediately and cannot work around; a cull that is
//! too lax costs frame time, which is a number. So every inequality here is `>=` or
//! signed the safe way, and the AABB test uses the corner *furthest along* each plane
//! normal rather than the box centre.

use glam::{Mat4, Vec3, Vec4};

/// Six world-space planes, `xyz` normal and `w` distance, pointing inward.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the planes from a view-projection matrix.
    pub fn new(view_proj: &Mat4) -> Self {
        let m = view_proj.transpose();
        // Reversed-Z with an infinite far plane: there is no far plane to extract, so
        // only five are meaningful and the sixth is left degenerate. A degenerate plane
        // has a zero normal and a positive `w`, so every test passes it -- which is the
        // safe direction, and why it can be left in rather than special-cased.
        let planes = [
            m.w_axis + m.x_axis,
            m.w_axis - m.x_axis,
            m.w_axis + m.y_axis,
            m.w_axis - m.y_axis,
            m.w_axis - m.z_axis,
            m.w_axis,
        ];
        Self { planes }
    }

    /// Whether a sphere is at least partly inside.
    ///
    /// Mirrors the test in `scatter_cull.wgsl` exactly. Only the shader culls instances
    /// in anger; this exists so the plane extraction is covered by a test that runs
    /// without a GPU.
    pub fn intersects_sphere(&self, centre: Vec3, radius: f32) -> bool {
        self.planes.iter().all(|p| p.truncate().dot(centre) + p.w >= -radius)
    }

    /// Whether an axis-aligned box is at least partly inside.
    ///
    /// The standard positive-vertex test: for each plane, take the box corner furthest
    /// along that plane's normal. If even *that* corner is behind the plane, no part of
    /// the box can be in front of it, so the box is out. Using the centre instead would
    /// reject boxes that straddle a plane, which is exactly the case a terrain patch at
    /// the edge of the view is in.
    ///
    /// This is what makes quadtree culling work: reject a node and its whole subtree goes
    /// with it, because a child's box is contained in its parent's.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|p| {
            let n = p.truncate();
            let far = Vec3::new(
                if n.x >= 0.0 { max.x } else { min.x },
                if n.y >= 0.0 { max.y } else { min.y },
                if n.z >= 0.0 { max.z } else { min.z },
            );
            n.dot(far) + p.w >= 0.0
        })
    }
}

/// Several frusta treated as one volume: inside any of them is inside.
///
/// What the shadow passes cull against. A caster only has to be in *some* cascade to
/// matter, and testing the union is both correct and cheaper than keeping a separate
/// visible set per cascade.
#[derive(Debug, Clone, Default)]
pub struct FrustumUnion {
    frusta: Vec<Frustum>,
}

impl FrustumUnion {
    pub fn new(frusta: impl IntoIterator<Item = Frustum>) -> Self {
        Self { frusta: frusta.into_iter().collect() }
    }

    /// Empty means "cull nothing", not "cull everything".
    ///
    /// The safe reading: a union with no frusta in it is a caller that has not been told
    /// where the light is, and the answer to that is to draw, not to make the world
    /// vanish.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.frusta.is_empty() || self.frusta.iter().any(|f| f.intersects_aabb(min, max))
    }

    pub fn is_empty(&self) -> bool {
        self.frusta.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;

    /// A camera at the origin looking along +X, which is what yaw 0 means here.
    fn cam() -> Camera {
        Camera { pos: Vec3::ZERO, yaw: 0.0, pitch: 0.0, ..Camera::default() }
    }

    fn frustum() -> Frustum {
        let c = cam();
        Frustum::new(&(c.projection(1.0) * c.look_at()))
    }

    #[test]
    fn something_straight_ahead_is_inside() {
        let f = frustum();
        assert!(f.intersects_sphere(Vec3::new(100.0, 0.0, 0.0), 1.0));
        assert!(f.intersects_aabb(Vec3::new(90.0, -5.0, -5.0), Vec3::new(110.0, 5.0, 5.0)));
    }

    #[test]
    fn something_behind_the_camera_is_outside() {
        // The case that matters most for the terrain: on a world centred on the camera,
        // half the quadtree is behind it.
        let f = frustum();
        assert!(!f.intersects_sphere(Vec3::new(-100.0, 0.0, 0.0), 1.0));
        assert!(!f.intersects_aabb(Vec3::new(-110.0, -5.0, -5.0), Vec3::new(-90.0, 5.0, 5.0)));
    }

    #[test]
    fn a_box_straddling_a_plane_is_kept() {
        // The positive-vertex test exists for this. A box half in view must not be
        // rejected, and a centre-only test would reject it whenever the centre fell
        // outside.
        let f = frustum();
        let min = Vec3::new(-10.0, -5.0, -5.0);
        let max = Vec3::new(200.0, 5.0, 5.0);
        assert!(f.intersects_aabb(min, max), "a box spanning the near plane was culled");
    }

    #[test]
    fn a_tall_box_whose_centre_is_out_of_view_is_kept() {
        // A terrain patch far to the side but tall enough to poke into view. Culling it
        // would clip a mountain out of the frame.
        let f = frustum();
        let min = Vec3::new(50.0, -2000.0, -5.0);
        let max = Vec3::new(70.0, 2000.0, 5.0);
        assert!(f.intersects_aabb(min, max));
    }

    #[test]
    fn a_child_box_is_never_kept_when_its_parent_is_culled() {
        // The property quadtree culling rests on: rejecting a node has to be safe for its
        // whole subtree. Checked over a grid of child boxes inside a culled parent.
        let f = frustum();
        let (pmin, pmax) = (Vec3::new(-400.0, -50.0, -400.0), Vec3::new(-200.0, 50.0, -200.0));
        assert!(!f.intersects_aabb(pmin, pmax), "the parent should be behind the camera");
        for i in 0..4 {
            for j in 0..4 {
                let step = (pmax - pmin) / 4.0;
                let min = pmin + Vec3::new(step.x * i as f32, 0.0, step.z * j as f32);
                let max = min + step;
                assert!(
                    !f.intersects_aabb(min, max),
                    "child {i},{j} survived a culled parent, so subtree pruning is unsound"
                );
            }
        }
    }

    #[test]
    fn a_union_of_one_matches_that_frustum() {
        let f = frustum();
        let u = FrustumUnion::new([f]);
        assert!(u.intersects_aabb(Vec3::new(90.0, -1.0, -1.0), Vec3::new(110.0, 1.0, 1.0)));
        assert!(!u.intersects_aabb(Vec3::new(-110.0, -1.0, -1.0), Vec3::new(-90.0, 1.0, 1.0)));
    }

    #[test]
    fn an_empty_union_culls_nothing() {
        // A caller that has not been told where the light is must draw, not make the world
        // vanish.
        let u = FrustumUnion::default();
        assert!(u.intersects_aabb(Vec3::splat(-1e6), Vec3::splat(-1e6 + 1.0)));
    }

    #[test]
    fn a_union_keeps_what_any_member_can_see() {
        // Two cameras looking opposite ways: everything either sees is in the union.
        let a = frustum();
        let mut back = cam();
        back.yaw = std::f32::consts::PI;
        let b = Frustum::new(&(back.projection(1.0) * back.look_at()));
        let u = FrustumUnion::new([a, b]);
        assert!(u.intersects_aabb(Vec3::new(90.0, -1.0, -1.0), Vec3::new(110.0, 1.0, 1.0)));
        assert!(
            u.intersects_aabb(Vec3::new(-110.0, -1.0, -1.0), Vec3::new(-90.0, 1.0, 1.0)),
            "the second frustum's view was culled"
        );
    }
}

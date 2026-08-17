//! Signed distance primitives and the boolean operators that combine them.
//!
//! Convention throughout this crate: **negative is solid**, positive is air,
//! and the zero level set is the surface. That sign choice is not arbitrary --
//! it makes `subtract` the operator that carves, so a cave modifier reads as
//! "subtract a tube" rather than the double negative the opposite convention
//! forces.
//!
//! Every primitive here is a *true* distance field where it can be (the
//! gradient has unit length), because the extraction pass in
//! [`crate::surface_nets`] places vertices by linear interpolation along cell
//! edges. That interpolation is only correct when the field is metric: a
//! field that merely has the right sign puts vertices in the wrong place and
//! the surface visibly ripples at cell boundaries.
//!
//! The one deliberate exception is [`heightfield`], which cannot be a true
//! distance field without a full 3D sweep. It is Lipschitz-bounded instead,
//! and the reason that is safe is documented there.

use glam::Vec3;

/// Distance to the surface of a sphere.
pub fn sphere(p: Vec3, center: Vec3, radius: f32) -> f32 {
    (p - center).length() - radius
}

/// Distance to a box with half-extents `half`, centred on `center`.
///
/// The `max(q, 0).length()` term handles the outside, where distance is to the
/// nearest corner or face; the `min(max3(q), 0)` term handles the inside,
/// where it is the distance to the nearest face. Both are needed -- an
/// exterior-only formula returns zero everywhere inside and the extraction
/// finds no surface at all.
pub fn box_sdf(p: Vec3, center: Vec3, half: Vec3) -> f32 {
    let q = (p - center).abs() - half;
    q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0)
}

/// Distance to a capsule: the swept volume of a sphere moved from `a` to `b`.
///
/// This is the workhorse for tunnels. A cave bored along a spline is just a
/// chain of these, one per spline segment, and because consecutive capsules
/// overlap at their shared endpoint the union has no seam to patch.
pub fn capsule(p: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    // Guard the degenerate segment: a zero-length capsule is a sphere, and
    // without this the division produces NaN that then poisons every boolean
    // op it flows into.
    let denom = ba.dot(ba);
    let h = if denom > 1e-12 { (pa.dot(ba) / denom).clamp(0.0, 1.0) } else { 0.0 };
    (pa - ba * h).length() - radius
}

/// Distance to a round cone: a capsule whose radius tapers from `ra` at `a` to
/// `rb` at `b`.
///
/// This, not the plain capsule, is what a cave segment is built from. A tunnel
/// of constant bore reads as extruded pipe; real passages pinch and open out,
/// and a tapering segment gets that from the spline for free. The surface is
/// the outer tangent of the two end spheres, so consecutive segments sharing
/// an endpoint radius join without a step.
pub fn round_cone(p: Vec3, a: Vec3, b: Vec3, ra: f32, rb: f32) -> f32 {
    let ba = b - a;
    let l2 = ba.dot(ba);
    // Degenerate segment: fall back to the larger end sphere.
    if l2 < 1e-12 {
        return (p - a).length() - ra.max(rb);
    }
    let rr = ra - rb;
    let a2 = l2 - rr * rr;
    // One end sphere swallows the other, so there is no tangent cone to speak
    // of. The union of the two spheres is the correct shape and stays metric.
    if a2 <= 0.0 {
        return ((p - a).length() - ra).min((p - b).length() - rb);
    }
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = pa.dot(ba);
    let z = y - l2;
    let x = pa * l2 - ba * y;
    let x2 = x.dot(x);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = rr.signum() * rr * rr * x2;

    if z.signum() * a2 * z2 > k {
        (x2 + z2).sqrt() * il2 - rb
    } else if y.signum() * a2 * y2 < k {
        (x2 + y2).sqrt() * il2 - ra
    } else {
        ((x2 * a2 * il2).sqrt() + y * rr) * il2 - ra
    }
}

/// Distance to an infinite plane through `point` with unit normal `normal`.
/// Solid on the side the normal points away from.
pub fn plane(p: Vec3, point: Vec3, normal: Vec3) -> f32 {
    (p - point).dot(normal)
}

/// A torus in the XZ plane, used for arch and doorway modifiers.
pub fn torus(p: Vec3, center: Vec3, major: f32, minor: f32) -> f32 {
    let q = p - center;
    let radial = (q.x * q.x + q.z * q.z).sqrt() - major;
    (radial * radial + q.y * q.y).sqrt() - minor
}

// ---------------------------------------------------------------------------
// Boolean operators
// ---------------------------------------------------------------------------

/// Union: everything solid in either operand stays solid.
pub fn union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Intersection: solid only where both are.
pub fn intersect(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Subtract `b` from `a`. This is the carve operator -- the whole basis of the
/// cave modifier.
pub fn subtract(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// Union with a blend of width `k`, so the join is a fillet rather than a
/// crease.
///
/// The polynomial form is used rather than the exponential one because it is
/// exact at `k = 0` and costs a single branchless mix. `k` is in metres, and
/// the result is only approximately metric inside the blend region -- which is
/// why callers must keep `k` well under a voxel or two, or extraction starts
/// misplacing vertices in exactly the region the blend was added to smooth.
pub fn smooth_union(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// Subtract with a fillet of width `k`. A tunnel mouth cut with this meets the
/// hillside in a rounded lip instead of a knife edge, which is both what caves
/// actually look like and what stops the extraction from producing slivers at
/// the intersection.
pub fn smooth_subtract(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.max(-b);
    }
    let h = (0.5 - 0.5 * (b + a) / k).clamp(0.0, 1.0);
    let m = a * (1.0 - h) + (-b) * h;
    m + k * h * (1.0 - h)
}

/// Intersection with a fillet of width `k`.
pub fn smooth_intersect(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.max(b);
    }
    let h = (0.5 - 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h + k * h * (1.0 - h)
}

// ---------------------------------------------------------------------------
// Heightfield
// ---------------------------------------------------------------------------

/// Signed distance to a heightfield surface, given the height directly below
/// `p` and the steepest local slope.
///
/// The naive form `p.y - height` is *not* a distance field on a slope: walk
/// horizontally toward a cliff and it reports the vertical gap, which on a 45°
/// face is 1.41x the true distance and on an overhang-steep face is unbounded.
/// Feeding that to a linear edge solve pushes vertices off the surface.
///
/// Dividing by `sqrt(1 + |grad h|^2)` -- the secant of the slope angle --
/// converts the vertical gap into the perpendicular one exactly for a plane,
/// and conservatively (never an overestimate) for curved ground. Never
/// overestimating is the property that matters: an underestimate costs a
/// little extraction accuracy, while an overestimate lets the surface be
/// skipped entirely between two samples.
pub fn heightfield(p_y: f32, height: f32, grad: glam::Vec2) -> f32 {
    (p_y - height) / (1.0 + grad.length_squared()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample the gradient by central differences and check it has unit length.
    /// This is the property the whole extraction pass depends on.
    fn gradient_len(f: impl Fn(Vec3) -> f32, p: Vec3) -> f32 {
        const H: f32 = 1e-3;
        let g = Vec3::new(
            f(p + Vec3::X * H) - f(p - Vec3::X * H),
            f(p + Vec3::Y * H) - f(p - Vec3::Y * H),
            f(p + Vec3::Z * H) - f(p - Vec3::Z * H),
        ) / (2.0 * H);
        g.length()
    }

    #[test]
    fn primitives_are_metric_outside() {
        let probes =
            [Vec3::new(3.0, 1.0, 2.0), Vec3::new(-4.0, 5.0, 1.5), Vec3::new(0.5, -6.0, 3.0)];
        for p in probes {
            for (name, d) in [
                ("sphere", gradient_len(|q| sphere(q, Vec3::ZERO, 1.0), p)),
                ("box", gradient_len(|q| box_sdf(q, Vec3::ZERO, Vec3::splat(1.0)), p)),
                ("capsule", gradient_len(|q| capsule(q, -Vec3::X, Vec3::X, 0.5), p)),
                ("torus", gradient_len(|q| torus(q, Vec3::ZERO, 2.0, 0.5), p)),
                ("cone", gradient_len(|q| round_cone(q, -Vec3::X, Vec3::X, 0.6, 0.3), p)),
            ] {
                assert!((d - 1.0).abs() < 1e-2, "{name} gradient {d} at {p}, want 1");
            }
        }
    }

    #[test]
    fn sphere_reports_true_distance() {
        // Trivially checkable by hand, and the anchor for everything else.
        assert!((sphere(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, 2.0) - 3.0).abs() < 1e-5);
        assert!((sphere(Vec3::ZERO, Vec3::ZERO, 2.0) + 2.0).abs() < 1e-5);
    }

    #[test]
    fn box_is_negative_inside_and_zero_on_the_face() {
        let half = Vec3::splat(1.0);
        assert!(box_sdf(Vec3::ZERO, Vec3::ZERO, half) < 0.0, "centre must be solid");
        assert!(box_sdf(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, half).abs() < 1e-6);
        assert!((box_sdf(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, half) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn degenerate_capsule_is_a_sphere_not_a_nan() {
        // A spline with a duplicated control point produces exactly this, and
        // one NaN here propagates through every boolean op downstream.
        let d = capsule(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, Vec3::ZERO, 1.0);
        assert!(d.is_finite(), "degenerate capsule produced {d}");
        assert!((d - 2.0).abs() < 1e-5);
    }

    #[test]
    fn equal_radius_cone_is_a_capsule() {
        // The taper is what makes a passage look bored rather than extruded,
        // but it must not change the shape when there is no taper -- otherwise
        // every uniform tunnel silently shifts the moment tapering ships.
        for p in [Vec3::new(2.0, 0.3, 0.1), Vec3::new(0.0, 1.4, 0.0), Vec3::new(-3.0, 2.0, 1.0)] {
            let cone = round_cone(p, -Vec3::X, Vec3::X, 0.5, 0.5);
            let cap = capsule(p, -Vec3::X, Vec3::X, 0.5);
            assert!((cone - cap).abs() < 1e-4, "at {p}: cone {cone} vs capsule {cap}");
        }
    }

    #[test]
    fn cone_reaches_both_end_radii() {
        // On the axis just past each cap the distance is the gap to that end
        // sphere, which pins the taper to the radii it was given.
        let (a, b) = (Vec3::ZERO, Vec3::X * 4.0);
        let d_a = round_cone(a - Vec3::X * 2.0, a, b, 1.0, 0.25);
        let d_b = round_cone(b + Vec3::X * 2.0, a, b, 1.0, 0.25);
        assert!((d_a - 1.0).abs() < 1e-3, "wide end: {d_a}");
        assert!((d_b - 1.75).abs() < 1e-3, "narrow end: {d_b}");
    }

    #[test]
    fn cone_with_a_swallowed_end_stays_finite() {
        // radius difference exceeds the segment length: there is no tangent
        // cone, and the unguarded formula takes the square root of a negative.
        let d = round_cone(Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO, Vec3::X * 0.1, 2.0, 0.1);
        assert!(d.is_finite(), "got {d}");
        assert!((d - 1.0).abs() < 1e-3, "should be the big sphere's distance, got {d}");
    }

    #[test]
    fn subtract_carves() {
        // Solid everywhere (-1), minus a sphere of radius 2 at the origin: the
        // origin must become air.
        assert!(subtract(-1.0, sphere(Vec3::ZERO, Vec3::ZERO, 2.0)) > 0.0);
        // ...and a point outside that sphere must stay solid.
        assert!(subtract(-1.0, sphere(Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO, 2.0)) < 0.0);
    }

    #[test]
    fn smooth_ops_degrade_to_sharp_ones_at_zero_width() {
        for (a, b) in [(-1.0f32, 2.0f32), (3.0, -0.5), (0.25, 0.75), (-2.0, -3.0)] {
            assert!((smooth_union(a, b, 0.0) - union(a, b)).abs() < 1e-6);
            assert!((smooth_subtract(a, b, 0.0) - subtract(a, b)).abs() < 1e-6);
            assert!((smooth_intersect(a, b, 0.0) - intersect(a, b)).abs() < 1e-6);
        }
    }

    #[test]
    fn smooth_union_never_exceeds_the_sharp_one() {
        // A fillet adds material; it must never remove any. If this inverts,
        // blended tunnel mouths eat into the rock instead of rounding it.
        for i in 0..50 {
            let a = -2.0 + i as f32 * 0.1;
            for j in 0..50 {
                let b = -2.0 + j as f32 * 0.1;
                let s = smooth_union(a, b, 0.5);
                assert!(s <= union(a, b) + 1e-6, "smooth_union({a},{b}) = {s} > {}", union(a, b));
            }
        }
    }

    #[test]
    fn heightfield_is_conservative_on_a_slope() {
        // A 45-degree plane through the origin: h(x) = x, so grad = 1. The
        // true perpendicular distance from (0, 1, 0) to that plane is
        // 1/sqrt(2); the naive vertical gap would claim 1.0 and overshoot.
        let d = heightfield(1.0, 0.0, glam::Vec2::new(1.0, 0.0));
        assert!((d - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5, "got {d}");
        assert!(d <= 1.0, "must never exceed the vertical gap");
    }

    #[test]
    fn heightfield_sign_matches_above_and_below() {
        let g = glam::Vec2::new(0.3, -0.7);
        assert!(heightfield(120.0, 100.0, g) > 0.0, "above ground must be air");
        assert!(heightfield(80.0, 100.0, g) < 0.0, "below ground must be solid");
    }
}

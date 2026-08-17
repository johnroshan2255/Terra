//! Volumetric brushes: clay, move, flatten, smooth and inflate.
//!
//! Every brush here reduces to the same three steps -- decide a *target*
//! distance for each voxel in range, blend the current value toward it by the
//! falloff, and store the difference in [`crate::volume::DeltaField`]. Only
//! the target differs between modes, which is why adding a brush is a match
//! arm rather than a new pass.
//!
//! ## Why the two-phase write
//!
//! Targets for the whole stroke are computed before any of them are stored.
//! Writing as we go would let a voxel's new value feed the next voxel's
//! target, making the result depend on iteration order -- a smooth brush would
//! smear along +X, and the GPU port, which has no defined order at all, would
//! produce something different again. Reading a consistent snapshot costs one
//! temporary Vec sized to the brush, not the world.
//!
//! ## Known limitation: the field drifts
//!
//! Repeated strokes leave a field that still has the right *sign* everywhere
//! but is no longer metric -- gradients stop having unit length far from the
//! surface. Surface Nets only interpolates across cells the surface actually
//! crosses, so this is harmless until it is severe, and [`Brush::Smooth`] is
//! the recovery tool. A proper fix is a redistancing sweep (fast sweeping, or
//! jump flooding on the GPU); it is not here yet, and heavy sculpting in one
//! spot will eventually show as slightly soft interpolation.

use crate::modifier::Aabb;
use crate::volume::VoxelVolume;
use glam::{IVec3, Vec3};

/// The 3D sculpt modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Brush {
    /// Build material up to, or cut it back from, a plane through the brush
    /// centre. The bread-and-butter mode: it deposits where there is a hollow
    /// and does nothing where the surface already stands proud, so strokes
    /// build mass instead of ballooning it.
    Clay,
    /// Grab the surface and drag it. Implemented as a domain shift of the
    /// field, so the surface translates instead of inflating -- detail already
    /// in the rock moves with it rather than being smeared flat.
    Move,
    /// Pull the surface onto a plane, from both sides. Cuts bumps and fills
    /// pits in the same stroke.
    Flatten,
    /// Relax toward the local average. Removes stair-stepping and the
    /// high-frequency noise that repeated clay strokes leave behind.
    Smooth,
    /// Offset the field uniformly, which moves the surface along its own
    /// normal everywhere at once. Balloons a shape outward or shrinks it.
    Inflate,
    /// Move the surface straight up, or down when inverted.
    ///
    /// Not the same tool as [`Brush::Inflate`], and the difference is the whole
    /// reason both exist: Inflate moves along the *surface normal*, this moves
    /// along *world up*. On a 45-degree hillside those diverge by 45 degrees,
    /// and world-up is what a landscape artist means by "raise". Inflate on a
    /// steep slope pushes material sideways out of the hill.
    Raise,
    /// Displace the surface by a noise field, for rock roughness and detail.
    /// The pattern comes from [`crate::noise::NoiseField`] -- either the
    /// built-in basis or an uploaded greyscale map.
    Noise,
    /// Pull the surface toward the brush axis, sharpening it into a crease.
    ///
    /// The only tool here that can *add* high-frequency detail rather than
    /// remove it, which is what makes crisp ridges and cave-mouth lips
    /// possible at all. Inverted, it rounds a crease off.
    Pinch,
}

impl Brush {
    pub const ALL: [Brush; 8] = [
        Brush::Clay,
        Brush::Raise,
        Brush::Move,
        Brush::Flatten,
        Brush::Smooth,
        Brush::Inflate,
        Brush::Noise,
        Brush::Pinch,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Brush::Clay => "Clay",
            Brush::Move => "Move",
            Brush::Flatten => "Flatten",
            Brush::Smooth => "Smooth",
            Brush::Inflate => "Inflate",
            Brush::Raise => "Raise",
            Brush::Noise => "Noise",
            Brush::Pinch => "Pinch",
        }
    }

    /// What the invert modifier does in this mode, for the tool tooltip.
    pub fn invert_label(self) -> Option<&'static str> {
        match self {
            Brush::Clay => Some("Carve"),
            Brush::Raise => Some("Lower"),
            Brush::Inflate => Some("Shrink"),
            Brush::Pinch => Some("Round off"),
            Brush::Noise => Some("Invert pattern"),
            // Flatten pulls from both sides already, and "un-smooth" is just
            // noise -- which is what the Noise brush is for.
            Brush::Flatten | Brush::Smooth | Brush::Move => None,
        }
    }

    /// Whether holding the invert modifier is meaningful.
    pub fn invertible(self) -> bool {
        self.invert_label().is_some()
    }

    /// Whether this mode reads the noise settings, so the UI can show them
    /// only where they apply.
    pub fn uses_noise(self) -> bool {
        matches!(self, Brush::Noise)
    }

    /// Whether this mode is driven by cursor drag rather than dwell time.
    pub fn uses_drag(self) -> bool {
        matches!(self, Brush::Move)
    }
}

/// One application of a brush. Strokes are dabs: the editor calls this once
/// per frame while the button is held, with `strength` already scaled by the
/// frame time so coverage does not depend on frame rate.
#[derive(Debug, Clone, Copy)]
pub struct Stroke {
    pub brush: Brush,
    pub center: Vec3,
    pub radius: f32,
    /// Blend fraction per application, 0..1.
    pub strength: f32,
    /// 0 gives a hard-edged disc, 1 a fully feathered one.
    pub falloff: f32,
    /// Surface normal under the cursor. Orients the Clay and Flatten planes.
    pub normal: Vec3,
    /// Drag vector for [`Brush::Move`], in world metres. Ignored otherwise.
    pub drag: Vec3,
    /// Dig instead of build.
    pub invert: bool,
    /// Peak displacement for [`Brush::Noise`], in metres. Separate from
    /// `strength`, which is how fast a held stroke converges: amplitude is how
    /// deep the pattern cuts once it has.
    pub noise_amplitude: f32,
}

impl Stroke {
    pub fn new(brush: Brush, center: Vec3, radius: f32) -> Self {
        Self {
            brush,
            center,
            radius,
            strength: 0.5,
            falloff: 0.5,
            normal: Vec3::Y,
            drag: Vec3::ZERO,
            invert: false,
            noise_amplitude: 1.5,
        }
    }

    /// Smooth radial weight, 1 at the centre and 0 at the rim.
    ///
    /// `falloff` sets how much of the radius is plateau. A hard edge leaves a
    /// visible cylinder wall in the rock, so the default is half-feathered.
    fn weight(&self, d: f32) -> f32 {
        if d >= self.radius {
            return 0.0;
        }
        let inner = self.radius * (1.0 - self.falloff.clamp(0.0, 1.0));
        if d <= inner {
            return 1.0;
        }
        let t = ((d - inner) / (self.radius - inner)).clamp(0.0, 1.0);
        // smoothstep, flipped: 1 at the plateau edge, 0 at the rim.
        let s = t * t * (3.0 - 2.0 * t);
        1.0 - s
    }

    /// World region this stroke can touch.
    ///
    /// [`Brush::Move`] reaches further than its radius, because it reads the
    /// field from where the drag came *from*. Missing that is how a grab
    /// leaves a torn edge at the far side of the brush.
    pub fn bounds(&self) -> Aabb {
        let reach = match self.brush {
            Brush::Move => self.radius + self.drag.length(),
            _ => self.radius,
        };
        Aabb::around(self.center, self.center, reach)
    }
}

/// Apply a stroke, returning the region whose extraction is now stale.
///
/// The returned box is grown by one voxel: a delta written at a lattice point
/// changes the trilinear field half a voxel either side of it, so a chunk that
/// merely touches the brush still needs re-extracting.
/// `noise` is only read by [`Brush::Noise`]. Passing `None` in that mode is a
/// no-op rather than an error: the editor always has a field to hand, and a
/// headless caller sculpting with the other seven modes should not have to
/// construct one.
pub fn apply(
    volume: &mut VoxelVolume<'_>,
    stroke: &Stroke,
    noise: Option<&crate::noise::NoiseField>,
) -> Aabb {
    let voxel = volume.voxel_size();
    let bounds = stroke.bounds();

    // Lattice range covering the brush, inclusive on both ends.
    let lo = (bounds.min / voxel).floor();
    let hi = (bounds.max / voxel).ceil();
    let lo = IVec3::new(lo.x as i32, lo.y as i32, lo.z as i32);
    let hi = IVec3::new(hi.x as i32, hi.y as i32, hi.z as i32);

    // Phase 1: read. Nothing is written to the volume inside this loop, so
    // every target sees the same pre-stroke field.
    let mut writes: Vec<(IVec3, f32)> = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let v = IVec3::new(x, y, z);
                let p = v.as_vec3() * voxel;
                let w =
                    stroke.weight((p - stroke.center).length()) * stroke.strength.clamp(0.0, 1.0);
                if w <= 0.0 {
                    continue;
                }
                let current = volume.sample(p);
                let Some(target) = target_for(volume, stroke, p, voxel, noise) else { continue };
                let delta = (target - current) * w;
                if delta.abs() > 1e-6 {
                    writes.push((v, delta));
                }
            }
        }
    }

    // Phase 2: write.
    for (v, d) in writes {
        volume.delta.add(v, d);
    }

    bounds.expand(voxel)
}

/// The distance this voxel should move toward, or `None` to leave it alone.
fn target_for(
    volume: &VoxelVolume<'_>,
    stroke: &Stroke,
    p: Vec3,
    voxel: f32,
    noise: Option<&crate::noise::NoiseField>,
) -> Option<f32> {
    let sign = if stroke.invert { -1.0 } else { 1.0 };
    let current = volume.sample(p);

    match stroke.brush {
        // Offset the whole field. Because the surface is the zero set, adding
        // a constant moves it along its own normal by exactly that amount.
        Brush::Inflate => Some(current - sign * voxel),

        // Sample the field from directly below (or above) instead of offsetting
        // it. Shifting the *domain* along world Y translates the surface
        // vertically whatever its slope, where a plain offset would move it
        // along the normal -- see the note on the variant.
        Brush::Raise => Some(volume.sample(p - Vec3::Y * sign * voxel)),

        // Displace along the surface normal by the noise pattern. The
        // displacement is a straight field offset for the same reason Inflate
        // is: the zero set moves by exactly the amount added.
        Brush::Noise => {
            let field = noise?;
            let n = field.sample(p, stroke.normal);
            Some(current - sign * n * stroke.noise_amplitude)
        }

        // Sharpen toward the brush axis. Sampling the field *closer to* the
        // axis pulls the surface inward there, which concentrates it into a
        // crease; inverted it samples further out and rounds the crease off.
        Brush::Pinch => {
            let axis = stroke.normal.normalize_or(Vec3::Y);
            let radial = {
                let d = p - stroke.center;
                d - axis * d.dot(axis)
            };
            // Nothing to pull toward on the axis itself.
            let len = radial.length();
            if len < 1e-5 {
                return Some(current);
            }
            let step = (voxel * 0.75).min(len);
            // Reading the field *further out* than `p` moves the surface
            // *inward*, toward the axis -- that is the direction that narrows a
            // ridge into a crease. Sampling inward is the opposite operation
            // and widens the feature, which is what invert selects.
            //
            // Clamped by `len` so the sample never crosses the axis and comes
            // back out the far side, which would fold the surface.
            Some(volume.sample(p + radial / len * step * sign))
        }

        // Union with a half-space through the brush centre, so material is
        // only ever added where the plane is *outside* the current surface.
        // Inverted, it intersects instead and only ever cuts.
        Brush::Clay => {
            let plane = crate::sdf::plane(p, stroke.center, stroke.normal);
            Some(if stroke.invert { current.max(-plane) } else { current.min(plane) })
        }

        // Pull onto the plane from whichever side the voxel is on.
        Brush::Flatten => Some(crate::sdf::plane(p, stroke.center, stroke.normal)),

        // Read the field from where the drag started. Sampling the *pre-stroke*
        // field at a shifted position translates the surface with its detail
        // intact, which is what separates a grab from an inflate.
        Brush::Move => Some(volume.sample(p - stroke.drag)),

        // Six-neighbour Laplacian. The average of the face neighbours is the
        // discrete mean curvature flow that relaxes a surface without
        // shrinking it the way a full 26-neighbour box blur does.
        Brush::Smooth => {
            let mut sum = 0.0;
            for d in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
                sum += volume.sample(p + d * voxel);
            }
            Some(sum / 6.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::BaseField;

    fn flat_volume(height: f32) -> (Vec<f32>, u32, f32) {
        (vec![height; 32 * 32], 32, 200.0)
    }

    /// Find the surface height above a column by bisection.
    fn surface_y(v: &VoxelVolume<'_>, x: f32, z: f32, lo: f32, hi: f32) -> Option<f32> {
        let (mut lo, mut hi) = (lo, hi);
        if v.sample(Vec3::new(x, lo, z)) >= 0.0 || v.sample(Vec3::new(x, hi, z)) <= 0.0 {
            return None;
        }
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if v.sample(Vec3::new(x, mid, z)) < 0.0 { lo = mid } else { hi = mid }
        }
        Some(0.5 * (lo + hi))
    }

    #[test]
    fn inflate_raises_the_surface_and_invert_lowers_it() {
        let (h, res, extent) = flat_volume(100.0);
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
        let before = surface_y(&v, 0.0, 0.0, 90.0, 110.0).unwrap();

        let mut s = Stroke::new(Brush::Inflate, Vec3::new(0.0, 100.0, 0.0), 8.0);
        s.strength = 1.0;
        apply(&mut v, &s, None);
        let up = surface_y(&v, 0.0, 0.0, 90.0, 115.0).unwrap();
        assert!(up > before + 0.3, "inflate should raise: {before} -> {up}");

        s.invert = true;
        apply(&mut v, &s, None);
        apply(&mut v, &s, None);
        let down = surface_y(&v, 0.0, 0.0, 85.0, 115.0).unwrap();
        assert!(down < up, "inverted inflate should lower: {up} -> {down}");
    }

    #[test]
    fn a_stroke_only_touches_its_own_radius() {
        let (h, res, extent) = flat_volume(100.0);
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
        let far = Vec3::new(60.0, 100.0, 0.0);
        let before = v.sample(far);

        let mut s = Stroke::new(Brush::Inflate, Vec3::new(0.0, 100.0, 0.0), 8.0);
        s.strength = 1.0;
        apply(&mut v, &s, None);

        assert_eq!(v.sample(far), before, "a brush must not reach 60 m for an 8 m radius");
    }

    #[test]
    fn falloff_is_monotonic_from_centre_to_rim() {
        let mut s = Stroke::new(Brush::Clay, Vec3::ZERO, 10.0);
        s.falloff = 0.7;
        let mut prev = f32::INFINITY;
        for i in 0..=20 {
            let w = s.weight(i as f32 * 0.5);
            assert!(w <= prev + 1e-6, "weight rose at d = {}", i as f32 * 0.5);
            assert!((0.0..=1.0).contains(&w));
            prev = w;
        }
        assert_eq!(s.weight(0.0), 1.0);
        assert_eq!(s.weight(10.0), 0.0);
    }

    #[test]
    fn zero_falloff_is_a_hard_disc() {
        let mut s = Stroke::new(Brush::Clay, Vec3::ZERO, 10.0);
        s.falloff = 0.0;
        assert_eq!(s.weight(9.99), 1.0);
        assert_eq!(s.weight(10.0), 0.0);
    }

    #[test]
    fn flatten_pulls_both_a_bump_and_a_pit_toward_the_plane() {
        let res = 32u32;
        let extent = 200.0f32;
        // Ground with a bump on one side and a pit on the other.
        let mut h = vec![100.0f32; (res * res) as usize];
        for z in 0..res {
            for x in 0..res {
                let wx = -extent * 0.5 + x as f32 / (res - 1) as f32 * extent;
                h[(z * res + x) as usize] = 100.0 + if wx > 0.0 { 6.0 } else { -6.0 };
            }
        }
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        let mut s = Stroke::new(Brush::Flatten, Vec3::new(0.0, 100.0, 0.0), 40.0);
        s.strength = 1.0;
        s.falloff = 0.0;
        for _ in 0..6 {
            apply(&mut v, &s, None);
        }

        let bump = surface_y(&v, 12.0, 0.0, 80.0, 120.0).unwrap();
        let pit = surface_y(&v, -12.0, 0.0, 80.0, 120.0).unwrap();
        assert!((bump - 100.0).abs() < 2.5, "bump not flattened: {bump}");
        assert!((pit - 100.0).abs() < 2.5, "pit not filled: {pit}");
    }

    #[test]
    fn smooth_reduces_roughness() {
        let res = 64u32;
        let extent = 200.0f32;
        // Alternating high-frequency ridges -- exactly what repeated clay
        // strokes leave and what Smooth exists to clean up.
        let mut h = vec![0.0f32; (res * res) as usize];
        for z in 0..res {
            for x in 0..res {
                h[(z * res + x) as usize] = 100.0 + if x % 2 == 0 { 1.5 } else { -1.5 };
            }
        }
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        let probe: Vec<Vec3> = (-6..6).map(|i| Vec3::new(i as f32 * 1.5, 100.0, 0.0)).collect();
        let roughness = |v: &VoxelVolume<'_>| -> f32 {
            let s: Vec<f32> = probe.iter().map(|p| v.sample(*p)).collect();
            s.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
        };
        let before = roughness(&v);

        let mut st = Stroke::new(Brush::Smooth, Vec3::new(0.0, 100.0, 0.0), 30.0);
        st.strength = 1.0;
        st.falloff = 0.0;
        for _ in 0..4 {
            apply(&mut v, &st, None);
        }
        let after = roughness(&v);
        assert!(after < before * 0.8, "smooth did not relax: {before} -> {after}");
    }

    #[test]
    fn clay_builds_up_but_never_digs() {
        // The distinction from Inflate. Clay unions with a plane, so where the
        // rock already stands above that plane it must be left alone.
        let (h, res, extent) = flat_volume(100.0);
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        // Brush plane sits below the surface: there is nothing to add.
        let mut s = Stroke::new(Brush::Clay, Vec3::new(0.0, 94.0, 0.0), 10.0);
        s.strength = 1.0;
        s.normal = Vec3::Y;
        apply(&mut v, &s, None);

        let y = surface_y(&v, 0.0, 0.0, 85.0, 115.0).unwrap();
        assert!(y >= 99.9, "clay below the surface must not cut into it, got {y}");
    }

    #[test]
    fn move_translates_rather_than_inflating() {
        // A grab must carry the surface sideways. Check that material appears
        // on the drag side and is removed from the trailing side -- an inflate
        // would add to both.
        let (h, res, extent) = flat_volume(100.0);
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        let mut s = Stroke::new(Brush::Move, Vec3::new(0.0, 100.0, 0.0), 12.0);
        s.strength = 1.0;
        s.falloff = 0.2;
        s.drag = Vec3::new(0.0, 5.0, 0.0);
        apply(&mut v, &s, None);

        let y = surface_y(&v, 0.0, 0.0, 85.0, 125.0).unwrap();
        assert!(y > 102.0, "dragging up by 5 m should lift the surface, got {y}");
    }

    #[test]
    fn a_stroke_allocates_only_near_the_brush() {
        let (h, res, extent) = flat_volume(100.0);
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
        let mut s = Stroke::new(Brush::Inflate, Vec3::new(0.0, 100.0, 0.0), 6.0);
        s.strength = 1.0;
        apply(&mut v, &s, None);

        // A 6 m brush at 1 m voxels spans ~13 voxels, so at most a handful of
        // 16^3 bricks. If this grows, the brush is writing outside its radius.
        assert!(v.delta.brick_count() <= 8, "{} bricks for a 6 m dab", v.delta.brick_count());
        assert!(v.delta.brick_count() > 0, "the stroke stored nothing at all");
    }

    #[test]
    fn stroke_result_does_not_depend_on_iteration_order() {
        // Guards the two-phase write. Smoothing is the mode most sensitive to
        // reading a partially-updated field; applying the same stroke to two
        // volumes must give identical results, and would not if phase 1 saw
        // phase 2's writes.
        let (h, res, extent) = flat_volume(100.0);
        let mut a = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
        let mut b = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        let mut seed = Stroke::new(Brush::Inflate, Vec3::new(0.0, 100.0, 0.0), 10.0);
        seed.strength = 1.0;
        apply(&mut a, &seed, None);
        apply(&mut b, &seed, None);

        let mut s = Stroke::new(Brush::Smooth, Vec3::new(0.0, 100.0, 0.0), 10.0);
        s.strength = 0.8;
        apply(&mut a, &s, None);
        apply(&mut b, &s, None);

        for i in -12..12 {
            let p = Vec3::new(i as f32, 100.0, 0.0);
            assert_eq!(a.sample(p), b.sample(p), "diverged at {p}");
        }
    }

    #[test]
    fn move_bounds_cover_the_drag() {
        let mut s = Stroke::new(Brush::Move, Vec3::ZERO, 5.0);
        s.drag = Vec3::new(20.0, 0.0, 0.0);
        let b = s.bounds();
        assert!(
            b.contains(Vec3::new(-24.0, 0.0, 0.0)),
            "must reach back to where the drag read from"
        );
        let plain = Stroke::new(Brush::Inflate, Vec3::ZERO, 5.0).bounds();
        assert!(!plain.contains(Vec3::new(-24.0, 0.0, 0.0)), "other brushes must not over-reach");
    }
}

#[cfg(test)]
mod new_mode_tests {
    use super::tests_support::*;
    use super::*;
    use crate::noise::{NoiseField, NoiseImage};
    use crate::volume::BaseField;

    #[test]
    fn raise_is_slope_independent_and_inflate_is_not() {
        // The regression this brush exists to fix, stated as the property that
        // actually distinguishes the two.
        //
        // Inflate offsets the field *value*, so it moves the surface by a fixed
        // perpendicular distance -- and on a tilted plane a perpendicular step
        // of 1 m raises the surface by 1/cos(theta) vertically, which grows
        // without bound as the ground steepens. Raise offsets the field
        // *domain* along world Y, so it lifts by the amount asked for whatever
        // the slope.
        //
        // Asserting one lifts "more" would be backwards: on a 45-degree face
        // Inflate actually wins, at 1.41 m against 1.00 m. Slope-independence
        // is the real invariant.
        let (res, extent) = (64u32, 200.0f32);
        let lift = |brush: Brush, slope: f32| {
            let h = ramp(res, extent, slope);
            let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
            let before = surface_y(&v, 0.0, 0.0, 40.0, 160.0).unwrap();
            let mut s = Stroke::new(brush, Vec3::new(0.0, before, 0.0), 10.0);
            s.strength = 1.0;
            s.falloff = 0.0;
            s.normal = Vec3::new(-slope, 1.0, 0.0).normalize();
            apply(&mut v, &s, None);
            surface_y(&v, 0.0, 0.0, 40.0, 160.0).unwrap() - before
        };

        let raise_flat = lift(Brush::Raise, 0.0);
        let raise_steep = lift(Brush::Raise, 1.0);
        let inflate_flat = lift(Brush::Inflate, 0.0);
        let inflate_steep = lift(Brush::Inflate, 1.0);

        assert!(raise_flat > 0.5, "Raise did nothing on flat ground: {raise_flat}");
        assert!(
            (raise_steep - raise_flat).abs() < 0.1,
            "Raise must lift the same on flat ({raise_flat}) and 45-degree ({raise_steep}) ground"
        );
        // sqrt(2) more on a 45-degree face, from the 1/cos(theta) factor.
        assert!(
            inflate_steep > inflate_flat * 1.3,
            "Inflate should scale with slope: flat {inflate_flat}, steep {inflate_steep}"
        );
    }

    #[test]
    fn raise_inverted_lowers() {
        let (res, extent) = (32u32, 200.0f32);
        let h = vec![100.0f32; (res * res) as usize];
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
        let before = surface_y(&v, 0.0, 0.0, 80.0, 120.0).unwrap();

        let mut s = Stroke::new(Brush::Raise, Vec3::new(0.0, 100.0, 0.0), 10.0);
        s.strength = 1.0;
        s.invert = true;
        for _ in 0..3 {
            apply(&mut v, &s, None);
        }
        let after = surface_y(&v, 0.0, 0.0, 70.0, 120.0).unwrap();
        assert!(after < before - 0.5, "inverted Raise should dig: {before} -> {after}");
    }

    #[test]
    fn noise_brush_roughens_a_flat_surface() {
        let (res, extent) = (64u32, 200.0f32);
        let h = vec![100.0f32; (res * res) as usize];
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        let probe: Vec<Vec3> = (-8..8).map(|i| Vec3::new(i as f32 * 1.5, 100.0, 0.0)).collect();
        let spread = |v: &VoxelVolume<'_>| {
            let s: Vec<f32> = probe.iter().map(|p| v.sample(*p)).collect();
            let mean = s.iter().sum::<f32>() / s.len() as f32;
            (s.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / s.len() as f32).sqrt()
        };
        let before = spread(&v);

        let field = NoiseField::procedural(11, 4, false, 6.0);
        let mut s = Stroke::new(Brush::Noise, Vec3::new(0.0, 100.0, 0.0), 20.0);
        s.strength = 1.0;
        s.falloff = 0.0;
        s.noise_amplitude = 2.0;
        apply(&mut v, &s, Some(&field));

        assert!(spread(&v) > before + 0.2, "noise did not roughen: {before} -> {}", spread(&v));
    }

    #[test]
    fn noise_brush_without_a_field_does_nothing() {
        // `None` must be a no-op, not a panic and not an accidental offset.
        let (res, extent) = (32u32, 200.0f32);
        let h = vec![100.0f32; (res * res) as usize];
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
        let mut s = Stroke::new(Brush::Noise, Vec3::new(0.0, 100.0, 0.0), 10.0);
        s.strength = 1.0;
        apply(&mut v, &s, None);
        assert_eq!(v.delta.brick_count(), 0, "a noise stroke with no pattern stored something");
    }

    #[test]
    fn a_mid_grey_upload_leaves_the_surface_alone() {
        // End to end through the brush, not just the sampler: an uploaded map
        // that averages mid-grey must add roughness symmetrically rather than
        // shifting the surface.
        let (res, extent) = (32u32, 200.0f32);
        let h = vec![100.0f32; (res * res) as usize];
        let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

        let flat = NoiseImage::new("mid", 4, 4, vec![0.5; 16]).unwrap();
        let field = NoiseField::image(flat, 8.0);
        let mut s = Stroke::new(Brush::Noise, Vec3::new(0.0, 100.0, 0.0), 12.0);
        s.strength = 1.0;
        apply(&mut v, &s, Some(&field));

        assert_eq!(
            v.delta.brick_count(),
            0,
            "mid-grey should be a no-op, stored {} bricks",
            v.delta.brick_count()
        );
    }

    #[test]
    fn noise_amplitude_scales_the_displacement() {
        let (res, extent) = (32u32, 200.0f32);
        let h = vec![100.0f32; (res * res) as usize];
        let field = NoiseField::procedural(5, 3, false, 6.0);

        let displaced = |amp: f32| {
            let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
            let mut s = Stroke::new(Brush::Noise, Vec3::new(0.0, 100.0, 0.0), 12.0);
            s.strength = 1.0;
            s.falloff = 0.0;
            s.noise_amplitude = amp;
            apply(&mut v, &s, Some(&field));
            (-6..6).map(|i| v.sample(Vec3::new(i as f32, 100.0, 0.0)).abs()).fold(0.0f32, f32::max)
        };
        let small = displaced(0.5);
        let large = displaced(4.0);
        assert!(large > small * 2.0, "amplitude 4 gave {large}, amplitude 0.5 gave {small}");
    }

    #[test]
    fn pinch_narrows_a_dome_and_inverted_pinch_widens_it() {
        // Pinch needs an existing feature to sharpen -- on a flat plane there
        // is nothing to pull toward the axis and it correctly does nothing, so
        // the test has to build a dome first.
        let (res, extent) = (64u32, 200.0f32);
        let h = ramp(res, extent, 0.0);

        let with_dome = || {
            let mut v = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);
            let mut dome = Stroke::new(Brush::Inflate, Vec3::new(0.0, 100.0, 0.0), 16.0);
            dome.strength = 1.0;
            dome.falloff = 1.0;
            for _ in 0..4 {
                apply(&mut v, &dome, None);
            }
            v
        };

        // Sharpness as the apex-to-flank height ratio: narrowing the dome
        // keeps the apex and drops the flanks.
        let sharpness = |v: &VoxelVolume<'_>| {
            let apex = surface_y(v, 0.0, 0.0, 90.0, 140.0).unwrap();
            let flank = surface_y(v, 9.0, 0.0, 90.0, 140.0).unwrap();
            apex - flank
        };

        let base = sharpness(&with_dome());

        let mut sharp = with_dome();
        let mut s = Stroke::new(Brush::Pinch, Vec3::new(0.0, 100.0, 0.0), 14.0);
        s.strength = 1.0;
        s.falloff = 0.4;
        for _ in 0..6 {
            apply(&mut sharp, &s, None);
        }
        assert!(
            sharpness(&sharp) > base,
            "pinch did not sharpen the dome: {base} -> {}",
            sharpness(&sharp)
        );

        let mut round = with_dome();
        s.invert = true;
        for _ in 0..6 {
            apply(&mut round, &s, None);
        }
        assert!(
            sharpness(&round) < base,
            "inverted pinch did not round the dome off: {base} -> {}",
            sharpness(&round)
        );

        // And nothing may go non-finite on the axis, where the radial
        // direction is undefined.
        assert!(sharp.sample(Vec3::new(0.0, 100.0, 0.0)).is_finite());
        assert!(round.sample(Vec3::new(0.0, 100.0, 0.0)).is_finite());
    }

    #[test]
    fn every_brush_is_labelled_and_reachable() {
        // A mode added to the enum but left out of ALL is invisible in the UI
        // and looks like it was never implemented.
        assert_eq!(Brush::ALL.len(), 8);
        for b in Brush::ALL {
            assert!(!b.label().is_empty());
        }
        assert!(Brush::Noise.uses_noise());
        assert!(!Brush::Clay.uses_noise());
        assert!(Brush::Move.uses_drag());
        assert!(Brush::Raise.invertible(), "Raise must offer Lower");
        assert_eq!(Brush::Raise.invert_label(), Some("Lower"));
        assert!(!Brush::Smooth.invertible());
    }
}

#[cfg(test)]
mod tests_support {
    use crate::volume::VoxelVolume;
    use glam::Vec3;

    /// Ground at 100 m with a constant dh/dx of `slope`.
    pub fn ramp(res: u32, extent: f32, slope: f32) -> Vec<f32> {
        let mut h = vec![0.0f32; (res * res) as usize];
        for z in 0..res {
            for x in 0..res {
                let wx = -extent * 0.5 + x as f32 / (res - 1) as f32 * extent;
                h[(z * res + x) as usize] = 100.0 + wx * slope;
            }
        }
        h
    }

    /// Bisect for the surface height above a column.
    pub fn surface_y(v: &VoxelVolume<'_>, x: f32, z: f32, lo: f32, hi: f32) -> Option<f32> {
        let (mut lo, mut hi) = (lo, hi);
        if v.sample(Vec3::new(x, lo, z)) >= 0.0 || v.sample(Vec3::new(x, hi, z)) <= 0.0 {
            return None;
        }
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if v.sample(Vec3::new(x, mid, z)) < 0.0 { lo = mid } else { hi = mid }
        }
        Some(0.5 * (lo + hi))
    }
}

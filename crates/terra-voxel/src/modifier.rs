//! Non-destructive boolean modifiers: the cave and tunnel layer.
//!
//! A modifier is a shape plus an operator, kept as data and re-evaluated every
//! time the field is sampled. Nothing is ever baked into the voxel delta, so a
//! tunnel can be moved, re-radiused, re-ordered or deleted long after it was
//! placed and the rock closes back up behind it exactly as it was.
//!
//! That is the whole reason this is a stack rather than a brush. A subtractive
//! *brush* would write air into the delta field, and the material it removed
//! would be gone -- the only way back is undo history, which does not survive
//! a save. A subtractive *modifier* removes nothing; it just answers "air"
//! when asked, and stops answering when switched off.
//!
//! The cost is that every sample walks the list. That is why every shape
//! carries [`Shape::bounds`] and the stack skips anything the sample point
//! falls outside of -- with a bounding-box reject, a hundred tunnels cost
//! roughly what the one nearest tunnel costs.

use crate::sdf;
use glam::Vec3;

/// An axis-aligned bounding box in world metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min: min.min(max), max: min.max(max) }
    }

    /// Box around a point pair, grown by `r` on every axis.
    pub fn around(a: Vec3, b: Vec3, r: f32) -> Self {
        Self { min: a.min(b) - Vec3::splat(r), max: a.max(b) + Vec3::splat(r) }
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    pub fn expand(&self, r: f32) -> Self {
        Self { min: self.min - Vec3::splat(r), max: self.max + Vec3::splat(r) }
    }

    pub fn union(&self, other: &Aabb) -> Self {
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.cmple(other.max).all() && self.max.cmpge(other.min).all()
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

/// One control point of a tube: where the passage goes and how wide it is
/// there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TubePoint {
    pub pos: Vec3,
    pub radius: f32,
}

impl TubePoint {
    pub fn new(pos: Vec3, radius: f32) -> Self {
        Self { pos, radius }
    }
}

/// The carving shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Box {
        center: Vec3,
        half: Vec3,
    },
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: f32,
    },
    Torus {
        center: Vec3,
        major: f32,
        minor: f32,
    },
    /// A tapering tube through a Catmull-Rom spline. The authored control
    /// points and the polyline evaluated from them are both kept: the control
    /// points are what the user drags, the polyline is what gets sampled.
    Tube(Tube),
}

impl Shape {
    pub fn distance(&self, p: Vec3) -> f32 {
        match self {
            Shape::Sphere { center, radius } => sdf::sphere(p, *center, *radius),
            Shape::Box { center, half } => sdf::box_sdf(p, *center, *half),
            Shape::Capsule { a, b, radius } => sdf::capsule(p, *a, *b, *radius),
            Shape::Torus { center, major, minor } => sdf::torus(p, *center, *major, *minor),
            Shape::Tube(t) => t.distance(p),
        }
    }

    pub fn bounds(&self) -> Aabb {
        match self {
            Shape::Sphere { center, radius } => Aabb::around(*center, *center, *radius),
            Shape::Box { center, half } => Aabb::new(*center - *half, *center + *half),
            Shape::Capsule { a, b, radius } => Aabb::around(*a, *b, *radius),
            Shape::Torus { center, major, minor } => {
                Aabb::around(*center, *center, *major + *minor)
            }
            Shape::Tube(t) => t.bounds(),
        }
    }
}

/// A swept tube through a spline, stored as the polyline it evaluates to.
#[derive(Debug, Clone, PartialEq)]
pub struct Tube {
    control: Vec<TubePoint>,
    /// Spline samples per control-point interval. Higher is smoother and
    /// linearly more expensive to sample.
    subdivisions: u32,
    polyline: Vec<TubePoint>,
    bounds: Aabb,
}

impl Tube {
    pub const DEFAULT_SUBDIVISIONS: u32 = 8;

    pub fn new(control: Vec<TubePoint>, subdivisions: u32) -> Self {
        let mut t = Self {
            control,
            subdivisions: subdivisions.max(1),
            polyline: Vec::new(),
            bounds: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        };
        t.rebuild();
        t
    }

    /// Straight tube of uniform bore between two points -- the quick way to
    /// punch an adit into a hillside.
    pub fn straight(a: Vec3, b: Vec3, radius: f32) -> Self {
        Self::new(vec![TubePoint::new(a, radius), TubePoint::new(b, radius)], 1)
    }

    pub fn control(&self) -> &[TubePoint] {
        &self.control
    }

    pub fn control_mut(&mut self) -> &mut Vec<TubePoint> {
        &mut self.control
    }

    /// Re-evaluate the spline. Must be called after touching `control_mut`;
    /// the polyline and bounds are derived state and go stale otherwise.
    pub fn rebuild(&mut self) {
        self.polyline = catmull_rom(&self.control, self.subdivisions);
        self.bounds = self
            .polyline
            .iter()
            .fold(None::<Aabb>, |acc, p| {
                let b = Aabb::around(p.pos, p.pos, p.radius);
                Some(acc.map_or(b, |a| a.union(&b)))
            })
            .unwrap_or(Aabb::new(Vec3::ZERO, Vec3::ZERO));
    }

    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    pub fn segments(&self) -> usize {
        self.polyline.len().saturating_sub(1)
    }

    /// Distance to the swept surface: the union of one round cone per polyline
    /// segment. Consecutive segments share an endpoint *and* its radius, so
    /// the union is seamless without any joint handling.
    pub fn distance(&self, p: Vec3) -> f32 {
        if self.polyline.is_empty() {
            return f32::INFINITY;
        }
        if self.polyline.len() == 1 {
            let s = self.polyline[0];
            return sdf::sphere(p, s.pos, s.radius);
        }
        let mut d = f32::INFINITY;
        for w in self.polyline.windows(2) {
            d = d.min(sdf::round_cone(p, w[0].pos, w[1].pos, w[0].radius, w[1].radius));
        }
        d
    }
}

/// Catmull-Rom through every control point, with the endpoints duplicated so
/// the curve starts and ends exactly where it was authored.
///
/// Catmull-Rom rather than Bezier because it interpolates its control points:
/// a user dragging a cave waypoint expects the tunnel to pass through the
/// handle, not to be pulled vaguely toward it.
fn catmull_rom(control: &[TubePoint], subdivisions: u32) -> Vec<TubePoint> {
    if control.len() < 2 {
        return control.to_vec();
    }
    let n = control.len();
    let at = |i: isize| -> TubePoint { control[(i.clamp(0, n as isize - 1)) as usize] };

    let mut out = Vec::with_capacity((n - 1) * subdivisions as usize + 1);
    for i in 0..n - 1 {
        let (p0, p1, p2, p3) =
            (at(i as isize - 1), at(i as isize), at(i as isize + 1), at(i as isize + 2));
        for s in 0..subdivisions {
            let t = s as f32 / subdivisions as f32;
            out.push(TubePoint {
                pos: catmull_point(p0.pos, p1.pos, p2.pos, p3.pos, t),
                // The radius rides the same basis, so a passage that widens
                // does it smoothly instead of in per-segment steps.
                radius: catmull_scalar(p0.radius, p1.radius, p2.radius, p3.radius, t).max(0.01),
            });
        }
    }
    out.push(control[n - 1]);
    out
}

fn catmull_point(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn catmull_scalar(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// How a modifier combines with everything beneath it in the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Add the shape as solid rock -- a buttress, a plug, a bridge deck.
    Union,
    /// Carve the shape out. Caves and tunnels.
    Subtract,
    /// Keep only what falls inside the shape. Used to clip a world to a
    /// playable island.
    Intersect,
}

impl Op {
    pub const ALL: [Op; 3] = [Op::Union, Op::Subtract, Op::Intersect];

    pub fn label(self) -> &'static str {
        match self {
            Op::Union => "Add",
            Op::Subtract => "Carve",
            Op::Intersect => "Clip",
        }
    }
}

/// One entry in the stack.
#[derive(Debug, Clone, PartialEq)]
pub struct Modifier {
    pub name: String,
    pub shape: Shape,
    pub op: Op,
    /// Fillet width in metres. Zero is a hard boolean.
    pub blend: f32,
    /// Switched off modifiers stay in the list and cost one bool to skip.
    /// Toggling one is how you check what a cave is doing to the silhouette
    /// without losing the spline.
    pub enabled: bool,
}

impl Modifier {
    pub fn new(name: impl Into<String>, shape: Shape, op: Op) -> Self {
        Self { name: name.into(), shape, op, blend: 0.0, enabled: true }
    }

    /// A carve with a rounded lip, which is what almost every cave wants.
    pub fn carve(name: impl Into<String>, shape: Shape, blend: f32) -> Self {
        Self { name: name.into(), shape, op: Op::Subtract, blend, enabled: true }
    }

    /// World region this modifier can possibly affect, blend included.
    ///
    /// An `Intersect` is the exception: it removes material *everywhere
    /// outside* itself, so its influence is unbounded and it cannot be
    /// bounds-rejected. Returning `None` says exactly that.
    pub fn bounds(&self) -> Option<Aabb> {
        match self.op {
            Op::Intersect => None,
            _ => Some(self.shape.bounds().expand(self.blend.max(0.0))),
        }
    }

    fn apply(&self, p: Vec3, d: f32) -> f32 {
        if !self.enabled {
            return d;
        }
        // Bounds reject before evaluating the shape. For a Tube this skips a
        // loop over every polyline segment, which is the difference between a
        // stack that scales to a cave system and one that does not.
        //
        // The reject preserves the *sign* everywhere, and the exact value
        // everywhere inside the box -- but not the value outside it. Outside,
        // `subtract` would still have clamped a deeply-solid distance toward
        // the far-off cave wall (`max(-100, -8)` is `-8`), and skipping it
        // leaves the larger magnitude.
        //
        // That is safe here and only here: extraction samples a dense lattice
        // and only ever interpolates across cells that already contain a sign
        // change, which by definition are inside the box. It would *not* be
        // safe for sphere tracing, where an overestimated distance lets a ray
        // step straight through a wall. Anything added later that marches this
        // field has to evaluate the stack unrejected.
        if let Some(b) = self.bounds()
            && !b.contains(p)
        {
            return d;
        }
        let s = self.shape.distance(p);
        match self.op {
            Op::Union => sdf::smooth_union(d, s, self.blend),
            Op::Subtract => sdf::smooth_subtract(d, s, self.blend),
            Op::Intersect => sdf::smooth_intersect(d, s, self.blend),
        }
    }
}

/// The ordered modifier list. Order matters: carving a tunnel and then adding
/// a plug is a blocked passage, while adding the plug first and carving after
/// is an open one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModifierStack {
    pub items: Vec<Modifier>,
}

impl ModifierStack {
    pub fn push(&mut self, m: Modifier) -> usize {
        self.items.push(m);
        self.items.len() - 1
    }

    pub fn remove(&mut self, i: usize) -> Option<Modifier> {
        (i < self.items.len()).then(|| self.items.remove(i))
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Fold every enabled modifier over an incoming distance.
    pub fn apply(&self, p: Vec3, mut d: f32) -> f32 {
        for m in &self.items {
            d = m.apply(p, d);
        }
        d
    }

    /// Combined influence region, or `None` if any enabled modifier is
    /// unbounded. Used to decide which chunks a stack edit dirties.
    pub fn bounds(&self) -> Option<Aabb> {
        let mut acc: Option<Aabb> = None;
        for m in self.items.iter().filter(|m| m.enabled) {
            let b = m.bounds()?;
            acc = Some(acc.map_or(b, |a| a.union(&b)));
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid everywhere, so a carve is the only thing that can make air.
    const SOLID: f32 = -100.0;

    #[test]
    fn a_carve_makes_air_and_leaves_the_rest_alone() {
        let mut s = ModifierStack::default();
        s.push(Modifier::carve("cave", Shape::Sphere { center: Vec3::ZERO, radius: 5.0 }, 0.0));
        assert!(s.apply(Vec3::ZERO, SOLID) > 0.0, "inside the sphere must be air");
        assert!(s.apply(Vec3::new(50.0, 0.0, 0.0), SOLID) < 0.0, "far away must stay solid");
    }

    #[test]
    fn disabling_a_modifier_restores_the_rock() {
        // The non-destructive claim, stated as a test: the same point reads
        // air with the modifier on and solid with it off, and nothing else
        // had to change.
        let mut s = ModifierStack::default();
        s.push(Modifier::carve("cave", Shape::Sphere { center: Vec3::ZERO, radius: 5.0 }, 0.0));
        assert!(s.apply(Vec3::ZERO, SOLID) > 0.0);
        s.items[0].enabled = false;
        assert_eq!(s.apply(Vec3::ZERO, SOLID), SOLID, "rock must come back untouched");
    }

    #[test]
    fn removing_a_modifier_is_the_same_as_never_adding_it() {
        let mut s = ModifierStack::default();
        s.push(Modifier::carve("a", Shape::Sphere { center: Vec3::ZERO, radius: 5.0 }, 0.0));
        let probe = Vec3::new(1.0, 2.0, 0.5);
        let with = s.apply(probe, SOLID);
        s.remove(0);
        assert_eq!(s.apply(probe, SOLID), SOLID);
        assert_ne!(with, SOLID, "the test is vacuous unless the carve did something");
    }

    #[test]
    fn stack_order_decides_whether_a_passage_is_blocked() {
        let tunnel = Shape::Capsule {
            a: Vec3::new(-20.0, 0.0, 0.0),
            b: Vec3::new(20.0, 0.0, 0.0),
            radius: 3.0,
        };
        let plug = Shape::Sphere { center: Vec3::ZERO, radius: 4.0 };

        let mut carve_then_plug = ModifierStack::default();
        carve_then_plug.push(Modifier::carve("t", tunnel.clone(), 0.0));
        carve_then_plug.push(Modifier::new("p", plug.clone(), Op::Union));

        let mut plug_then_carve = ModifierStack::default();
        plug_then_carve.push(Modifier::new("p", plug, Op::Union));
        plug_then_carve.push(Modifier::carve("t", tunnel, 0.0));

        assert!(carve_then_plug.apply(Vec3::ZERO, SOLID) < 0.0, "plug last must block");
        assert!(plug_then_carve.apply(Vec3::ZERO, SOLID) > 0.0, "carve last must open");
    }

    #[test]
    fn bounds_reject_never_moves_the_surface() {
        // The optimization is allowed to change the distance *magnitude*
        // outside its box -- see the note in `Modifier::apply` -- but it must
        // never change a sign, because the sign is what the extracted surface
        // is made of. Inside the box it must be exact.
        let m = Modifier::carve("c", Shape::Sphere { center: Vec3::ZERO, radius: 3.0 }, 0.0);
        let bounds = m.bounds().unwrap();
        let mut inside_checked = 0;
        for i in 0..20 {
            for j in 0..20 {
                let p = Vec3::new(-6.0 + i as f32 * 0.6, 0.0, -6.0 + j as f32 * 0.6);
                let fast = m.apply(p, SOLID);
                let slow = sdf::smooth_subtract(SOLID, m.shape.distance(p), 0.0);
                assert_eq!(
                    fast < 0.0,
                    slow < 0.0,
                    "at {p} the reject flipped solid/air: {fast} vs {slow}"
                );
                if bounds.contains(p) {
                    assert!((fast - slow).abs() < 1e-4, "at {p}, inside bounds: {fast} vs {slow}");
                    inside_checked += 1;
                }
            }
        }
        assert!(inside_checked > 50, "only {inside_checked} points fell inside the box");
    }

    #[test]
    fn intersect_is_never_bounds_rejected() {
        // A clip removes everything outside itself, so treating its bounding
        // box as its influence would leave the whole rest of the world solid.
        let m =
            Modifier::new("clip", Shape::Sphere { center: Vec3::ZERO, radius: 5.0 }, Op::Intersect);
        assert!(m.bounds().is_none());
        let far = Vec3::new(500.0, 0.0, 0.0);
        assert!(m.apply(far, SOLID) > 0.0, "outside a clip must become air");
    }

    #[test]
    fn tube_passes_through_its_control_points() {
        // The reason for Catmull-Rom over Bezier. A dragged waypoint must be
        // on the tunnel, not near it.
        let pts = vec![
            TubePoint::new(Vec3::new(-10.0, 0.0, 0.0), 2.0),
            TubePoint::new(Vec3::new(0.0, 5.0, 3.0), 3.0),
            TubePoint::new(Vec3::new(10.0, 0.0, 0.0), 2.0),
        ];
        let t = Tube::new(pts.clone(), 8);
        for p in &pts {
            let d = t.distance(p.pos);
            assert!(
                d < -p.radius * 0.9,
                "control point {:?} should be deep inside, d = {d}",
                p.pos
            );
        }
    }

    #[test]
    fn tube_radius_interpolates_between_control_points() {
        let t = Tube::new(
            vec![
                TubePoint::new(Vec3::new(-10.0, 0.0, 0.0), 1.0),
                TubePoint::new(Vec3::new(10.0, 0.0, 0.0), 5.0),
            ],
            8,
        );
        // On the axis the distance is -radius, so the bore is readable directly.
        let narrow = -t.distance(Vec3::new(-10.0, 0.0, 0.0));
        let wide = -t.distance(Vec3::new(10.0, 0.0, 0.0));
        assert!((narrow - 1.0).abs() < 0.2, "narrow end {narrow}");
        assert!((wide - 5.0).abs() < 0.2, "wide end {wide}");
        assert!(wide > narrow, "the tube must actually taper");
    }

    #[test]
    fn tube_surface_is_continuous_along_its_length() {
        // Segment joins are where a swept tube cracks. Walk the axis and check
        // the reported bore never jumps -- a step here is a visible ring in
        // the extracted mesh.
        let t = Tube::new(
            vec![
                TubePoint::new(Vec3::new(-10.0, 0.0, 0.0), 2.0),
                TubePoint::new(Vec3::new(0.0, 4.0, 0.0), 2.5),
                TubePoint::new(Vec3::new(10.0, 0.0, 2.0), 2.0),
            ],
            8,
        );
        let mut prev: Option<f32> = None;
        for i in 0..=200 {
            let x = -10.0 + i as f32 * 0.1;
            // Sample well outside so we read the tube's envelope, not its core.
            let d = t.distance(Vec3::new(x, 30.0, 0.0));
            if let Some(p) = prev {
                assert!((d - p).abs() < 0.5, "discontinuity at x = {x}: {p} -> {d}");
            }
            prev = Some(d);
        }
    }

    #[test]
    fn empty_and_single_point_tubes_are_harmless() {
        assert_eq!(Tube::new(vec![], 4).distance(Vec3::ZERO), f32::INFINITY);
        let one = Tube::new(vec![TubePoint::new(Vec3::ZERO, 2.0)], 4);
        assert!((one.distance(Vec3::new(5.0, 0.0, 0.0)) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn tube_bounds_contain_the_whole_sweep() {
        let t = Tube::new(
            vec![
                TubePoint::new(Vec3::new(-10.0, 0.0, 0.0), 2.0),
                TubePoint::new(Vec3::new(0.0, 8.0, 0.0), 3.0),
                TubePoint::new(Vec3::new(10.0, 0.0, 0.0), 2.0),
            ],
            8,
        );
        let b = t.bounds();
        // Anything the tube calls solid must be inside the box, or the bounds
        // reject in `Modifier::apply` would silently clip the cave.
        for i in 0..40 {
            for j in 0..40 {
                let p = Vec3::new(-15.0 + i as f32 * 0.75, -2.0 + j as f32 * 0.35, 0.0);
                if t.distance(p) < 0.0 {
                    assert!(b.contains(p), "{p} is inside the tube but outside its bounds");
                }
            }
        }
    }

    #[test]
    fn straight_tube_matches_a_plain_capsule() {
        let (a, b) = (Vec3::new(-5.0, 1.0, 0.0), Vec3::new(5.0, 1.0, 0.0));
        let t = Tube::straight(a, b, 1.5);
        for p in [Vec3::new(0.0, 4.0, 0.0), Vec3::new(-7.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 3.0)] {
            let want = sdf::capsule(p, a, b, 1.5);
            assert!((t.distance(p) - want).abs() < 1e-4, "at {p}");
        }
    }
}

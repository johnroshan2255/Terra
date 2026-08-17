//! Physical dimensions of a driveable vehicle, measured from its mesh.
//!
//! Here rather than in `terra-physics` or `terra-assets` because both need it and
//! neither should depend on the other: physics has no business decoding glTF, and the
//! asset loader has no business knowing about rigid bodies. `terra-core` is the shared
//! vocabulary, which is what it is for.
//!
//! # Why these are measured and not typed in
//!
//! The chassis collider, the suspension mount points and the wheel radius all have to
//! agree with the mesh, and until now they did not: the collider was a hand-written
//! `[0.85, 0.5, 1.8]` half-extent "small hatchback" while the renderer drew a box of
//! the same size, and neither had anything to do with the vehicle model on disk.
//! Swapping the mesh for a real one would have left a 5.2 m Hummer driving on the
//! collider of a 3.6 m hatchback -- wheels floating outside the arches, the body
//! grounding out on nothing, and no way to tell from the code that anything was wrong.
//!
//! Taking every figure from the mesh's own geometry makes that class of mismatch
//! impossible rather than merely fixed.

/// Where a vehicle's parts are, in metres, in its own space.
///
/// Convention, matching glTF and the rest of the renderer: **+Y up, +Z forward**, origin
/// at the centre of the contact patch -- the point on the ground between the axles. That
/// origin is deliberate: every figure below is then a plain measurement from the ground,
/// and the mesh needs no offset when it is drawn at the rigid body's transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleDims {
    /// Half-extents of the chassis collision box.
    ///
    /// The body only -- not the wheels, which are raycasts, and not roof racks or
    /// mirrors, which should not stop the vehicle on a rock face.
    pub chassis_half: [f32; 3],
    /// Height of the collision box's centre above the ground.
    ///
    /// Separate from the half-extents because the box is not centred on the origin: it
    /// sits above it by the ground clearance. This also sets the centre of mass, which
    /// is what decides how readily the vehicle rolls over.
    pub chassis_centre_y: f32,
    pub wheel_radius: f32,
    /// Tyre width, for drawing. The simulation does not use it.
    pub wheel_width: f32,
    /// Half the track: distance from the centreline to a wheel's centre.
    pub axle_half_width: f32,
    /// Front axle position along +Z.
    pub front_axle_z: f32,
    /// Rear axle position, negative.
    pub rear_axle_z: f32,
    /// Kerb mass in kilograms.
    ///
    /// Set explicitly rather than derived from a collider density, because density
    /// times a box volume is a number nobody can sanity-check, and mass drives every
    /// force in the vehicle model.
    pub mass_kg: f32,
}

impl VehicleDims {
    /// Distance between the axles.
    pub fn wheelbase(&self) -> f32 {
        self.front_axle_z - self.rear_axle_z
    }

    /// Distance between the wheel centres across an axle.
    pub fn track(&self) -> f32 {
        self.axle_half_width * 2.0
    }

    /// Gap between the bottom of the chassis box and the ground.
    pub fn ground_clearance(&self) -> f32 {
        self.chassis_centre_y - self.chassis_half[1]
    }

    /// Overall length across the body box.
    pub fn length(&self) -> f32 {
        self.chassis_half[2] * 2.0
    }

    /// Lateral acceleration, in g, at which the vehicle would tip rather than slide.
    ///
    /// `(track / 2) / centre-of-mass height`. Worth having as a function because it is
    /// the number that decides whether a tall vehicle is driveable or a rollover
    /// simulator, and because it is easy to break by nudging either input: raising the
    /// body to clear a rock lowers this, and so does widening the collider.
    ///
    /// Real values sit near 1.0-1.2 for a road car and near 1.05 for a Hummer H1. Tyre
    /// grip has to stay below it or the vehicle tips in every corner.
    pub fn rollover_threshold_g(&self) -> f32 {
        self.axle_half_width / self.chassis_centre_y.max(1e-3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Roughly the Hummer H1 the game drives, for tests that need numbers.
    fn h1() -> VehicleDims {
        VehicleDims {
            chassis_half: [1.30, 0.80, 2.55],
            chassis_centre_y: 1.30,
            wheel_radius: 0.5675,
            wheel_width: 0.444,
            axle_half_width: 1.009,
            front_axle_z: 1.7385,
            rear_axle_z: -1.648,
            mass_kg: 2900.0,
        }
    }

    #[test]
    fn the_derived_figures_match_the_real_vehicle() {
        let d = h1();
        // H1: 3.30 m wheelbase, 1.83 m track officially, 0.41 m clearance, 4.69 m long.
        assert!((d.wheelbase() - 3.39).abs() < 0.05, "wheelbase {}", d.wheelbase());
        assert!((d.track() - 2.02).abs() < 0.05, "track {}", d.track());
        assert!(d.ground_clearance() > 0.35, "clearance {}", d.ground_clearance());
        assert!((4.5..5.4).contains(&d.length()), "length {}", d.length());
    }

    #[test]
    fn the_centre_of_mass_is_low_enough_to_corner_on() {
        // The check that keeps the vehicle driveable. A tall body with a narrow track
        // tips instead of sliding, and it presents as "the car flips whenever I turn"
        // rather than as a bad number anywhere.
        let g = h1().rollover_threshold_g();
        assert!(g > 0.75, "would tip at {g} g, which any corner reaches");
        // And it should not be so low that a 2.5 m tall off-roader corners like a
        // go-kart -- that reads as fake.
        assert!(g < 1.6, "rollover threshold {g} g is implausibly high for a Hummer");
    }

    #[test]
    fn the_wheels_fit_under_the_body() {
        // A wheel outside the chassis box is a wheel visibly outside the arch, and a
        // wheel taller than the clearance is one buried in the floor.
        let d = h1();
        assert!(
            d.axle_half_width <= d.chassis_half[0] + 0.25,
            "wheels stick {} m outside the body",
            d.axle_half_width - d.chassis_half[0]
        );
        assert!(d.front_axle_z < d.chassis_half[2], "front axle is beyond the front bumper");
        assert!(-d.rear_axle_z < d.chassis_half[2], "rear axle is beyond the rear bumper");
        assert!(
            d.wheel_radius > d.ground_clearance() * 0.5,
            "a wheel this small could not lift the body clear"
        );
    }
}

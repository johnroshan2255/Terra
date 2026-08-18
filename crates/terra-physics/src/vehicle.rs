//! A four-wheeled vehicle on a raycast suspension.
//!
//! Wheels are raycasts against the world, not rolling cylinder colliders. That is the
//! standard approach and the right one here: a cylinder rolling on a heightfield catches
//! on texel edges, and at 2-4 m/texel those edges are metres apart. The suspension is a
//! real spring-damper -- stiffness, separate compression and rebound damping, a travel
//! limit and a force ceiling -- so the body squats under power, dives under braking and
//! rolls into corners without any of that being scripted.
//!
//! # The dimensions are measured, not typed in
//!
//! Every figure comes from [`VehicleDims`], which `terra-assets` measures off the vehicle
//! mesh itself. This file used to carry `CHASSIS_HALF: [0.85, 0.5, 1.8]` -- a "small
//! hatchback" -- while the renderer drew a box of the same size, and neither had anything
//! to do with a vehicle model. Bolting a real 5.2 m Hummer onto that would have left the
//! wheels a metre outside the arches with nothing in the code looking wrong.
//!
//! # Collision shape and mass distribution are separate on purpose
//!
//! The collider is where the body *is*: a box over the passenger compartment, its floor at
//! the chassis rail. The centre of mass is much lower, because in a body-on-frame vehicle
//! the engine, transfer case and axles all sit near the floor.
//!
//! Letting the collider's own centre serve as the centre of mass -- which is what happens
//! if you give it a density and stop there -- put the mass 1.24 m up on a 2.02 m track.
//! That is a static stability factor of 0.81 g, and since tyres can pull more than that,
//! the vehicle would tip in any hard corner. Real H1 figures are near 1.05 g. So the
//! collider is massless and the body carries explicit mass properties.

use crate::PhysicsWorld;
use rapier3d::control::{DynamicRayCastVehicleController, WheelTuning};
use rapier3d::prelude::*;
use terra_core::VehicleDims;

/// Height of the centre of mass above the collision box's floor, as a fraction of that
/// box's height.
///
/// A third. The engine, gearbox, transfer case, propshafts and both live axles of a
/// body-on-frame off-roader sit at or below the floor pan, so the mass is nothing like
/// uniformly distributed through the bounding box. On the measured Hummer this puts the
/// centre of mass 0.96 m up against a 1.01 m half-track, for a static stability factor of
/// 1.05 g -- which is the published figure for the real vehicle.
const COM_FRACTION_OF_BODY: f32 = 1.0 / 3.0;

/// Scale applied to the inertia of a uniform box of the collider's size.
///
/// A uniform box overestimates: real mass is concentrated low and towards the centre, not
/// spread to the corners. Unscaled, the measured Hummer comes out at 8200 kg m^2 in yaw,
/// where a vehicle of its size and mass measures nearer 5500. 0.7 lands on that, and it is
/// what decides how willingly the vehicle changes direction -- too high and steering feels
/// like it is filtered through treacle.
const INERTIA_SCALE: f32 = 0.7;

/// Peak tractive force at the contact patches, in newtons, summed across the axles.
///
/// 20 kN against 28.5 kN of vehicle weight, so it will hold itself on a 45% grade and pull
/// away up one. A Hummer H1 is rated for 60%; getting there needs more grip than the tyre
/// model gives, so this is the honest ceiling rather than the brochure one.
const MAX_TRACTIVE_N: f32 = 20_000.0;

/// Crank power reaching the ground, in watts. An H1's 195 hp less drivetrain losses.
///
/// Force is capped by [`MAX_TRACTIVE_N`] up to about 6.5 m/s and falls as power over speed
/// above it, which is what a real drivetrain does in aggregate across its gears. Modelling
/// it as one constant force cannot do both jobs at once: the force needed to climb a steep
/// grade is roughly four times the force that gives a 2.9 t vehicle a plausible 0-100, so a
/// single figure either cannot climb or accelerates like a supercar. The first attempt here
/// used 5.2 kN and could not get up a 20 degree slope.
const ENGINE_POWER_W: f32 = 130_000.0;

/// Suspension length with the wheel hanging free.
///
/// Long, because this is an off-roader. It is also what sets ride height: the spring
/// compresses under the vehicle's weight until it supports it, so the height it settles at
/// is a result rather than a setting, and `the_hummer_settles_at_its_designed_ride_height`
/// is what checks the result is right.
const SUSPENSION_REST: f32 = 0.42;

/// Driver input for one step, each in `-1..=1` or `0..=1`.
#[derive(Debug, Default, Clone, Copy)]
pub struct VehicleInput {
    /// Forward is positive, reverse negative.
    pub throttle: f32,
    pub brake: f32,
    /// Left is positive.
    pub steer: f32,
    pub handbrake: bool,
}

/// Where to draw one wheel this frame.
#[derive(Debug, Clone, Copy)]
pub struct WheelPose {
    pub position: [f32; 3],
    /// Steering angle in radians, about the chassis up axis.
    pub steer: f32,
    /// Rolling angle in radians, about the axle.
    pub roll: f32,
    pub in_contact: bool,
    /// How far the spring is compressed from its free length, in metres. Zero when the
    /// wheel is hanging.
    pub compression: f32,
}

pub struct Vehicle {
    controller: DynamicRayCastVehicleController,
    dims: VehicleDims,
    /// Steering lock at a standstill, in radians.
    max_steer: f32,
    /// Accumulated wheel rotation, purely for rendering.
    roll: [f32; 4],
    /// Forward speed measured from the chassis body each step.
    speed: f32,
}

impl Vehicle {
    /// Spawn a vehicle with the centre of its contact patch at `ground_position`.
    ///
    /// That is the point on the ground between the axles, matching the origin of the mesh
    /// the dimensions were measured from -- so the same translation places the body and
    /// the collider with no offset to keep in step.
    pub fn spawn(world: &mut PhysicsWorld, ground_position: [f32; 3], dims: &VehicleDims) -> Self {
        let d = *dims;
        let half = d.chassis_half;

        // Mass properties, explicit and entirely on the body: see the note above on why
        // the collider must not supply them.
        let com_y = (d.chassis_centre_y - half[1]) + half[1] * 2.0 * COM_FRACTION_OF_BODY;
        let m = d.mass_kg;
        // Principal inertia of a box about its own axes, then scaled. Y up, Z forward, so
        // this is (pitch, yaw, roll).
        let k = m / 3.0 * INERTIA_SCALE;
        let inertia = Vector::new(
            k * (half[1] * half[1] + half[2] * half[2]),
            k * (half[0] * half[0] + half[2] * half[2]),
            k * (half[0] * half[0] + half[1] * half[1]),
        );

        let body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(ground_position[0], ground_position[1], ground_position[2]))
            // Without damping the chassis keeps spinning after a bad landing.
            .linear_damping(0.05)
            .angular_damping(0.6)
            .additional_mass_properties(MassProperties::new(
                Vector::new(0.0, com_y, 0.0),
                m,
                inertia,
            ))
            // A five-metre body at 30 m/s covers half its own length in a step; without
            // this it can pass through a barrier between two frames.
            .ccd_enabled(true)
            .build();
        let handle = world.bodies.insert(body);

        let collider = ColliderBuilder::cuboid(half[0], half[1], half[2])
            // The box sits above the origin by the ground clearance, not centred on it.
            .translation(Vector::new(0.0, d.chassis_centre_y, 0.0))
            // Massless: every mass property is set on the body above. A density here would
            // silently add a second, higher centre of mass to the one chosen there.
            .density(0.0)
            .friction(0.4)
            // Sheet metal does not bounce.
            .restitution(0.05)
            .build();
        world.colliders.insert_with_parent(collider, handle, &mut world.bodies);

        // Rapier folds `additional_mass_properties` into the body at the *next* step, so
        // without this the first step runs with a mass of zero -- and a zero-mass chassis
        // takes any force as infinite acceleration, which fires the vehicle off the map on
        // the frame it spawns. Forcing the recompute here also means `mass()` and the
        // centre of mass are readable immediately, which is what the tests check.
        let colliders = &world.colliders;
        world.bodies[handle].recompute_mass_properties_from_colliders(colliders);

        let mut controller = DynamicRayCastVehicleController::new(handle);
        // Chassis axes: Y up, Z forward. Must match the cuboid above and the mesh.
        controller.index_up_axis = 1;
        controller.index_forward_axis = 2;

        let tuning = WheelTuning {
            // Bullet's lineage: the spring force this produces is already proportional to
            // the chassis mass, so the number does not have to be rescaled for a 2.9 t
            // vehicle. What does have to be rescaled is the ceiling below.
            suspension_stiffness: 26.0,
            // Rebound damped harder than compression. A spring that extends as fast as it
            // compresses makes the vehicle pogo after every bump, and on a heavy one that
            // turns into a bounce it never settles out of.
            suspension_compression: 1.1,
            suspension_damping: 2.3,
            // Long travel, because this is an off-roader and because a wheel that runs out
            // of travel transmits the impact straight into the body.
            max_suspension_travel: 0.34,
            side_friction_stiffness: 1.0,
            // Grip has to stay under the rollover threshold or the vehicle tips instead of
            // sliding. See `a_hard_turn_at_speed_slides_rather_than_flipping`.
            friction_slip: 2.2,
            // Static load per wheel is mass * g / 4, about 7.1 kN here. A landing needs
            // several times that, and a ceiling too low makes the suspension bottom out on
            // ground it should absorb.
            max_suspension_force: 40_000.0,
        };

        let down = Vector::NEG_Y;
        // The axle points LEFT, not right, and that is not a typo.
        //
        // Rapier derives a wheel's forward direction as `contact_normal x axle`. On level
        // ground that is `+Y x axle`, and `+Y x +X` is **-Z** -- so an axle pointing right
        // makes positive engine force drive the vehicle backwards. It did: under full
        // throttle the chassis accelerated to -40 m/s down a slope it was supposed to
        // climb. Nothing caught it because the tests measured distance travelled as an
        // absolute value, which is exactly as large going the wrong way.
        //
        // `+Y x -X` is `+Z`, which is the forward the mesh, the collider and
        // `index_forward_axis` all already agreed on.
        let axle = -Vector::X;
        // Corner order matches `VehicleRig::wheels`: front-left, front-right, rear-left,
        // rear-right. Wheels 0 and 1 steer.
        for (sx, z) in [
            (-1.0, d.front_axle_z),
            (1.0, d.front_axle_z),
            (-1.0, d.rear_axle_z),
            (1.0, d.rear_axle_z),
        ] {
            // The hard point is where the spring bolts to the body: directly above the
            // wheel's centre by the free length of the spring, so that at rest the wheel
            // hangs into exactly the place the mesh puts it.
            let mount = Vector::new(d.axle_half_width * sx, d.wheel_radius + SUSPENSION_REST, z);
            controller.add_wheel(mount, down, axle, SUSPENSION_REST, d.wheel_radius, &tuning);
        }

        Self { controller, dims: d, max_steer: 0.60, roll: [0.0; 4], speed: 0.0 }
    }

    pub fn dims(&self) -> &VehicleDims {
        &self.dims
    }

    /// Tractive force available at the contact patches at a given speed, in newtons.
    ///
    /// Flat below the speed at which full power would exceed [`MAX_TRACTIVE_N`], then
    /// falling as `power / speed`. Public because the shape of this curve is what
    /// acceleration and top speed both come out of, and it is worth checking directly.
    pub fn tractive_force(&self, speed_ms: f32) -> f32 {
        let v = speed_ms.abs().max(0.5);
        (ENGINE_POWER_W / v).min(MAX_TRACTIVE_N)
    }

    /// Steering lock available at a given speed, in radians.
    ///
    /// A real steering rack has a fixed ratio, but a driver does not use full lock at
    /// speed and a car that did would be uncontrollable -- at 30 m/s, 34 degrees of lock
    /// asks for a 5 m radius turn, which is a spin. Tapering the available angle with
    /// speed is what every driving game does, and it is the single change that makes a
    /// keyboard-steered vehicle feel like a car rather than a shopping trolley.
    ///
    /// Public because it is worth testing directly: the taper is easy to get inverted, and
    /// inverted it would give full lock at speed and none when parking.
    pub fn steer_limit(&self, speed_ms: f32) -> f32 {
        // Half the lock by 25 m/s, a third of it by 50.
        let taper = 1.0 / (1.0 + (speed_ms.abs() / 25.0));
        self.max_steer * taper.clamp(0.25, 1.0)
    }

    /// Advance one fixed step. Call before [`PhysicsWorld::step`].
    pub fn update(&mut self, world: &mut PhysicsWorld, input: VehicleInput, dt: f32) {
        // Passed through, *not* negated. Rapier steers about `-wheel_direction_ws` --
        // the suspension points down, so that axis is world up -- and a positive
        // rotation about up takes +Z forward towards +X. With `right = forward x up`,
        // which is the convention `Camera::right` uses and therefore the one the
        // driver sees, +X on a +Z heading is *left*. So positive steer is already
        // left and the sign needs nothing done to it.
        //
        // This was negated, with a comment asserting the opposite handedness, and the
        // test that covered it asserted the direction it observed rather than the one
        // a driver would name -- so the two agreed with each other and disagreed with
        // the keyboard. `steering_direction` below pins it to `up x forward` instead,
        // which is checkable without a mental rotation.
        let steer = input.steer.clamp(-1.0, 1.0) * self.steer_limit(self.speed);
        let drive = input.throttle.clamp(-1.0, 1.0) * self.tractive_force(self.speed);
        let pedal = input.brake.clamp(0.0, 1.0);

        for (i, wheel) in self.controller.wheels_mut().iter_mut().enumerate() {
            let front = i < 2;
            wheel.steering = if front { steer } else { 0.0 };
            // Permanent four-wheel drive, which is what a Hummer has and what makes it
            // climb. Rear-wheel drive on a 2.9 t vehicle with this much weight over the
            // back axle just spins the rears on anything loose. A quarter each, because
            // `drive` is the force for the whole vehicle.
            wheel.engine_force = drive * 0.25;
            // Front-biased braking, about 60/40. Weight transfers forward under braking,
            // so the front tyres have the grip to use it; an even split locks the rears
            // first and the tail steps out every time you slow down.
            let bias = if front { 1.2 } else { 0.8 };
            wheel.brake = pedal * 6_000.0 * bias
                // Handbrake on the rear axle only -- that is what makes it a handbrake
                // rather than a second brake pedal.
                + if input.handbrake && !front { 9_000.0 } else { 0.0 };
        }

        let chassis = self.controller.chassis;

        // Wake the body ourselves on any input. Rapier only wakes it for a *positive*
        // engine force, so once the vehicle had come to a stop and fallen asleep, reverse
        // did nothing at all -- the throttle went in, the wheels were given their force,
        // and the sleeping body ignored all of it. Braking and steering a rolling vehicle
        // have the same problem in principle, so this covers all three.
        if input.throttle != 0.0 || input.brake > 0.0 || input.steer != 0.0 || input.handbrake {
            world.bodies[chassis].wake_up(true);
        }

        self.controller.update_vehicle(dt, world.query_excluding(chassis));

        // Measure speed from the chassis rather than reading the controller's
        // `current_vehicle_speed`. That field keeps reporting motion after the body has
        // gone to sleep -- it will sit at 1.5 m/s while the vehicle has not moved a
        // millimetre for seconds.
        let body = &world.bodies[self.controller.chassis];
        let forward = *body.rotation() * Vector::Z;
        self.speed = body.linvel().dot(forward);

        // Roll is for rendering only; the simulation never reads it.
        for r in self.roll.iter_mut() {
            *r = (*r + self.speed / self.dims.wheel_radius * dt) % std::f32::consts::TAU;
        }
    }

    pub fn chassis(&self) -> RigidBodyHandle {
        self.controller.chassis
    }

    /// Speed along the chassis forward axis, in m/s. Measured from the body, so it reads
    /// zero when the vehicle is actually stationary.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// World transform of the chassis: translation and rotation quaternion
    /// `[x, y, z, w]`.
    /// Put the vehicle back on its wheels at `ground_position`, facing `yaw`.
    ///
    /// The recovery for a car on its roof, wedged in a gully, or dropped through
    /// something. A raycast vehicle has no way out of those by itself: upside down
    /// the wheel rays point at the sky, so there is no contact, no traction and
    /// nothing the controller can do with throttle or steering.
    ///
    /// Velocities are zeroed rather than kept. Carrying momentum through a reset is
    /// how a car that rolled at 30 m/s immediately rolls again.
    /// `heading` is in the same convention [`Self::heading`] returns, so the two
    /// round-trip: reading a heading and resetting to it leaves the car pointing the
    /// way it already was.
    ///
    /// That conversion is the whole reason this takes a heading rather than a body
    /// yaw. A body yaw of `y` puts forward at `(sin y, 0, cos y)`, whose `atan2(z, x)`
    /// is `pi/2 - y` -- so passing a heading straight to `from_rotation_y` mirrors the
    /// car about the 45 degree line, which on a reset looks like it turned by itself.
    pub fn reset(&mut self, world: &mut PhysicsWorld, ground_position: [f32; 3], heading: f32) {
        let body = &mut world.bodies[self.controller.chassis];
        body.set_translation(
            Vector::new(ground_position[0], ground_position[1], ground_position[2]),
            true,
        );
        // Upright, keeping only the heading. Rapier 0.35 is glam-based, so this is
        // its rotation type directly.
        let body_yaw = std::f32::consts::FRAC_PI_2 - heading;
        body.set_rotation(glam::Quat::from_rotation_y(body_yaw), true);
        body.set_linvel(Vector::ZERO, true);
        body.set_angvel(Vector::ZERO, true);
        // The suspension state is derived from the raycasts each step, but the roll
        // angles are ours and would otherwise carry the old spin into the new pose.
        self.roll = [0.0; 4];
        self.speed = 0.0;
    }

    /// Heading in radians, in the same convention `Camera::yaw` uses: the angle of
    /// the forward vector in the XZ plane, measured as `atan2(z, x)`.
    pub fn heading(&self, world: &PhysicsWorld) -> f32 {
        let body = &world.bodies[self.controller.chassis];
        let forward = body.rotation() * glam::Vec3::Z;
        forward.z.atan2(forward.x)
    }

    /// Whether the vehicle is far enough from upright to be unrecoverable.
    ///
    /// Its own up axis against world up. Past 90 degrees no wheel can reach the
    /// ground, so throttle and steering do nothing and the only way out is a reset.
    pub fn is_overturned(&self, world: &PhysicsWorld) -> bool {
        let body = &world.bodies[self.controller.chassis];
        (body.rotation() * glam::Vec3::Y).y < 0.0
    }

    pub fn chassis_pose(&self, world: &PhysicsWorld) -> ([f32; 3], [f32; 4]) {
        let body = &world.bodies[self.controller.chassis];
        let t = body.translation();
        let r = body.rotation();
        ([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
    }

    /// Where each wheel currently sits, accounting for suspension travel.
    pub fn wheel_poses(&self) -> Vec<WheelPose> {
        self.controller
            .wheels()
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let info = w.raycast_info();
                // The wheel hangs below its hard point by however far the spring is
                // currently extended -- this is what makes the wheels visibly move in the
                // arches instead of being welded to the body.
                //
                // Taken from Rapier rather than recomputed. This was
                // `info.hard_point_ws + w.direction_cs * info.suspension_length`, which
                // mixes frames: the hard point is world space and `direction_cs` is
                // *chassis* space. Level ground hid it, because the suspension axis is
                // (0, -1, 0) in both frames when the body is flat -- but the moment the
                // chassis pitched or rolled, the wheels dropped along world down instead
                // of along the car's own suspension travel, so they slid vertically out
                // of the arches. `center()` is `hard_point_ws + wheel_direction_ws *
                // suspension_length`, which is the same expression with the world-space
                // direction it should always have used.
                let centre = w.center();
                WheelPose {
                    position: [centre.x, centre.y, centre.z],
                    steer: w.steering,
                    roll: self.roll[i.min(3)],
                    in_contact: info.is_in_contact,
                    compression: (SUSPENSION_REST - info.suspension_length).max(0.0),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FIXED_DT;

    /// The measured Hummer, so the tests exercise the vehicle the game actually drives.
    ///
    /// Written out rather than loaded from the glb, because `terra-physics` has no glTF
    /// decoder and should not grow one. `terra-assets` has the test that these figures
    /// match the file; this one is about what happens once they reach the simulation.
    fn hummer() -> VehicleDims {
        VehicleDims {
            chassis_half: [1.3050119, 0.8158015, 2.5982199],
            chassis_centre_y: 1.2414056,
            wheel_radius: 0.5674721,
            wheel_width: 0.44394073,
            axle_half_width: 1.0087726,
            front_axle_z: 1.7380618,
            rear_axle_z: -1.6483529,
            mass_kg: 2900.0,
        }
    }

    const GROUND: f32 = 50.0;

    pub(super) fn flat_world() -> PhysicsWorld {
        let mut w = PhysicsWorld::new();
        w.set_terrain(&vec![GROUND; 64 * 64], 64, 400.0);
        w
    }

    pub(super) fn spawn(world: &mut PhysicsWorld) -> Vehicle {
        Vehicle::spawn(world, [0.0, GROUND + 0.6, 0.0], &hummer())
    }

    pub(super) fn settle(
        world: &mut PhysicsWorld,
        car: &mut Vehicle,
        steps: usize,
        input: VehicleInput,
    ) {
        for _ in 0..steps {
            car.update(world, input, FIXED_DT);
            world.step();
        }
    }

    fn drive(world: &mut PhysicsWorld, car: &mut Vehicle) {
        settle(world, car, 180, VehicleInput::default());
        settle(world, car, 400, VehicleInput { throttle: 1.0, ..Default::default() });
    }

    // --- the suspension ---

    #[test]
    fn the_hummer_settles_at_its_designed_ride_height() {
        // Ride height is a *result*: the spring compresses under 2.9 tonnes until it
        // supports them. If the stiffness or the force ceiling is wrong for the mass, the
        // vehicle either sits on its belly or floats, and both look like a broken model
        // rather than a mis-tuned spring.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 400, VehicleInput::default());

        let (pos, _) = car.chassis_pose(&world);
        let height = pos[1] - GROUND;
        // The body origin is the contact patch, so this should settle near zero -- within
        // the spring's working range either way.
        assert!(height.abs() < 0.25, "the contact patch settled {height} m off the ground");

        let poses = car.wheel_poses();
        assert!(poses.iter().all(|w| w.in_contact), "a wheel is off flat ground");
        for (i, w) in poses.iter().enumerate() {
            let wheel_height = w.position[1] - GROUND;
            assert!(
                (wheel_height - hummer().wheel_radius).abs() < 0.2,
                "wheel {i} centre sits {wheel_height} m up, tyre radius is {}",
                hummer().wheel_radius
            );
        }
    }

    #[test]
    fn the_springs_carry_the_weight_without_bottoming_out() {
        // The failure this catches: a force ceiling tuned for a light car cannot hold a
        // 2.9 t one up, so every spring pins itself fully compressed and the vehicle rides
        // on its bump stops -- which feels like driving a brick and shows no suspension
        // movement at all.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 400, VehicleInput::default());

        for (i, w) in car.wheel_poses().iter().enumerate() {
            assert!(w.compression > 0.01, "wheel {i} is not carrying any load");
            assert!(
                w.compression < SUSPENSION_REST * 0.85,
                "wheel {i} is compressed {} m of {SUSPENSION_REST} and has run out of travel",
                w.compression
            );
        }
    }

    #[test]
    fn the_body_dives_under_braking_and_squats_under_power() {
        // Load transfer, which is the whole reason to simulate a spring at each corner
        // rather than glue the body to the ground.
        //
        // Measured as the *difference* between the axles, not as each axle's change from
        // rest. Two earlier versions of this got it wrong in instructive ways. Comparing
        // front against rear after ten seconds of full throttle measured the static weight
        // split, because at terminal speed there is no acceleration to transfer anything.
        // Comparing each axle against its own settled value then failed too: applying
        // throttle lifts the whole body for the first tenth of a second, so both axles
        // extend before either starts to squat. The gap between them has neither problem.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 400, VehicleInput::default());
        // Positive means the rear is more compressed than the front, i.e. squatting.
        let rear_bias = |car: &Vehicle| {
            let p = car.wheel_poses();
            (p[2].compression + p[3].compression) * 0.5
                - (p[0].compression + p[1].compression) * 0.5
        };
        let settled = rear_bias(&car);

        settle(&mut world, &mut car, 24, VehicleInput { throttle: 1.0, ..Default::default() });
        let accelerating = rear_bias(&car);
        assert!(
            accelerating > settled + 0.02,
            "the rear did not squat under power: bias {settled} -> {accelerating}"
        );

        // Then hard braking from speed: the transfer reverses and the nose dives.
        settle(&mut world, &mut car, 400, VehicleInput { throttle: 1.0, ..Default::default() });
        let rolling = rear_bias(&car);
        settle(&mut world, &mut car, 24, VehicleInput { brake: 1.0, ..Default::default() });
        let braking = rear_bias(&car);
        assert!(
            braking < rolling - 0.02,
            "the nose did not dive under braking: bias {rolling} -> {braking}"
        );
        assert!(braking < 0.0, "under full braking the front should be the loaded axle");
    }

    #[test]
    fn a_wheel_over_a_hole_hangs_instead_of_following_the_body() {
        // The visible test of a working shock absorber: an unloaded wheel extends to the
        // spring's free length rather than staying where the body is.
        //
        // A pit under one wheel, not a step under half the vehicle. The step version let the
        // whole vehicle slide into the low side and settle there with every wheel touching
        // again, which is a true statement about a step and no test of a spring.
        let n = 128usize;
        let extent = 48.0f32;
        let mut h = vec![GROUND; n * n];
        let d = hummer();
        let to_world = |i: usize| (i as f32 / (n - 1) as f32 - 0.5) * extent;
        for z in 0..n {
            for x in 0..n {
                let (wx, wz) = (to_world(x), to_world(z));
                // Around the front-left wheel only, and deeper than the spring can reach.
                if (wx - -d.axle_half_width).abs() < 0.9 && (wz - d.front_axle_z).abs() < 0.9 {
                    h[z * n + x] = GROUND - 3.0;
                }
            }
        }
        let mut world = PhysicsWorld::new();
        world.set_terrain(&h, n as u32, extent);

        let mut car = Vehicle::spawn(&mut world, [0.0, GROUND + 0.6, 0.0], &d);
        settle(&mut world, &mut car, 90, VehicleInput { brake: 1.0, ..Default::default() });

        let p = car.wheel_poses();
        assert!(!p[0].in_contact, "the front-left wheel found the bottom of a three metre pit");
        assert!(
            p[0].compression < 0.05,
            "an airborne wheel is still compressed {} m",
            p[0].compression
        );
        // And the spring extended it below the others, which is what shows on screen.
        assert!(
            p[0].position[1] < p[1].position[1],
            "the hanging wheel sits at {} against {} for its opposite number",
            p[0].position[1],
            p[1].position[1]
        );
        // The other three keep the vehicle up.
        assert!(
            p[1..].iter().filter(|w| w.in_contact).count() >= 2,
            "the rest of the vehicle fell into the pit too"
        );
    }

    // --- driving ---

    #[test]
    fn full_throttle_reaches_a_plausible_speed() {
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        let (start, _) = {
            settle(&mut world, &mut car, 180, VehicleInput::default());
            car.chassis_pose(&world)
        };
        settle(&mut world, &mut car, 600, VehicleInput { throttle: 1.0, ..Default::default() });
        let (end, _) = car.chassis_pose(&world);

        let travelled = ((end[0] - start[0]).powi(2) + (end[2] - start[2]).powi(2)).sqrt();
        assert!(travelled > 20.0, "only moved {travelled} m in ten seconds of full throttle");
        let kph = car.speed().abs() * 3.6;
        // Not a supercar and not a milk float. An H1 does about 130 km/h flat out.
        assert!((20.0..170.0).contains(&kph), "reached {kph:.0} km/h");
    }

    #[test]
    fn braking_brings_it_to_a_stop() {
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        drive(&mut world, &mut car);
        assert!(car.speed().abs() > 3.0, "never got moving");

        // Assert it stops *moving*, not that a derived number reads zero.
        let stop = VehicleInput { brake: 1.0, handbrake: true, ..Default::default() };
        settle(&mut world, &mut car, 300, stop);
        let (before, _) = car.chassis_pose(&world);
        settle(&mut world, &mut car, 120, stop);
        let (after, _) = car.chassis_pose(&world);

        let crept = ((after[0] - before[0]).powi(2) + (after[2] - before[2]).powi(2)).sqrt();
        assert!(crept < 0.6, "travelled {crept} m in the two seconds after stopping");
    }

    #[test]
    fn steering_turns_it_and_the_lock_tapers_with_speed() {
        // The taper is easy to invert, and inverted it gives full lock at 100 km/h and
        // almost none in a car park -- which feels like the steering is broken.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        let parked = car.steer_limit(0.0);
        let fast = car.steer_limit(30.0);
        assert!(parked > fast, "lock should shrink with speed: {parked} at rest, {fast} at 30 m/s");
        assert!(fast > 0.0, "no steering at all at speed");
        assert!(
            car.steer_limit(-20.0) == car.steer_limit(20.0),
            "reversing should not change the lock"
        );

        // And it actually turns.
        drive(&mut world, &mut car);
        let (_, before) = car.chassis_pose(&world);
        settle(
            &mut world,
            &mut car,
            240,
            VehicleInput { throttle: 0.6, steer: 1.0, ..Default::default() },
        );
        let (_, after) = car.chassis_pose(&world);
        let yaw = |q: [f32; 4]| {
            // Heading from the quaternion's forward axis.
            let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
            let fx = 2.0 * (x * z + w * y);
            let fz = 1.0 - 2.0 * (x * x + y * y);
            fz.atan2(fx)
        };
        let turned = (yaw(after) - yaw(before)).abs();
        assert!(turned > 0.3, "only changed heading by {turned} rad under full lock");
    }

    #[test]
    fn a_hard_turn_at_speed_slides_rather_than_flipping() {
        // The consequence of the centre of mass being where it is. Grip above the rollover
        // threshold tips a tall vehicle instead of sliding it, and the symptom is a car
        // that lands on its roof every time you turn.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        drive(&mut world, &mut car);
        settle(
            &mut world,
            &mut car,
            400,
            VehicleInput { throttle: 1.0, steer: 1.0, ..Default::default() },
        );

        let (_, q) = car.chassis_pose(&world);
        // The body's up-axis, from the quaternion.
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        let up_y = 1.0 - 2.0 * (x * x + z * z);
        let _ = (y, w);
        assert!(up_y > 0.55, "the vehicle rolled onto its side: up.y = {up_y}");
        assert!(
            car.wheel_poses().iter().filter(|p| p.in_contact).count() >= 2,
            "fewer than two wheels still on the ground through the corner"
        );
    }

    #[test]
    fn it_climbs_a_slope_that_a_two_wheel_drive_would_not() {
        // Four-wheel drive, which is what a Hummer has and what the previous rear-drive
        // setup could not do on anything loose.
        let n = 64;
        let extent = 400.0;
        let mut h = vec![0.0f32; n * n];
        // A 20 degree ramp rising along +Z.
        for z in 0..n {
            let world_z = (z as f32 / (n - 1) as f32 - 0.5) * extent;
            for x in 0..n {
                h[z * n + x] = GROUND + world_z * 0.36;
            }
        }
        let mut world = PhysicsWorld::new();
        world.set_terrain(&h, n as u32, extent);

        let start_ground = GROUND + 0.0 * 0.36;
        let mut car = Vehicle::spawn(&mut world, [0.0, start_ground + 0.6, 0.0], &hummer());
        // On the brakes while it settles. Released, a 2.9 t vehicle on a 20 degree slope
        // rolls backwards for the whole three seconds and starts the climb already moving
        // downhill -- which is correct behaviour and a useless starting point.
        settle(&mut world, &mut car, 200, VehicleInput { brake: 1.0, ..Default::default() });
        let (before, _) = car.chassis_pose(&world);
        settle(&mut world, &mut car, 600, VehicleInput { throttle: 1.0, ..Default::default() });
        let (after, _) = car.chassis_pose(&world);

        assert!(
            after[1] > before[1] + 2.0,
            "climbed only {} m of a 20 degree slope",
            after[1] - before[1]
        );
    }

    #[test]
    fn the_handbrake_works_the_rear_axle_only() {
        // Otherwise it is just a second brake pedal, and the one thing a handbrake is for
        // -- unsticking the tail -- does not happen.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 60, VehicleInput::default());
        car.update(&mut world, VehicleInput { handbrake: true, ..Default::default() }, FIXED_DT);

        let brakes: Vec<f32> = car.controller.wheels().iter().map(|w| w.brake).collect();
        assert_eq!(brakes[0], 0.0, "the front-left brake came on with the handbrake");
        assert_eq!(brakes[1], 0.0, "the front-right brake came on with the handbrake");
        assert!(brakes[2] > 0.0 && brakes[3] > 0.0, "the handbrake did not reach the rear axle");
    }

    #[test]
    fn all_four_wheels_are_driven_and_only_the_front_pair_steers() {
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 60, VehicleInput::default());
        car.update(
            &mut world,
            VehicleInput { throttle: 1.0, steer: 1.0, ..Default::default() },
            FIXED_DT,
        );

        for (i, w) in car.controller.wheels().iter().enumerate() {
            assert!(w.engine_force.abs() > 0.0, "wheel {i} is not driven");
            if i < 2 {
                assert!(w.steering.abs() > 0.0, "front wheel {i} does not steer");
            } else {
                assert_eq!(w.steering, 0.0, "rear wheel {i} steers");
            }
        }
    }

    #[test]
    fn throttle_drives_forward_and_reverse_drives_back() {
        // The regression test for a bug that shipped invisibly. Rapier derives a wheel's
        // forward direction as `contact_normal x axle`, so an axle pointing right makes
        // positive engine force drive the vehicle *backwards*. It did, and every test
        // passed, because they all measured distance travelled as an absolute value --
        // which is exactly as large in the wrong direction. Only a signed check catches it.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        let (start, _) = car.chassis_pose(&world);

        settle(&mut world, &mut car, 200, VehicleInput { throttle: 1.0, ..Default::default() });
        let (fwd, _) = car.chassis_pose(&world);
        assert!(
            fwd[2] > start[2] + 1.0,
            "throttle moved it from z {} to {}, which is not forward",
            start[2],
            fwd[2]
        );
        assert!(car.speed() > 0.0, "speed reads {} under throttle", car.speed());

        // And reverse goes the other way, rather than simply being ignored.
        let stop = VehicleInput { brake: 1.0, handbrake: true, ..Default::default() };
        settle(&mut world, &mut car, 400, stop);
        let (from, _) = car.chassis_pose(&world);
        settle(&mut world, &mut car, 200, VehicleInput { throttle: -1.0, ..Default::default() });
        let (back, _) = car.chassis_pose(&world);
        assert!(back[2] < from[2] - 0.5, "reverse moved it from z {} to {}", from[2], back[2]);
    }

    #[test]
    fn positive_steer_turns_left() {
        // Driving up +Z, a left turn moves towards **+X**: with `right = forward x up`,
        // `Z x Y = -X`, so right is -X and left is +X.
        //
        // This assertion used to demand the opposite, on a comment that asserted the
        // handedness the wrong way round. It passed, because the steering sign was
        // negated to match it -- the test and the code agreed with each other and the
        // car steered away from the key that was pressed. See `steering_direction`,
        // which checks the same thing against `up x forward` rather than against a
        // hard-coded axis.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        settle(&mut world, &mut car, 240, VehicleInput { throttle: 0.7, ..Default::default() });
        let (before, _) = car.chassis_pose(&world);
        settle(
            &mut world,
            &mut car,
            240,
            VehicleInput { throttle: 0.5, steer: 1.0, ..Default::default() },
        );
        let (after, _) = car.chassis_pose(&world);
        assert!(
            after[0] > before[0] + 1.0,
            "under positive steer it went from x {} to {}, which is a right turn",
            before[0],
            after[0]
        );
    }

    #[test]
    fn the_torque_curve_can_climb_without_making_it_a_supercar() {
        // Both ends of the same curve. A single constant force cannot do both, which is
        // what the first attempt got wrong: 5.2 kN gave a plausible 0-100 and could not get
        // up a 20 degree slope.
        let mut world = flat_world();
        let car = spawn(&mut world);
        let d = hummer();
        let weight = d.mass_kg * 9.81;

        // Pulling away: enough to hold itself on a steep grade.
        let launch = car.tractive_force(0.0);
        let grade = (launch / weight).asin().to_degrees();
        assert!(grade > 35.0, "can only hold a {grade} degree slope from rest");

        // At speed: force has fallen, so acceleration is nothing like a sports car's.
        let at_speed = car.tractive_force(28.0);
        assert!(at_speed < launch * 0.4, "force barely fell with speed: {launch} -> {at_speed}");
        let accel_g = at_speed / weight;
        assert!(accel_g < 0.25, "still pulling {accel_g} g at 100 km/h");

        // Monotonic, and never zero -- a curve that crossed zero would coast to a stop.
        let mut prev = f32::MAX;
        for v in [0.0f32, 2.0, 6.0, 10.0, 20.0, 40.0, 80.0] {
            let f = car.tractive_force(v);
            assert!(f > 0.0, "no force at {v} m/s");
            assert!(f <= prev + 1.0, "force rose from {prev} to {f} at {v} m/s");
            prev = f;
        }
    }

    // --- the rig contract ---

    #[test]
    fn the_wheels_are_mounted_in_the_corners_the_mesh_expects() {
        // Wheel order is a contract with `VehicleRig::wheels`: index 0 is front-left in
        // both. Break it and the drawn wheels swap corners, which looks like the steering
        // is coming from the back.
        let mut world = flat_world();
        let car = spawn(&mut world);
        let d = hummer();
        let expected = [
            (-d.axle_half_width, d.front_axle_z),
            (d.axle_half_width, d.front_axle_z),
            (-d.axle_half_width, d.rear_axle_z),
            (d.axle_half_width, d.rear_axle_z),
        ];
        for (i, w) in car.controller.wheels().iter().enumerate() {
            let c = w.chassis_connection_point_cs;
            assert!(
                (c.x - expected[i].0).abs() < 1e-4 && (c.z - expected[i].1).abs() < 1e-4,
                "wheel {i} is mounted at ({}, {}), expected {:?}",
                c.x,
                c.z,
                expected[i]
            );
        }
    }

    #[test]
    fn the_centre_of_mass_is_below_the_middle_of_the_body() {
        // The mass properties are the reason the vehicle corners instead of tipping, and
        // they are set in a place nothing else touches -- so if a density crept back onto
        // the collider, this is what would notice.
        let mut world = flat_world();
        let car = spawn(&mut world);
        let body = &world.bodies[car.chassis()];
        let com = body.mass_properties().local_mprops.local_com;
        let d = hummer();

        assert!(com.y < d.chassis_centre_y, "the centre of mass is not below the box centre");
        assert!(com.y > d.ground_clearance(), "the centre of mass is under the chassis rail");
        assert!((body.mass() - d.mass_kg).abs() < 1.0, "mass reads {} kg", body.mass());

        let ssf = d.axle_half_width / com.y;
        assert!((0.9..1.25).contains(&ssf), "static stability factor {ssf} g is not H1-like");
    }
}

#[cfg(test)]
mod wheel_frames {
    use super::tests::{flat_world, settle, spawn};
    use super::*;

    /// Wheel centres in the chassis's own frame.
    fn wheels_in_chassis_space(car: &Vehicle, world: &PhysicsWorld) -> Vec<glam::Vec3> {
        let (t, r) = car.chassis_pose(world);
        let inv = glam::Quat::from_xyzw(r[0], r[1], r[2], r[3]).inverse();
        car.wheel_poses()
            .iter()
            .map(|w| inv * (glam::Vec3::from_array(w.position) - glam::Vec3::from_array(t)))
            .collect()
    }

    #[test]
    fn a_tilted_chassis_keeps_its_wheels_in_the_arches() {
        // The bug this guards: the wheel centre was built from a world-space hard point
        // plus a *chassis*-space suspension direction. Flat ground hides it, because both
        // are (0, -1, 0) there. Tilt the body and the wheels slide vertically out of the
        // arches, which is what "the axle moves vertically sometimes" looks like.
        //
        // Measured in the chassis's own frame: whatever the body is doing, a wheel must
        // stay at its axle position and only travel along the suspension axis.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        let level = wheels_in_chassis_space(&car, &world);

        // Roll and pitch the chassis hard, then read the wheels again.
        let handle = car.chassis();
        let body = &mut world.bodies[handle];
        let tilt = glam::Quat::from_euler(glam::EulerRot::YXZ, 0.7, 0.35, 0.45);
        // Rapier 0.35 is glam-based, so our quaternion is its rotation type.
        body.set_rotation(tilt, true);
        // One step so the controller refreshes its hard points from the new pose.
        settle(&mut world, &mut car, 1, VehicleInput::default());
        let tilted = wheels_in_chassis_space(&car, &world);

        for (i, (a, b)) in level.iter().zip(&tilted).enumerate() {
            // Lateral and longitudinal offsets are fixed by the axle geometry and must
            // not move at all. Only Y -- the suspension axis -- is free.
            assert!(
                (a.x - b.x).abs() < 0.05,
                "wheel {i} moved {:.3} m sideways in chassis space when the body tilted",
                (a.x - b.x).abs()
            );
            assert!(
                (a.z - b.z).abs() < 0.05,
                "wheel {i} moved {:.3} m fore-aft in chassis space when the body tilted",
                (a.z - b.z).abs()
            );
        }
    }

    #[test]
    fn wheels_sit_where_the_axles_say_on_level_ground() {
        // The baseline the test above compares against: two wheels either side, front
        // pair ahead of the rear pair, all at the same height.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        let w = wheels_in_chassis_space(&car, &world);
        assert_eq!(w.len(), 4);

        // 0/1 front, 2/3 rear -- the order `spawn` adds them in.
        assert!(w[0].z > 0.0 && w[1].z > 0.0, "front wheels are not ahead: {w:?}");
        assert!(w[2].z < 0.0 && w[3].z < 0.0, "rear wheels are not behind: {w:?}");
        // One of each pair either side of the centreline.
        assert!(w[0].x * w[1].x < 0.0, "front pair is on one side: {w:?}");
        assert!(w[2].x * w[3].x < 0.0, "rear pair is on one side: {w:?}");
        // Level ground, so all four hang the same amount.
        for pair in w.windows(2) {
            assert!((pair[0].y - pair[1].y).abs() < 0.02, "uneven ride height: {w:?}");
        }
    }
}

#[cfg(test)]
mod recovery {
    use super::tests::{flat_world, settle, spawn};
    use super::*;

    /// Roll the chassis onto its roof.
    fn flip(world: &mut PhysicsWorld, car: &Vehicle) {
        let body = &mut world.bodies[car.chassis()];
        body.set_rotation(glam::Quat::from_rotation_z(std::f32::consts::PI), true);
        body.set_translation(Vector::new(0.0, 2.0, 0.0), true);
    }

    #[test]
    fn an_upside_down_car_is_detected() {
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 200, VehicleInput::default());
        assert!(!car.is_overturned(&world), "a settled car is not overturned");

        flip(&mut world, &car);
        assert!(car.is_overturned(&world), "a car on its roof must report overturned");
    }

    #[test]
    fn a_reset_puts_it_back_on_its_wheels() {
        // The recovery R exists for. A raycast vehicle upside down has its wheel rays
        // pointing at the sky, so there is no contact and no input does anything --
        // it can only be lifted out.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 200, VehicleInput::default());
        flip(&mut world, &car);
        settle(&mut world, &mut car, 5, VehicleInput::default());
        assert!(car.is_overturned(&world));

        car.reset(&mut world, [12.0, 1.0, -7.0], 0.9);
        assert!(!car.is_overturned(&world), "still overturned after a reset");

        let (t, _) = car.chassis_pose(&world);
        assert!((t[0] - 12.0).abs() < 0.01 && (t[2] + 7.0).abs() < 0.01, "moved to {t:?}");
        // And it stays up once the simulation runs again.
        settle(&mut world, &mut car, 200, VehicleInput::default());
        assert!(!car.is_overturned(&world), "it fell back over after the reset");
    }

    #[test]
    fn a_reset_keeps_the_heading_it_was_given() {
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 200, VehicleInput::default());
        // Round-trip: `reset` takes what `heading` gives.
        for want in [0.0f32, 1.2, -2.5, 3.0] {
            car.reset(&mut world, [0.0, 1.0, 0.0], want);
            let got = car.heading(&world);
            assert!((got - want).abs() < 0.01, "heading came back {got}, wanted {want}");
        }
    }

    #[test]
    fn a_reset_drops_the_momentum_it_crashed_with() {
        // Carrying velocity through a reset is how a car that rolled at speed
        // immediately rolls again.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        settle(&mut world, &mut car, 400, VehicleInput { throttle: 1.0, ..Default::default() });
        assert!(car.speed().abs() > 5.0, "the car never got moving");

        car.reset(&mut world, [0.0, 1.0, 0.0], 0.0);
        assert_eq!(car.speed(), 0.0, "speed survived the reset");
        let v = world.bodies[car.chassis()].linvel();
        assert!(v.length() < 1e-4, "velocity survived the reset: {v:?}");
    }

    #[test]
    fn heading_matches_the_camera_yaw_convention() {
        // `Camera::yaw` is `atan2(z, x)` of the forward vector, and the chase camera
        // sits at `heading + pi`. If these two disagree the camera starts up facing
        // the bonnet.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 200, VehicleInput::default());
        // Heading 0 means forward points along +X, because the convention is
        // `atan2(z, x)` -- the same one `Camera::yaw` uses, which is what lets the
        // chase camera sit at `heading + pi` and be behind the car.
        car.reset(&mut world, [0.0, 1.0, 0.0], 0.0);
        let forward = world.bodies[car.chassis()].rotation() * glam::Vec3::Z;
        assert!(forward.x > 0.99, "heading 0 should face +X, forward is {forward}");
        assert!(car.heading(&world).abs() < 0.01);
    }
}

#[cfg(test)]
mod steering_direction {
    use super::tests::{flat_world, settle, spawn};
    use super::*;

    /// Screen-left for a camera behind the car looking along `forward`.
    ///
    /// Tied to the same convention `terra_render::camera::Camera::right()` uses --
    /// `right = forward x up`, so `left = up x forward`. That function is what
    /// decides which way the world appears to move, so it is the only definition of
    /// "left" that can be checked against what a driver sees.
    fn screen_left(forward: glam::Vec3) -> glam::Vec3 {
        glam::Vec3::Y.cross(forward).normalize_or_zero()
    }

    #[test]
    fn left_is_plus_x_when_driving_towards_plus_z() {
        // The claim the steering sign rests on, isolated from the simulation.
        let l = screen_left(glam::Vec3::Z);
        assert!(l.x > 0.9, "left came out {l}, so the convention note is wrong");
    }

    #[test]
    fn pressing_left_moves_the_car_to_screen_left() {
        // `VehicleInput::steer` is documented as positive-is-left, and this is what
        // that has to mean: the car goes the way the driver's eye calls left.
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        settle(&mut world, &mut car, 240, VehicleInput { throttle: 0.7, ..Default::default() });

        let (before, rot) = car.chassis_pose(&world);
        let q = glam::Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
        let forward = q * glam::Vec3::Z;
        let left = screen_left(forward);

        settle(
            &mut world,
            &mut car,
            240,
            VehicleInput { throttle: 0.5, steer: 1.0, ..Default::default() },
        );
        let (after, _) = car.chassis_pose(&world);

        let moved = glam::Vec3::from_array(after) - glam::Vec3::from_array(before);
        let lateral = moved.dot(left);
        assert!(
            lateral > 1.0,
            "positive steer moved {lateral:.2} m along screen-left, so A steers right"
        );
    }

    #[test]
    fn pressing_right_moves_the_car_to_screen_right() {
        let mut world = flat_world();
        let mut car = spawn(&mut world);
        settle(&mut world, &mut car, 300, VehicleInput::default());
        settle(&mut world, &mut car, 240, VehicleInput { throttle: 0.7, ..Default::default() });

        let (before, rot) = car.chassis_pose(&world);
        let q = glam::Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
        let left = screen_left(q * glam::Vec3::Z);

        settle(
            &mut world,
            &mut car,
            240,
            VehicleInput { throttle: 0.5, steer: -1.0, ..Default::default() },
        );
        let (after, _) = car.chassis_pose(&world);
        let moved = glam::Vec3::from_array(after) - glam::Vec3::from_array(before);
        assert!(
            moved.dot(left) < -1.0,
            "negative steer went {:.2} m along screen-left, so D steers left",
            moved.dot(left)
        );
    }
}

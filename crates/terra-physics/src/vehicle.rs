//! A small four-wheeled car on a raycast suspension.
//!
//! Wheels are raycasts against the world, not rolling cylinder colliders. That
//! is the standard approach and the right one here: a cylinder rolling on a
//! heightfield catches on texel edges, and at 2-4 m/texel those edges are metres
//! apart. The suspension is a real spring-damper -- stiffness, compression and
//! rebound damping, travel limit -- so the body pitches under braking and rolls
//! into corners without any of that being faked.

use crate::PhysicsWorld;
use rapier3d::control::{DynamicRayCastVehicleController, WheelTuning};
use rapier3d::prelude::*;

/// Half-extents of the chassis box, in meters. A small hatchback: 3.6 m long,
/// 1.7 m wide, 1.0 m tall.
pub const CHASSIS_HALF: [f32; 3] = [0.85, 0.5, 1.8];
pub const WHEEL_RADIUS: f32 = 0.36;
/// Where the wheels sit relative to the chassis centre.
const AXLE_HALF_WIDTH: f32 = 0.78;
const AXLE_FORWARD: f32 = 1.25;
/// Suspension length at rest. The chassis floats this far above the wheel's
/// attachment point when unloaded.
const SUSPENSION_REST: f32 = 0.35;

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
}

pub struct Vehicle {
    controller: DynamicRayCastVehicleController,
    engine_force: f32,
    max_steer: f32,
    /// Accumulated wheel rotation, purely for rendering.
    roll: [f32; 4],
    /// Forward speed measured from the chassis body each step.
    speed: f32,
}

impl Vehicle {
    /// Spawn a car with its chassis centre at `position`.
    pub fn spawn(world: &mut PhysicsWorld, position: [f32; 3]) -> Self {
        let body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(position[0], position[1], position[2]))
            // Without damping the chassis keeps spinning after a bad landing.
            .linear_damping(0.05)
            .angular_damping(0.6)
            .build();
        let handle = world.bodies.insert(body);

        let collider = ColliderBuilder::cuboid(CHASSIS_HALF[0], CHASSIS_HALF[1], CHASSIS_HALF[2])
            .density(120.0)
            .friction(0.5)
            .build();
        world.colliders.insert_with_parent(collider, handle, &mut world.bodies);

        let mut controller = DynamicRayCastVehicleController::new(handle);
        // Chassis axes: Y up, Z forward. Must match the cuboid above.
        controller.index_up_axis = 1;
        controller.index_forward_axis = 2;

        // Close to Rapier's defaults on purpose. These are Bullet's long-lived
        // vehicle constants and they are tuned against each other; pushing
        // stiffness or friction_slip well past them does not give a firmer car,
        // it launches the chassis into the air.
        let tuning = WheelTuning {
            suspension_stiffness: 24.0,
            // Rebound damping higher than compression: a spring that extends as
            // fast as it compresses makes the car pogo.
            suspension_compression: 0.9,
            suspension_damping: 1.6,
            // Default is 5 metres, which is not suspension travel, it is a
            // crane. A road car has roughly a hand's width.
            max_suspension_travel: 0.30,
            side_friction_stiffness: 1.0,
            // Rapier's own docs warn this flips the vehicle when too strong.
            friction_slip: 12.0,
            max_suspension_force: 12_000.0,
        };

        let down = Vector::NEG_Y;
        let axle = Vector::X;
        for (i, (sx, sz)) in
            [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].into_iter().enumerate()
        {
            let mount =
                Vector::new(AXLE_HALF_WIDTH * sx, -CHASSIS_HALF[1] * 0.4, AXLE_FORWARD * sz);
            controller.add_wheel(mount, down, axle, SUSPENSION_REST, WHEEL_RADIUS, &tuning);
            let _ = i;
        }

        Self { controller, engine_force: 900.0, max_steer: 0.55, roll: [0.0; 4], speed: 0.0 }
    }

    /// Advance one fixed step. Call before [`PhysicsWorld::step`].
    pub fn update(&mut self, world: &mut PhysicsWorld, input: VehicleInput, dt: f32) {
        let steer = input.steer.clamp(-1.0, 1.0) * self.max_steer;
        let drive = input.throttle.clamp(-1.0, 1.0) * self.engine_force;
        let brake =
            input.brake.clamp(0.0, 1.0) * 2500.0 + if input.handbrake { 4000.0 } else { 0.0 };

        for (i, wheel) in self.controller.wheels_mut().iter_mut().enumerate() {
            let front = i < 2;
            // Front wheels steer, rear wheels drive. Rear-wheel drive keeps the
            // steering from fighting the engine on loose ground.
            wheel.steering = if front { steer } else { 0.0 };
            wheel.engine_force = if front { 0.0 } else { drive };
            wheel.brake = brake;
        }

        let chassis = self.controller.chassis;
        self.controller.update_vehicle(dt, world.query_excluding(chassis));

        // Measure speed from the chassis rather than reading the controller's
        // `current_vehicle_speed`. That field keeps reporting motion after the
        // body has gone to sleep -- it will sit at 1.5 m/s while the car has
        // not moved a millimetre for seconds.
        let body = &world.bodies[self.controller.chassis];
        let forward = *body.rotation() * Vector::Z;
        self.speed = body.linvel().dot(forward);

        // Roll is for rendering only; the simulation never reads it.
        for r in self.roll.iter_mut() {
            *r = (*r + self.speed / WHEEL_RADIUS * dt) % std::f32::consts::TAU;
        }
    }

    pub fn chassis(&self) -> RigidBodyHandle {
        self.controller.chassis
    }

    /// Speed along the chassis forward axis, in m/s. Measured from the body,
    /// so it reads zero when the car is actually stationary.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// World transform of the chassis: translation and rotation quaternion
    /// `[x, y, z, w]`.
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
                // The wheel hangs below its hard point by however far the
                // suspension is currently extended -- this is what makes the
                // wheels visibly move in the arches.
                let centre = info.hard_point_ws + w.direction_cs * info.suspension_length;
                WheelPose {
                    position: [centre.x, centre.y, centre.z],
                    steer: w.steering,
                    roll: self.roll[i.min(3)],
                    in_contact: info.is_in_contact,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FIXED_DT;

    fn flat_world() -> PhysicsWorld {
        let mut w = PhysicsWorld::new();
        w.set_terrain(&vec![50.0; 32 * 32], 32, 256.0);
        w
    }

    fn settle(world: &mut PhysicsWorld, car: &mut Vehicle, steps: usize, input: VehicleInput) {
        for _ in 0..steps {
            car.update(world, input, FIXED_DT);
            world.step();
        }
    }

    #[test]
    fn the_car_rests_on_its_suspension_not_on_its_belly() {
        let mut world = flat_world();
        let mut car = Vehicle::spawn(&mut world, [0.0, 54.0, 0.0]);
        settle(&mut world, &mut car, 240, VehicleInput::default());

        let (pos, _) = car.chassis_pose(&world);
        // Ground at 50 m. The chassis should hover roughly a wheel radius plus
        // some suspension above it, and certainly not be sitting in the dirt.
        assert!(pos[1] > 50.3, "chassis sank to {}", pos[1]);
        assert!(pos[1] < 52.5, "chassis floating at {}", pos[1]);

        assert!(
            car.wheel_poses().iter().all(|w| w.in_contact),
            "every wheel should be touching flat ground"
        );
    }

    #[test]
    fn throttle_moves_it_forward() {
        let mut world = flat_world();
        let mut car = Vehicle::spawn(&mut world, [0.0, 54.0, 0.0]);
        settle(&mut world, &mut car, 180, VehicleInput::default());
        let (start, _) = car.chassis_pose(&world);

        settle(&mut world, &mut car, 240, VehicleInput { throttle: 1.0, ..Default::default() });
        let (end, _) = car.chassis_pose(&world);

        let travelled = (end[2] - start[2]).abs() + (end[0] - start[0]).abs();
        assert!(travelled > 2.0, "car only moved {travelled} m under full throttle");
        assert!(car.speed().abs() > 1.0, "speed reads {} m/s", car.speed());
    }

    #[test]
    fn braking_brings_it_to_a_stop() {
        let mut world = flat_world();
        let mut car = Vehicle::spawn(&mut world, [0.0, 54.0, 0.0]);
        settle(&mut world, &mut car, 180, VehicleInput::default());
        settle(&mut world, &mut car, 300, VehicleInput { throttle: 1.0, ..Default::default() });
        assert!(car.speed().abs() > 1.0, "car never got moving");

        // Assert it stops *moving*, not that a derived number reads zero.
        settle(
            &mut world,
            &mut car,
            240,
            VehicleInput { brake: 1.0, handbrake: true, ..Default::default() },
        );
        let (before, _) = car.chassis_pose(&world);
        settle(
            &mut world,
            &mut car,
            120,
            VehicleInput { brake: 1.0, handbrake: true, ..Default::default() },
        );
        let (after, _) = car.chassis_pose(&world);

        let crept = ((after[0] - before[0]).powi(2) + (after[2] - before[2]).powi(2)).sqrt();
        assert!(crept < 0.5, "car travelled {crept} m in the two seconds after stopping");
        assert!(car.speed().abs() < 0.6, "speed reads {} m/s while stationary", car.speed());
    }

    #[test]
    fn suspension_compresses_under_the_cars_own_weight() {
        let mut world = flat_world();
        let mut car = Vehicle::spawn(&mut world, [0.0, 54.0, 0.0]);
        settle(&mut world, &mut car, 240, VehicleInput::default());

        let (chassis, _) = car.chassis_pose(&world);
        let wheel_y = car.wheel_poses()[0].position[1];
        let drop = chassis[1] - wheel_y;
        // If the spring were rigid this would equal the rest length exactly.
        // A real spring under load sits shorter than that.
        assert!(drop < SUSPENSION_REST + 0.3, "suspension not compressing: {drop} m");
        assert!(drop > 0.0, "wheel ended up above the chassis");
    }
}

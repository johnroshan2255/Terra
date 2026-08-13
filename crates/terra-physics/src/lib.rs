//! Rigid-body physics: terrain collision and a driveable vehicle.
//!
//! Follows the plan in `docs/physics.md`. Rapier is the engine; `parry3d` comes
//! with it and is not a separate choice.
//!
//! The world is stepped at a **fixed** rate by the caller (see [`FIXED_DT`]).
//! Everything here assumes that: varying the step changes the simulation
//! result, so a frame hitch would otherwise launch a car off the road and
//! nothing would be reproducible.

pub mod vehicle;

pub use vehicle::{Vehicle, VehicleInput, WheelPose};

use rapier3d::prelude::*;

/// Physics step. Rapier's default, and the rate every tuning value here
/// assumes.
pub const FIXED_DT: f32 = 1.0 / 60.0;

/// The simulation. Owns every Rapier set so the caller holds one thing.
pub struct PhysicsWorld {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    gravity: Vector,
    integration: IntegrationParameters,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
    terrain: Option<ColliderHandle>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let integration = IntegrationParameters { dt: FIXED_DT, ..Default::default() };
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            gravity: Vector::new(0.0, -9.81, 0.0),
            integration,
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            terrain: None,
        }
    }

    /// Build (or replace) the terrain collider from a heightfield.
    ///
    /// `heights` is our row-major `z * res + x` layout in meters, covering
    /// `extent_m` centred on the origin -- the same array the renderer samples,
    /// so no GPU readback is involved.
    ///
    /// The transpose here is load-bearing. Parry stores heightfields
    /// **column-major** (`flat_index(i, j) = i + j * nrows`) and treats rows as
    /// Z and columns as X. Passing our buffer straight through would mirror the
    /// terrain about its diagonal -- which looks almost right, so the way it
    /// shows up is a car climbing a hill that is not there.
    pub fn set_terrain(&mut self, heights: &[f32], res: u32, extent_m: f32) {
        let n = res as usize;
        assert_eq!(heights.len(), n * n, "heightfield is not square");

        if let Some(old) = self.terrain.take() {
            self.colliders.remove(old, &mut self.islands, &mut self.bodies, false);
        }

        // (row, col) = (z, x); `from_fn` places each value at parry's layout.
        let grid = rapier3d::parry::utils::Array2::from_fn(n, n, |z, x| heights[z * n + x]);
        let collider = ColliderBuilder::heightfield(grid, Vector::new(extent_m, 1.0, extent_m))
            .friction(0.9)
            .build();
        self.terrain = Some(self.colliders.insert(collider));
    }

    pub fn step(&mut self) {
        self.pipeline.step(
            self.gravity,
            &self.integration,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    /// Scene-query view for the vehicle's wheel raycasts.
    ///
    /// `exclude` **must** be the vehicle's own chassis. The rays start at the
    /// wheel mounts, which sit inside the bodywork, so without this every wheel
    /// immediately hits its own car, decides the ground is centimetres away,
    /// and the suspension launches the whole thing into the sky.
    pub fn query_excluding(&mut self, exclude: RigidBodyHandle) -> QueryPipelineMut<'_> {
        self.broad_phase.as_query_pipeline_mut(
            self.narrow_phase.query_dispatcher(),
            &mut self.bodies,
            &mut self.colliders,
            QueryFilter::default().exclude_rigid_body(exclude),
        )
    }

    pub fn has_terrain(&self) -> bool {
        self.terrain.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ramp rising along +X, so a transposed heightfield is obviously wrong.
    fn ramp(res: u32, extent: f32) -> Vec<f32> {
        let step = extent / (res - 1) as f32;
        (0..res * res)
            .map(|i| {
                let x = (i % res) as f32 * step - extent * 0.5;
                100.0 + x * 0.5
            })
            .collect()
    }

    #[test]
    fn terrain_is_not_transposed() {
        const RES: u32 = 32;
        const EXTENT: f32 = 128.0;
        let heights = ramp(RES, EXTENT);

        let mut world = PhysicsWorld::new();
        world.set_terrain(&heights, RES, EXTENT);
        assert!(world.has_terrain());

        // Drop a ball on the +X side, where the ramp is high, and another on
        // the -X side, where it is low. If rows and columns were swapped the
        // two would rest at the same height instead.
        let drop_at = |world: &mut PhysicsWorld, x: f32, z: f32| -> (f32, f32) {
            let body = world
                .bodies
                .insert(RigidBodyBuilder::dynamic().translation(Vector::new(x, 260.0, z)).build());
            let ball = ColliderBuilder::ball(1.0).build();
            world.colliders.insert_with_parent(ball, body, &mut world.bodies);
            for _ in 0..400 {
                world.step();
            }
            let t = world.bodies[body].translation();
            (t.y, t.x)
        };

        let (high, high_x) = drop_at(&mut world, 48.0, 0.0);
        let (low, low_x) = drop_at(&mut world, -48.0, 0.0);

        assert!(high > low + 20.0, "ramp not oriented along X: high={high} low={low}");

        // Check against the ramp height where each ball actually ended up, not
        // where it was dropped -- a ball on a 26 degree slope rolls downhill,
        // and asserting the drop position would only be testing that.
        for (y, x) in [(high, high_x), (low, low_x)] {
            let expected = 100.0 + x * 0.5 + 1.0; // ramp + ball radius
            assert!((y - expected).abs() < 2.0, "ball at x={x} rested at {y}, ramp is {expected}");
        }
    }

    #[test]
    fn a_dropped_body_comes_to_rest_on_the_terrain() {
        const RES: u32 = 16;
        const EXTENT: f32 = 64.0;
        let mut world = PhysicsWorld::new();
        world.set_terrain(&vec![50.0; (RES * RES) as usize], RES, EXTENT);

        let body = world
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 90.0, 0.0)).build());
        let ball = ColliderBuilder::ball(0.5).build();
        world.colliders.insert_with_parent(ball, body, &mut world.bodies);

        for _ in 0..600 {
            world.step();
        }
        let y = world.bodies[body].translation().y;
        assert!((y - 50.5).abs() < 1.0, "ball rested at {y}, expected ~50.5");
    }

    #[test]
    fn replacing_terrain_does_not_leak_colliders() {
        let mut world = PhysicsWorld::new();
        let flat = vec![10.0; 16 * 16];
        world.set_terrain(&flat, 16, 64.0);
        world.set_terrain(&flat, 16, 64.0);
        world.set_terrain(&flat, 16, 64.0);
        assert_eq!(world.colliders.len(), 1, "each rebuild must replace, not accumulate");
    }
}

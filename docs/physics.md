# Physics and world effects plan

Companion to [culling.md](culling.md). Same rule: each piece lands with the
system that needs it, not before.

## Stack

| Concern | Choice | Version |
|---|---|---|
| Rigid bodies, vehicles | `rapier3d` | 0.35 |
| Collision shapes | `parry3d` (via Rapier) | 0.30 |
| Math | `nalgebra` (via Rapier) | 0.35 |
| Rain, debris | WebGPU compute particles | — |
| Grass, foliage | Instanced draw + vertex wind | — |

**Parry is not a separate choice.** `rapier3d` depends on `parry3d`, and
`ColliderBuilder::cylinder(..)` already builds a Parry shape. Depend on Rapier;
reach for Parry directly only for standalone queries with no rigid body
involved.

Rapier is f32 by default. At our largest world the camera sits at most 8 km from
the origin, where f32 still resolves well under a millimetre — precision is not
a concern at this scale, and the f64 feature is not needed.

Gravity and distances are already SI in this codebase (`gravity: 9.81` in
`ErosionParams`, everything in meters), so Rapier's defaults line up with no
unit conversion layer.

## Terrain collision

`ColliderBuilder::heightfield` built from the heightmap — correct, and cheaper
here than it usually is: **`Terrain::heights` is already the authoritative CPU
copy**, so the collider is built from memory we hold, with no GPU readback. That
was the reason sculpting was kept CPU-side with partial uploads rather than as a
compute pass.

Two things that will bite:

**One collider for the whole world does not work.** A 16 km world is 4096²,
i.e. 16.7 M cells and ~33 M collision triangles in a single shape. Build one
heightfield collider **per tile** and stream them in a radius around the player,
matching the tile layout already in `terra-core::coords`. A 1024 m tile at
2 m/texel is 512², which is a reasonable collider.

**Sculpting invalidates colliders.** `Terrain::sculpt` already computes the
texel rectangle it touched; that rectangle maps to a set of tiles which must be
marked dirty and rebuilt. Rebuild on stroke end, not per dab — a held brush
would otherwise rebuild the same collider 75 times a second.

Regenerating terrain invalidates every collider at once, which is fine because
generation is already a blocking operation.

## Vehicles

Rapier ships `DynamicRayCastVehicleController`: wheels are raycasts against the
world rather than rolling colliders. That is the standard approach and the right
one — real cylinder wheels on a heightfield catch on texel edges and need far
more tuning to feel stable.

Suspension, friction slip and engine/brake forces are parameters on that
controller. Expect the feel to come from tuning those, not from the physics
engine choice.

## Trees and obstacles

Primitive colliders (cylinder for a trunk, box for a rock) are right. The
constraint is count: scatter will place 10⁵–10⁶ instances, and a collider per
instance is not viable.

Instantiate colliders only within a radius of the player, from the same instance
data the renderer scatters from, and despawn them behind. The scatter is
generated deterministically from a seed plus the erosion masks, so physics and
rendering can derive the identical set without storing per-instance transforms.

## Fixed timestep — a real change to the app loop

The current loop is variable-dt: `update(dt)` is called with whatever the frame
took. Physics cannot run that way. Varying the step changes the simulation
result, so a frame hitch becomes a car launching off the road, and nothing is
reproducible.

Physics needs a fixed step (Rapier defaults to 1/60 s) accumulated across
frames, with render transforms interpolated between the last two physics states.
That is a change to `App::update`, and it should land with the runtime rather
than being retrofitted after vehicles exist.

## Rain

A compute particle system will work, but for rain specifically it is usually the
wrong tool.

The cheap standard approach is a **camera-locked volume** — a box that follows
the camera, with scrolling animated texture layers or a small set of instanced
streaks, plus a screen-space splash pass using the depth buffer. It costs a
fraction of 100 k simulated particles and is hard to tell apart, because rain
has no interesting trajectory: it falls, and every drop looks the same.

Reserve GPU compute particles for effects where the trajectory *is* the effect —
sparks, debris, dust kicked up by wheels. Those genuinely need per-particle
state, collision and varied lifetimes.

If storms need to affect the world (puddles forming, surfaces darkening), that
is a material-layer concern driven by a wetness mask, not a particle count.

## Grass and foliage

Instanced draw with wind animation in the vertex shader — correct, and this is
where the erosion work pays off a second time.

Density comes from the masks the solver already produces: grass where sediment
was deposited and slope is gentle, sparse or absent on scoured rock and in
riverbeds. That is `global_sediment.r16` and `global_flow.r16`, already written
to disk. No hand-painted density map is needed for a plausible first pass.

Wind is a scrolling noise field sampled per instance and per vertex height, so
blade tips move more than bases. Reuse `warp_basis` from
`assets/shaders/common/noise.wgsl` rather than adding another noise function.

Distance handling matters more than the wind: full blades near the camera,
cross-quad billboards at mid range, and folded into the terrain material beyond
that. This is the same instance buffer the Phase B culling in
[culling.md](culling.md) operates on, so the two should be built together.

## Suggested order

1. **Fixed timestep** in the app loop — everything else depends on it.
2. **Heightfield colliders**, per tile, with dirty tracking from sculpt.
3. **Vehicle controller** — the first thing that makes the world feel like a
   place rather than a picture.
4. **Scatter + grass**, sharing an instance buffer with culling.
5. **Prop colliders**, streamed from that same scatter.
6. **Weather** last: it is the cheapest to add and the easiest to over-invest in.

Physics belongs in a new `terra-physics` crate sitting beside `terra-render`,
depending on `terra-core` for tile addressing. Not created yet — it should
arrive with step 1 rather than as an empty placeholder.

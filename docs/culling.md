# Visibility and culling plan

Target: 5 ms/frame at 1080p (see the frame budget in `README.md`).

Status at time of writing: **2 draw calls per frame**, ~524k triangles, 1.39 ms
GPU. Nothing is instanced. There is nothing to cull yet — every technique below
is staged behind the feature that creates the work it removes.

**Update, with scatter landed:** Phase B's GPU instance culling is implemented
(`scatter.rs`, `assets/shaders/render/scatter_cull.wgsl`). Measured on an M4 at
1600x900 with four species covering a 4 km world, ~140k instances: 3.10 ms GPU,
1.73 ms CPU. Phase A (CDLOD) is still the larger outstanding win and remains
unbuilt.

**Update, with grass landed:** Hi-Z occlusion is implemented (`hiz.rs`,
`assets/shaders/render/hiz.wgsl`) and used by the grass placement pass. It
carries one lesson worth recording. Applied literally -- cull anything behind
the farthest surface in its footprint -- it culled about seventy per cent of
the near blades and visibly thinned the field, because *a field of grass is not
an occluder*. The depth buffer records only the frontmost blade, and at the
coarse pyramid levels the test reads, the gaps between blades are gone: grass
culls grass. The test now compares in metres and demands real slack, so a
landform still culls what is behind it while blades no longer cull each other.
Measured at 620 blades/m2, 34 m: 4.81 ms with, 5.18 ms without.

Building culling before the geometry exists costs complexity and measures as
zero. Each phase below lands *with* the system it serves.

## Already done

- **Back-face culling** — `cull_mode: Some(Face::Back)` on the terrain
  pipeline. Free, fixed-function.

## The thing missing from most culling lists

**CDLOD is not culling, and it matters more than all of it.**

The terrain is one uniform 512² grid covering the whole world. At 2 km that's
fine. At 16 km it means ~4 cm of screen detail near the camera and a triangle
per 30 m at the horizon — simultaneously too much work and too little detail.
No amount of culling fixes that, because none of those triangles are *hidden*;
they're just badly distributed.

Continuous LOD is the first thing to build. Everything below assumes the
quadtree it introduces.

## Phase A — with CDLOD

| Technique | Why here |
|---|---|
| **Quadtree bounding-volume culling** | Falls out of CDLOD for free. Each node already carries a min/max height, so its AABB is known without extra work. |
| **Frustum culling** | Test node AABBs during quadtree descent. Reject a node and its whole subtree goes with it. |
| **Horizon culling** | Underrated, and specific to heightfields. A ray from the camera along the ground either clears the ridgeline in front or it doesn't; anything below that horizon angle is hidden. Cheap, and in mountain terrain it removes a large fraction of the map. |
| **Distance culling** | Trivial once nodes are being walked. Mostly subsumed by LOD selection — a node past max range simply never gets a mesh. |

Expected: this is what makes 8–16 km worlds viable at all.

## Phase B — with scatter (grass, rocks, trees)

This is where the real win lives, because this is where the object count goes
from 1 to 10⁵–10⁶.

- **GPU instance culling.** *Done.* A compute pass tests every instance against
  the frustum and a distance threshold, compacts the survivors into an instance
  buffer, and writes a draw count. One `draw_indexed_indirect` per prop type.
  The CPU never sees an instance. Per-species draw distance is exposed in the
  foliage tool, because it is the dial that decides how much there is to cull.
- **Hi-Z occlusion culling.** *Done, for grass.* A depth mip pyramid built from
  last frame's depth, each level the minimum reversed-Z of the block below it --
  the farthest surface drawn there. Each blade's bounding sphere tests against
  the level whose texels cover its screen footprint. In mountain terrain a ridge
  hides an enormous amount, and this is the technique that finds it. Scatter's
  props do not use it yet; the same pyramid is already there for them.

Both are compute + indirect draw, which wgpu supports on every backend.

## Phase C — with shadows

- **Light frustum culling** — cull per cascade against the light's frustum, not
  the camera's. An object outside the camera can still cast into it.
- Cascade fitting is the bigger win here and is not culling: sizing each cascade
  to the visible depth range beats culling more objects out of a badly-fit one.

## Phase D — only if many lights

- **Tile / clustered light culling.** Currently there is one directional sun,
  so this would optimize a loop that runs once. Revisit past ~20 dynamic lights.

## Deliberately not planned

These are on the general list but wrong for *this* renderer. Recorded so the
decision isn't relitigated.

**Portal / PVS culling** — designed for rooms connected by doorways, where
visibility is precomputable and highly restrictive. This is open-world terrain
with no cells and no portals. Nothing to precompute.

**Hardware occlusion queries** — asks the GPU whether a bounding box drew any
pixels, then reads the answer back. That read is a CPU↔GPU round trip, so the
result is either a frame stale or it stalls the pipeline. Superseded by Hi-Z,
which answers the same question entirely on the GPU with no sync. WebGPU does
expose occlusion queries, and we still should not use them.

**Meshlet / cluster culling and normal-cone culling** — needs mesh and task
shaders. wgpu 30 has `EXPERIMENTAL_MESH_SHADER`, but it is native-only and
experimental, which forfeits the web target that motivated choosing WebGPU. It
can be emulated with compute plus indirect draws, but it only pays off on dense
static meshes, and terrain LOD plus instance culling already covers our
geometry. Revisit if authored props ever dominate the frame.

**Sub-pixel triangle culling** — only meaningful inside a meshlet pipeline. LOD
selection is the correct place to prevent sub-pixel triangles: don't generate
them rather than reject them after the fact.

## How to decide a phase is done

Turn the perf overlay on (Settings → Performance) and read GPU milliseconds
against the 5 ms budget, with V-Sync off so frame time reflects work rather
than the 75 Hz refresh. A culling technique that does not move that number is
not paying for its complexity, regardless of how many objects it reports
rejecting.

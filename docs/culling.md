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

| Technique | State |
|---|---|
| **Quadtree bounding-volume culling** | *Done.* `Cdlod::descend` tests each node's AABB and drops the node **and its whole subtree** — sound because a child's box is contained in its parent's. |
| **Frustum culling** | *Done.* The same test. Y comes from the whole heightfield's range rather than per-node, which is conservative; tightening it wants a min/max height pyramid. |
| **Horizon culling** | **Not done.** Wants the same per-node height pyramid, so it is one change with the point above rather than two. |
| **Distance culling** | *Subsumed.* A node past the finest band simply gets a coarser level; there is no maximum range, because the terrain is the world. |

Measured share of terrain patches culled, camera at 400 m:

| World | level | looking down | looking up |
|---|---|---|---|
| 2 km | **68%** | 27% | 34% |
| 4 km | **69%** | 33% | 39% |
| 8 km | **69%** | 38% | 43% |
| 16 km | **69%** | 43% | 46% |

Level is the common case and roughly two thirds of the quadtree goes. That is not a
clever result — on a world centred on the camera, half of it is simply behind you.

The water surface reuses the same selection, so it is culled by the same test.

**The shadow passes get their own patch set.** Culling the terrain against the camera
alone would drop a ridge behind the camera that casts *into* the view, so
`Cdlod::select_culled` produces two sets: one culled against the camera, one against
the union of the shadow cascades. Both are descended from the same eye, so a patch in
both is identical in both — which matters, because a caster morphed from a different
eye than the surface it shades puts a band of acne along every level boundary. This is
Phase C's light frustum culling, arriving early because it was the correctness
condition for doing Phase A at all.

## Phase B — with scatter (grass, rocks, trees)

This is where the real win lives, because this is where the object count goes
from 1 to 10⁵–10⁶.

- **GPU instance culling.** *Done.* A compute pass tests every instance against
  the frustum and a distance threshold, compacts the survivors into an instance
  buffer, and writes a draw count. One `draw_indexed_indirect` per prop type.
  The CPU never sees an instance. Per-species draw distance is exposed in the
  foliage tool, because it is the dial that decides how much there is to cull.
- **Mesh LOD for scatter.** *Done.* Three levels per species, built at import by
  the same vertex-cluster decimator, at roughly 6000 / 1500 / 400 triangles. The
  cull pass bins each survivor by horizontal distance into one of three output
  buffers and writes three sets of indirect draw arguments, so LOD selection
  costs no extra pass over the instances -- it is the same single dispatch, with
  three counters instead of one. Switch distances are per species and runtime.

  Measured on a 12 m grid inside a 900 m draw distance, 17,665 instances drawn:

  | switch distances | LOD 0 / 1 / 2 | triangles | vs one level |
  |---|---|---|---|
  | 60 / 200 m | 81 / 796 / 16788 | 8.4 M | **-92%** |
  | 120 / 350 m (default) | 317 / 2376 / 14972 | 11.5 M | **-89%** |
  | 300 / 600 m | 1961 / 5884 / 9820 | 24.5 M | -77% |

  The instance *count* is identical in all three rows: banding changes detail,
  never visibility. Most of the saving is geometric rather than clever -- instance
  count grows with the square of the radius, so the far field is most of the
  scatter and it is the cheapest level that draws it.

  This was affordable only after the instance record shrank from 80 bytes to 32
  (`mesh::Instance`): a 4x4 matrix replaced by a quaternion, an f16 scale and a
  position, since every instance here is a rigid transform with uniform scale.
  Three 32-byte output buffers plus a 32-byte source is *less* memory than the
  one 80-byte source and one 80-byte output they replaced.

  A dithered cross-fade between levels is not implemented. It would need an
  instance inside a transition band emitted into **both** adjacent buffers, so
  peak occupancy would exceed the instance count and the buffers would have to
  grow by the width of the widest band; they are not sized for that.
- **Hi-Z occlusion culling.** *Done, for scatter.* A depth mip pyramid built from
  last frame's depth, each level the minimum reversed-Z of the block below it --
  the farthest surface drawn there. Each instance's bounding sphere tests against
  the level whose texels cover its screen footprint. In mountain terrain a ridge
  hides an enormous amount, and this is the technique that finds it.

  It carries one lesson worth recording. Applied literally -- cull anything behind
  the farthest surface in its footprint -- it culled about seventy per cent of near
  grass and visibly thinned the field, because *a field of grass is not an
  occluder*. The depth buffer records only the frontmost blade, and at the coarse
  pyramid levels the test reads, the gaps between blades are gone: grass culls
  grass. The test therefore compares in **metres** with two metres of slack, which
  lets a landform occlude what is behind it while foliage no longer occludes
  itself. A depth epsilon would have been meaningless across a reversed-Z range
  spanning kilometres.

  Read with `textureLoad` rather than a sampler: `R32Float` is not filterable, and
  the test wants the exact texel covering a footprint rather than an interpolation.
  Tested against the *previous* frame's view-projection, because that is the matrix
  the depths correspond to -- using this frame's makes it flicker. Occlusion stays
  off on the first frame, when there is no pyramid to correspond to anything.

  This pyramid was being built every frame and read by nothing for some time: its
  only consumer had been a dense-grass pass that was later removed. It is wired to
  the scatter cull now.

Both are compute + indirect draw, which wgpu supports on every backend.

## Phase C — with shadows

- **Light frustum culling** — *done for the terrain*, against the union of the
  cascades rather than per cascade: a caster only has to be in *some* cascade to
  matter, and the union needs one visible set instead of three. **Not done for
  scatter**, which still casts from the camera-culled set, so foliage entering from
  an off-screen edge casts no shadow until it is on screen.
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

//! CDLOD -- Continuous Distance-Dependent Level of Detail.
//!
//! A quadtree over the world picks a level per patch; every patch draws the same
//! unit grid mesh, instanced, and morphs its vertices toward the parent level by
//! camera distance. That morph is the "continuous" part and it is what removes
//! both the cracks between levels and the pop when one changes -- no stitch
//! meshes, no skirts, no per-patch vertex buffers.
//!
//! # Why this is the prerequisite for displacement
//!
//! The uniform 512-square grid this replaces put 7.8 m between vertices on a 4 km
//! world, while a material tiles every 3.5 m -- two full texture repeats inside a
//! single quad. Displacing geometry by a material height map at that spacing does
//! not make bumps; it makes the whole 8 m quad lurch by one arbitrary texel.
//!
//! A uniform grid fine enough for material-scale detail is about 84 M vertices,
//! which is not viable. Adaptive density is: patches near the camera reach
//! sub-metre spacing while the far field stays coarse, and the total stays in the
//! low hundreds of thousands.
//!
//! # The selection rule
//!
//! Level `L` covers camera distances from `range(L-1)` to `range(L)`, where the
//! ranges double each level. Descending is therefore "if any part of this patch is
//! inside the child level's range, split it" -- which is a distance test against
//! the patch's nearest point, not its centre. Using the centre makes a large patch
//! straddling the boundary pick a level that is wrong for half of itself.

use crate::frustum::{Frustum, FrustumUnion};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// Quads along one edge of the grid every patch draws.
///
/// 32 rather than a larger patch: the whole point is adaptivity, and a coarse
/// patch grid means the quadtree has to descend further to reach a given vertex
/// spacing, which costs draw instances instead of vertices. 32 keeps the index
/// buffer at 6k indices and the instance count in the low hundreds.
pub const PATCH_QUADS: u32 = 32;

/// Vertices along one edge of a patch.
pub const PATCH_VERTS: u32 = PATCH_QUADS + 1;

/// Where the morph toward the parent level begins, as a fraction of the level's
/// range.
///
/// Not 0: morphing across the whole range means a patch is always partly morphed,
/// which throws away most of the detail the level was selected for. Not 1 either
/// -- the morph has to finish before the level boundary or the seam reappears.
const MORPH_START: f32 = 0.62;

/// A level's outer range, in multiples of that level's patch size.
///
/// This is not a free parameter. The morph reconciles a patch with its *immediate*
/// parent only, so a two-level jump across a shared edge is a crack the morph
/// cannot close, and the selection rule has to make that impossible.
///
/// It does, above a threshold. Take a patch `A` at level `L`, side `s`, that was
/// not split -- so `dist(A) > range(L-1)`. For a level `L-2` patch to touch it,
/// some level `L-1` patch `P` of side `s/2` adjacent to `A` must have been split,
/// so `dist(P) <= range(L-2)`. `P` shares a full side with `A`, and no point of `P`
/// is further than `s/2` from that side, so `dist(A) <= dist(P) + s/2`. Chaining:
///
/// ```text
/// range(L-1) < dist(A) <= range(L-2) + s/2 = range(L-1)/2 + s/2
///   =>  range(L-1) < s   =>  RANGE_IN_PATCHES / 2 < 1
/// ```
///
/// So a two-level jump requires `RANGE_IN_PATCHES < 2`, and anything at or above 2
/// forbids it outright. The same chain run one level deeper shows three-level jumps
/// need a factor below 0.67, so the two-level case is the binding one.
///
/// 2.4 rather than exactly 2 for margin against float error. Larger is not free --
/// it pushes each level's band outward, so more of the world is drawn at fine
/// levels and the patch count rises -- which is why this sits just above the bound
/// rather than comfortably clear of it.
const RANGE_IN_PATCHES: f32 = 2.4;

/// Ceiling on tree depth.
///
/// A backstop against an absurd target spacing, not a quality knob: the patch count
/// grows roughly linearly in depth, so this is generous. 14 levels puts the finest
/// patch 8192x smaller than the root.
const MAX_LEVELS: u32 = 14;

/// One patch, as the vertex shader reads it.
///
/// A storage buffer indexed by `instance_index` rather than a vertex buffer: the
/// grid has no per-vertex attributes at all -- position comes from the vertex
/// index and height from the heightfield -- so introducing a vertex buffer just to
/// carry per-instance data would be the only one in the pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq)]
pub struct Patch {
    /// World XZ of the patch's minimum corner.
    pub origin: [f32; 2],
    /// World size of one patch edge, in metres.
    pub size: f32,
    /// Quadtree level. 0 is finest.
    pub level: u32,
    /// Camera distance at which the morph toward the parent begins.
    pub morph_start: f32,
    /// Camera distance at which it completes.
    pub morph_end: f32,
    _pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<Patch>() == 32);
// Two `f32` pairs and four scalars. No `vec3` anywhere, deliberately -- see the
// note on `material::LayerParams` for what a misaligned one costs.
const _: () = assert!(std::mem::offset_of!(Patch, morph_start) == 16);

impl Patch {
    /// A patch with the padding zeroed.
    ///
    /// The padding stays private -- it exists only to make the Rust and WGSL layouts
    /// agree, and letting callers set it would invite someone to store something
    /// there that the shader does not read.
    pub fn new(origin: Vec2, size: f32, level: u32, morph_start: f32, morph_end: f32) -> Self {
        Self { origin: origin.to_array(), size, level, morph_start, morph_end, _pad: [0.0; 2] }
    }

    /// World distance between adjacent vertices of this patch.
    pub fn step(&self) -> f32 {
        self.size / PATCH_QUADS as f32
    }

    /// World XZ of a grid vertex before the morph.
    pub fn grid_xz(&self, gx: u32, gz: u32) -> Vec2 {
        Vec2::from(self.origin) + Vec2::new(gx as f32, gz as f32) * self.step()
    }

    /// How far this vertex has morphed toward the parent level, 0 to 1.
    ///
    /// Taken from the *unmorphed* position deliberately. Two patches of the same
    /// level that share an edge must compute the same factor for the vertices they
    /// share, or the edge splits; the unmorphed position is the same in both, the
    /// morphed one is not until the factors already agree.
    pub fn morph_factor(&self, unmorphed_xz: Vec2, eye_xz: Vec2, gap: f32) -> f32 {
        let d = (unmorphed_xz - eye_xz).length_squared() + gap * gap;
        ((d.sqrt() - self.morph_start) / (self.morph_end - self.morph_start).max(1e-3))
            .clamp(0.0, 1.0)
    }

    /// World XZ of a grid vertex after the morph toward the parent level.
    ///
    /// Odd grid indices slide back onto the even ones, which are exactly the
    /// parent level's vertices: a patch is one quadrant of its parent and has the
    /// same vertex count, so its even indices land on parent vertices. At `k == 1`
    /// the patch is geometrically identical to the parent's quadrant, which is what
    /// makes the level change invisible.
    ///
    /// `cdlod_vertex_xz` in `assets/shaders/common/cdlod.wgsl` is the shader's copy
    /// of this, and `tests/cdlod_gpu.rs` requires the two to agree bit for bit.
    pub fn vertex_xz(&self, gx: u32, gz: u32, eye_xz: Vec2, gap: f32) -> Vec2 {
        let g = Vec2::new(gx as f32, gz as f32);
        let base = Vec2::from(self.origin) + g * self.step();
        let k = self.morph_factor(base, eye_xz, gap);
        // fract(g/2)*2 is 1 on odd indices and 0 on even ones.
        let odd = (g * 0.5).fract() * 2.0;
        Vec2::from(self.origin) + (g - odd * k) * self.step()
    }
}

/// Vertical distance from the eye to the slab that contains all terrain.
///
/// Selection has to account for camera altitude or a camera 2 km up picks the
/// finest level for the ground directly beneath it -- half-metre triangles that
/// cover a pixel between them. Rather than track a height range per patch, the
/// whole terrain is treated as one slab and the eye's distance to it is folded
/// into every patch's distance.
///
/// Zero when the eye is inside the slab, which is the common case for an editing
/// camera and makes this reduce to the plain XZ metric.
pub fn vertical_gap(eye_y: f32, height_range: (f32, f32)) -> f32 {
    (eye_y - eye_y.clamp(height_range.0, height_range.1)).abs()
}

/// A node during selection.
#[derive(Debug, Clone, Copy)]
struct Node {
    origin: Vec2,
    size: f32,
    level: u32,
}

impl Node {
    /// Distance used to pick this node's level.
    ///
    /// Horizontal distance to the patch's *nearest point*, not its centre: a 2 km
    /// patch straddling a range boundary would otherwise pick a level that is
    /// wrong for half of itself. `gap` folds in camera altitude.
    ///
    /// This is a lower bound on the true 3D distance to the patch's surface -- the
    /// surface lies inside the slab, so it cannot be nearer than this -- which
    /// makes the metric conservative: it can select finer than strictly needed,
    /// never coarser.
    ///
    /// It also leaves the neighbour-level proof in [`RANGE_IN_PATCHES`] intact.
    /// That proof needs `dist(A) <= dist(P) + s/2` given `dxz(A) <= dxz(P) + s/2`,
    /// and `sqrt((a+b)^2 + c^2) <= sqrt(a^2 + c^2) + b` holds for every `b >= 0`,
    /// so adding a shared vertical term to both sides cannot break it.
    fn distance_to(&self, p: Vec2, gap: f32) -> f32 {
        let max = self.origin + Vec2::splat(self.size);
        let dxz = (self.origin - p).max(p - max).max(Vec2::ZERO).length();
        (dxz * dxz + gap * gap).sqrt()
    }
}

/// Quadtree selection and the patch list it produces.
pub struct Cdlod {
    /// Levels in the tree. Level `levels - 1` is the root patch.
    levels: u32,
    /// Camera distance the finest level reaches to, in metres.
    near_range: f32,
    /// Selected patches for the colour pass, rebuilt each frame.
    patches: Vec<Patch>,
    /// Selected patches for the shadow passes.
    ///
    /// A separate set, because the two are culled against different volumes: the colour
    /// pass wants what the camera can see, the shadow passes want what the *light* can
    /// see -- a ridge behind the camera still casts into the view. Both are descended
    /// with the same camera position, so the morph is identical and the overlapping
    /// patches are bit-for-bit the same. That matters: a caster morphed from a different
    /// eye than the surface it shades puts a band of acne along every level boundary.
    shadow_patches: Vec<Patch>,
    /// Highest patch count seen, so the buffer is grown rather than reallocated
    /// every frame.
    capacity: usize,
}

impl Cdlod {
    /// Levels are chosen so the finest patch resolves close to `target_spacing_m`.
    ///
    /// Derived rather than configured: the point of the finest level is that its
    /// vertices are close enough together to carry material-scale displacement,
    /// and that is a property of the world size and the patch grid, not a taste.
    pub fn new(world_extent_m: f32, target_spacing_m: f32) -> Self {
        // Root patch spans the world; each level halves the patch size, so the
        // spacing at level L is extent / (PATCH_QUADS * 2^(levels-1-L)).
        let root_spacing = world_extent_m / PATCH_QUADS as f32;
        let ratio = (root_spacing / target_spacing_m.max(0.01)).max(1.0);
        // +1 because level 0 exists.
        let levels = (ratio.log2().ceil() as u32 + 1).clamp(1, MAX_LEVELS);

        // Every level's band is the same multiple of its own patch size, which is
        // what keeps the neighbour constraint uniform across the tree.
        let finest_patch = world_extent_m / 2f32.powi(levels as i32 - 1);
        Self {
            levels,
            near_range: finest_patch * RANGE_IN_PATCHES,
            patches: Vec::new(),
            shadow_patches: Vec::new(),
            capacity: 0,
        }
    }

    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Vertex spacing at the finest level, in metres. What displacement can
    /// resolve.
    pub fn finest_spacing(&self, world_extent_m: f32) -> f32 {
        world_extent_m / 2f32.powi(self.levels as i32 - 1) / PATCH_QUADS as f32
    }

    /// Outer distance of a level's band, in metres.
    fn range(&self, level: u32) -> f32 {
        self.near_range * 2f32.powi(level as i32)
    }

    /// Largest angular size one quad can reach on screen, in radians.
    ///
    /// CDLOD's defining property, and the reason a single pair of constants tunes
    /// the whole tree: because every level's band is the same multiple of its own
    /// patch size, this is a constant, independent of level, of world size, and of
    /// camera position. A quad looks largest at the *inner* edge of its level's
    /// band, where that level has just taken over -- distance `range(L)/2`, spacing
    /// `size(L)/PATCH_QUADS`.
    ///
    /// The uniform grid this replaces had no such bound: the same 7.81 m quad
    /// subtended 1.3 px at 4 km and a third of the screen at 20 m.
    pub fn worst_quad_angle_rad(&self) -> f32 {
        2.0 / (RANGE_IN_PATCHES * PATCH_QUADS as f32)
    }

    pub fn patches(&self) -> &[Patch] {
        &self.patches
    }

    /// Triangles the current selection draws.
    pub fn triangle_count(&self) -> u32 {
        self.patches.len() as u32 * PATCH_QUADS * PATCH_QUADS * 2
    }

    /// Vertices the current selection draws. Shaded per patch, so patches that
    /// overlap on screen pay twice -- this is the submitted count, not the visible
    /// one.
    pub fn vertex_count(&self) -> u32 {
        self.patches.len() as u32 * PATCH_VERTS * PATCH_VERTS
    }

    /// Rebuild the patch list for a camera.
    ///
    /// `height_range` is the terrain's own min/max, which only matters when the
    /// camera is above or below all of it -- see [`vertical_gap`].
    /// Select patches with no visibility culling. Kept for callers -- and tests -- that
    /// want the whole quadtree.
    pub fn select(&mut self, eye: Vec3, height_range: (f32, f32), world_extent_m: f32) -> &[Patch] {
        self.select_culled(eye, height_range, world_extent_m, None, &FrustumUnion::default());
        &self.patches
    }

    /// Select both patch sets: the colour pass's, culled against `camera`, and the shadow
    /// passes', culled against `lights`.
    ///
    /// Passing `None` for the camera frustum culls nothing, which is the safe reading of
    /// "the caller has not said where it is looking".
    ///
    /// Both sets are descended from the same eye, so a patch appearing in both is
    /// identical in both -- the morph is a function of the eye alone. Culling only ever
    /// *removes* patches, and CDLOD's crack-freeness comes from the morph rather than from
    /// a patch's neighbours being present, so a culled set has no seams.
    pub fn select_culled(
        &mut self,
        eye: Vec3,
        height_range: (f32, f32),
        world_extent_m: f32,
        camera: Option<&Frustum>,
        lights: &FrustumUnion,
    ) {
        self.patches.clear();
        self.shadow_patches.clear();
        let half = world_extent_m * 0.5;
        let root =
            Node { origin: Vec2::splat(-half), size: world_extent_m, level: self.levels - 1 };
        let gap = vertical_gap(eye.y, height_range);
        self.descend(root, Vec2::new(eye.x, eye.z), gap, height_range, camera, lights);
        self.capacity = self.capacity.max(self.patches.len().max(self.shadow_patches.len()));
        debug_assert!(
            self.patches.len() <= self.max_patches()
                && self.shadow_patches.len() <= self.max_patches(),
            "selected {}/{} patches, past the {} the instance buffer is sized for",
            self.patches.len(),
            self.shadow_patches.len(),
            self.max_patches()
        );
        // Release builds clamp rather than overrun the buffer: losing the tail of a
        // selection is a missing patch, overrunning it is a validation failure that
        // kills the frame.
        self.patches.truncate(self.max_patches());
        self.shadow_patches.truncate(self.max_patches());
    }

    /// The world-space box a node occupies.
    ///
    /// Y comes from the whole heightfield's range rather than from the node's own, which
    /// is conservative: a node is kept whenever *any* ground in the world reaches into
    /// view above it. Tightening this needs a per-node min/max height pyramid, which is
    /// also what horizon culling would want -- so it is one change, not two.
    fn node_aabb(node: &Node, height_range: (f32, f32)) -> (Vec3, Vec3) {
        (
            Vec3::new(node.origin.x, height_range.0, node.origin.y),
            Vec3::new(node.origin.x + node.size, height_range.1, node.origin.y + node.size),
        )
    }

    fn descend(
        &mut self,
        node: Node,
        cam: Vec2,
        gap: f32,
        height_range: (f32, f32),
        camera: Option<&Frustum>,
        lights: &FrustumUnion,
    ) {
        // Quadtree bounding-volume culling: test the node's box once and, if neither the
        // camera nor any light can see it, drop the node *and its whole subtree*. A
        // child's box is contained in its parent's, so this is sound -- and it is what
        // makes the test cost proportional to what is visible rather than to the tree.
        let (min, max) = Self::node_aabb(&node, height_range);
        let in_camera = camera.is_none_or(|f| f.intersects_aabb(min, max));
        let in_light = lights.intersects_aabb(min, max);
        if !in_camera && !in_light {
            return;
        }

        // Finest level, or far enough away that the child level's band does not
        // reach this patch: render it as-is.
        if node.level == 0 || node.distance_to(cam, gap) > self.range(node.level - 1) {
            self.emit(node, in_camera, in_light);
            return;
        }
        let half = node.size * 0.5;
        for (ox, oz) in [(0.0, 0.0), (half, 0.0), (0.0, half), (half, half)] {
            self.descend(
                Node { origin: node.origin + Vec2::new(ox, oz), size: half, level: node.level - 1 },
                cam,
                gap,
                height_range,
                camera,
                lights,
            );
        }
    }

    fn emit(&mut self, node: Node, in_camera: bool, in_light: bool) {
        let outer = self.range(node.level);
        let inner = if node.level == 0 { 0.0 } else { self.range(node.level - 1) };
        // The morph has to finish by the outer edge of the band, where the next
        // level takes over: at that point this patch's vertices must coincide with
        // the parent's or the seam is a crack.
        let start = inner + (outer - inner) * MORPH_START;
        // One patch, pushed into whichever sets can see it. Built once rather than twice
        // so the two sets cannot disagree about a patch they share.
        let patch = Patch {
            origin: node.origin.to_array(),
            size: node.size,
            level: node.level,
            morph_start: start,
            morph_end: outer,
            _pad: [0.0; 2],
        };
        if in_camera {
            self.patches.push(patch);
        }
        if in_light {
            self.shadow_patches.push(patch);
        }
    }

    /// Patches the shadow passes should draw.
    pub fn shadow_patches(&self) -> &[Patch] {
        &self.shadow_patches
    }

    /// Upper bound on how many patches any camera position can select.
    ///
    /// A bound rather than a high-water mark, because the instance buffer is
    /// allocated once and a selection that outgrew it would be a mid-session
    /// overflow -- the worst kind, since it needs the camera to reach one particular
    /// place before it shows up.
    ///
    /// Per level, an emitted patch's parent must have been split, so its parent lies
    /// within `range(L)` of the camera. Level-`L+1` nodes that close lie inside a
    /// square of half-extent `range(L) + size(L+1)`, and since
    /// `range(L) = RANGE_IN_PATCHES * size(L+1) / 2` that is at most
    /// `(RANGE_IN_PATCHES + 2)^2` nodes, each contributing four children. The root
    /// level adds one.
    pub fn max_patches(&self) -> usize {
        let per_level = 4.0 * (RANGE_IN_PATCHES + 2.0).powi(2);
        // `levels` is at least 1, so the subtraction cannot wrap.
        1 + (per_level.ceil() as usize) * (self.levels as usize - 1)
    }

    /// Bytes the instance buffer needs, sized for the worst case.
    pub fn buffer_bytes(&self) -> u64 {
        (self.max_patches() * std::mem::size_of::<Patch>()) as u64
    }

    /// Largest selection actually seen, for reporting.
    pub fn high_water(&self) -> usize {
        self.capacity
    }
}

/// Triangle-list indices for a `quads x quads` grid with `quads + 1` verts a side.
///
/// Counter-clockwise viewed from above, matching `cull_mode: Back`.
///
/// Note the split: `[a, c, b, b, c, d]` makes the shared edge of the two triangles
/// `b`-`c`, the *anti*-diagonal. `grid_wire_indices` has to draw that same one.
pub fn grid_indices(quads: u32) -> Vec<u32> {
    let verts = quads + 1;
    let mut idx = Vec::with_capacity((quads * quads * 6) as usize);
    for z in 0..quads {
        for x in 0..quads {
            let a = z * verts + x;
            let b = a + 1;
            let c = a + verts;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    idx
}

/// Edge indices for the same grid, as a line list.
///
/// The fallback wireframe, used when the adapter has no `POLYGON_MODE_LINE` --
/// which is every Metal adapter, so this is the live path on macOS rather than a
/// contingency.
///
/// # Why a line list rather than barycentric edge detection
///
/// The usual barycentric trick -- give each triangle corner a basis vector and
/// shade a pixel dark when any coordinate is near zero -- cannot work on an indexed
/// mesh. There is no per-corner attribute to interpolate, WGSL has no barycentric
/// builtin and no geometry stage, and `@builtin(vertex_index)` under an index buffer
/// is the *index value*, not which corner of the triangle it is. Getting
/// barycentrics would mean expanding the grid to three unshared vertices per
/// triangle, tripling the vertex work to draw a debug view.
///
/// A line list needs none of that. It is exact rather than a screen-space
/// approximation, needs no optional feature, and the buffer is built once at load.
///
/// Only two of each quad's four sides are emitted -- the shared ones -- plus the far
/// sides at the grid boundary. Drawing all four would submit every interior edge
/// twice, which doubles the vertex work and shows as visibly denser lines where
/// quads meet.
///
/// # The diagonals matter
///
/// Each quad's diagonal is emitted too, because the renderer draws *triangles* and a
/// wireframe that hides that is showing the wrong topology. Unreal's wireframe shows
/// triangle edges, and the first version of this drew only the quad grid -- which
/// looked like graph paper and told you nothing about the actual mesh.
///
/// The diagonal has to be the one `grid_indices` actually splits on: `b`-`c`, the
/// anti-diagonal, not `a`-`d`. Drawing the other one would be a wireframe of a mesh
/// that does not exist.
pub fn grid_wire_indices(quads: u32) -> Vec<u32> {
    let verts = quads + 1;
    let mut idx = Vec::with_capacity((quads * quads * 6 + quads * 4) as usize);
    for z in 0..verts {
        for x in 0..verts {
            let a = z * verts + x;
            if x + 1 < verts {
                idx.extend_from_slice(&[a, a + 1]);
            }
            if z + 1 < verts {
                idx.extend_from_slice(&[a, a + verts]);
            }
            // The split, for quads that have one.
            if x + 1 < verts && z + 1 < verts {
                idx.extend_from_slice(&[a + 1, a + verts]);
            }
        }
    }
    idx
}

/// Indices for one patch's grid, shared by every instance.
pub fn patch_indices() -> Vec<u32> {
    grid_indices(PATCH_QUADS)
}

/// Line-list indices for one patch, for the Wireframe view mode.
pub fn patch_wire_indices() -> Vec<u32> {
    grid_wire_indices(PATCH_QUADS)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTENT: f32 = 4000.0;
    /// A flat world, so the altitude term is whatever the eye's own Y makes it.
    const FLAT: (f32, f32) = (0.0, 0.0);

    fn lod() -> Cdlod {
        // Half a metre: fine enough that a 3.5 m material repeat spans seven
        // vertices, which is what displacement needs.
        Cdlod::new(EXTENT, 0.5)
    }

    #[test]
    fn the_finest_level_resolves_material_scale_detail() {
        // The whole reason this exists. The uniform grid it replaces had 7.8 m
        // between vertices while a material tiles every 3.5 m -- two repeats
        // inside one quad, so displacement could only make the quad lurch.
        let l = lod();
        let spacing = l.finest_spacing(EXTENT);
        assert!(spacing < 1.0, "finest spacing is {spacing} m, too coarse to displace");
        let repeats_per_vertex = 3.5 / spacing;
        assert!(
            repeats_per_vertex > 4.0,
            "a 3.5 m material repeat spans only {repeats_per_vertex} vertices"
        );
    }

    #[test]
    fn the_patch_count_stays_bounded_as_the_world_grows() {
        // Adaptivity, restated: sixteen times the area must not cost sixteen times
        // the patches. If it does, this is a uniform grid with extra steps.
        let count = |extent: f32| {
            let mut l = Cdlod::new(extent, 0.5);
            l.select(Vec3::ZERO, FLAT, extent).len()
        };
        let small = count(4000.0);
        let huge = count(16000.0);
        assert!(small > 0);
        assert!(huge < small * 3, "4 km selected {small} patches and 16 km selected {huge}");
    }

    #[test]
    fn the_vertex_budget_is_a_fraction_of_a_uniform_grid() {
        let mut l = lod();
        let patches = l.select(Vec3::ZERO, FLAT, EXTENT).len();
        let verts = patches * (PATCH_VERTS * PATCH_VERTS) as usize;
        // A uniform grid at the finest spacing would be this many.
        let uniform = (EXTENT / l.finest_spacing(EXTENT)).powi(2) as usize;
        assert!(
            verts < uniform / 100,
            "selected {verts} vertices against {uniform} for a uniform grid"
        );
        // And still a sane absolute number.
        assert!(verts < 2_000_000, "{verts} vertices is too many to draw");
    }

    #[test]
    fn detail_concentrates_at_the_camera() {
        let mut l = lod();
        let eye = Vec3::new(-1800.0, 0.0, -1800.0);
        let cam = Vec2::new(eye.x, eye.z);
        let patches = l.select(eye, FLAT, EXTENT).to_vec();

        let nearest = patches
            .iter()
            .min_by(|a, b| {
                let da = Vec2::from(a.origin).distance(cam);
                let db = Vec2::from(b.origin).distance(cam);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        let farthest = patches
            .iter()
            .max_by(|a, b| {
                let da = Vec2::from(a.origin).distance(cam);
                let db = Vec2::from(b.origin).distance(cam);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        assert_eq!(nearest.level, 0, "the patch under the camera must be finest");
        assert!(farthest.level > nearest.level, "the far field must be coarser");
    }

    #[test]
    fn selection_tiles_the_world_exactly() {
        // The invariant that matters: a quadtree selection is a partition, so the
        // patch areas must sum to the world's area. A shortfall is a hole in the
        // terrain; an excess is overdraw and z-fighting.
        let mut l = lod();
        for cam in [
            Vec3::ZERO,
            Vec3::new(1900.0, 0.0, -1900.0),
            Vec3::new(-500.0, 900.0, 300.0),
            Vec3::new(50_000.0, 0.0, 0.0),
        ] {
            let patches = l.select(cam, FLAT, EXTENT);
            let area: f64 = patches.iter().map(|p| (p.size as f64).powi(2)).sum();
            let expect = (EXTENT as f64).powi(2);
            assert!(
                (area - expect).abs() / expect < 1e-9,
                "camera {cam}: patches cover {area} of {expect}"
            );
        }
    }

    #[test]
    fn patches_never_overlap() {
        let mut l = lod();
        let patches = l.select(Vec3::new(600.0, 0.0, -200.0), FLAT, EXTENT).to_vec();
        for (i, a) in patches.iter().enumerate() {
            for b in &patches[i + 1..] {
                let (amin, amax) =
                    (Vec2::from(a.origin), Vec2::from(a.origin) + Vec2::splat(a.size));
                let (bmin, bmax) =
                    (Vec2::from(b.origin), Vec2::from(b.origin) + Vec2::splat(b.size));
                let overlap = amin.x < bmax.x - 1e-3
                    && bmin.x < amax.x - 1e-3
                    && amin.y < bmax.y - 1e-3
                    && bmin.y < amax.y - 1e-3;
                assert!(!overlap, "patches at {amin} and {bmin} overlap");
            }
        }
    }

    #[test]
    fn neighbouring_patches_differ_by_at_most_one_level() {
        // The morph only reconciles a patch with its immediate parent, so a
        // two-level jump across an edge is a crack the morph cannot close. The
        // distance-based descent guarantees this; the test is what keeps it true.
        let mut l = lod();
        let patches = l.select(Vec3::new(-1200.0, 0.0, 900.0), FLAT, EXTENT).to_vec();
        for a in &patches {
            let (amin, amax) = (Vec2::from(a.origin), Vec2::from(a.origin) + Vec2::splat(a.size));
            for b in &patches {
                let (bmin, bmax) =
                    (Vec2::from(b.origin), Vec2::from(b.origin) + Vec2::splat(b.size));
                // Share an edge: touching in one axis, overlapping in the other.
                let touch_x = (amax.x - bmin.x).abs() < 1e-3 || (bmax.x - amin.x).abs() < 1e-3;
                let touch_z = (amax.y - bmin.y).abs() < 1e-3 || (bmax.y - amin.y).abs() < 1e-3;
                let span_x = amin.x < bmax.x - 1e-3 && bmin.x < amax.x - 1e-3;
                let span_z = amin.y < bmax.y - 1e-3 && bmin.y < amax.y - 1e-3;
                if (touch_x && span_z) || (touch_z && span_x) {
                    let d = a.level.abs_diff(b.level);
                    assert!(d <= 1, "adjacent patches jump {d} levels");
                }
            }
        }
    }

    #[test]
    fn the_morph_band_finishes_before_the_level_changes() {
        // If the morph is not complete by the outer edge of a level's band, the
        // patch does not line up with the parent that takes over there, and the
        // seam is a visible crack.
        let mut l = lod();
        let patches = l.select(Vec3::new(300.0, 0.0, 300.0), FLAT, EXTENT).to_vec();
        for p in &patches {
            assert!(p.morph_end > p.morph_start, "empty morph band on level {}", p.level);
            assert!(
                (p.morph_end - l.range(p.level)).abs() < 1e-3,
                "level {} morph ends at {} but its band ends at {}",
                p.level,
                p.morph_end,
                l.range(p.level)
            );
            // And it must not start before the band does, or a patch arrives
            // already partly morphed and wastes the detail it was selected for.
            let inner = if p.level == 0 { 0.0 } else { l.range(p.level - 1) };
            assert!(p.morph_start >= inner - 1e-3, "level {} morphs before its band", p.level);
        }
    }

    #[test]
    fn a_camera_far_outside_the_world_still_gets_one_patch() {
        // Reachable: the wheel can put the camera 40 km out. Returning nothing
        // would mean an empty screen rather than a distant world.
        let mut l = lod();
        let coarsest = l.levels() - 1;
        let patches = l.select(Vec3::new(500_000.0, 0.0, 500_000.0), FLAT, EXTENT);
        assert!(!patches.is_empty());
        assert!(patches.iter().all(|p| p.level == coarsest), "all should be coarsest");
    }

    #[test]
    fn the_patch_grid_is_a_closed_triangle_mesh() {
        let idx = patch_indices();
        assert_eq!(idx.len(), (PATCH_QUADS * PATCH_QUADS * 6) as usize);
        let verts = PATCH_VERTS * PATCH_VERTS;
        assert!(idx.iter().all(|i| *i < verts), "an index points past the patch grid");
        assert_eq!(grid_indices(4).iter().copied().max().unwrap(), 4 * 5 + 4);
    }

    #[test]
    fn the_line_list_covers_every_grid_edge_once() {
        // A wireframe that draws each interior edge twice doubles the vertex work
        // and shows as visibly denser lines where quads meet, so the builder emits
        // only the two edges each vertex owns.
        let n = 4;
        let idx = grid_wire_indices(n);
        assert!(idx.len().is_multiple_of(2), "a line list needs an even index count");

        // n*(n+1) horizontal sides, as many vertical, plus one diagonal per quad.
        let expected_edges = 2 * n * (n + 1) + n * n;
        assert_eq!(idx.len() / 2, expected_edges as usize, "expected {expected_edges} edges");

        let mut seen = std::collections::HashSet::new();
        for e in idx.chunks_exact(2) {
            let key = (e[0].min(e[1]), e[0].max(e[1]));
            assert!(seen.insert(key), "edge {key:?} emitted twice");
        }
    }

    #[test]
    fn every_wire_index_is_a_real_vertex() {
        // Out-of-range indices are undefined behaviour on some backends and a
        // silent garbage triangle on others.
        for n in [1u32, 6, PATCH_QUADS] {
            let verts = (n + 1) * (n + 1);
            assert!(
                grid_wire_indices(n).iter().all(|i| *i < verts),
                "an index points past the {n}-quad vertex grid"
            );
        }
    }

    #[test]
    fn wire_edges_are_sides_or_the_real_diagonal() {
        // Every edge must be a grid side or the anti-diagonal the triangulation
        // actually splits on. A stray edge is a wireframe of a mesh that is not
        // being drawn.
        let n = 5;
        let verts = n + 1;
        for e in grid_wire_indices(n).chunks_exact(2) {
            let (a, b) = (e[0].min(e[1]), e[0].max(e[1]));
            let step = b - a;
            let same_row = a / verts == b / verts;
            let side = (step == 1 && same_row) || step == verts;
            // The anti-diagonal runs b -> c where b = quad + 1 and c = quad + verts,
            // so it spans verts - 1 and its lower index is never in column zero.
            // That column check is what separates it from a horizontal run of
            // verts - 1 across a row, which would not be a real edge.
            let diagonal = step == verts - 1 && (a % verts) >= 1;
            assert!(side || diagonal, "edge {a}->{b} is neither a side nor the split");
        }
    }

    #[test]
    fn the_patch_wireframe_covers_every_triangle_edge() {
        // The bug this pins: the wireframe drew only the quad grid, so it looked
        // like graph paper and said nothing about the actual mesh. Every triangle
        // edge the renderer submits has to appear in the wireframe.
        for n in [3u32, PATCH_QUADS] {
            let tris = grid_indices(n);
            let wire = grid_wire_indices(n);
            let edges: std::collections::HashSet<(u32, u32)> =
                wire.chunks_exact(2).map(|e| (e[0].min(e[1]), e[0].max(e[1]))).collect();
            for t in tris.chunks_exact(3) {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let key = (a.min(b), a.max(b));
                    assert!(
                        edges.contains(&key),
                        "triangle edge {key:?} missing from the {n}-quad wireframe"
                    );
                }
            }
        }
    }

    #[test]
    fn the_patch_struct_matches_a_gpu_layout() {
        assert_eq!(std::mem::size_of::<Patch>(), 32);
        assert_eq!(std::mem::offset_of!(Patch, origin), 0);
        assert_eq!(std::mem::offset_of!(Patch, size), 8);
        assert_eq!(std::mem::offset_of!(Patch, level), 12);
        assert_eq!(std::mem::offset_of!(Patch, morph_start), 16);
        assert_eq!(std::mem::offset_of!(Patch, morph_end), 20);
    }

    #[test]
    fn levels_scale_with_the_world_not_the_target() {
        // A 16 km world needs two more levels than a 4 km one to reach the same
        // spacing, and the constructor has to work that out rather than be told.
        let small = Cdlod::new(4000.0, 0.5);
        let huge = Cdlod::new(16000.0, 0.5);
        assert!(huge.levels() > small.levels());
        // Both land near the requested spacing.
        assert!(small.finest_spacing(4000.0) <= 0.5 * 1.05);
        assert!(huge.finest_spacing(16000.0) <= 0.5 * 1.05);
    }

    #[test]
    fn the_triangle_count_stays_near_the_grid_it_replaces() {
        // The trade this whole module makes: 16x finer vertices where the camera is,
        // for roughly the same number of triangles. If the count blows up, the trade
        // is not worth making and `RANGE_IN_PATCHES` or `PATCH_QUADS` is wrong.
        //
        // The uniform 512-square grid drew 512 * 512 * 2 triangles.
        const UNIFORM_TRIS: u32 = 512 * 512 * 2;
        let mut l = lod();
        for cam in [Vec3::ZERO, Vec3::new(1900.0, 0.0, 1900.0), Vec3::new(-700.0, 0.0, 250.0)] {
            l.select(cam, FLAT, EXTENT);
            let tris = l.triangle_count();
            assert!(
                tris < UNIFORM_TRIS * 3 / 2,
                "camera {cam}: {tris} triangles against {UNIFORM_TRIS} for the uniform grid"
            );
        }
        // And the spacing it buys near the camera, against the grid's flat 7.81 m.
        assert!(l.finest_spacing(EXTENT) < 7.8125 / 8.0);
    }

    #[test]
    fn the_morph_closes_every_seam_between_levels() {
        // The claim the whole scheme rests on, and the one that shows up as a
        // flickering hairline of sky through the terrain when it is false.
        //
        // Where a fine patch meets a coarse one, the fine patch's edge polyline must
        // be *identical* to the coarse patch's -- same vertices, same spacing, so
        // the two linear interpolations agree everywhere between them.
        //
        // Two things make that true, and this test checks both:
        //
        //  1. Every vertex on such an edge is fully morphed. The coarse patch was
        //     only emitted at a distance past the fine level's `morph_end`, and the
        //     shared edge belongs to the coarse patch, so it is at least that far.
        //  2. Fully morphed, the odd indices sit on the even ones, leaving vertices
        //     spaced one coarse step apart and aligned to the coarse lattice -- so
        //     the fine edge has no vertex the coarse edge lacks.
        //
        // Membership in one *specific* coarse patch is deliberately not the
        // assertion: the morph shifts vertices toward the patch origin, so the last
        // vertex on a seam legitimately lands in the next coarse patch along. The
        // lattice is what matters, not which neighbour owns the point.
        let mut l = lod();
        let eye = Vec3::new(400.0, 30.0, -900.0);
        let range = (0.0, 60.0);
        let patches = l.select(eye, range, EXTENT).to_vec();
        let eye_xz = Vec2::new(eye.x, eye.z);
        let gap = vertical_gap(eye.y, range);
        let world_min = -EXTENT * 0.5;

        let mut checked = 0;
        for fine in &patches {
            let fmin = Vec2::from(fine.origin);
            let fmax = fmin + Vec2::splat(fine.size);
            // The lattice the parent level's vertices sit on.
            let parent_step = fine.step() * 2.0;
            for coarse in patches.iter().filter(|c| c.level == fine.level + 1) {
                let cmin = Vec2::from(coarse.origin);
                let cmax = cmin + Vec2::splat(coarse.size);
                let on_x = (fmax.x - cmin.x).abs() < 1e-3 || (cmax.x - fmin.x).abs() < 1e-3;
                let on_z = (fmax.y - cmin.y).abs() < 1e-3 || (cmax.y - fmin.y).abs() < 1e-3;
                let span_x = fmin.x < cmax.x - 1e-3 && cmin.x < fmax.x - 1e-3;
                let span_z = fmin.y < cmax.y - 1e-3 && cmin.y < fmax.y - 1e-3;
                let vertical_seam = on_x && span_z;
                if !vertical_seam && !(on_z && span_x) {
                    continue;
                }
                // The coarse patch's own spacing must match the lattice we check
                // against, or the two levels were never one step apart to begin with.
                assert!((coarse.step() - parent_step).abs() < 1e-4);

                let edge: Vec<(u32, u32)> = if vertical_seam {
                    let gx = if (fmax.x - cmin.x).abs() < 1e-3 { PATCH_QUADS } else { 0 };
                    (0..=PATCH_QUADS).map(|gz| (gx, gz)).collect()
                } else {
                    let gz = if (fmax.y - cmin.y).abs() < 1e-3 { PATCH_QUADS } else { 0 };
                    (0..=PATCH_QUADS).map(|gx| (gx, gz)).collect()
                };
                let mut distinct = std::collections::HashSet::new();
                for (gx, gz) in edge {
                    let base = fine.grid_xz(gx, gz);
                    let k = fine.morph_factor(base, eye_xz, gap);
                    assert!(
                        k > 0.999,
                        "seam vertex at {base} is only {k} morphed, so it sits off the coarse edge"
                    );
                    let p = fine.vertex_xz(gx, gz, eye_xz, gap);
                    // On the parent lattice, in both axes.
                    for axis in [p.x, p.y] {
                        let n = (axis - world_min) / parent_step;
                        assert!(
                            (n - n.round()).abs() < 1e-3,
                            "morphed seam vertex {p} is {} of a coarse step off the lattice",
                            (n - n.round()).abs()
                        );
                    }
                    distinct.insert(((p.x * 64.0).round() as i64, (p.y * 64.0).round() as i64));
                    checked += 1;
                }
                // Collapsed to the coarse spacing: half as many vertices, plus the
                // shared endpoint. Any more and the fine edge bends where the coarse
                // one does not.
                assert_eq!(
                    distinct.len(),
                    (PATCH_QUADS / 2 + 1) as usize,
                    "the seam kept {} distinct vertices against the coarse edge's {}",
                    distinct.len(),
                    PATCH_QUADS / 2 + 1
                );
            }
        }
        // Guard against the test passing because it found no seams to check.
        assert!(checked > 100, "only {checked} seam vertices examined");
    }

    #[test]
    fn the_morph_is_a_no_op_at_the_near_edge_of_a_band() {
        // The other half of the seam argument: a patch that has just become the
        // selected level must be unmorphed, or it arrives already collapsed toward
        // its parent and the finer level buys nothing.
        let mut l = lod();
        let eye = Vec3::new(0.0, 20.0, 0.0);
        let range = (0.0, 40.0);
        l.select(eye, range, EXTENT);
        let gap = vertical_gap(eye.y, range);
        let p =
            *l.patches().iter().find(|p| p.level == 1).expect("a level-1 patch should be selected");
        // A point just inside this level's band.
        let inner = p.morph_start * 0.999;
        let k = p.morph_factor(Vec2::new(inner, 0.0), Vec2::ZERO, gap);
        assert_eq!(k, 0.0, "a patch at the near edge of its band is already morphed");
    }

    #[test]
    fn only_odd_grid_indices_move_when_morphing() {
        // Even indices are the parent level's own vertices, so moving them would
        // pull the patch away from the surface it is supposed to converge to.
        let p = Patch::new(Vec2::ZERO, 32.0, 0, 0.0, 1.0);
        // morph_start 0 and morph_end 1 means anything past 1 m is fully morphed.
        let far = Vec2::splat(-10_000.0);
        for g in [0u32, 1, 2, 3, 16, 31, 32] {
            let base = p.grid_xz(g, 0);
            let moved = p.vertex_xz(g, 0, far, 0.0);
            if g % 2 == 0 {
                assert_eq!(moved, base, "even index {g} moved");
            } else {
                assert_eq!(moved, p.grid_xz(g - 1, 0), "odd index {g} did not reach {}", g - 1);
            }
        }
    }

    #[test]
    fn a_camera_high_above_does_not_select_the_finest_level_beneath_it() {
        // Without the altitude term, a camera 2 km up gets half-metre triangles on
        // the ground directly below, where they cover a fraction of a pixel.
        let mut l = lod();
        let range = (0.0, 50.0);
        let low = l.select(Vec3::new(0.0, 25.0, 0.0), range, EXTENT).len();
        let mut high = Cdlod::new(EXTENT, 0.5);
        let high_n = high.select(Vec3::new(0.0, 2000.0, 0.0), range, EXTENT).len();
        assert!(
            high_n < low,
            "at 2 km up the selection is {high_n} patches against {low} at ground level"
        );
        // The gap itself: zero inside the terrain slab, real above it.
        assert_eq!(vertical_gap(25.0, range), 0.0);
        assert_eq!(vertical_gap(2000.0, range), 1950.0);
        assert_eq!(vertical_gap(-10.0, range), 10.0);
    }

    #[test]
    fn no_camera_position_outgrows_the_instance_buffer() {
        // The buffer is allocated once from `max_patches`, so a position that
        // selected more would be a mid-session overflow -- and one that only appears
        // when the camera reaches a particular spot.
        let mut l = lod();
        let bound = l.max_patches();
        let mut worst = 0;
        let step = EXTENT / 24.0;
        for i in -14..=14 {
            for j in -14..=14 {
                for y in [-500.0, 0.0, 40.0, 900.0, 9000.0] {
                    let eye = Vec3::new(i as f32 * step, y, j as f32 * step);
                    let n = l.select(eye, (0.0, 80.0), EXTENT).len();
                    worst = worst.max(n);
                    assert!(n <= bound, "{eye} selected {n} patches against a bound of {bound}");
                }
            }
        }
        // The bound must be loose enough to never bite, but not so loose it stops
        // being a statement about the algorithm.
        assert!(worst * 4 > bound, "worst case {worst} against a bound of {bound}");
    }

    #[test]
    fn the_worst_quad_angle_is_the_same_at_every_scale() {
        // The property that makes one constant tune the whole tree. If this ever
        // depends on world size, some level's band stopped being proportional to
        // its patch size and the level transitions will not look uniform.
        let a = Cdlod::new(4000.0, 0.5).worst_quad_angle_rad();
        let b = Cdlod::new(16000.0, 0.5).worst_quad_angle_rad();
        let c = Cdlod::new(4000.0, 4.0).worst_quad_angle_rad();
        assert!((a - b).abs() < 1e-9 && (a - c).abs() < 1e-9);

        // And it has to be small enough to look like geometry rather than facets.
        // At 720p over a ~60 degree vertical field, one radian is about 690 px.
        let px = a * 690.0;
        assert!(px < 20.0, "a quad can reach {px} px, which reads as faceting");
    }
}

#[cfg(test)]
mod culling_tests {
    use super::*;
    use crate::camera::Camera;

    const EXTENT: f32 = 4096.0;
    const RANGE: (f32, f32) = (0.0, 900.0);

    fn camera_at(pos: Vec3, yaw: f32) -> Camera {
        Camera { pos, yaw, pitch: 0.0, ..Camera::default() }
    }

    fn frustum_of(cam: &Camera) -> Frustum {
        Frustum::new(&(cam.projection(1.6) * cam.look_at()))
    }

    /// Patches selected with and without culling, from the same viewpoint.
    fn counts(cam: &Camera) -> (usize, usize) {
        let mut c = Cdlod::new(EXTENT, 0.5);
        let uncut = c.select(cam.pos, RANGE, EXTENT).len();
        let f = frustum_of(cam);
        c.select_culled(cam.pos, RANGE, EXTENT, Some(&f), &FrustumUnion::default());
        (uncut, c.patches().len())
    }

    #[test]
    fn culling_removes_a_large_share_of_the_quadtree() {
        // On a world centred on the camera, roughly half the tree is behind it. Anything
        // less than a quarter removed would mean the test is not reaching the tree.
        let cam = camera_at(Vec3::new(0.0, 400.0, 0.0), 0.0);
        let (uncut, cut) = counts(&cam);
        assert!(uncut > 0, "nothing was selected at all");
        assert!(cut < uncut * 3 / 4, "culling kept {cut} of {uncut} patches, which is not culling");
        assert!(cut > 0, "culling removed everything, so the camera sees no ground");
    }

    #[test]
    fn culling_never_drops_a_patch_the_frustum_can_see() {
        // The property that matters. A cull that is too eager pops geometry out of view,
        // which a user sees immediately; too lax only costs frame time. So every patch the
        // unculled selection produced must survive unless its own box is outside.
        let cam = camera_at(Vec3::new(300.0, 250.0, -700.0), 1.1);
        let f = frustum_of(&cam);

        let mut c = Cdlod::new(EXTENT, 0.5);
        let all: Vec<Patch> = c.select(cam.pos, RANGE, EXTENT).to_vec();
        c.select_culled(cam.pos, RANGE, EXTENT, Some(&f), &FrustumUnion::default());
        let kept: Vec<Patch> = c.patches().to_vec();

        for p in &all {
            let min = Vec3::new(p.origin[0], RANGE.0, p.origin[1]);
            let max = Vec3::new(p.origin[0] + p.size, RANGE.1, p.origin[1] + p.size);
            if f.intersects_aabb(min, max) {
                assert!(
                    kept.iter().any(|k| k.origin == p.origin && k.size == p.size),
                    "a visible patch at {:?} size {} was culled",
                    p.origin,
                    p.size
                );
            }
        }
    }

    #[test]
    fn a_kept_patch_is_identical_to_its_unculled_self() {
        // Culling must only *remove*. If it changed a patch's level or morph band, the
        // shading would differ from frame to frame as the camera turned.
        let cam = camera_at(Vec3::new(-500.0, 300.0, 200.0), 2.4);
        let mut c = Cdlod::new(EXTENT, 0.5);
        let all: Vec<Patch> = c.select(cam.pos, RANGE, EXTENT).to_vec();
        c.select_culled(cam.pos, RANGE, EXTENT, Some(&frustum_of(&cam)), &FrustumUnion::default());
        for k in c.patches() {
            let same = all
                .iter()
                .find(|p| p.origin == k.origin && p.size == k.size)
                .expect("a culled selection invented a patch");
            assert_eq!(same.level, k.level, "level changed under culling");
            assert_eq!(same.morph_start, k.morph_start, "morph band changed under culling");
            assert_eq!(same.morph_end, k.morph_end);
        }
    }

    #[test]
    fn no_frustum_means_no_culling() {
        // The safe reading of "the caller has not said where it is looking".
        let cam = camera_at(Vec3::new(0.0, 400.0, 0.0), 0.0);
        let mut c = Cdlod::new(EXTENT, 0.5);
        let uncut = c.select(cam.pos, RANGE, EXTENT).len();
        c.select_culled(cam.pos, RANGE, EXTENT, None, &FrustumUnion::default());
        assert_eq!(c.patches().len(), uncut);
    }

    #[test]
    fn the_shadow_set_keeps_casters_the_camera_cannot_see() {
        // The Phase C fix. A ridge behind the camera casts into the view, so it has to
        // survive the *camera's* cull -- which is why the shadow set is culled against the
        // light instead.
        let cam = camera_at(Vec3::new(0.0, 400.0, 0.0), 0.0);
        // A "light" looking the opposite way, standing in for a sun behind the camera.
        let light = camera_at(Vec3::new(0.0, 400.0, 0.0), std::f32::consts::PI);
        let lights = FrustumUnion::new([frustum_of(&light)]);

        let mut c = Cdlod::new(EXTENT, 0.5);
        c.select_culled(cam.pos, RANGE, EXTENT, Some(&frustum_of(&cam)), &lights);

        assert!(!c.patches().is_empty(), "the camera set is empty");
        assert!(!c.shadow_patches().is_empty(), "the shadow set is empty");
        // The shadow set has to contain something the camera set does not, or the light
        // frustum is being ignored.
        let extra = c
            .shadow_patches()
            .iter()
            .any(|s| !c.patches().iter().any(|p| p.origin == s.origin && p.size == s.size));
        assert!(extra, "the shadow set is a subset of the camera set, so nothing off-screen casts");
    }

    #[test]
    fn an_empty_light_union_leaves_the_shadow_set_uncut() {
        // Shadows off, or cascades not yet fitted: draw every caster rather than none.
        let cam = camera_at(Vec3::new(0.0, 400.0, 0.0), 0.0);
        let mut c = Cdlod::new(EXTENT, 0.5);
        let uncut = c.select(cam.pos, RANGE, EXTENT).len();
        c.select_culled(cam.pos, RANGE, EXTENT, Some(&frustum_of(&cam)), &FrustumUnion::default());
        assert_eq!(c.shadow_patches().len(), uncut, "an empty light union culled casters");
    }

    #[test]
    fn culling_holds_up_from_every_direction() {
        // Sweeping the yaw, because a plane-extraction sign error usually shows up at one
        // orientation and not at the one that was tried by hand.
        for i in 0..12 {
            let yaw = i as f32 * std::f32::consts::TAU / 12.0;
            let cam = camera_at(Vec3::new(120.0, 350.0, -80.0), yaw);
            let (uncut, cut) = counts(&cam);
            assert!(cut > 0, "at yaw {yaw} everything was culled");
            assert!(cut <= uncut, "at yaw {yaw} culling invented patches");
        }
    }
}

#[cfg(test)]
mod culling_report {
    use super::*;
    use crate::camera::Camera;

    /// What culling removes, printed rather than asserted.
    ///
    /// `cargo test -p terra-render --lib culling_report -- --ignored --nocapture`
    ///
    /// A share rather than a frame time: the saving is in patches submitted, and what that
    /// is worth depends on the fill cost of the terrain shader on the day.
    #[test]
    #[ignore]
    fn how_much_the_terrain_cull_removes() {
        println!("world   pitch   patches   culled   kept    saved");
        for extent in [2048.0f32, 4096.0, 8192.0, 16384.0] {
            for (label, pitch) in [("level", 0.0f32), ("down", -0.6), ("up", 0.3)] {
                let cam = Camera {
                    pos: Vec3::new(0.0, 400.0, 0.0),
                    yaw: 0.7,
                    pitch,
                    ..Camera::default()
                };
                let range = (0.0, 900.0);
                let mut c = Cdlod::new(extent, 0.5);
                let all = c.select(cam.pos, range, extent).len();
                let f = Frustum::new(&(cam.projection(1.6) * cam.look_at()));
                c.select_culled(cam.pos, range, extent, Some(&f), &FrustumUnion::default());
                let kept = c.patches().len();
                println!(
                    "{:>5.0}m  {label:<6} {all:>7}  {:>6}  {kept:>5}   {:>4.0}%",
                    extent,
                    all - kept,
                    100.0 - (kept as f64 / all as f64) * 100.0
                );
            }
        }
    }
}

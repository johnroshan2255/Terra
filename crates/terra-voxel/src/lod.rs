//! Chunk selection, frustum culling, and the indirect draw arguments the
//! whole visible set is submitted with.
//!
//! Cave interiors break the assumption the heightfield renderer was built on.
//! A heightfield has one surface per column, so its working set is bounded by
//! screen area. A cave system has many, most of them hidden behind rock, and
//! the visible set stops correlating with distance -- standing at a tunnel
//! mouth, the ten metres of passage ahead matter more than the kilometre of
//! hillside behind it.
//!
//! Two things follow. Selection is an octree descent rather than a distance
//! ring, so detail concentrates where the camera actually is rather than
//! spreading evenly over a radius. And submission is indirect: the chunk list
//! changes every frame as the camera moves through a passage, and re-recording
//! a draw call per chunk on the CPU would put the cost of the cave system into
//! the command encoder rather than the GPU.

use crate::modifier::Aabb;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// One selected chunk: where it is, how big, and at which detail level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub min: Vec3,
    /// World metres along one edge.
    pub size: f32,
    /// 0 is the finest level. Equal to the number of subdivisions remaining
    /// that were *not* taken.
    pub lod: u8,
}

impl Node {
    pub fn max(&self) -> Vec3 {
        self.min + Vec3::splat(self.size)
    }

    pub fn center(&self) -> Vec3 {
        self.min + Vec3::splat(self.size * 0.5)
    }

    pub fn bounds(&self) -> Aabb {
        Aabb::new(self.min, self.max())
    }

    /// Metres per voxel once this node is diced into `chunk_dim` cells. Every
    /// LOD extracts the same cell count, so a coarse chunk costs the same
    /// vertex budget as a fine one and covers eight times the volume.
    pub fn voxel_size(&self, chunk_dim: u32) -> f32 {
        self.size / chunk_dim as f32
    }

    /// Shortest distance from a point to this node's box, zero inside.
    pub fn distance_to(&self, p: Vec3) -> f32 {
        (self.min - p).max(p - self.max()).max(Vec3::ZERO).length()
    }
}

/// How the octree is diced.
#[derive(Debug, Clone, Copy)]
pub struct LodConfig {
    /// Cells along a chunk edge, at every level.
    pub chunk_dim: u32,
    /// Levels below the root. Depth 0 means the root is the only chunk.
    pub max_depth: u8,
    /// Subdivide while the camera is nearer than this many node-widths.
    ///
    /// At 1.0 a node splits only once the camera is inside a box its own size,
    /// which pops visibly. 2.5 keeps roughly two rings of finer detail in view
    /// ahead of the transition, which is what makes the change unnoticeable
    /// while still halving the chunk count per level.
    pub detail: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self { chunk_dim: crate::volume::CHUNK_DIM, max_depth: 5, detail: 2.5 }
    }
}

/// The six clip-space planes of a camera, for rejecting chunks.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// `xyz` is the outward normal, `w` the offset: a point is inside when
    /// `dot(n, p) + w >= 0` for all six.
    planes: [Vec4; 6],
}

impl Frustum {
    /// Gribb-Hartmann extraction from a view-projection matrix.
    ///
    /// Works unchanged under the reversed-Z projection the renderer uses: near
    /// and far swap roles, but both planes are still produced and the enclosed
    /// volume is identical.
    pub fn from_view_proj(m: Mat4) -> Self {
        let r = m.transpose();
        let (r0, r1, r2, r3) = (r.x_axis, r.y_axis, r.z_axis, r.w_axis);
        let mut planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
        for p in &mut planes {
            // Normalize so the plane offset is a real distance. Without this a
            // near-plane test can be off by the projection's scale factor.
            let len = Vec3::new(p.x, p.y, p.z).length();
            if len > 1e-9 {
                *p /= len;
            }
        }
        Self { planes }
    }

    /// Conservative box test: rejects only boxes wholly outside one plane.
    ///
    /// The classic false positive -- a box outside the frustum but straddling
    /// all six planes -- costs one wasted chunk draw and is not worth a more
    /// expensive test.
    pub fn intersects(&self, b: &Aabb) -> bool {
        for p in &self.planes {
            let n = Vec3::new(p.x, p.y, p.z);
            // The box corner furthest along the plane normal. If even that is
            // behind the plane, nothing in the box can be in front of it.
            let far = Vec3::new(
                if n.x >= 0.0 { b.max.x } else { b.min.x },
                if n.y >= 0.0 { b.max.y } else { b.min.y },
                if n.z >= 0.0 { b.max.z } else { b.min.z },
            );
            if n.dot(far) + p.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Select the visible chunk set by descending an octree.
///
/// `occupied` is asked whether a node contains any surface at all. Returning
/// `false` prunes the whole subtree, which is what keeps solid rock and open
/// sky off the draw list -- in a cave world that is the overwhelming majority
/// of space, and skipping it early is worth more than any amount of frustum
/// culling further down.
pub fn select(
    root: Node,
    camera: Vec3,
    cfg: &LodConfig,
    frustum: Option<&Frustum>,
    occupied: &impl Fn(&Node) -> bool,
) -> Vec<Node> {
    let mut out = Vec::new();
    descend(root, camera, cfg, frustum, occupied, &mut out);
    out
}

fn descend(
    node: Node,
    camera: Vec3,
    cfg: &LodConfig,
    frustum: Option<&Frustum>,
    occupied: &impl Fn(&Node) -> bool,
    out: &mut Vec<Node>,
) {
    let bounds = node.bounds();
    if let Some(f) = frustum
        && !f.intersects(&bounds)
    {
        return;
    }
    if !occupied(&node) {
        return;
    }

    let split = node.lod > 0 && node.distance_to(camera) < cfg.detail * node.size;
    if !split {
        out.push(node);
        return;
    }

    let half = node.size * 0.5;
    for i in 0..8 {
        let offset = Vec3::new(
            if i & 1 != 0 { half } else { 0.0 },
            if i & 2 != 0 { half } else { 0.0 },
            if i & 4 != 0 { half } else { 0.0 },
        );
        descend(
            Node { min: node.min + offset, size: half, lod: node.lod - 1 },
            camera,
            cfg,
            frustum,
            occupied,
            out,
        );
    }
}

/// Arguments for one `draw_indexed_indirect` call.
///
/// Field order and size are fixed by the graphics API, not by us: this must
/// stay a 20-byte struct in exactly this order or the GPU reads garbage. The
/// compile-time assertion below is the guard.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

const _: () = assert!(std::mem::size_of::<DrawIndexedIndirect>() == 20);

impl DrawIndexedIndirect {
    /// One instance, drawn from a slice of a shared index buffer.
    pub fn single(index_count: u32, first_index: u32, base_vertex: i32) -> Self {
        Self { index_count, instance_count: 1, first_index, base_vertex, first_instance: 0 }
    }

    /// An entry the GPU will skip. Culling writes these rather than compacting
    /// the buffer, because compaction needs a prefix sum and a zeroed
    /// instance count costs the command processor almost nothing.
    pub fn skipped() -> Self {
        Self { instance_count: 0, ..Default::default() }
    }

    pub fn is_skipped(&self) -> bool {
        self.instance_count == 0 || self.index_count == 0
    }
}

/// Where one chunk's geometry lives inside the shared vertex and index
/// buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

/// Build the indirect argument list for a set of allocated chunks.
///
/// Every selected chunk gets an entry whether or not it survives culling, so
/// the argument buffer index matches the chunk index and a per-chunk uniform
/// can be looked up by `instance_index` in the shader.
pub fn build_draw_list(chunks: &[(Allocation, bool)]) -> Vec<DrawIndexedIndirect> {
    chunks
        .iter()
        .map(|(a, visible)| {
            if *visible && a.index_count > 0 {
                DrawIndexedIndirect::single(a.index_count, a.first_index, a.first_vertex as i32)
            } else {
                DrawIndexedIndirect::skipped()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use glam::Vec4;

    fn root(size: f32, depth: u8) -> Node {
        Node { min: Vec3::splat(-size * 0.5), size, lod: depth }
    }

    const ALL: fn(&Node) -> bool = |_| true;

    /// The renderer's own view and projection, rebuilt here rather than
    /// imported: `terra-render` sits *above* this crate, and glam's `look_at`
    /// and `perspective` helpers are deprecated, which is why `camera.rs`
    /// builds both by hand too. Testing against the real reversed-Z infinite
    /// projection is also the only way to know the extraction survives it.
    fn view_proj(eye: Vec3, forward: Vec3, aspect: f32, fov_y: f32, znear: f32) -> Mat4 {
        let f = forward.normalize();
        let s = f.cross(Vec3::Y).normalize();
        let u = s.cross(f);
        let view = Mat4::from_cols(
            Vec4::new(s.x, u.x, -f.x, 0.0),
            Vec4::new(s.y, u.y, -f.y, 0.0),
            Vec4::new(s.z, u.z, -f.z, 0.0),
            Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        );
        let c = 1.0 / (fov_y * 0.5).tan();
        let proj = Mat4::from_cols(
            Vec4::new(c / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, c, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, znear, 0.0),
        );
        proj * view
    }

    #[test]
    fn far_camera_keeps_the_root_whole() {
        let cfg = LodConfig { max_depth: 4, detail: 2.5, ..Default::default() };
        let r = root(1024.0, 4);
        let sel = select(r, Vec3::new(0.0, 0.0, 100_000.0), &cfg, None, &ALL);
        assert_eq!(sel.len(), 1, "nothing near the camera should stay subdivided");
        assert_eq!(sel[0].lod, 4);
    }

    #[test]
    fn detail_concentrates_at_the_camera() {
        let cfg = LodConfig { max_depth: 4, detail: 2.5, ..Default::default() };
        let cam = Vec3::new(-500.0, -500.0, -500.0);
        let sel = select(root(1024.0, 4), cam, &cfg, None, &ALL);

        let near =
            sel.iter().min_by(|a, b| a.distance_to(cam).partial_cmp(&b.distance_to(cam)).unwrap());
        let far =
            sel.iter().max_by(|a, b| a.distance_to(cam).partial_cmp(&b.distance_to(cam)).unwrap());
        assert!(near.unwrap().lod < far.unwrap().lod, "the nearest chunk must be the finest");
        assert_eq!(near.unwrap().lod, 0, "the camera corner should reach full detail");
    }

    #[test]
    fn selection_tiles_the_root_without_gaps_or_overlaps() {
        // Volume is the invariant: an octree selection is a partition, so the
        // selected boxes must sum to exactly the root's volume. A gap shows up
        // as a hole in the world; an overlap as z-fighting.
        let cfg = LodConfig { max_depth: 3, detail: 2.0, ..Default::default() };
        let r = root(512.0, 3);
        let sel = select(r, Vec3::new(-100.0, 0.0, 30.0), &cfg, None, &ALL);
        let total: f64 = sel.iter().map(|n| (n.size as f64).powi(3)).sum();
        let expect = (r.size as f64).powi(3);
        assert!((total - expect).abs() / expect < 1e-9, "{total} vs {expect}");
    }

    #[test]
    fn an_unoccupied_subtree_is_pruned_entirely() {
        // The cave-world optimization: most of the volume is solid rock or
        // open sky and must never reach the draw list.
        let cfg = LodConfig { max_depth: 3, detail: 10.0, ..Default::default() };
        // Only accept nodes touching the +X half.
        let occupied = |n: &Node| n.max().x > 0.0;
        let sel = select(root(512.0, 3), Vec3::ZERO, &cfg, None, &occupied);
        assert!(!sel.is_empty());
        assert!(sel.iter().all(|n| n.max().x > 0.0), "pruned region leaked into the draw list");
        let total: f64 = sel.iter().map(|n| (n.size as f64).powi(3)).sum();
        assert!(total < (512.0f64).powi(3) * 0.6, "pruning saved nothing: {total}");
    }

    #[test]
    fn depth_zero_root_never_subdivides() {
        let cfg = LodConfig { max_depth: 0, detail: 100.0, ..Default::default() };
        let sel = select(root(64.0, 0), Vec3::ZERO, &cfg, None, &ALL);
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn voxel_size_halves_with_each_level() {
        let a = Node { min: Vec3::ZERO, size: 128.0, lod: 2 };
        let b = Node { min: Vec3::ZERO, size: 64.0, lod: 1 };
        assert_eq!(a.voxel_size(32), 4.0);
        assert_eq!(b.voxel_size(32), 2.0);
    }

    #[test]
    fn frustum_keeps_what_is_in_front_and_drops_what_is_behind() {
        let f =
            Frustum::from_view_proj(view_proj(Vec3::ZERO, -Vec3::Z, 1.0, 60f32.to_radians(), 0.1));

        let ahead = Aabb::new(Vec3::new(-5.0, -5.0, -60.0), Vec3::new(5.0, 5.0, -50.0));
        let behind = Aabb::new(Vec3::new(-5.0, -5.0, 50.0), Vec3::new(5.0, 5.0, 60.0));
        let far_side = Aabb::new(Vec3::new(900.0, -5.0, -60.0), Vec3::new(910.0, 5.0, -50.0));

        assert!(f.intersects(&ahead), "a box in front must be kept");
        assert!(!f.intersects(&behind), "a box behind the camera must be culled");
        assert!(!f.intersects(&far_side), "a box far off to the side must be culled");
    }

    #[test]
    fn frustum_has_no_far_plane_to_cull_against() {
        // The projection is infinite-far by design, so distance alone must
        // never cull. If a far plane crept back in, the 16 km world would
        // silently lose its horizon.
        let f =
            Frustum::from_view_proj(view_proj(Vec3::ZERO, -Vec3::Z, 1.0, 60f32.to_radians(), 0.1));
        let very_far =
            Aabb::new(Vec3::new(-50.0, -50.0, -80_000.0), Vec3::new(50.0, 50.0, -79_000.0));
        assert!(f.intersects(&very_far), "an infinite projection must not cull on distance");
    }

    #[test]
    fn frustum_keeps_a_box_that_straddles_the_camera() {
        // A chunk the camera is standing inside is trivially visible, and is
        // the case a naive centre-point test gets wrong.
        let f =
            Frustum::from_view_proj(view_proj(Vec3::ZERO, -Vec3::Z, 1.0, 60f32.to_radians(), 0.1));
        assert!(f.intersects(&Aabb::new(Vec3::splat(-20.0), Vec3::splat(20.0))));
    }

    #[test]
    fn indirect_args_are_the_layout_the_gpu_expects() {
        // Wrong field order here is silent: the GPU draws a plausible-looking
        // wrong number of indices and the bug reads as corrupt geometry.
        let a = DrawIndexedIndirect::single(300, 12, 40);
        let raw: &[u32; 5] = bytemuck::cast_ref(&a);
        assert_eq!(*raw, [300, 1, 12, 40, 0]);
    }

    #[test]
    fn culled_chunks_become_zero_instance_entries() {
        let alloc =
            Allocation { first_vertex: 10, vertex_count: 50, first_index: 20, index_count: 90 };
        let list = build_draw_list(&[(alloc, true), (alloc, false)]);
        assert_eq!(list.len(), 2, "entries must line up with chunk indices, not be compacted");
        assert!(!list[0].is_skipped());
        assert!(list[1].is_skipped());
        assert_eq!(list[0].index_count, 90);
        assert_eq!(list[0].base_vertex, 10);
    }

    #[test]
    fn an_empty_chunk_is_skipped_even_when_visible() {
        let empty = Allocation { first_vertex: 0, vertex_count: 0, first_index: 0, index_count: 0 };
        assert!(build_draw_list(&[(empty, true)])[0].is_skipped());
    }
}

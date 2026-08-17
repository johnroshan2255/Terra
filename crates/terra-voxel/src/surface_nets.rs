//! Surface Nets: turning a sampled distance field back into triangles.
//!
//! Surface Nets rather than Marching Cubes, for three reasons that all matter
//! here:
//!
//! * **Uniform density.** At most one vertex per cell, wherever the surface
//!   goes. Marching Cubes emits between one and five triangles per cell
//!   depending on which of its 256 cases fires, so triangle density swings by
//!   5x across a single cave wall and the LOD budget has to be sized for the
//!   worst case.
//! * **Manifold by construction.** Every quad is generated from a sign change
//!   on one lattice edge and joins the four cells around that edge, so each
//!   interior mesh edge is shared by exactly two triangles. Marching Cubes
//!   needs a disambiguation table to avoid holes on saddle cases.
//! * **It maps onto compute cleanly.** One thread per cell, one atomic bump
//!   for the vertex index, and a second pass that reads the index map. There
//!   is no per-case triangle table to keep in registers.
//!
//! The cost is that Surface Nets rounds off hard edges -- a cube comes back
//! with bevelled corners. For eroded rock and cave interiors that is a feature.
//!
//! ## Chunk seams
//!
//! A grid of `dim` cells needs `dim + 1` samples per axis, and can only emit
//! quads for lattice points that have all four surrounding cells. That leaves
//! the outermost layer of quads unemitted, so tiling chunks edge-to-edge would
//! leave a one-cell crack at every seam.
//!
//! The fix is overlap, not stitching: a chunk that owns `n` cells samples
//! `n + 1` cells and emits the quads for the extra layer too. Its neighbour
//! samples the identical field at the identical lattice points, so the two
//! chunks compute bit-identical vertex positions there and the seam closes.
//! Vertices are duplicated across the seam; cracks are not. [`chunk_grid`]
//! applies that overlap.

use crate::volume::VoxelVolume;
use glam::{IVec3, Vec3};

/// Corner offsets of a cell, indexed so bit 0 is X, bit 1 is Y, bit 2 is Z.
const CORNERS: [IVec3; 8] = [
    IVec3::new(0, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(1, 1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(1, 0, 1),
    IVec3::new(0, 1, 1),
    IVec3::new(1, 1, 1),
];

/// The 12 cell edges, as pairs of corner indices: four along X, four along Y,
/// four along Z.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// A dense block of field samples on a regular lattice.
///
/// `dim` counts **cells**; there are `dim + 1` samples along each axis.
pub struct SampleGrid {
    pub dim: u32,
    pub voxel: f32,
    /// World position of sample (0, 0, 0).
    pub origin: Vec3,
    pub values: Vec<f32>,
}

impl SampleGrid {
    pub fn samples_per_axis(&self) -> u32 {
        self.dim + 1
    }

    pub fn index(&self, x: u32, y: u32, z: u32) -> usize {
        let n = self.samples_per_axis() as usize;
        (z as usize * n + y as usize) * n + x as usize
    }

    pub fn at(&self, x: u32, y: u32, z: u32) -> f32 {
        self.values[self.index(x, y, z)]
    }

    /// Allocate and fill by evaluating `f` at every lattice point.
    pub fn sample(dim: u32, voxel: f32, origin: Vec3, f: impl Fn(Vec3) -> f32 + Sync) -> Self {
        let n = (dim + 1) as usize;
        let mut values = vec![0.0f32; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;
                    values[(z * n + y) * n + x] = f(p);
                }
            }
        }
        Self { dim, voxel, origin, values }
    }

    /// True when every sample has the same sign, so no surface can cross this
    /// block. Checked before extraction because most chunks in a world are
    /// solid rock or open air and can be rejected for the cost of one scan.
    pub fn is_uniform(&self) -> bool {
        let first = self.values[0] < 0.0;
        self.values.iter().all(|v| (*v < 0.0) == first)
    }

    /// Trilinear sample in lattice units, used for vertex normals.
    fn lerped(&self, p: Vec3) -> f32 {
        let n = self.dim as f32;
        let c = p.clamp(Vec3::ZERO, Vec3::splat(n));
        let b = c.floor();
        let f = c - b;
        let (x0, y0, z0) = (b.x as u32, b.y as u32, b.z as u32);
        let lim = self.dim;
        let g = |dx: u32, dy: u32, dz: u32| {
            self.at((x0 + dx).min(lim), (y0 + dy).min(lim), (z0 + dz).min(lim))
        };
        let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = mix(g(0, 0, 0), g(1, 0, 0), f.x);
        let c10 = mix(g(0, 1, 0), g(1, 1, 0), f.x);
        let c01 = mix(g(0, 0, 1), g(1, 0, 1), f.x);
        let c11 = mix(g(0, 1, 1), g(1, 1, 1), f.x);
        mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z)
    }

    /// Field gradient at a lattice-space position, as the un-normalized
    /// surface normal.
    fn gradient(&self, p: Vec3) -> Vec3 {
        const H: f32 = 0.5;
        Vec3::new(
            self.lerped(p + Vec3::X * H) - self.lerped(p - Vec3::X * H),
            self.lerped(p + Vec3::Y * H) - self.lerped(p - Vec3::Y * H),
            self.lerped(p + Vec3::Z * H) - self.lerped(p - Vec3::Z * H),
        )
    }
}

/// An extracted triangle mesh, in world metres.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned bounds, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.positions.iter();
        let first = *it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p))))
    }
}

/// Sentinel for "this cell produced no vertex".
const NO_VERTEX: u32 = u32::MAX;

/// Extract the zero level set of `grid`.
pub fn extract(grid: &SampleGrid) -> Mesh {
    let mut mesh = Mesh::default();
    if grid.dim == 0 || grid.is_uniform() {
        return mesh;
    }

    let dim = grid.dim;
    let cells = dim as usize;
    let mut cell_vertex = vec![NO_VERTEX; cells * cells * cells];
    let cell_index =
        |x: u32, y: u32, z: u32| (z as usize * cells + y as usize) * cells + x as usize;

    // --- pass 1: one vertex per cell the surface passes through ---
    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let base = IVec3::new(x as i32, y as i32, z as i32);
                let mut s = [0.0f32; 8];
                for (i, c) in CORNERS.iter().enumerate() {
                    let p = base + *c;
                    s[i] = grid.at(p.x as u32, p.y as u32, p.z as u32);
                }
                // A cell only holds a vertex if the surface crosses it, which
                // is what caps the mesh at one vertex per cell and keeps
                // triangle density uniform.
                let inside = s[0] < 0.0;
                if s.iter().all(|v| (*v < 0.0) == inside) {
                    continue;
                }

                let mut sum = Vec3::ZERO;
                let mut count = 0.0f32;
                for (a, b) in EDGES {
                    let (sa, sb) = (s[a], s[b]);
                    if (sa < 0.0) == (sb < 0.0) {
                        continue;
                    }
                    // Linear crossing. The denominator cannot vanish: the
                    // signs differ, so sa != sb.
                    let t = sa / (sa - sb);
                    let ca = CORNERS[a].as_vec3();
                    let cb = CORNERS[b].as_vec3();
                    sum += ca + (cb - ca) * t;
                    count += 1.0;
                }
                if count == 0.0 {
                    continue;
                }

                let local = sum / count;
                let lattice = base.as_vec3() + local;
                let index = mesh.positions.len() as u32;
                mesh.positions.push(grid.origin + lattice * grid.voxel);
                // Gradient normals rather than accumulated face normals: the
                // field already knows the true surface orientation, and
                // reading it costs six trilinear taps instead of a second
                // pass over every triangle.
                mesh.normals.push(grid.gradient(lattice).normalize_or(Vec3::Y));
                cell_vertex[cell_index(x, y, z)] = index;
            }
        }
    }

    if mesh.positions.is_empty() {
        return mesh;
    }

    // --- pass 2: one quad per lattice edge that changes sign ---
    //
    // Each interior lattice edge is surrounded by exactly four cells. If the
    // field changes sign along the edge, those four cells all hold a vertex
    // and the quad joining them is part of the surface. Winding follows the
    // sign direction so the front face always points into the air.
    let quad = |mesh: &mut Mesh, cells4: [Option<u32>; 4], flip: bool| {
        let (Some(a), Some(b), Some(c), Some(d)) = (cells4[0], cells4[1], cells4[2], cells4[3])
        else {
            // Cannot happen for a genuine sign change, but a missing corner
            // must drop the quad rather than index out of bounds.
            return;
        };
        let v = if flip { [a, d, c, b] } else { [a, b, c, d] };
        mesh.indices.extend_from_slice(&[v[0], v[1], v[2], v[0], v[2], v[3]]);
    };

    let get = |cv: &[u32], x: u32, y: u32, z: u32| -> Option<u32> {
        let i = cv[cell_index(x, y, z)];
        (i != NO_VERTEX).then_some(i)
    };

    for z in 0..=dim {
        for y in 0..=dim {
            for x in 0..=dim {
                let s0 = grid.at(x, y, z);

                // +X edge. The four cells around it vary in Y and Z; ordering
                // Y then Z is counter-clockwise seen from +X.
                if x < dim && y >= 1 && z >= 1 && y < dim && z < dim {
                    let s1 = grid.at(x + 1, y, z);
                    if (s0 < 0.0) != (s1 < 0.0) {
                        let c = [
                            get(&cell_vertex, x, y - 1, z - 1),
                            get(&cell_vertex, x, y, z - 1),
                            get(&cell_vertex, x, y, z),
                            get(&cell_vertex, x, y - 1, z),
                        ];
                        quad(&mut mesh, c, s0 >= 0.0);
                    }
                }

                // +Y edge. Z then X is counter-clockwise seen from +Y.
                if y < dim && x >= 1 && z >= 1 && x < dim && z < dim {
                    let s1 = grid.at(x, y + 1, z);
                    if (s0 < 0.0) != (s1 < 0.0) {
                        let c = [
                            get(&cell_vertex, x - 1, y, z - 1),
                            get(&cell_vertex, x - 1, y, z),
                            get(&cell_vertex, x, y, z),
                            get(&cell_vertex, x, y, z - 1),
                        ];
                        quad(&mut mesh, c, s0 >= 0.0);
                    }
                }

                // +Z edge. X then Y is counter-clockwise seen from +Z.
                if z < dim && x >= 1 && y >= 1 && x < dim && y < dim {
                    let s1 = grid.at(x, y, z + 1);
                    if (s0 < 0.0) != (s1 < 0.0) {
                        let c = [
                            get(&cell_vertex, x - 1, y - 1, z),
                            get(&cell_vertex, x, y - 1, z),
                            get(&cell_vertex, x, y, z),
                            get(&cell_vertex, x - 1, y, z),
                        ];
                        quad(&mut mesh, c, s0 >= 0.0);
                    }
                }
            }
        }
    }

    mesh
}

/// Sample one chunk of a volume, with the one-cell overlap that closes seams.
///
/// `chunk` is in chunk units, `dim` is how many cells a chunk owns. The grid
/// returned covers `dim + 1` cells, the last of which belongs to the
/// neighbour and exists only so this chunk can emit the boundary quads.
pub fn chunk_grid(volume: &VoxelVolume<'_>, chunk: IVec3, dim: u32, voxel: f32) -> SampleGrid {
    let origin = chunk.as_vec3() * (dim as f32 * voxel);
    SampleGrid::sample(dim + 1, voxel, origin, |p| volume.sample(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sphere_grid(dim: u32, radius: f32) -> SampleGrid {
        let voxel = 4.0 / dim as f32;
        let origin = Vec3::splat(-2.0);
        SampleGrid::sample(dim, voxel, origin, |p| p.length() - radius)
    }

    /// Every undirected edge and how many triangles use it.
    fn edge_use(mesh: &Mesh) -> HashMap<(u32, u32), u32> {
        let mut m = HashMap::new();
        for t in mesh.indices.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *m.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        m
    }

    #[test]
    fn a_uniform_block_produces_nothing() {
        let all_air = SampleGrid::sample(8, 1.0, Vec3::ZERO, |_| 5.0);
        let all_rock = SampleGrid::sample(8, 1.0, Vec3::ZERO, |_| -5.0);
        assert!(extract(&all_air).is_empty());
        assert!(extract(&all_rock).is_empty());
    }

    #[test]
    fn sphere_extracts_a_closed_surface() {
        let mesh = extract(&sphere_grid(24, 1.0));
        assert!(!mesh.is_empty(), "a sphere crossing the block must produce triangles");
        // Watertight: every edge shared by exactly two triangles. A hole shows
        // up as an edge with one user, a non-manifold fold as three or more.
        for (edge, n) in edge_use(&mesh) {
            assert_eq!(n, 2, "edge {edge:?} used by {n} triangles, want 2");
        }
    }

    #[test]
    fn sphere_vertices_land_on_the_sphere() {
        // The point of interpolating along cell edges rather than snapping to
        // cell centres. With a 24-cell grid over 4 units, a cell is 1/6 unit;
        // vertices should sit far closer to the true radius than that.
        let grid = sphere_grid(24, 1.0);
        let mesh = extract(&grid);
        let worst = mesh.positions.iter().map(|p| (p.length() - 1.0).abs()).fold(0.0f32, f32::max);
        assert!(worst < grid.voxel * 0.5, "worst radial error {worst}, voxel {}", grid.voxel);
    }

    #[test]
    fn triangles_wind_outward() {
        // Winding is the one thing in the quad pass that cannot be derived
        // from first principles without picking a convention, so it is pinned
        // here: on a sphere centred at the origin, every face normal must
        // point away from the centre.
        let mesh = extract(&sphere_grid(20, 1.0));
        let mut checked = 0;
        for t in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            let n = (b - a).cross(c - a);
            // Degenerate slivers carry no orientation; skip rather than fail.
            if n.length_squared() < 1e-12 {
                continue;
            }
            let outward = (a + b + c) / 3.0;
            assert!(
                n.dot(outward) > 0.0,
                "inward-facing triangle at {outward}, normal {}",
                n.normalize()
            );
            checked += 1;
        }
        assert!(checked > 100, "only {checked} triangles checked; test is too weak");
    }

    #[test]
    fn vertex_normals_point_outward_too() {
        let mesh = extract(&sphere_grid(20, 1.0));
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            assert!(n.dot(p.normalize()) > 0.85, "normal {n} at {p} is not radial");
            assert!((n.length() - 1.0).abs() < 1e-3, "normal not unit: {}", n.length());
        }
    }

    #[test]
    fn density_is_uniform_one_vertex_per_crossed_cell() {
        // The property Surface Nets is chosen for. Count cells the surface
        // actually crosses and check the vertex count matches exactly.
        let grid = sphere_grid(16, 1.0);
        let mesh = extract(&grid);
        let dim = grid.dim;
        let mut crossed = 0;
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let mut neg = false;
                    let mut pos = false;
                    for c in CORNERS {
                        let v = grid.at(x + c.x as u32, y + c.y as u32, z + c.z as u32);
                        if v < 0.0 { neg = true } else { pos = true }
                    }
                    if neg && pos {
                        crossed += 1;
                    }
                }
            }
        }
        assert_eq!(mesh.vertex_count(), crossed, "must be exactly one vertex per crossed cell");
    }

    #[test]
    fn a_carved_sphere_is_still_watertight() {
        // The case the whole crate exists for: two surfaces meeting. A ball of
        // rock with a tunnel bored through it has an outer surface, an inner
        // surface, and two mouths where they join -- historically where naive
        // extraction leaves holes.
        let grid = SampleGrid::sample(32, 0.125, Vec3::splat(-2.0), |p| {
            let rock = p.length() - 1.5;
            let bore =
                crate::sdf::capsule(p, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), 0.4);
            crate::sdf::subtract(rock, bore)
        });
        let mesh = extract(&grid);
        assert!(!mesh.is_empty());
        for (edge, n) in edge_use(&mesh) {
            assert_eq!(n, 2, "carved mesh leaks at edge {edge:?}: {n} triangles");
        }
    }

    #[test]
    fn adjacent_chunks_agree_on_their_shared_boundary() {
        // The seam claim. Two neighbouring chunks, extracted independently,
        // must produce coincident vertices in the overlap -- otherwise the
        // world cracks along every chunk line.
        // A sphere straddling the seam. Centring it on the origin instead
        // would leave the second chunk entirely in open air, and the test
        // would pass by extracting nothing from either side of a seam that
        // was never tested.
        let (dim, voxel) = (8u32, 1.0f32);
        let field = |p: Vec3| (p - Vec3::new(8.0, 5.0, 5.0)).length() - 3.5;

        let a = SampleGrid::sample(dim + 1, voxel, Vec3::new(0.0, 0.0, 0.0), field);
        let b = SampleGrid::sample(dim + 1, voxel, Vec3::new(dim as f32 * voxel, 0.0, 0.0), field);

        // Chunk A's overlap column is chunk B's first column. Same field, same
        // lattice, so the samples must be bit-identical.
        for z in 0..=dim {
            for y in 0..=dim {
                assert_eq!(a.at(dim, y, z), b.at(0, y, z), "seam sample mismatch at y{y} z{z}");
            }
        }

        let (ma, mb) = (extract(&a), extract(&b));
        assert!(!ma.is_empty() && !mb.is_empty());

        // And the geometry in the shared slab coincides. The shared slab is
        // exactly the overlap cell -- chunk A's last cell and chunk B's first
        // -- which spans x in [seam, seam + voxel]. Widening this to either
        // side picks up cells only one chunk owns, which legitimately have no
        // counterpart.
        let seam_x = dim as f32 * voxel;
        let near = |m: &Mesh| {
            let mut v: Vec<Vec3> = m
                .positions
                .iter()
                .copied()
                .filter(|p| p.x >= seam_x && p.x <= seam_x + voxel)
                .collect();
            v.sort_by(|p, q| p.to_array().partial_cmp(&q.to_array()).unwrap());
            v
        };
        let (va, vb) = (near(&ma), near(&mb));
        assert!(!va.is_empty(), "no geometry at the seam; test proves nothing");
        for p in &va {
            assert!(
                vb.iter().any(|q| (*q - *p).length() < 1e-4),
                "chunk A has a seam vertex at {p} that chunk B does not"
            );
        }
    }

    #[test]
    fn a_plane_comes_back_flat_and_at_the_right_height() {
        let grid = SampleGrid::sample(16, 0.5, Vec3::new(-4.0, -4.0, -4.0), |p| p.y - 1.0);
        let mesh = extract(&grid);
        assert!(!mesh.is_empty());
        for p in &mesh.positions {
            assert!((p.y - 1.0).abs() < 1e-3, "vertex off the plane at {p}");
        }
        for n in &mesh.normals {
            assert!(n.dot(Vec3::Y) > 0.99, "plane normal should be +Y, got {n}");
        }
    }
}

//! The composed volumetric field, and the sparse storage behind its editable
//! layer.
//!
//! Three layers are combined at sample time:
//!
//! ```text
//! base      the existing 2.5D heightfield, read as an SDF
//! delta     sparse voxel offsets -- freeform clay, overhangs, arches
//! stack     non-destructive boolean modifiers -- caves and tunnels
//! ```
//!
//! evaluated as `stack(base(p) + delta(p))`.
//!
//! The ordering is the whole design. Clay lands *under* the modifiers, so a
//! tunnel bored through a sculpted overhang stays bored no matter how much the
//! clay around it is reworked afterwards -- and deleting the tunnel modifier
//! restores solid rock without touching a single stored voxel. Putting the
//! delta last would bake the carve into the clay the first time anyone
//! smoothed near a cave mouth.
//!
//! Only the delta layer costs memory, and only where someone has actually
//! sculpted. An untouched world stores zero voxels: the base is the
//! heightfield that was already in memory, and the stack is a short list of
//! shapes.

use crate::modifier::ModifierStack;
use glam::{IVec3, Vec2, Vec3};
use std::collections::HashMap;

/// Voxels along one edge of an extraction chunk.
///
/// 32 gives 32,768 cells and at most 32,768 vertices per chunk, which keeps a
/// chunk's vertex buffer under a megabyte and lets the GPU extraction pass use
/// one workgroup per 4x4x4 block without a second dispatch level.
pub const CHUNK_DIM: u32 = 32;

/// Voxels along one edge of a sparse delta brick.
///
/// Deliberately smaller than a chunk. Brush strokes are local, and allocating
/// a full chunk of storage for a 2 m dab would waste 30x the memory it needs.
/// 16^3 f32 is 16 KiB per brick -- small enough that a stray allocation is
/// cheap, large enough that the hash lookup is amortized over 4096 voxels.
pub const BRICK_DIM: u32 = 16;

const BRICK_VOXELS: usize = (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize;

/// The 2.5D heightfield, presented as a sampler.
///
/// Holds a borrowed view rather than a copy: the editor already keeps an
/// authoritative CPU heightfield for raycasting and saving, and duplicating a
/// 4096^2 map to sculpt on it would double the largest allocation in the
/// process.
pub struct BaseField<'a> {
    heights: &'a [f32],
    res: u32,
    extent_m: f32,
}

impl<'a> BaseField<'a> {
    pub fn new(heights: &'a [f32], res: u32, extent_m: f32) -> Self {
        debug_assert_eq!(heights.len(), (res as usize) * (res as usize));
        Self { heights, res, extent_m }
    }

    /// Metres covered by one heightfield texel.
    pub fn cell_m(&self) -> f32 {
        self.extent_m / (self.res - 1).max(1) as f32
    }

    fn texel(&self, x: i32, z: i32) -> f32 {
        let n = self.res as i32;
        let x = x.clamp(0, n - 1) as usize;
        let z = z.clamp(0, n - 1) as usize;
        self.heights[z * self.res as usize + x]
    }

    /// Grid coordinates for a world position. The heightfield is
    /// origin-centred, matching the tile addressing in `terra-core`.
    fn grid_coords(&self, x: f32, z: f32) -> (f32, f32) {
        let half = self.extent_m * 0.5;
        let u = (x + half) / self.extent_m * (self.res - 1) as f32;
        let v = (z + half) / self.extent_m * (self.res - 1) as f32;
        (u, v)
    }

    /// Bilinear height at a world XZ position.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let (u, v) = self.grid_coords(x, z);
        let (x0, z0) = (u.floor() as i32, v.floor() as i32);
        let (fx, fz) = (u - x0 as f32, v - z0 as f32);
        let h00 = self.texel(x0, z0);
        let h10 = self.texel(x0 + 1, z0);
        let h01 = self.texel(x0, z0 + 1);
        let h11 = self.texel(x0 + 1, z0 + 1);
        let a = h00 + (h10 - h00) * fx;
        let b = h01 + (h11 - h01) * fx;
        a + (b - a) * fz
    }

    /// Slope, as dh/dx and dh/dz in metres per metre. Central differences on
    /// the texel lattice: sampling `height_at` either side would re-run the
    /// bilinear filter four times for the same answer.
    pub fn gradient_at(&self, x: f32, z: f32) -> Vec2 {
        let c = self.cell_m();
        let dx = (self.height_at(x + c, z) - self.height_at(x - c, z)) / (2.0 * c);
        let dz = (self.height_at(x, z + c) - self.height_at(x, z - c)) / (2.0 * c);
        Vec2::new(dx, dz)
    }

    /// Signed distance to the terrain surface, negative below ground.
    pub fn distance(&self, p: Vec3) -> f32 {
        crate::sdf::heightfield(p.y, self.height_at(p.x, p.z), self.gradient_at(p.x, p.z))
    }
}

/// Sparse voxel offsets, stored as bricks in a hash map.
///
/// A value is added to the base distance, so a negative entry means "more
/// solid here than the heightfield says" -- which is exactly what an overhang
/// is, and why this layer is what lifts the engine out of 2.5D.
#[derive(Default, Clone)]
pub struct DeltaField {
    bricks: HashMap<IVec3, Vec<f32>>,
    voxel_size: f32,
}

impl DeltaField {
    pub fn new(voxel_size: f32) -> Self {
        Self { bricks: HashMap::new(), voxel_size }
    }

    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Allocated bricks. The editor shows this as the sculpt memory cost.
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    pub fn bytes(&self) -> usize {
        self.bricks.len() * BRICK_VOXELS * std::mem::size_of::<f32>()
    }

    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    pub fn iter_bricks(&self) -> impl Iterator<Item = (&IVec3, &Vec<f32>)> {
        self.bricks.iter()
    }

    /// Split a voxel coordinate into its brick and the index within it.
    ///
    /// `div_euclid` rather than integer division: at negative coordinates
    /// truncating division rounds toward zero, so voxels -15..=-1 would all
    /// land in brick 0 alongside 0..=15 and overwrite each other. Every world
    /// here is origin-centred, so half of it is at negative coordinates and
    /// this is the common case, not an edge case.
    fn split(v: IVec3) -> (IVec3, usize) {
        let d = BRICK_DIM as i32;
        let brick = IVec3::new(v.x.div_euclid(d), v.y.div_euclid(d), v.z.div_euclid(d));
        let l = IVec3::new(v.x.rem_euclid(d), v.y.rem_euclid(d), v.z.rem_euclid(d));
        let index = (l.z * d * d + l.y * d + l.x) as usize;
        (brick, index)
    }

    /// Stored offset at a voxel lattice point. Unallocated bricks read as
    /// zero, which is what makes an untouched world free.
    pub fn get(&self, v: IVec3) -> f32 {
        let (brick, i) = Self::split(v);
        self.bricks.get(&brick).map_or(0.0, |b| b[i])
    }

    pub fn set(&mut self, v: IVec3, value: f32) {
        let (brick, i) = Self::split(v);
        // Do not allocate a brick to store a zero. A smooth brush passing over
        // untouched ground writes mostly zeros, and without this check a
        // single pass would allocate the entire swept volume.
        if value == 0.0 && !self.bricks.contains_key(&brick) {
            return;
        }
        self.bricks.entry(brick).or_insert_with(|| vec![0.0; BRICK_VOXELS])[i] = value;
    }

    pub fn add(&mut self, v: IVec3, value: f32) {
        if value == 0.0 {
            return;
        }
        let (brick, i) = Self::split(v);
        self.bricks.entry(brick).or_insert_with(|| vec![0.0; BRICK_VOXELS])[i] += value;
    }

    /// Trilinear offset at an arbitrary world position.
    ///
    /// Eight independent lattice lookups, each resolving its own brick. That
    /// is up to eight hash probes for one sample, which sounds wasteful and is
    /// -- but it is also the only version that stays correct across a brick
    /// boundary without an apron, and the GPU path in [`crate::gpu`] uploads
    /// dense chunks anyway, so this is authoring-time code, not frame-time
    /// code.
    pub fn sample(&self, p: Vec3) -> f32 {
        if self.bricks.is_empty() {
            return 0.0;
        }
        let g = p / self.voxel_size;
        let base = IVec3::new(g.x.floor() as i32, g.y.floor() as i32, g.z.floor() as i32);
        let f = g - Vec3::new(base.x as f32, base.y as f32, base.z as f32);

        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { f.x } else { 1.0 - f.x })
                        * (if dy == 1 { f.y } else { 1.0 - f.y })
                        * (if dz == 1 { f.z } else { 1.0 - f.z });
                    if w > 0.0 {
                        acc += w * self.get(base + IVec3::new(dx, dy, dz));
                    }
                }
            }
        }
        acc
    }

    /// Drop bricks that hold nothing but zeros.
    ///
    /// Sculpting toward flat, then smoothing, leaves bricks that are allocated
    /// but semantically empty. Without a sweep they accumulate for the life of
    /// the session -- the memory equivalent of a leak, even though every byte
    /// is still reachable.
    pub fn compact(&mut self) -> usize {
        let before = self.bricks.len();
        self.bricks.retain(|_, b| b.iter().any(|v| *v != 0.0));
        before - self.bricks.len()
    }
}

/// Base, delta and modifiers, composed.
pub struct VoxelVolume<'a> {
    pub base: BaseField<'a>,
    pub delta: DeltaField,
    pub stack: ModifierStack,
}

impl<'a> VoxelVolume<'a> {
    pub fn new(base: BaseField<'a>, voxel_size: f32) -> Self {
        Self { base, delta: DeltaField::new(voxel_size), stack: ModifierStack::default() }
    }

    pub fn voxel_size(&self) -> f32 {
        self.delta.voxel_size()
    }

    /// The composed field. Negative is solid.
    pub fn sample(&self, p: Vec3) -> f32 {
        let d = self.base.distance(p) + self.delta.sample(p);
        self.stack.apply(p, d)
    }

    /// Surface normal, as the normalized gradient of the composed field.
    ///
    /// Central differences at half a voxel. Forward differences were tried and
    /// bias the normal by half a cell toward +XYZ, which on a smooth cave wall
    /// reads as a directional sheen that moves when the camera does.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        let h = self.voxel_size() * 0.5;
        let g = Vec3::new(
            self.sample(p + Vec3::X * h) - self.sample(p - Vec3::X * h),
            self.sample(p + Vec3::Y * h) - self.sample(p - Vec3::Y * h),
            self.sample(p + Vec3::Z * h) - self.sample(p - Vec3::Z * h),
        );
        g.normalize_or(Vec3::Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_base(res: u32, height: f32) -> Vec<f32> {
        vec![height; (res * res) as usize]
    }

    #[test]
    fn flat_ground_reads_as_a_plane() {
        let h = flat_base(16, 100.0);
        let b = BaseField::new(&h, 16, 1000.0);
        assert!((b.height_at(0.0, 0.0) - 100.0).abs() < 1e-4);
        assert!(b.gradient_at(0.0, 0.0).length() < 1e-4, "flat ground must have no slope");
        assert!(b.distance(Vec3::new(0.0, 130.0, 0.0)) > 0.0, "above ground is air");
        assert!(b.distance(Vec3::new(0.0, 70.0, 0.0)) < 0.0, "below ground is solid");
    }

    #[test]
    fn base_gradient_matches_a_known_ramp() {
        // h(x) = x, sampled over a 100 m extent. dh/dx must come back as 1.
        let res = 64;
        let extent = 100.0f32;
        let mut h = vec![0.0; (res * res) as usize];
        for z in 0..res {
            for x in 0..res {
                let wx = -extent * 0.5 + x as f32 / (res - 1) as f32 * extent;
                h[(z * res + x) as usize] = wx;
            }
        }
        let b = BaseField::new(&h, res, extent);
        let g = b.gradient_at(0.0, 0.0);
        assert!((g.x - 1.0).abs() < 1e-3, "dh/dx = {}", g.x);
        assert!(g.y.abs() < 1e-3, "ramp has no Z slope, got {}", g.y);
    }

    #[test]
    fn empty_delta_costs_nothing() {
        let d = DeltaField::new(1.0);
        assert_eq!(d.brick_count(), 0);
        assert_eq!(d.bytes(), 0);
        assert_eq!(d.sample(Vec3::new(3.0, -9.0, 42.0)), 0.0);
    }

    #[test]
    fn negative_voxels_get_their_own_brick() {
        // The bug this guards: with truncating division, voxel -1 and voxel 0
        // share brick 0 and land on different indices only by luck. Every
        // origin-centred world puts half its voxels here.
        let mut d = DeltaField::new(1.0);
        d.set(IVec3::new(-1, -1, -1), -5.0);
        d.set(IVec3::new(0, 0, 0), 7.0);
        assert_eq!(d.get(IVec3::new(-1, -1, -1)), -5.0);
        assert_eq!(d.get(IVec3::new(0, 0, 0)), 7.0);
        assert_eq!(d.brick_count(), 2, "opposite sides of the origin are different bricks");
    }

    #[test]
    fn delta_round_trips_across_a_brick_boundary() {
        let mut d = DeltaField::new(1.0);
        let n = BRICK_DIM as i32;
        for v in [IVec3::new(n - 1, 0, 0), IVec3::new(n, 0, 0), IVec3::new(-n, 5, -1)] {
            d.set(v, v.x as f32 + 0.5);
            assert_eq!(d.get(v), v.x as f32 + 0.5, "at {v}");
        }
    }

    #[test]
    fn trilinear_sample_hits_lattice_points_exactly() {
        let mut d = DeltaField::new(2.0);
        d.set(IVec3::new(1, 2, 3), -4.0);
        // Lattice point (1,2,3) is at world (2,4,6) with a 2 m voxel.
        assert!((d.sample(Vec3::new(2.0, 4.0, 6.0)) + 4.0).abs() < 1e-5);
        // Halfway to an empty neighbour is half the value.
        assert!((d.sample(Vec3::new(3.0, 4.0, 6.0)) + 2.0).abs() < 1e-5);
    }

    #[test]
    fn setting_zero_does_not_allocate() {
        let mut d = DeltaField::new(1.0);
        d.set(IVec3::new(4, 4, 4), 0.0);
        assert_eq!(d.brick_count(), 0, "a zero write must not allocate 16 KiB");
    }

    #[test]
    fn compact_reclaims_zeroed_bricks() {
        let mut d = DeltaField::new(1.0);
        d.set(IVec3::new(0, 0, 0), 1.0);
        d.set(IVec3::new(100, 0, 0), 2.0);
        assert_eq!(d.brick_count(), 2);
        d.set(IVec3::new(0, 0, 0), 0.0);
        assert_eq!(d.compact(), 1);
        assert_eq!(d.brick_count(), 1);
    }

    #[test]
    fn delta_can_produce_an_overhang() {
        // The claim that this is no longer 2.5D, tested directly: put solid
        // material in the air above ground level, with air beneath it. No
        // heightfield can represent that column.
        let h = flat_base(16, 100.0);
        let mut v = VoxelVolume::new(BaseField::new(&h, 16, 1000.0), 1.0);
        for y in 118..=122 {
            for x in -2..=2 {
                for z in -2..=2 {
                    // Push the field strongly negative -- solid -- up in the air.
                    v.delta.set(IVec3::new(x, y, z), -30.0);
                }
            }
        }
        let above = v.sample(Vec3::new(0.0, 120.0, 0.0));
        let between = v.sample(Vec3::new(0.0, 110.0, 0.0));
        assert!(above < 0.0, "the floating slab must be solid, got {above}");
        assert!(between > 0.0, "the gap under it must stay air, got {between}");
    }

    #[test]
    fn normal_points_up_on_flat_ground() {
        let h = flat_base(16, 100.0);
        let v = VoxelVolume::new(BaseField::new(&h, 16, 1000.0), 1.0);
        let n = v.normal(Vec3::new(0.0, 100.0, 0.0));
        assert!(n.dot(Vec3::Y) > 0.99, "flat ground normal should be +Y, got {n}");
    }
}

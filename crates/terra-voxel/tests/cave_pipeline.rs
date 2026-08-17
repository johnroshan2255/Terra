//! End-to-end: a real heightfield, sculpted into an overhang, bored through
//! with a cave, extracted, and selected for drawing.
//!
//! Each unit test in the crate checks one layer in isolation. This one checks
//! that the layers actually compose -- that a tunnel carved through sculpted
//! rock over generated terrain comes out as one watertight surface with a
//! draw list in front of it.

use glam::{IVec3, Vec3};
use std::collections::HashMap;
use terra_voxel::brush::{self, Brush, Stroke};
use terra_voxel::lod::{self, Allocation, LodConfig, Node};
use terra_voxel::modifier::{Modifier, Shape, Tube, TubePoint};
use terra_voxel::surface_nets::{self, Mesh, SampleGrid};
use terra_voxel::volume::{BaseField, VoxelVolume};

/// A hill: smooth, tall enough to bore through, and not flat anywhere the
/// tests probe. Generated rather than loaded so the test has no fixtures.
fn hill(res: u32, extent: f32) -> Vec<f32> {
    let mut h = vec![0.0f32; (res * res) as usize];
    for z in 0..res {
        for x in 0..res {
            let wx = -extent * 0.5 + x as f32 / (res - 1) as f32 * extent;
            let wz = -extent * 0.5 + z as f32 / (res - 1) as f32 * extent;
            let r = (wx * wx + wz * wz).sqrt();
            // A raised cosine out to 60 m, flat ground beyond.
            let bump = if r < 60.0 {
                40.0 * (0.5 + 0.5 * (r / 60.0 * std::f32::consts::PI).cos())
            } else {
                0.0
            };
            h[(z * res + x) as usize] = 100.0 + bump;
        }
    }
    h
}

/// Check that every *interior* edge is shared by exactly two triangles.
///
/// Edges on the block boundary are excluded, and legitimately so: an extracted
/// block clips whatever surface runs out of it, leaving an open boundary loop
/// by construction. That loop is closed by the neighbouring chunk's overlap,
/// not by this block. Requiring closure here would only be satisfiable by a
/// surface that happens to fit entirely inside the sample block, which real
/// terrain never does.
fn interior_manifold(m: &Mesh, min: Vec3, max: Vec3, margin: f32) -> Result<(), String> {
    let key = |p: Vec3| {
        let q = |v: f32| (v as f64 * 10_000.0).round() as i64;
        (q(p.x), q(p.y), q(p.z))
    };
    let interior = |p: Vec3| {
        p.cmpgt(min + Vec3::splat(margin)).all() && p.cmplt(max - Vec3::splat(margin)).all()
    };

    let mut edges: HashMap<_, (u32, bool)> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (pa, pb) = (m.positions[a as usize], m.positions[b as usize]);
            let (ka, kb) = (key(pa), key(pb));
            let e = edges.entry((ka.min(kb), ka.max(kb))).or_insert((0, false));
            e.0 += 1;
            e.1 = interior(pa) && interior(pb);
        }
    }
    let mut interior_edges = 0;
    for (e, (n, is_interior)) in &edges {
        if !is_interior {
            continue;
        }
        interior_edges += 1;
        if *n != 2 {
            return Err(format!("interior edge {e:?} used by {n} triangles, want 2"));
        }
    }
    if interior_edges < 100 {
        return Err(format!("only {interior_edges} interior edges; the check is vacuous"));
    }
    Ok(())
}

/// Count how many times a vertical ray through (x, z) crosses the surface.
/// Two crossings is ordinary ground with a cave under it; a heightfield can
/// only ever produce one.
fn crossings(v: &VoxelVolume<'_>, x: f32, z: f32, lo: f32, hi: f32, steps: u32) -> u32 {
    let mut n = 0;
    let mut prev = v.sample(Vec3::new(x, lo, z)) < 0.0;
    for i in 1..=steps {
        let y = lo + (hi - lo) * i as f32 / steps as f32;
        let cur = v.sample(Vec3::new(x, y, z)) < 0.0;
        if cur != prev {
            n += 1;
        }
        prev = cur;
    }
    n
}

#[test]
fn a_tunnel_bored_through_a_hill_is_one_watertight_surface() {
    let (res, extent) = (128u32, 400.0f32);
    let h = hill(res, extent);
    let mut volume = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

    // Bore a passage straight through the hill, entering low on one side and
    // leaving low on the other, dipping in the middle.
    let tube = Tube::new(
        vec![
            TubePoint::new(Vec3::new(-70.0, 108.0, 0.0), 6.0),
            TubePoint::new(Vec3::new(-20.0, 112.0, 0.0), 8.0),
            TubePoint::new(Vec3::new(20.0, 112.0, 0.0), 8.0),
            TubePoint::new(Vec3::new(70.0, 108.0, 0.0), 6.0),
        ],
        8,
    );
    volume.stack.push(Modifier::carve("main passage", Shape::Tube(tube), 1.5));

    // A vertical ray through the middle of the hill starts underground and
    // crosses the surface three times on the way out: cave floor, cave roof,
    // hilltop. A heightfield has one height per column and can therefore
    // produce exactly one crossing, so this count *is* the 3D claim.
    let n = crossings(&volume, 0.0, 0.0, 95.0, 150.0, 4000);
    assert_eq!(n, 3, "expected floor/roof/hilltop, got {n} crossings through the cave");

    // Extract the block containing the passage and check the cave surface
    // closes up everywhere except where the block itself clips it.
    let origin = Vec3::new(-24.0, 96.0, -24.0);
    let size = 48.0;
    let grid = SampleGrid::sample(48, 1.0, origin, |p| volume.sample(p));
    let mesh = surface_nets::extract(&grid);
    assert!(!mesh.is_empty(), "the cave interior extracted to nothing");
    interior_manifold(&mesh, origin, origin + Vec3::splat(size), 1.5)
        .expect("carved hill is not manifold in its interior");
}

#[test]
fn deleting_the_modifier_fills_the_cave_back_in() {
    let (res, extent) = (64u32, 400.0f32);
    let h = hill(res, extent);
    let mut volume = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

    let probe = Vec3::new(0.0, 115.0, 0.0);
    let solid_before = volume.sample(probe);
    assert!(solid_before < 0.0, "the probe must start inside the hill");

    let id = volume.stack.push(Modifier::carve(
        "passage",
        Shape::Tube(Tube::straight(Vec3::new(-70.0, 115.0, 0.0), Vec3::new(70.0, 115.0, 0.0), 8.0)),
        0.0,
    ));
    assert!(volume.sample(probe) > 0.0, "the carve did not open the passage");

    volume.stack.remove(id);
    assert_eq!(
        volume.sample(probe),
        solid_before,
        "removing the modifier must restore the field exactly"
    );
}

#[test]
fn sculpted_overhang_survives_a_carve_through_it() {
    // The ordering claim from the module docs, end to end: clay goes under the
    // modifiers, so boring a tunnel through sculpted rock leaves the sculpt
    // intact once the tunnel is switched off.
    let (res, extent) = (64u32, 400.0f32);
    let h = hill(res, extent);
    let mut volume = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

    // Build a slab of rock out into the air, past where the ground is -- an
    // overhang no heightfield could hold.
    for y in 150..=154 {
        for x in -8..=8 {
            for z in -8..=8 {
                volume.delta.set(IVec3::new(x, y, z), -25.0);
            }
        }
    }
    let in_slab = Vec3::new(0.0, 152.0, 0.0);
    let under_slab = Vec3::new(0.0, 145.0, 0.0);
    assert!(volume.sample(in_slab) < 0.0, "the sculpted slab should be solid");
    assert!(volume.sample(under_slab) > 0.0, "the gap under it should be air");

    let sculpt_value = volume.sample(in_slab);

    volume.stack.push(Modifier::carve(
        "bore",
        Shape::Tube(Tube::straight(Vec3::new(-40.0, 152.0, 0.0), Vec3::new(40.0, 152.0, 0.0), 3.0)),
        0.0,
    ));
    assert!(volume.sample(in_slab) > 0.0, "the bore should have opened the slab");

    volume.stack.items[0].enabled = false;
    assert_eq!(volume.sample(in_slab), sculpt_value, "the sculpt must come back untouched");
}

#[test]
fn brushes_and_modifiers_compose_into_a_drawable_chunk_set() {
    let (res, extent) = (128u32, 400.0f32);
    let h = hill(res, extent);
    let mut volume = VoxelVolume::new(BaseField::new(&h, res, extent), 1.0);

    // Sculpt: pull a lip out of the hillside, then relax it.
    let mut clay = Stroke::new(Brush::Clay, Vec3::new(30.0, 118.0, 0.0), 14.0);
    clay.strength = 1.0;
    clay.normal = Vec3::new(0.6, 0.8, 0.0).normalize();
    brush::apply(&mut volume, &clay, None);

    let mut relax = Stroke::new(Brush::Smooth, Vec3::new(30.0, 118.0, 0.0), 14.0);
    relax.strength = 0.6;
    brush::apply(&mut volume, &relax, None);
    assert!(volume.delta.brick_count() > 0, "sculpting stored nothing");

    // Carve.
    volume.stack.push(Modifier::carve(
        "passage",
        Shape::Tube(Tube::straight(Vec3::new(-70.0, 112.0, 0.0), Vec3::new(70.0, 112.0, 0.0), 7.0)),
        1.0,
    ));

    // Select chunks around the cave mouth, pruning anything with no surface in
    // it -- the step that keeps solid rock and open sky off the draw list.
    let root = Node { min: Vec3::new(-128.0, 64.0, -128.0), size: 256.0, lod: 3 };
    let cfg = LodConfig { chunk_dim: 16, max_depth: 3, detail: 2.0 };
    let occupied = |n: &Node| {
        // Cheap conservative test: sample the node's corners and centre, and
        // keep it if they disagree about being solid.
        let mut neg = false;
        let mut pos = false;
        for i in 0..9 {
            let p = if i == 8 {
                n.center()
            } else {
                n.min
                    + Vec3::new(
                        if i & 1 != 0 { n.size } else { 0.0 },
                        if i & 2 != 0 { n.size } else { 0.0 },
                        if i & 4 != 0 { n.size } else { 0.0 },
                    )
            };
            if volume.sample(p) < 0.0 { neg = true } else { pos = true }
        }
        neg && pos
    };
    let selected = lod::select(root, Vec3::new(-80.0, 112.0, 0.0), &cfg, None, &occupied);
    assert!(!selected.is_empty(), "nothing was selected to draw");

    // Extract each selected chunk and build the indirect draw list, packing
    // into one shared pair of buffers the way the renderer would.
    let mut allocations = Vec::new();
    let (mut first_vertex, mut first_index) = (0u32, 0u32);
    let mut total_triangles = 0;
    for node in &selected {
        let voxel = node.voxel_size(cfg.chunk_dim);
        let grid = SampleGrid::sample(cfg.chunk_dim + 1, voxel, node.min, |p| volume.sample(p));
        let mesh = surface_nets::extract(&grid);
        total_triangles += mesh.triangle_count();
        allocations.push((
            Allocation {
                first_vertex,
                vertex_count: mesh.vertex_count() as u32,
                first_index,
                index_count: mesh.indices.len() as u32,
            },
            true,
        ));
        first_vertex += mesh.vertex_count() as u32;
        first_index += mesh.indices.len() as u32;
    }

    assert!(total_triangles > 0, "the selected chunks extracted no geometry at all");
    let draws = lod::build_draw_list(&allocations);
    assert_eq!(draws.len(), selected.len(), "one argument slot per selected chunk");

    // The draw list must cover every index exactly once, contiguously. A gap
    // or an overlap here draws one chunk's triangles with another's vertices.
    let mut cursor = 0u32;
    for d in draws.iter().filter(|d| !d.is_skipped()) {
        assert_eq!(d.first_index, cursor, "draw ranges are not contiguous");
        cursor += d.index_count;
    }
    assert_eq!(cursor, first_index, "draw list does not cover the whole index buffer");
}

#[test]
fn lod_keeps_the_chunk_count_bounded_as_the_world_grows() {
    // The reason selection is an octree descent. Doubling the world must not
    // double the draw calls -- it should add one ring of coarse chunks.
    let cfg = LodConfig { chunk_dim: 32, max_depth: 6, detail: 2.5 };
    let count = |size: f32, depth: u8| {
        let root = Node { min: Vec3::splat(-size * 0.5), size, lod: depth };
        lod::select(root, Vec3::ZERO, &cfg, None, &|_| true).len()
    };
    let small = count(1024.0, 4);
    let big = count(4096.0, 6);
    assert!(
        big < small * 3,
        "16x the volume produced {big} chunks against {small}; LOD is not holding"
    );
}

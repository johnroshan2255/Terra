// GPU instance culling for scatter.
//
// Phase B of docs/culling.md. Every instance of one species is tested against
// the frustum and a distance threshold; survivors are compacted into a second
// buffer and the count is written straight into the indirect draw arguments.
// The CPU never sees an instance and never learns how many were drawn.
//
// The compaction is an atomic bump on the draw's own `instance_count` field,
// which is what makes this one pass rather than a count pass plus a scan.
// Instance order in the output is therefore arbitrary -- irrelevant for opaque
// geometry, and the reason this cannot be reused for anything alpha-blended.

struct Cull {
    // Six frustum planes in world space, xyz = normal, w = distance.
    planes: array<vec4f, 6>,
    // Last frame's view-projection, for projecting an instance into the Hi-Z pyramid.
    // Last frame's, not this frame's, because the pyramid was built from last frame's
    // depth and testing against a matrix the depths do not correspond to is how
    // occlusion culling starts flickering.
    prev_view_proj: mat4x4f,
    eye: vec4f,
    cull_distance: f32,
    // Bounding radius of the source mesh at unit scale.
    radius: f32,
    count: u32,
    // Instances one output buffer holds.
    capacity: u32,
    // Squared LOD switch distances in xy; z = Hi-Z levels, w = occlusion on.
    lod_bands: vec4f,
    // xy = Hi-Z level-0 size in texels, zw unused.
    hiz_size: vec4f,
};

/// Levels per species. Must match `scatter::LOD_COUNT`.
const LOD_COUNT: u32 = 3u;

// Matches `crate::mesh::Instance`: 32 bytes, held as two `vec4u` rather than as
// named fields.
//
// Naming the fields would be nicer to read and would not match. WGSL aligns
// `vec3<f32>` to 16 bytes in a storage buffer, so a struct starting with the
// position would pad it out to 16 and every field after would sit four bytes
// past where Rust wrote it. Two `vec4u` have the right size, the right
// alignment, and are copied verbatim by the compaction anyway -- only the
// position and the scale are ever read here.
//
//   d[0].xyz  position, 3 x f32 bitcast
//   d[0].w    rotation x,y  -- packed i16 pair, not read here
//   d[1].x    rotation z,w  -- packed i16 pair, not read here
//   d[1].y    scale f16 in the low half, padding in the high half
//   d[1].z    colour, 4 x u8
//   d[1].w    seed
struct Instance {
    d: array<vec4u, 2>,
};

fn inst_pos(i: Instance) -> vec3f {
    return bitcast<vec3f>(i.d[0].xyz);
}

fn inst_scale(i: Instance) -> f32 {
    return unpack2x16float(i.d[1].y).x;
}

// Exactly the layout `draw_indexed_indirect` reads.
struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

/// Instances that could not be written because a band's buffer was full.
///
/// Structurally unreachable: each instance lands in exactly one band and every
/// band's buffer is sized to the whole species, so the three counters sum to at
/// most the instance count. It exists so that a bad switch distance -- or a later
/// change that sizes the buffers smaller -- degrades into missing instances and a
/// log line rather than into an out-of-bounds write.
struct Overflow {
    n: atomic<u32>,
};

@group(0) @binding(0) var<uniform> c: Cull;
// The depth pyramid, built from the previous frame. Group 1, matching
// `HiZ::cull_layout`.
// Read with `textureLoad`, not `textureSample`: the pyramid is `R32Float`, which is not a
// filterable format, and a Hi-Z lookup wants the exact texel covering a footprint rather
// than an interpolation between two of them. That also means no sampler is needed here,
// even though `HiZ::cull_layout` provides one -- an unused layout entry is allowed.
@group(1) @binding(0) var hiz: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> src: array<Instance>;
// One output buffer per LOD. Separate bindings rather than one buffer carved into
// regions: the regions would need a prefix sum over the counts, which is a second
// pass over the instances to save memory that the 32-byte record already saved.
@group(0) @binding(2) var<storage, read_write> dst0: array<Instance>;
@group(0) @binding(3) var<storage, read_write> dst1: array<Instance>;
@group(0) @binding(4) var<storage, read_write> dst2: array<Instance>;
@group(0) @binding(5) var<storage, read_write> args: array<DrawArgs, LOD_COUNT>;
@group(0) @binding(6) var<storage, read_write> ov: Overflow;

/// Metres of slack the occlusion test demands before it will reject an instance.
///
/// Not a depth epsilon: a depth-buffer epsilon is meaningless across a reversed-Z range
/// spanning kilometres. Two metres of *world* distance is enough to absorb the one frame
/// of latency in the pyramid and the coarseness of the level being read, and it is far
/// less than the depth of a ridge -- so a landform still occludes and a bush standing in
/// front of another bush does not.
const OCCLUSION_SLACK_M: f32 = 2.0;

/// Whether an instance's bounding sphere is in front of the depth already recorded over
/// its screen footprint.
///
/// Conservative in both the ways that matter: anything off-screen, behind the camera, or
/// at a level the pyramid does not have is kept rather than rejected. Culling something
/// visible pops it out of the world, which is the failure a user sees.
fn visible_against_hiz(centre: vec3f, radius: f32) -> bool {
    let clip = c.prev_view_proj * vec4f(centre, 1.0);
    // Behind the previous frame's camera, so there is nothing to compare against.
    if (clip.w <= 0.0) {
        return true;
    }
    let ndc = clip.xyz / clip.w;
    // The sphere's screen extent, from its radius projected the same way.
    let edge = c.prev_view_proj * vec4f(centre + vec3f(radius, 0.0, 0.0), 1.0);
    var uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
        return true;
    }
    let extent_ndc = abs(edge.x / max(edge.w, 1e-6) - ndc.x);
    let extent_px = extent_ndc * 0.5 * c.hiz_size.x;

    // The level whose texels cover the footprint, so one fetch answers for the whole
    // sphere rather than needing a loop over its pixels.
    let level = i32(clamp(ceil(log2(max(extent_px * 2.0, 1.0))), 0.0, max(c.lod_bands.z - 1.0, 0.0)));
    // Texel coordinates at that level, from the level's own dimensions rather than by
    // shifting level 0's -- a pyramid over an odd size does not halve exactly.
    let dim = vec2f(textureDimensions(hiz, level));
    let texel = vec2i(clamp(uv * dim, vec2f(0.0), dim - vec2f(1.0)));
    let far_depth = textureLoad(hiz, texel, level).r;
    // Nothing was drawn there last frame, so nothing can be occluding.
    if (far_depth <= 0.0) {
        return true;
    }

    // Reversed-Z: larger is nearer. Compared as metres by turning both back into view
    // distance, because a depth difference means nothing across a kilometre of range.
    let near_plane = 0.1;
    let inst_dist = near_plane / max(ndc.z, 1e-6);
    let occluder_dist = near_plane / max(far_depth, 1e-6);
    return inst_dist - radius - OCCLUSION_SLACK_M <= occluder_dist;
}

/// Write one record into a band's buffer. WGSL has no pointer into a binding, so
/// the band has to be a switch rather than an index.
fn emit(band: u32, slot: u32, inst: Instance) {
    switch (band) {
        case 0u: { dst0[slot] = inst; }
        case 1u: { dst1[slot] = inst; }
        default: { dst2[slot] = inst; }
    }
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= c.count) {
        return;
    }
    let inst = src[i];

    // Position and uniform scale are stored directly now, so neither has to be
    // recovered from a matrix column.
    let centre = inst_pos(inst);
    let radius = c.radius * inst_scale(inst);

    // Distance first: it rejects the most for the least work, and it is the
    // horizontal distance that decides whether a prop is worth drawing --
    // height above or below the camera does not make a tree smaller on screen.
    let d = centre.xz - c.eye.xz;
    if (dot(d, d) > c.cull_distance * c.cull_distance) {
        return;
    }

    // Sphere against each plane. Fully outside any one plane means outside.
    for (var p = 0u; p < 6u; p++) {
        if (dot(c.planes[p].xyz, centre) + c.planes[p].w < -radius) {
            return;
        }
    }

    // --- LOD band ---
    //
    // By the same horizontal distance the cull above used, so an instance cannot
    // be culled by one measure and shaded by another. Compared squared, which is
    // why `Cull::lod_bands` arrives pre-squared.
    //
    // A dithered cross-fade between levels is deliberately not here. When it is
    // added, an instance inside a transition band has to be emitted into *both*
    // adjacent buffers -- so peak total occupancy will exceed the instance count
    // and the buffers will have to grow by the width of the widest band. They are
    // not sized for that now.
    let d2 = dot(d, d);
    var band = 0u;
    if (d2 > c.lod_bands.y) {
        band = 2u;
    } else if (d2 > c.lod_bands.x) {
        band = 1u;
    }

    // --- occlusion ---
    //
    // Hi-Z, against the previous frame's depth. Applied *after* the frustum and distance
    // tests because it is the most expensive of the three and they reject far more.
    //
    // The lesson this carries, recorded in `docs/culling.md`: applied literally -- cull
    // anything behind the farthest surface in its footprint -- this culled seventy per
    // cent of near grass and visibly thinned the field, because **a field of grass is not
    // an occluder**. The depth buffer records only the frontmost blade, and at the coarse
    // levels this test reads, the gaps between blades are gone: grass culls grass. So the
    // comparison is in metres with real slack, which lets a landform occlude what is
    // behind it while foliage no longer occludes itself.
    if (c.lod_bands.w > 0.5 && !visible_against_hiz(centre, radius)) {
        return;
    }

    let slot = atomicAdd(&args[band].instance_count, 1u);
    if (slot >= c.capacity) {
        // Undo, so `instance_count` stays a number the indirect draw can be
        // trusted with. Leaving it over capacity would draw past the buffer.
        atomicSub(&args[band].instance_count, 1u);
        atomicAdd(&ov.n, 1u);
        return;
    }
    emit(band, slot, inst);
}

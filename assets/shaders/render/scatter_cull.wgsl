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
    eye: vec4f,
    cull_distance: f32,
    // Bounding radius of the source mesh at unit scale.
    radius: f32,
    count: u32,
    _pad: u32,
};

// Matches crate::mesh::Instance: a model matrix and a colour.
struct Instance {
    m0: vec4f,
    m1: vec4f,
    m2: vec4f,
    m3: vec4f,
    color: vec4f,
};

// Exactly the layout `draw_indexed_indirect` reads.
struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> c: Cull;
@group(0) @binding(1) var<storage, read> src: array<Instance>;
@group(0) @binding(2) var<storage, read_write> dst: array<Instance>;
@group(0) @binding(3) var<storage, read_write> args: DrawArgs;

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= c.count) {
        return;
    }
    let inst = src[i];

    // Translation is the fourth column; the instance's own scale is the length
    // of the first, since these are rigid transforms with uniform scale.
    let centre = inst.m3.xyz;
    let radius = c.radius * length(inst.m0.xyz);

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

    let slot = atomicAdd(&args.instance_count, 1u);
    dst[slot] = inst;
}

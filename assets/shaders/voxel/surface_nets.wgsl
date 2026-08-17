// Surface Nets isosurface extraction.
//
// Three dispatches over one sampled block, matching the CPU reference in
// terra-voxel/src/surface_nets.rs step for step. The CPU version is the
// oracle: any change here has to keep `gpu_matches_cpu_reference` passing, or
// the editor preview and the baked result drift apart.
//
//   place_vertices   one thread per cell    -> at most one vertex per cell
//   emit_quads       one thread per lattice -> six indices per sign change
//   write_args       one thread             -> the draw_indexed_indirect args
//
// Negative is solid. The vertex pass and the quad pass are separate dispatches
// because the quad pass reads the cell-to-vertex map the vertex pass writes,
// and a workgroup barrier cannot synchronize across the whole grid.

struct Params {
    // Cells per axis. There are dim + 1 samples per axis.
    dim: u32,
    voxel: f32,
    max_vertices: u32,
    max_indices: u32,
    origin: vec3<f32>,
    _pad: f32,
}

struct Vertex {
    position: vec3<f32>,
    _p0: f32,
    normal: vec3<f32>,
    _p1: f32,
}

struct Counters {
    vertex_count: atomic<u32>,
    index_count: atomic<u32>,
    // Sticky flags. A chunk that overflows its allocation must be visibly
    // wrong in a log line, not silently truncated into a hole in the world.
    vertex_overflow: atomic<u32>,
    index_overflow: atomic<u32>,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<f32>;
@group(0) @binding(2) var<storage, read_write> cell_vertex: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: Counters;
@group(0) @binding(6) var<storage, read_write> args: DrawIndexedIndirect;

const NO_VERTEX: u32 = 0xffffffffu;

fn samples_per_axis() -> u32 {
    return params.dim + 1u;
}

fn sample_index(x: u32, y: u32, z: u32) -> u32 {
    let n = samples_per_axis();
    return (z * n + y) * n + x;
}

fn sample_at(x: u32, y: u32, z: u32) -> f32 {
    return samples[sample_index(x, y, z)];
}

fn cell_index(x: u32, y: u32, z: u32) -> u32 {
    let d = params.dim;
    return (z * d + y) * d + x;
}

// Corner offsets: bit 0 is X, bit 1 is Y, bit 2 is Z. Same order as CORNERS in
// the Rust reference, and the edge table below indexes into it.
fn corner(i: u32) -> vec3<u32> {
    return vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
}

// The 12 cell edges as corner-index pairs: four along X, four along Y, four
// along Z. Packed as a pair of nibbles so the table costs one u32 lookup.
const EDGES = array<u32, 12>(
    0x01u, 0x23u, 0x45u, 0x67u,
    0x02u, 0x13u, 0x46u, 0x57u,
    0x04u, 0x15u, 0x26u, 0x37u,
);

// --- trilinear field access, for gradient normals -------------------------

fn lerped(p: vec3<f32>) -> f32 {
    let n = f32(params.dim);
    let c = clamp(p, vec3<f32>(0.0), vec3<f32>(n));
    let b = floor(c);
    let f = c - b;
    let lim = params.dim;
    let x0 = u32(b.x);
    let y0 = u32(b.y);
    let z0 = u32(b.z);

    let g000 = sample_at(min(x0, lim), min(y0, lim), min(z0, lim));
    let g100 = sample_at(min(x0 + 1u, lim), min(y0, lim), min(z0, lim));
    let g010 = sample_at(min(x0, lim), min(y0 + 1u, lim), min(z0, lim));
    let g110 = sample_at(min(x0 + 1u, lim), min(y0 + 1u, lim), min(z0, lim));
    let g001 = sample_at(min(x0, lim), min(y0, lim), min(z0 + 1u, lim));
    let g101 = sample_at(min(x0 + 1u, lim), min(y0, lim), min(z0 + 1u, lim));
    let g011 = sample_at(min(x0, lim), min(y0 + 1u, lim), min(z0 + 1u, lim));
    let g111 = sample_at(min(x0 + 1u, lim), min(y0 + 1u, lim), min(z0 + 1u, lim));

    let c00 = mix(g000, g100, f.x);
    let c10 = mix(g010, g110, f.x);
    let c01 = mix(g001, g101, f.x);
    let c11 = mix(g011, g111, f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn gradient(p: vec3<f32>) -> vec3<f32> {
    let h = 0.5;
    return vec3<f32>(
        lerped(p + vec3<f32>(h, 0.0, 0.0)) - lerped(p - vec3<f32>(h, 0.0, 0.0)),
        lerped(p + vec3<f32>(0.0, h, 0.0)) - lerped(p - vec3<f32>(0.0, h, 0.0)),
        lerped(p + vec3<f32>(0.0, 0.0, h)) - lerped(p - vec3<f32>(0.0, 0.0, h)),
    );
}

// --- pass 1: one vertex per crossed cell ----------------------------------

@compute @workgroup_size(4, 4, 4)
fn place_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = params.dim;
    if (gid.x >= d || gid.y >= d || gid.z >= d) {
        return;
    }
    let ci = cell_index(gid.x, gid.y, gid.z);
    cell_vertex[ci] = NO_VERTEX;

    var s: array<f32, 8>;
    var neg = false;
    var pos = false;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let c = corner(i);
        let v = sample_at(gid.x + c.x, gid.y + c.y, gid.z + c.z);
        s[i] = v;
        if (v < 0.0) { neg = true; } else { pos = true; }
    }
    // No sign change means the surface misses this cell entirely. This is the
    // early out that keeps vertex density at one per crossed cell.
    if (!(neg && pos)) {
        return;
    }

    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var e = 0u; e < 12u; e = e + 1u) {
        let packed = EDGES[e];
        let ia = packed >> 4u;
        let ib = packed & 0xfu;
        let sa = s[ia];
        let sb = s[ib];
        if ((sa < 0.0) == (sb < 0.0)) {
            continue;
        }
        // Signs differ, so sa - sb cannot be zero.
        let t = sa / (sa - sb);
        let ca = vec3<f32>(corner(ia));
        let cb = vec3<f32>(corner(ib));
        sum = sum + ca + (cb - ca) * t;
        count = count + 1.0;
    }
    if (count == 0.0) {
        return;
    }

    let local = sum / count;
    let lattice = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) + local;

    let vi = atomicAdd(&counters.vertex_count, 1u);
    if (vi >= params.max_vertices) {
        atomicStore(&counters.vertex_overflow, 1u);
        return;
    }

    var v: Vertex;
    v.position = params.origin + lattice * params.voxel;
    let g = gradient(lattice);
    let len = length(g);
    v.normal = select(vec3<f32>(0.0, 1.0, 0.0), g / len, len > 1e-12);
    v._p0 = 0.0;
    v._p1 = 0.0;
    vertices[vi] = v;
    cell_vertex[ci] = vi;
}

// --- pass 2: one quad per lattice edge that changes sign ------------------

fn cell_vertex_at(x: u32, y: u32, z: u32) -> u32 {
    return cell_vertex[cell_index(x, y, z)];
}

fn emit_quad(a: u32, b: u32, c: u32, d: u32, flip: bool) {
    if (a == NO_VERTEX || b == NO_VERTEX || c == NO_VERTEX || d == NO_VERTEX) {
        return;
    }
    var v0 = a;
    var v1 = b;
    var v2 = c;
    var v3 = d;
    if (flip) {
        v1 = d;
        v3 = b;
    }
    let base = atomicAdd(&counters.index_count, 6u);
    if (base + 6u > params.max_indices) {
        atomicStore(&counters.index_overflow, 1u);
        return;
    }
    indices[base + 0u] = v0;
    indices[base + 1u] = v1;
    indices[base + 2u] = v2;
    indices[base + 3u] = v0;
    indices[base + 4u] = v2;
    indices[base + 5u] = v3;
}

@compute @workgroup_size(4, 4, 4)
fn emit_quads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = params.dim;
    if (gid.x > d || gid.y > d || gid.z > d) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    let s0 = sample_at(x, y, z);

    // +X edge. Ordering Y then Z is counter-clockwise seen from +X.
    if (x < d && y >= 1u && z >= 1u && y < d && z < d) {
        let s1 = sample_at(x + 1u, y, z);
        if ((s0 < 0.0) != (s1 < 0.0)) {
            emit_quad(
                cell_vertex_at(x, y - 1u, z - 1u),
                cell_vertex_at(x, y, z - 1u),
                cell_vertex_at(x, y, z),
                cell_vertex_at(x, y - 1u, z),
                s0 >= 0.0,
            );
        }
    }

    // +Y edge. Z then X is counter-clockwise seen from +Y.
    if (y < d && x >= 1u && z >= 1u && x < d && z < d) {
        let s1 = sample_at(x, y + 1u, z);
        if ((s0 < 0.0) != (s1 < 0.0)) {
            emit_quad(
                cell_vertex_at(x - 1u, y, z - 1u),
                cell_vertex_at(x - 1u, y, z),
                cell_vertex_at(x, y, z),
                cell_vertex_at(x, y, z - 1u),
                s0 >= 0.0,
            );
        }
    }

    // +Z edge. X then Y is counter-clockwise seen from +Z.
    if (z < d && x >= 1u && y >= 1u && x < d && y < d) {
        let s1 = sample_at(x, y, z + 1u);
        if ((s0 < 0.0) != (s1 < 0.0)) {
            emit_quad(
                cell_vertex_at(x - 1u, y - 1u, z),
                cell_vertex_at(x, y - 1u, z),
                cell_vertex_at(x, y, z),
                cell_vertex_at(x - 1u, y, z),
                s0 >= 0.0,
            );
        }
    }
}

// --- pass 3: the indirect draw arguments ----------------------------------

@compute @workgroup_size(1)
fn write_args() {
    let n = min(atomicLoad(&counters.index_count), params.max_indices);
    args.index_count = n;
    // Zero indices means an empty chunk. Writing instance_count = 0 rather
    // than skipping the entry keeps argument slots aligned with chunk slots,
    // so a shader can still index per-chunk data by instance_index.
    args.instance_count = select(0u, 1u, n > 0u);
    args.first_index = 0u;
    args.base_vertex = 0;
    args.first_instance = 0u;
}

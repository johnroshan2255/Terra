// Grass placement.
//
// One thread per grid slot, one blade per slot, and the slot's *world* cell
// decides its hash -- so blades are anchored to the ground rather than to the
// camera, and the grid slides underneath a moving view without anything
// swimming.
//
// Concentric rings rather than one grid. A single uniform grid cannot serve
// both ends of the range: fine enough for the near field and it needs millions
// of threads to reach the horizon, coarse enough to reach the horizon and the
// near field is bare. Each ring doubles its spacing and doubles its reach, so
// four of them cover sixty-four times the area of one, and a distance-squared
// thinning term inside each ring makes the hand-off between them continuous
// instead of a visible step.
//
// Survivors are sorted into three levels of detail by distance, each with its
// own indirect draw. Sorting here is what makes the LOD free: a near blade and
// a far blade end up in different draws with different vertex counts, and
// neither pays for the other.

struct GrassU {
    // x,z camera position, z spacing of the innermost ring, w cells per side.
    grid: vec4f,
    eye: vec4f,
    blade: vec4f,
    // x = full-density radius, y = pixels per metre at 1 m, z,w = lean range.
    thinning: vec4f,
    // rgb = the ground albedo blades converge on, w = the terrain's mesh
    // resolution.
    ground: vec4f,
    // x = height res, y = extent, z = time, w = grass layer.
    world: vec4f,
    // x = half width, y = curve, z = tip taper, w = colour variation.
    shape: vec4f,
    planes: array<vec4f, 6>,
    // Carried here rather than bound as the camera group: the camera's uniform
    // belongs to the world's terrain, and this pass already receives everything
    // else it needs through one buffer.
    view_proj: mat4x4f,
};

struct Blade {
    // xyz root position, w scale.
    pos_scale: vec4f,
    // x yaw, y dissolve, z lean, w ground blend.
    params: vec4f,
};

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> g: GrassU;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var splat_a: texture_2d<f32>;
@group(0) @binding(3) var splat_b: texture_2d<f32>;
@group(0) @binding(4) var splat_samp: sampler;
@group(0) @binding(5) var<storage, read_write> blades: array<Blade>;
@group(0) @binding(6) var<storage, read_write> args: array<DrawArgs, 3>;
// Last frame's depth as a min-pyramid, owned by HiZ so a resize rebuilds it
// without this pass having to know. See hiz.wgsl.
@group(1) @binding(0) var hiz: texture_2d<f32>;
@group(1) @binding(1) var hiz_samp: sampler;

/// Blades one level of detail can hold, so each stream knows where its slice of
/// the shared buffer begins. Kept in step with `LOD_CAPACITY` in grass.rs.
const LOD_CAPACITY: u32 = 260000u;
const RINGS: u32 = 4u;

fn hash2(p: vec2i) -> vec2f {
    var h = u32(p.x) * 0x8DA6B343u ^ u32(p.y) * 0xD8163841u;
    h ^= h >> 15u;
    h = h * 0x2545F491u;
    h ^= h >> 13u;
    return vec2f(f32(h & 0xFFFFu), f32((h >> 16u) & 0xFFFFu)) / 65535.0;
}

fn height_at(xz: vec2f) -> f32 {
    let res = u32(g.world.x);
    let uv = xz / g.world.y + 0.5;
    let p = clamp(uv, vec2f(0.0), vec2f(1.0)) * f32(res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = u32(i.x);
    let z = u32(i.y);
    let x1 = min(x + 1u, res - 1u);
    let z1 = min(z + 1u, res - 1u);
    let h00 = heights[z * res + x];
    let h10 = heights[z * res + x1];
    let h01 = heights[z1 * res + x];
    let h11 = heights[z1 * res + x1];
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

/// Height of the ground *as drawn*.
///
/// Not `height_at`. The terrain is rendered as a grid coarser than its own
/// heightfield, so the surface a player sees is the piecewise-linear
/// interpolation between grid corners -- and on eroded ground those chords
/// bridge straight over the incised channels the heightfield contains. A blade
/// rooted at the heightfield therefore starts *below* the visible ground and
/// is never seen at all: the near field came out bare while distant grass, seen
/// edge-on past the silhouette, looked fine.
///
/// Reproducing the mesh exactly -- its cell, its diagonal, its two triangles --
/// puts every root on the surface it will be drawn against.
fn ground_at(xz: vec2f) -> vec3f {
    let cells = g.ground.w;
    let step = g.world.y / cells;
    let p = (xz / g.world.y + 0.5) * cells;
    let i = floor(p);
    let f = p - i;
    let c = (i / cells - 0.5) * g.world.y;

    let h00 = height_at(c);
    let h10 = height_at(c + vec2f(step, 0.0));
    let h01 = height_at(c + vec2f(0.0, step));
    let h11 = height_at(c + vec2f(step, step));

    // The quad's two triangles meet along the 10--01 diagonal; see
    // `build_indices` in terrain.rs. Interpolating across the quad instead
    // would leave a blade a few centimetres out at the cell centres, which is
    // most of a blade.
    var h = h00 + (h10 - h00) * f.x + (h01 - h00) * f.y;
    if (f.x + f.y > 1.0) {
        h = h11 + (h01 - h11) * (1.0 - f.x) + (h10 - h11) * (1.0 - f.y);
    }
    // Gradient of the same mesh, so a blade stands normal to the ground it is
    // standing on rather than to a surface nobody can see.
    return vec3f(h, (h00 + h01 - h10 - h11) * 0.5 / step, (h00 + h10 - h01 - h11) * 0.5 / step);
}

/// Painted weight of the grass layer -- the same splat the terrain shades by,
/// so blades grow exactly where the ground is already green and stop where a
/// path was painted over it. There is no second mask to keep in agreement.
fn grass_weight(xz: vec2f) -> f32 {
    let uv = clamp(xz / g.world.y + 0.5, vec2f(0.0), vec2f(1.0));
    let a = textureSampleLevel(splat_a, splat_samp, uv, 0.0);
    let b = textureSampleLevel(splat_b, splat_samp, uv, 0.0);
    let layer = u32(g.world.w);
    let v = select(b, a, layer < 4u);
    let j = layer % 4u;
    let w = select(select(select(v.w, v.z, j == 2u), v.y, j == 1u), v.x, j == 0u);
    let total = dot(a, vec4f(1.0)) + dot(b, vec4f(1.0));
    // Unpainted ground still grows grass, so a freshly generated world is not
    // bare. Painting is an override, not a prerequisite.
    return select(w / max(total, 1e-5), 1.0, total < 0.001);
}

/// Is this blade behind something substantial?
///
/// The occlusion half of Phase B in docs/culling.md, which was the piece left
/// outstanding. What it is worth having for is terrain: a ridge hides an
/// enormous amount of grass, and every blade behind it otherwise runs a vertex
/// shader and shades fragments that are immediately thrown away.
///
/// The pyramid holds the *minimum* reversed-Z over each region -- the farthest
/// surface drawn there -- so a blade farther than that is behind everything in
/// its own footprint. Applied literally that culls far too much, because a
/// field of blades is not an occluder: the depth buffer records only the
/// frontmost blade, and at the coarse levels this test reads, the gaps between
/// blades are gone. Grass then culls grass, and the field visibly thins.
///
/// So the comparison is made in metres and given real slack. Blades hide each
/// other across a metre or two; a landform hides things by tens. Requiring the
/// blade to be well behind the occluder keeps the case worth having and drops
/// the case that costs coverage.
fn occluded(centre: vec3f, radius: f32) -> bool {
    let clip = g.view_proj * vec4f(centre, 1.0);
    if (clip.w <= radius) {
        return false;
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    // Screen extent of the bounding sphere, used to pick a level coarse enough
    // that a single tap covers the whole footprint. Coarser is the safe way to
    // be wrong: a wider region is more likely to contain something far, and the
    // farthest sample is what protects a visible blade.
    let radius_px = radius * g.thinning.y / clip.w;
    let level = clamp(ceil(log2(max(radius_px * 2.0, 1.0))) + 1.0, 0.0, 11.0);
    let far_depth = textureSampleLevel(hiz, hiz_samp, uv, level).r;
    if (far_depth <= 1e-6) {
        // Nothing was drawn there at all -- open sky.
        return false;
    }

    // Reversed-Z with an infinite far plane is exactly `near / distance`, so
    // `ndc.z * clip.w` recovers the near plane and the whole test converts back
    // into metres.
    let near_plane = ndc.z * clip.w;
    let occluder = near_plane / far_depth;
    let blade = clip.w - radius;
    return blade > occluder + max(3.0, occluder * 0.35);
}

@compute @workgroup_size(64)
fn place(@builtin(global_invocation_id) gid: vec3u) {
    let side = u32(g.grid.w);
    let per_ring = side * side;
    if (gid.x >= per_ring * RINGS) {
        return;
    }
    let ring = gid.x / per_ring;
    let i = gid.x % per_ring;

    // Each ring is twice as coarse and twice as wide as the one inside it.
    let step = f32(1u << ring);
    let spacing = g.grid.z * step;
    let cell = vec2i(i32(i % side), i32(i / side));
    // Snapped to this ring's own spacing, so its cells stay put as the camera
    // moves rather than sliding with it.
    let corner = floor(g.grid.xy / spacing - f32(side) * 0.5);
    let world_cell = vec2i(corner) + cell + vec2i(i32(ring) * 4096, 0);
    let base = (vec2f(corner) + vec2f(cell)) * spacing;

    let r = hash2(world_cell);
    // Pull blades toward a coarse clump centre instead of spreading them
    // evenly. Uniform placement reads as a lawn however dense it is; real grass
    // grows in tufts with thinner ground between them, and that unevenness is
    // most of what separates a field from a carpet.
    let clump_cell = vec2i(floor((vec2f(corner) + vec2f(cell)) / 5.0));
    let clump = (vec2f(clump_cell) + 0.5 + (hash2(clump_cell) - 0.5) * 0.7) * 5.0 * spacing;
    let jitter = base + (r - 0.5) * spacing * 1.8;
    let pos_xz = mix(jitter, clump, 0.32);

    let dist = length(pos_xz - g.eye.xz);
    if (dist > g.eye.w) {
        return;
    }

    // Density falls as 1/d^2, which holds the count of blades per *pixel*
    // roughly constant instead of per square metre -- otherwise almost the
    // whole budget goes to blades landing inside a single pixel.
    //
    // Each ring's grid is already 4^ring sparser, so multiplying by 4^ring is
    // what it takes to reach the target. Where that product exceeds one the
    // ring is too coarse to be the right one and a finer ring is covering this
    // distance, so it bails. That single test is both the ring's inner edge and
    // a seamless hand-off: the coarse ring starts at exactly the density the
    // fine ring has thinned down to.
    let full = g.thinning.x;
    let falloff = min(1.0, (full * full) / max(dist * dist, 1.0));
    let keep = falloff * step * step;
    if (keep > 1.0) {
        return;
    }
    let r2 = hash2(world_cell + vec2i(7919, 104729));
    if (r2.x > keep) {
        return;
    }

    // Only now is it worth touching memory: the tests above have already
    // rejected the great majority of threads.
    let weight = grass_weight(pos_xz);
    if (r2.x > keep * weight) {
        return;
    }

    let ground = ground_at(pos_xz);
    let y = ground.x;
    let normal = normalize(vec3f(ground.y, 1.0, ground.z));
    // Grass does not hold on a cliff. Fading it out across a band rather than
    // cutting at one angle keeps the edge from tracing the heightfield's grid.
    let slope = smoothstep(0.55, 0.75, normal.y);
    if (r.y > slope) {
        return;
    }

    // Height varies per blade *and* per tuft. Per blade alone averages out at
    // any distance and the field flattens into one level surface; the tuft term
    // is what keeps a silhouette once the individual blades stop resolving.
    let tuft = mix(0.72, 1.28, hash2(clump_cell + vec2i(613, 977)).x);
    let scale = mix(0.6, 1.3, r2.y) * tuft * mix(0.7, 1.0, weight);
    let height = g.blade.x * scale;
    let centre = vec3f(pos_xz.x, y + height * 0.5, pos_xz.y);
    let radius = height * 0.6;

    for (var p = 0u; p < 6u; p++) {
        if (dot(g.planes[p].xyz, centre) + g.planes[p].w < -radius) {
            return;
        }
    }
    if (occluded(centre, radius)) {
        return;
    }

    let fade = 1.0 - smoothstep(g.blade.y, g.eye.w, dist);
    let lean = mix(g.thinning.z, g.thinning.w, hash2(world_cell + vec2i(31, 17)).x);

    // Pick the level of detail by distance. The near band is a small fraction
    // of the ground but most of the screen, which is why it can afford the
    // vertices the far band cannot. The far band's boundary is also where the
    // dissolve begins, so it is the only one that needs the discarding shader.
    var lod = 2u;
    if (dist < g.eye.w * 0.18) {
        lod = 0u;
    } else if (dist < g.blade.y) {
        lod = 1u;
    }

    let slot = atomicAdd(&args[lod].instance_count, 1u);
    if (slot >= LOD_CAPACITY) {
        // Hand the slot back. Every thread that overflows subtracts exactly the
        // one it added, so the counter settles at the capacity rather than at
        // however many threads raced past it -- an indirect draw asking for more
        // instances than the buffer holds reads past the end of it.
        atomicSub(&args[lod].instance_count, 1u);
        return;
    }
    blades[lod * LOD_CAPACITY + slot] = Blade(
        vec4f(pos_xz.x, y, pos_xz.y, scale),
        vec4f(r.x * 6.2831853, fade * fade, lean, 1.0 - fade),
    );
}

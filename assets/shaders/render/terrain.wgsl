// Terrain draw.
//
// Vertices are generated from the index buffer alone -- there is no vertex
// buffer. Height comes from a storage buffer sampled in the vertex shader,
// which is what lets a sculpt stroke change the surface with zero CPU upload.
//
// Materials are driven by the erosion solver's own outputs rather than by
// hand-painted masks: where water actually flowed gets riverbed gravel, where
// sediment settled gets soil, and what the water scoured gets bare rock. That
// is why erosion is worth running even for a scene that never shows a river.

struct Camera {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    eye: vec4f,
};

struct TerrainU {
    world_extent: f32,
    height_res: u32,
    grid_res: u32,
    brush_radius: f32,
    brush_center: vec2f,
    brush_active: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> t: TerrainU;
@group(1) @binding(1) var<storage, read> heights: array<f32>;
// Accumulated discharge, log-normalized to 0..1.
@group(1) @binding(2) var<storage, read> flow: array<f32>;
// Net height change, 0.5 = unchanged, >0.5 deposited, <0.5 scoured.
@group(1) @binding(3) var<storage, read> deposition: array<f32>;
// Road carriageway coverage, 1 at the centre fading across the shoulder.
@group(1) @binding(4) var<storage, read> road: array<f32>;
// Wheel ruts, 1 at the bottom of a track. Sub-texel as geometry, so this is
// where ruts actually get rendered.
@group(1) @binding(5) var<storage, read> rut: array<f32>;

fn texel(ix: i32, iz: i32) -> u32 {
    let n = i32(t.height_res);
    // Clamp rather than wrap: wrapping would join the north edge to the south.
    return u32(clamp(iz, 0, n - 1) * n + clamp(ix, 0, n - 1));
}

fn h_at(uv: vec2f) -> f32 {
    let p = uv * f32(t.height_res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let z = i32(i.y);
    return mix(
        mix(heights[texel(x, z)],     heights[texel(x + 1, z)],     f.x),
        mix(heights[texel(x, z + 1)], heights[texel(x + 1, z + 1)], f.x),
        f.y
    );
}

fn flow_at(uv: vec2f) -> f32 {
    let p = uv * f32(t.height_res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let z = i32(i.y);
    return mix(
        mix(flow[texel(x, z)],     flow[texel(x + 1, z)],     f.x),
        mix(flow[texel(x, z + 1)], flow[texel(x + 1, z + 1)], f.x),
        f.y
    );
}

fn road_at(uv: vec2f) -> f32 {
    let p = uv * f32(t.height_res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let z = i32(i.y);
    return mix(
        mix(road[texel(x, z)],     road[texel(x + 1, z)],     f.x),
        mix(road[texel(x, z + 1)], road[texel(x + 1, z + 1)], f.x),
        f.y
    );
}

fn rut_at(uv: vec2f) -> f32 {
    let p = uv * f32(t.height_res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let z = i32(i.y);
    return mix(
        mix(rut[texel(x, z)],     rut[texel(x + 1, z)],     f.x),
        mix(rut[texel(x, z + 1)], rut[texel(x + 1, z + 1)], f.x),
        f.y
    );
}

fn dep_at(uv: vec2f) -> f32 {
    let p = uv * f32(t.height_res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let z = i32(i.y);
    return mix(
        mix(deposition[texel(x, z)],     deposition[texel(x + 1, z)],     f.x),
        mix(deposition[texel(x, z + 1)], deposition[texel(x + 1, z + 1)], f.x),
        f.y
    );
}

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let verts_per_side = t.grid_res + 1u;
    let gx = vi % verts_per_side;
    let gz = vi / verts_per_side;

    let uv = vec2f(f32(gx), f32(gz)) / f32(t.grid_res);
    let xz = (uv - 0.5) * t.world_extent;
    let h = h_at(uv);

    // Central differences in world units. One heightfield texel of UV, not one
    // grid cell -- the grid is coarser, and using its step flattens normals.
    let texel_m = t.world_extent / f32(t.height_res - 1u);
    let d = 1.0 / f32(t.height_res - 1u);
    let hl = h_at(uv - vec2f(d, 0.0));
    let hr = h_at(uv + vec2f(d, 0.0));
    let hd = h_at(uv - vec2f(0.0, d));
    let hu = h_at(uv + vec2f(0.0, d));

    var out: VsOut;
    out.world = vec3f(xz.x, h, xz.y);
    out.normal = normalize(vec3f(hl - hr, 2.0 * texel_m, hd - hu));
    out.uv = uv;
    out.clip = cam.view_proj * vec4f(out.world, 1.0);
    return out;
}

const SUN: vec3f = vec3f(0.42, 0.34, 0.60);
const SKY: vec3f = vec3f(0.235, 0.360, 0.560);

// Linear-space albedos.
const GRASS:    vec3f = vec3f(0.105, 0.150, 0.062);
const DRY_GRASS:vec3f = vec3f(0.190, 0.175, 0.088);
const SOIL:     vec3f = vec3f(0.135, 0.098, 0.062);
const ROCK:     vec3f = vec3f(0.150, 0.147, 0.140);
const SCREE:    vec3f = vec3f(0.205, 0.192, 0.172);
const RIVERBED: vec3f = vec3f(0.115, 0.112, 0.098);
const SNOW:     vec3f = vec3f(0.780, 0.810, 0.860);
// Dry track, and the same earth saturated. Wet mud is roughly half the albedo
// of dry -- that ratio is most of what reads as "wet".
const MUD_DRY:  vec3f = vec3f(0.098, 0.076, 0.052);
const MUD_WET:  vec3f = vec3f(0.042, 0.033, 0.024);

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let slope = 1.0 - clamp(n.y, 0.0, 1.0);

    let wet = flow_at(in.uv);
    // Remap deposition to -1..1: negative scoured, positive settled.
    let dep = (dep_at(in.uv) - 0.5) * 2.0;

    // Base layer by slope. Vegetation holds on gentle ground and gives way to
    // soil, then to bare rock as the face steepens.
    var albedo = GRASS;
    albedo = mix(albedo, SOIL, smoothstep(0.06, 0.26, slope));
    albedo = mix(albedo, ROCK, smoothstep(0.28, 0.55, slope));

    // Where the solver removed material, rock is exposed; where it settled,
    // loose sediment covers whatever was underneath.
    albedo = mix(albedo, ROCK, clamp(-dep, 0.0, 1.0) * 0.55);
    albedo = mix(albedo, SCREE, clamp(dep, 0.0, 1.0) * 0.35);
    // Flat, heavily silted ground reads as dry meadow rather than bare soil.
    let meadow = clamp(dep, 0.0, 1.0) * (1.0 - smoothstep(0.05, 0.20, slope));
    albedo = mix(albedo, DRY_GRASS, meadow * 0.5);

    // Channels override everything: this is where water demonstrably ran.
    albedo = mix(albedo, RIVERBED, smoothstep(0.30, 0.72, wet));

    // --- road surface ---
    //
    // Applied after the natural materials so a track overrides them, but before
    // snow, which should still cover a road at altitude.
    let on_road = road_at(in.uv);
    if (on_road > 0.001) {
        let track = rut_at(in.uv);

        // Puddles collect where a rut is deep and the ground is close to level.
        // Both conditions matter: a rut on a slope drains.
        let level = 1.0 - smoothstep(0.02, 0.10, slope);
        let puddle = smoothstep(0.45, 0.9, track) * level;

        var mud = mix(MUD_DRY, MUD_WET, max(track * 0.55, puddle));
        // Centre strip between the wheel tracks keeps its vegetation on a
        // lightly used road.
        mud = mix(mud, GRASS, (1.0 - smoothstep(0.15, 0.5, track)) * 0.35);

        // Scuffed edge rather than a hard line -- a crisp mud/grass boundary is
        // the clearest tell of a painted-on road.
        let edge = smoothstep(0.0, 0.55, on_road);
        albedo = mix(albedo, mud, edge);
    }

    // Snow high up, and only where it can sit.
    let cold = smoothstep(900.0, 1350.0, in.world.y);
    let flat_enough = 1.0 - smoothstep(0.30, 0.58, slope);
    albedo = mix(albedo, SNOW, cold * flat_enough * 0.9);

    let ndl = clamp(dot(n, normalize(SUN)), 0.0, 1.0);
    let ambient = mix(vec3f(0.055, 0.065, 0.080), SKY * 0.35, clamp(n.y, 0.0, 1.0));
    var color = albedo * (ndl * 1.6 + 0.25) + ambient;

    // Wet ground is darker and glossier. A cheap specular lobe does more for
    // "there is water here" than any amount of extra geometry -- and standing
    // water in a wheel rut is the single strongest cue that a track is mud.
    let puddle_shine = smoothstep(0.45, 0.9, rut_at(in.uv))
        * (1.0 - smoothstep(0.02, 0.10, slope))
        * road_at(in.uv);
    let shine = max(smoothstep(0.45, 0.9, wet), puddle_shine);
    let h_vec = normalize(normalize(SUN) + normalize(cam.eye.xyz - in.world));
    color += vec3f(0.5, 0.55, 0.6) * pow(clamp(dot(n, h_vec), 0.0, 1.0), 48.0) * shine * 0.5;

    // Brush ring, drawn on the surface so the cursor reads at any camera angle.
    if (t.brush_active > 0.5) {
        let dist = distance(in.world.xz, t.brush_center);
        let edge = abs(dist - t.brush_radius);
        let ring = 1.0 - smoothstep(0.0, t.brush_radius * 0.02 + 0.5, edge);
        let fill = 1.0 - smoothstep(0.0, t.brush_radius, dist);
        color = mix(color, vec3f(1.0, 0.85, 0.35), ring * 0.85 + fill * 0.06);
    }

    // Aerial perspective. Cheap, and it does more for perceived scale than any
    // amount of extra geometry.
    let view_dist = length(in.world - cam.eye.xyz);
    let fog = 1.0 - exp(-view_dist * 0.00035);
    color = mix(color, SKY, clamp(fog, 0.0, 1.0));

    return vec4f(linear_to_srgb(color), 1.0);
}

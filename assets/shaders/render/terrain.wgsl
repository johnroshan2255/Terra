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
    // Metres covered by one repeat of a material tile.
    mat_scale_m: f32,
    layer_count: u32,
    grass_layer: u32,
    shadow_grid_res: u32,
    // Three scalars, not a vec3u: a vec3 carries 16-byte alignment and would
    // push the roles array to offset 48, making the block 96 bytes against the
    // 80 the Rust struct packs.
    _pad1: u32,
    // Automatic role per layer, or ROLE_NONE. Two vec4s = eight slots.
    layer_roles: array<vec4u, 2>,
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

// Material layers. rgb albedo + a height, and rg normal + b roughness + a AO.
@group(2) @binding(0) var mat_albedo: texture_2d_array<f32>;
@group(2) @binding(1) var mat_surface: texture_2d_array<f32>;
@group(2) @binding(2) var mat_samp: sampler;
// Cheaper sampler for the normal/roughness/occlusion array -- see material.rs.
@group(2) @binding(3) var mat_samp_fast: sampler;

@group(3) @binding(0) var<uniform> light: Light;
@group(3) @binding(1) var shadow_map: texture_depth_2d_array;
@group(3) @binding(2) var shadow_samp: sampler_comparison;
@group(3) @binding(3) var fog_grid: texture_3d<f32>;
@group(3) @binding(4) var fog_samp: sampler;

// Painted layer weights, four per texture. All-zero means "not painted here".
@group(1) @binding(6) var splat_a: texture_2d<f32>;
@group(1) @binding(7) var splat_b: texture_2d<f32>;
@group(1) @binding(8) var splat_samp: sampler;

// Roles, not layer indices. Which palette slot fills which role is decided at
// load time from the material's name and passed in `t.layer_roles`.
const R_SOIL: u32 = 0u;
const R_GRASS: u32 = 1u;
const R_ROCK: u32 = 2u;
const R_GRAVEL: u32 = 3u;
const R_SNOW: u32 = 4u;
const R_MUD: u32 = 5u;
const R_NONE: u32 = 6u;
const ROLES: u32 = 6u;

/// Palette slots. The array sizes below depend on this.
const MAX_LAYERS: u32 = 8u;

fn role_of_layer(i: u32) -> u32 {
    let v = t.layer_roles[i / 4u];
    switch (i % 4u) {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        case 2u: { return v.z; }
        default: { return v.w; }
    }
}

/// One slot out of the two already-sampled splat texels.
///
/// Pure arithmetic on purpose. Sampling the textures inside this, once per
/// slot, made the fragment shader fetch the same two texels sixteen times.
fn pick8(a: vec4f, b: vec4f, i: u32) -> f32 {
    let v = select(b, a, i < 4u);
    let j = i % 4u;
    return select(select(select(v.w, v.z, j == 2u), v.y, j == 1u), v.x, j == 0u);
}

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


// Albedo now comes from the material layers rather than from constants here;
// the per-layer base colours live in `material.rs` beside the grain they belong
// to, so a material is described in one place.

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

// ---------------------------------------------------------------------------
// Material layers
// ---------------------------------------------------------------------------

struct Surf {
    albedo: vec3f,
    // Detail normal in world space, as an offset added to the geometric normal.
    bump: vec3f,
    rough: f32,
    ao: f32,
    // The layer's own relief at this point. This is what the blend arbitrates.
    height: f32,
};

// Triplanar projection weights. A high exponent keeps the transition tight, so
// flat ground is sampled almost purely top-down and only genuine cliffs pay for
// the blend.
fn tri_weights(n: vec3f) -> vec3f {
    var w = pow(abs(n), vec3f(6.0));
    return w / max(w.x + w.y + w.z, 1e-5);
}

// Sample one layer, projected on all three axes and recombined.
//
// Planar UVs alone would smear the texture into vertical streaks on every
// cliff face -- the single most recognisable failure of a first terrain
// shader. Projecting on all three axes and weighting by the normal costs three
// fetches instead of one and removes the stretching entirely.
fn sample_layer(layer: u32, world: vec3f, n: vec3f, w: vec3f) -> Surf {
    let uv = world / t.mat_scale_m;

    let ax = textureSample(mat_albedo, mat_samp, uv.zy, layer);
    let ay = textureSample(mat_albedo, mat_samp, uv.xz, layer);
    let az = textureSample(mat_albedo, mat_samp, uv.xy, layer);

    let sx = textureSample(mat_surface, mat_samp_fast, uv.zy, layer);
    let sy = textureSample(mat_surface, mat_samp_fast, uv.xz, layer);
    let sz = textureSample(mat_surface, mat_samp_fast, uv.xy, layer);

    let a = ax * w.x + ay * w.y + az * w.z;
    let s = sx * w.x + sy * w.y + sz * w.z;

    // Each projection's tangent normal is lifted into world space by dropping
    // it into the two axes that projection actually spans, then summed. This
    // is a perturbation of the geometric normal, not a replacement for it, so
    // the terrain's own shape always dominates.
    let tx = (sx.xy - 0.5) * 2.0;
    let ty = (sy.xy - 0.5) * 2.0;
    let tz = (sz.xy - 0.5) * 2.0;
    let bump = vec3f(0.0, tx.y, tx.x) * w.x
        + vec3f(ty.x, 0.0, ty.y) * w.y
        + vec3f(tz.x, tz.y, 0.0) * w.z;

    var out: Surf;
    out.albedo = a.rgb;
    out.height = a.a;
    out.bump = bump;
    out.rough = s.b;
    out.ao = s.a;
    return out;
}

// Height-aware blend factor for layer B over layer A.
//
// This is the whole point of the stack. A plain `mix(a, b, mask)` fades two
// materials through a grey no-man's-land that exists in neither; blending by
// height instead lets whichever material physically stands higher occupy each
// texel, so the boundary follows the grass clumps and the gaps between the
// stones. `depth` is the width of the band where both are still in play --
// at 0 the transition is a hard per-texel cut, and too wide is just a fade
// again.
fn height_blend(ha: f32, wa: f32, hb: f32, wb: f32) -> f32 {
    let depth = 0.22;
    let top = max(ha + wa, hb + wb) - depth;
    let ba = max(ha + wa - top, 0.0);
    let bb = max(hb + wb - top, 0.0);
    return bb / max(ba + bb, 1e-5);
}

fn mix_surf(a: Surf, b: Surf, k: f32) -> Surf {
    var out: Surf;
    out.albedo = mix(a.albedo, b.albedo, k);
    out.bump = mix(a.bump, b.bump, k);
    out.rough = mix(a.rough, b.rough, k);
    out.ao = mix(a.ao, b.ao, k);
    out.height = mix(a.height, b.height, k);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let geo_n = normalize(in.normal);
    let slope = 1.0 - clamp(geo_n.y, 0.0, 1.0);

    let wet = flow_at(in.uv);
    // Remap deposition to -1..1: negative scoured, positive settled.
    let dep = (dep_at(in.uv) - 0.5) * 2.0;
    let on_road = road_at(in.uv);
    let track = rut_at(in.uv);

    // --- automatic weights, by role ---
    //
    // The masks decide what kind of ground this is; the height blend below
    // decides where exactly the boundary falls. Keeping those two jobs apart is
    // what lets this stay a handful of smoothsteps.
    var role_w: array<f32, 6>;

    // Soil is the base coat. It never drops to zero, so there is always
    // something underneath for the others to break through.
    role_w[R_SOIL] = 0.45 + 0.35 * smoothstep(0.06, 0.26, slope);

    // Grass holds on gentle ground, gives up on steep ground, and thickens
    // where the solver left sediment.
    let gentle = 1.0 - smoothstep(0.08, 0.34, slope);
    role_w[R_GRASS] = gentle * (0.75 + 0.45 * clamp(dep, 0.0, 1.0));

    // Rock is the steep face, and anything the water scoured back to bare.
    role_w[R_ROCK] = smoothstep(0.26, 0.58, slope) * 1.3 + clamp(-dep, 0.0, 1.0) * 0.8;

    // Loose material: where sediment settled, and along the channels where the
    // water actually ran.
    role_w[R_GRAVEL] = clamp(dep, 0.0, 1.0) * 0.55 + smoothstep(0.30, 0.72, wet) * 1.5;

    // Snow high up, and only where it can sit.
    let cold = smoothstep(900.0, 1350.0, in.world.y);
    role_w[R_SNOW] = cold * (1.0 - smoothstep(0.30, 0.58, slope)) * 2.0;

    // The carriageway. Weighted well above the naturals so a road wins, but
    // still blended by height, which gives the scuffed edge a hard mask never
    // does.
    role_w[R_MUD] = smoothstep(0.0, 0.55, on_road) * 2.2;

    // --- fold in what has been painted ---
    //
    // An unpainted texel reads all-zero and keeps the automatic weights
    // untouched, so a world nobody has painted still looks like the solver
    // built it. Paint takes over in proportion to how much of it is there,
    // which makes a half-strength stroke a genuine blend rather than a switch.
    //
    // Both halves of the palette are fetched once, here, and everything below
    // reads the results. Slots past `layer_count` are never written, so they
    // are zero and need no masking.
    let sp_a = textureSample(splat_a, splat_samp, in.uv);
    let sp_b = textureSample(splat_b, splat_samp, in.uv);
    let paint_sum = dot(sp_a, vec4f(1.0)) + dot(sp_b, vec4f(1.0));
    let paint_k = clamp(paint_sum, 0.0, 1.0);
    let paint_norm = 1.0 / max(paint_sum, 1e-5);

    var w: array<f32, 8>;
    var total = 0.0;
    for (var i = 0u; i < MAX_LAYERS; i++) {
        var auto_w = 0.0;
        if (i < t.layer_count) {
            let role = role_of_layer(i);
            if (role < ROLES) {
                auto_w = role_w[role];
            }
        }
        let hand = pick8(sp_a, sp_b, i) * paint_norm;
        let v = mix(auto_w, hand, paint_k);
        w[i] = v;
        total += v;
    }
    // A palette whose roles leave a texel with nothing at all -- every set
    // named "Ground", say, on a cliff -- would divide by zero below. Fall back
    // to the first slot, which is always the base coat.
    if (total < 1e-4) {
        w[0] = 1.0;
    }

    // --- pick the two strongest ---
    //
    // Sampling every layer would cost six fetches each. Two is what actually
    // shows: a third material at a boundary is below the noise floor, and every
    // terrain renderer worth the name makes this same cut.
    var i0 = 0u;
    var i1 = 0u;
    var w0 = -1.0;
    var w1 = -1.0;
    for (var i = 0u; i < MAX_LAYERS; i++) {
        let wi = w[i];
        if (wi > w0) {
            w1 = w0; i1 = i0;
            w0 = wi; i0 = i;
        } else if (wi > w1) {
            w1 = wi; i1 = i;
        }
    }
    // Normalise against the winner so the blend depth means the same thing
    // regardless of how large the raw weights happened to get.
    let k1 = clamp(w1 / max(w0, 1e-5), 0.0, 1.0);

    let tw = tri_weights(geo_n);
    let s0 = sample_layer(i0, in.world, geo_n, tw);
    let s1 = sample_layer(i1, in.world, geo_n, tw);
    var surf = mix_surf(s0, s1, height_blend(s0.height, 1.0, s1.height, k1));

    // --- tints that are not worth their own layer ---
    //
    // Silted flats read as dry meadow rather than fresh growth, and a channel
    // bed is the gravel layer soaked rather than a different stone.
    let meadow = clamp(dep, 0.0, 1.0) * (1.0 - smoothstep(0.05, 0.20, slope));
    surf.albedo = mix(surf.albedo, surf.albedo * vec3f(1.35, 1.18, 0.72), meadow * 0.5);

    // Puddles collect where a rut is deep and the ground is close to level.
    // Both conditions matter: a rut on a slope drains.
    let level = 1.0 - smoothstep(0.02, 0.10, slope);
    let puddle = smoothstep(0.45, 0.9, track) * level * step(0.001, on_road);
    // Wet earth is roughly half the albedo of dry, and much smoother. That
    // ratio is most of what reads as "wet".
    let soak = max(smoothstep(0.30, 0.72, wet), max(track * 0.55, puddle) * step(0.001, on_road));
    surf.albedo *= mix(1.0, 0.45, soak);
    surf.rough = mix(surf.rough, 0.12, soak);

    // Break up the tiling. Even a well-made texture repeats visibly across a
    // kilometre of ground; a very low frequency wobble in brightness costs one
    // noise evaluation and hides the grid without touching the detail.
    let macro_v = noise2(in.world.xz * 0.004) * 0.5 + 0.5;
    surf.albedo *= mix(0.88, 1.12, macro_v);

    // Contact shadow under the grass.
    //
    // The blades darken toward their own roots, but the ground they stand on
    // knows nothing about them, so the surface between clumps stays fully lit
    // and the field looks like blades resting on a bright floor. Darkening the
    // ambient wherever grass grows costs nothing and is most of what makes the
    // two read as one surface. It uses the same weights the grass pass places
    // from, so the two cannot disagree about where grass is.
    let grass_here = w[t.grass_layer] * (1.0 - smoothstep(0.10, 0.40, slope));
    surf.albedo *= mix(1.0, 0.62, clamp(grass_here, 0.0, 1.0));

    // Detail normal, applied as a perturbation so the landform still leads.
    let n = normalize(geo_n + surf.bump * 0.55);

    let sun_dir = normalize(light.sun_direction.xyz);
    let ndl = clamp(dot(n, sun_dir), 0.0, 1.0);
    let view_depth = length(in.world - cam.eye.xyz);
    let shadow = sun_visibility(in.world, view_depth, ndl);

    // Occlusion belongs to the ambient term: a crevice sees less of the sky,
    // but sunlight reaching it arrives at full strength. Folding it into the
    // direct term as well is the usual shortcut, and it reads as grime.
    let ambient = mix(light.ambient.rgb * 0.55, light.ambient.rgb, clamp(n.y, 0.0, 1.0)) * surf.ao;
    var color = surf.albedo * light.sun_color.rgb * ndl * shadow + surf.albedo * ambient;

    // Specular from the material's own roughness. Standing water in a wheel
    // rut is the single strongest cue that a track is mud, and it now falls
    // out of the roughness channel instead of being a special case.
    let gloss = 1.0 - surf.rough;
    let h_vec = normalize(sun_dir + normalize(cam.eye.xyz - in.world));
    let spec = pow(clamp(dot(n, h_vec), 0.0, 1.0), 4.0 + gloss * gloss * 220.0);
    color += light.sun_color.rgb * spec * gloss * gloss * 0.8 * shadow;

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
    if (fog_enabled()) {
        let uvw = fog_lookup(in.world, cam.eye.xyz, in.clip.xy);
        color = apply_fog(color, textureSampleLevel(fog_grid, fog_samp, uvw, 0.0));
    }

    // Alpha 0: geometry occludes the sun.
    // Linear out: exposure and the sRGB transfer happen once, in the post pass.
    return vec4f(color, 0.0);
}

// --- shadow pass ---
//
// Same vertex work as `vs_main`, but group 0 is bound to a cascade's light
// matrix instead of the camera. Sharing the module means the heightfield can
// never be sampled differently by the two passes, which is how shadows detach
// from the ground they belong to.
@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let verts_per_side = t.shadow_grid_res + 1u;
    let gx = vi % verts_per_side;
    let gz = vi / verts_per_side;
    let uv = vec2f(f32(gx), f32(gz)) / f32(t.shadow_grid_res);
    let xz = (uv - 0.5) * t.world_extent;
    return cam.view_proj * vec4f(xz.x, h_at(uv), xz.y, 1.0);
}

// Grass blades.
//
// One blade per instance, following the approach Sucker Punch described for
// Ghost of Tsushima: the compute pass places and culls, and the vertex shader
// evaluates the blade's shape from almost no data at all. A vertex carries only
// its height along the blade and which edge it is; everything else -- the curve,
// the width, the normal, the wind -- is computed here.
//
// Three levels of detail, chosen per blade by distance. Near blades get eight
// segments of curvature, far blades get two. The alternative, a fixed vertex
// count, spends the same geometry on a blade filling a pixel as on one filling
// the screen.

struct Camera {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    eye: vec4f,
};

struct GrassU {
    grid: vec4f,
    eye: vec4f,
    // x = height, y = fade start, z = wind strength, w = wind speed.
    blade: vec4f,
    // x = full-density radius, y = pixels per metre at 1 m, z,w = lean range.
    thinning: vec4f,
    ground: vec4f,
    // x = height res, y = extent, z = time, w = grass layer.
    world: vec4f,
    // x = half width, y = curve, z = tip taper, w = colour variation.
    shape: vec4f,
    planes: array<vec4f, 6>,
    view_proj: mat4x4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> g: GrassU;

@group(2) @binding(0) var<uniform> light: Light;
@group(2) @binding(1) var shadow_map: texture_depth_2d_array;
@group(2) @binding(2) var shadow_samp: sampler_comparison;
@group(2) @binding(3) var fog_grid: texture_3d<f32>;
@group(2) @binding(4) var fog_samp: sampler;

struct VsIn {
    // x = height fraction along the blade, y = -1 or +1 for the edge.
    @location(0) vert: vec2f,
    // xyz root position, w scale.
    @location(1) pos_scale: vec4f,
    // x yaw, y dissolve, z lean, w ground blend.
    @location(2) params: vec4f,
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) tint: vec3f,
    // x = height fraction, y = dissolve.
    @location(3) shade: vec2f,
    @location(4) tangent: vec3f,
};

fn rot_y(v: vec3f, a: f32) -> vec3f {
    let s = sin(a);
    let c = cos(a);
    return vec3f(v.x * c - v.z * s, v.y, v.x * s + v.z * c);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let root = in.pos_scale.xyz;
    let scale = in.pos_scale.w;
    let yaw = in.params.x;
    let lean = in.params.z;
    let t = in.vert.x;
    let edge = in.vert.y;

    let len = g.blade.x * scale;

    // Wind. Two scales: a slow swell moving the whole field together, and a
    // faster per-blade flutter so neighbours do not march in lockstep.
    let time = g.world.z * g.blade.w;
    let swell = warp_basis(root.xz * 0.012 + vec2f(time * 0.35, 0.0));
    let phase = fract(sin(dot(root.xz, vec2f(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    let gust = swell + sin(time * 2.3 + phase) * 0.3;

    // Quadratic Bezier from root to tip. The control point carries both the
    // blade's own lean and the wind, so a gust bends the whole curve rather
    // than displacing the tip and leaving a kink at the base.
    let bend = lean + gust * g.blade.z;
    let p0 = vec2f(0.0, 0.0);
    let p1 = vec2f(bend * g.shape.y, 0.55);
    let p2 = vec2f(bend, 0.92);
    let u = 1.0 - t;
    let point = u * u * p0 + 2.0 * u * t * p1 + t * t * p2;
    // Derivative of the same curve: the blade's own direction, for the normal
    // and the anisotropic highlight.
    let deriv = normalize(2.0 * u * (p1 - p0) + 2.0 * t * (p2 - p1));

    // Width holds most of the way up and tapers only near the tip. A quadratic
    // taper spends the top half of every blade below a pixel, and the top half
    // is the silhouette.
    var half_width = g.shape.x * scale * (1.0 - g.shape.z * t * t * t * t);

    // Widen blades that have fallen below the sample grid. A blade thinner than
    // a pixel does not get thinner on screen -- it starts missing pixel centres
    // entirely, so a fraction of the field is simply not drawn.
    let to_cam = distance(root, cam.eye.xyz);
    let px = half_width * 2.0 * g.thinning.y / max(to_cam, 0.5);
    half_width *= clamp(1.1 / max(px, 0.02), 1.0, 6.0);

    // The blade grows in its own plane; yaw turns that plane to face outward.
    let spine = vec3f(point.x * len, point.y * len, 0.0);
    let side = vec3f(0.0, 0.0, half_width * edge);
    let local = rot_y(spine + side, yaw);

    var out: VsOut;
    out.world = root + local;
    out.tangent = normalize(rot_y(vec3f(deriv.x, deriv.y, 0.0), yaw));
    // Face across the blade, perpendicular to both its length and its width.
    out.normal = normalize(rot_y(vec3f(-deriv.y, deriv.x, 0.0), yaw + 1.5707963));
    out.shade = vec2f(t, in.params.y);

    // Colour. A field of one green reads as painted no matter how many blades
    // are in it, so the variation runs at three scales at once -- and in hue,
    // not only in brightness. Brightness alone just looks like uneven lighting;
    // it is the drift between a lush blue-green and a dry straw-green that the
    // eye reads as a living field.
    let vary = g.shape.w;
    let drift = warp_basis(root.xz * 0.035);
    let blade_rand = fract(phase * 0.159);
    let dry = clamp(0.5 + drift * 0.85 + (blade_rand - 0.5) * 0.75, 0.0, 1.0) * vary;
    let lush = vec3f(0.72, 1.06, 0.70);
    let straw = vec3f(1.18, 1.02, 0.56);
    var tint = g.ground.rgb * mix(lush, straw, dry);
    // Dark at the root, bright at the tip. This gradient is most of what gives
    // a dense field depth: without it the mat lights as one flat surface.
    //
    // The tip stays close to the ground's own albedo rather than well above it.
    // A canopy is *darker* than the flat albedo of the same material, because
    // most of what the eye sees is blades shadowing each other -- pushing the
    // tips brighter to make the field read as lush is what turns it neon.
    tint *= mix(0.34, 1.14, t * t * 0.45 + t * 0.55);

    // Base colour matching: as the blade dissolves it converges on the ground's
    // own albedo, so the pixels it stops covering look the same as the ones it
    // covered.
    out.tint = mix(tint, g.ground.rgb, in.params.w);
    out.clip = cam.view_proj * vec4f(out.world, 1.0);
    return out;
}

/// Interleaved gradient noise. Fine and high frequency rather than an ordered
/// grid: no tiling of its own, and it is what temporal accumulation resolves
/// most cleanly.
fn ign(p: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

fn shade(in: VsOut, front: bool) -> vec4f {
    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    // Bend the shading normal toward vertical. A blade is a thin ribbon, and
    // lighting it by its true normal makes a field flicker between lit and
    // unlit as blades turn; leaning the normal up treats the clump as a
    // surface, which is what the eye reads anyway.
    n = normalize(mix(n, vec3f(0.0, 1.0, 0.0), 0.5));

    let sun_dir = normalize(light.sun_direction.xyz);
    let ndl = clamp(dot(n, sun_dir), 0.0, 1.0);
    let dist = length(in.world - cam.eye.xyz);
    let shadow = sun_visibility(in.world, dist, ndl);

    // Occlusion down the blade. This is the whole of "thick": without a dark
    // root the field reads as wires standing on soil rather than as a mat with
    // depth between the blades.
    let ao = mix(0.30, 1.0, in.shade.x * in.shade.x);
    let ambient = light.ambient.rgb * ao;

    var color = in.tint * light.sun_color.rgb * ndl * shadow + in.tint * ambient;

    // Translucency. Grass is thin enough to glow when lit from behind, and it
    // is most of why a real field looks alive at a low sun.
    let back = clamp(dot(-n, sun_dir), 0.0, 1.0);
    color += in.tint * light.sun_color.rgb * pow(back, 3.0) * 0.42 * in.shade.x * shadow;

    // Anisotropic sheen. A blade is a fibre, not a surface: its highlight runs
    // as a band across it rather than as a spot, because the normal sweeps
    // around the length. Kajiya-Kay -- the half vector's angle to the tangent
    // rather than to the normal.
    let view = normalize(cam.eye.xyz - in.world);
    let half_v = normalize(sun_dir + view);
    let th = dot(normalize(in.tangent), half_v);
    let sheen = pow(sqrt(max(1.0 - th * th, 0.0)), 26.0);
    color += light.sun_color.rgb * sheen * 0.20 * in.shade.x * shadow;

    if (fog_enabled()) {
        let uvw = fog_lookup(in.world, cam.eye.xyz, in.clip.xy);
        color = apply_fog(color, textureSampleLevel(fog_grid, fog_samp, uvw, 0.0));
    }
    // Alpha 0: grass occludes the sun, same as any other geometry.
    return vec4f(color, 0.0);
}

/// The near field. No `discard` anywhere in this path, so the tile GPU can
/// reject occluded fragments before shading them -- which is most of the cost
/// of a dense carpet.
@fragment
fn fs_solid(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    return shade(in, front);
}

/// The fade band. Discarding against a screen-space pattern spreads the
/// transition over pixels instead of over time, which is what removes the pop.
@fragment
fn fs_fade(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    if (ign(in.clip.xy) > in.shade.y) {
        discard;
    }
    return shade(in, front);
}

// --- shadow pass ---
@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4f {
    let root = in.pos_scale.xyz;
    let scale = in.pos_scale.w;
    let t = in.vert.x;
    let len = g.blade.x * scale;
    let bend = in.params.z;
    let u = 1.0 - t;
    let point = 2.0 * u * t * vec2f(bend * g.shape.y, 0.55) + t * t * vec2f(bend, 0.92);
    let side = vec3f(0.0, 0.0, g.shape.x * scale * in.vert.y);
    let local = rot_y(vec3f(point.x * len, point.y * len, 0.0) + side, in.params.x);
    return cam.view_proj * vec4f(root + local, 1.0);
}

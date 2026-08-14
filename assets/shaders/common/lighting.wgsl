// Shared light state and shadow lookup.
//
// Prepended to every shading pass, so the sun the terrain is lit by is the same
// sun the sky draws. Three hardcoded `SUN` constants that had to be kept in
// agreement by hand is what this replaces.

struct Light {
    // xyz toward the sun, w = daylight (0 at night).
    sun_direction: vec4f,
    // rgb radiance already scaled by intensity, w = intensity.
    sun_color: vec4f,
    sky_zenith: vec4f,
    sky_horizon: vec4f,
    // rgb ambient, w = exposure.
    ambient: vec4f,
    cascade_view_proj: array<mat4x4f, 3>,
    // Far distance of each cascade.
    cascade_split: vec4f,
    // x = shadows on, y = 1/resolution, z = haze, w = night.
    params: vec4f,
    // x = fog near, y = fog far, z = fog on, w = slices.
    fog: vec4f,
    // xy = 1/viewport.
    fog_screen: vec4f,
};

fn light_split(i: u32) -> f32 {
    switch (i) {
        case 0u: { return light.cascade_split.x; }
        case 1u: { return light.cascade_split.y; }
        default: { return light.cascade_split.z; }
    }
}

/// Fraction of the sun reaching `world`, 1 = fully lit.
///
/// `n_dot_l` scales the depth bias: a surface edge-on to the light needs far
/// more slack than one facing it, and a constant bias either acnes the steep
/// faces or detaches the shadows from the flat ones.
fn sun_visibility(world: vec3f, view_depth: f32, n_dot_l: f32) -> f32 {
    if (light.params.x < 0.5) {
        return 1.0;
    }

    // Pick the tightest cascade that still contains this fragment.
    var cascade = 2u;
    if (view_depth < light_split(0u)) {
        cascade = 0u;
    } else if (view_depth < light_split(1u)) {
        cascade = 1u;
    } else if (view_depth >= light_split(2u)) {
        // Past the last cascade there is no shadow data. Returning lit is the
        // only honest answer; returning shadowed would draw a dark band across
        // the distance.
        return 1.0;
    }

    let clip = light.cascade_view_proj[cascade] * vec4f(world, 1.0);
    let ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > vec2f(1.0)) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    // Clip xy is -1..1 with y up; texture uv is 0..1 with y down.
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    let slope = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = 0.0009 + 0.0045 * slope;
    let depth = ndc.z + bias;

    // 3x3 PCF. The comparison sampler already filters the test results, so
    // this is nine hardware-filtered taps rather than nine manual ones.
    let texel = light.params.y;
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2f(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_samp, uv + o, cascade, depth);
        }
    }
    return sum / 9.0;
}

/// Apply the froxel grid to a shaded colour.
///
/// Replaces the analytic `1 - exp(-d)` fade every pass used to do. That one had
/// no idea whether the sun reached a given point in the air, so it fogged the
/// inside of a shadow exactly as brightly as the sunlit air beside it. This
/// carries both: `rgb` is the light scattered toward the eye along the way, `a`
/// is how much of the surface behind it survives.
/// Where in the froxel grid a fragment sits.
///
/// `frag_coord` is its position in pixels: the grid's x and y are screen space,
/// so the lookup comes from where the fragment actually is rather than from
/// anything uniform across the draw. Deliberately takes no bindings -- the fog
/// compute pass includes this file too, and it cannot name the grid it is in
/// the middle of writing.
fn fog_lookup(world: vec3f, eye: vec3f, frag_coord: vec2f) -> vec3f {
    let dist = length(world - eye);
    // z is distance, distributed exponentially -- the same mapping the compute
    // pass filled the grid with.
    let t = log(max(dist, light.fog.x) / light.fog.x) / log(light.fog.y / light.fog.x);
    return vec3f(frag_coord * light.fog_screen.xy, clamp(t, 0.0, 1.0));
}

fn fog_enabled() -> bool {
    return light.fog.z >= 0.5;
}

/// `v` is the grid sample: rgb is the light scattered toward the eye along the
/// way, a is how much of the surface behind it survives.
fn apply_fog(color: vec3f, v: vec4f) -> vec3f {
    return color * v.a + v.rgb;
}

/// Sky colour along a view ray. Shared so the fog the terrain fades into is
/// the sky it is standing under.
fn sky_color(dir: vec3f) -> vec3f {
    let t = clamp(dir.y, -1.0, 1.0);
    let sun = normalize(light.sun_direction.xyz);

    // Two-stage gradient: the sharp near-horizon band is what reads as
    // atmosphere. A single mix from zenith to horizon looks like a backdrop.
    let mid = mix(light.sky_horizon.rgb, light.sky_zenith.rgb, 0.45);
    var color = mix(light.sky_horizon.rgb, mid, smoothstep(0.0, 0.22, t));
    color = mix(color, light.sky_zenith.rgb, smoothstep(0.18, 0.85, t));

    // Warm scattering toward the sun, strongest near the horizon.
    let sd = clamp(dot(dir, sun), 0.0, 1.0);
    let haze = pow(sd, 6.0) * (1.0 - smoothstep(0.0, 0.55, t));
    color += light.sun_color.rgb * haze * 0.35 * light.params.z;

    // Disc plus bloom. The moon gets a smaller, colder one.
    let disc = select(900.0, 2400.0, light.params.w > 0.5);
    color += light.sun_color.rgb * pow(sd, disc) * 1.6;
    color += light.sun_color.rgb * pow(sd, 48.0) * 0.30 * light.params.z;

    return color;
}

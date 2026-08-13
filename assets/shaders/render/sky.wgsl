// Sky dome, drawn as a fullscreen triangle before the terrain.
//
// Depth test is Always with writes off, so it fills every pixel the terrain
// does not cover without interfering with reversed-Z depth.

struct Camera {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    eye: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Single oversized triangle -- no vertex buffer, and no diagonal seam the
    // way a two-triangle quad has.
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    let p = uv * 2.0 - 1.0;

    var out: VsOut;
    out.clip = vec4f(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

const SUN_DIR: vec3f = vec3f(0.42, 0.34, 0.60);

const ZENITH:  vec3f = vec3f(0.055, 0.115, 0.255);
const MID:     vec3f = vec3f(0.235, 0.360, 0.560);
const HORIZON: vec3f = vec3f(0.640, 0.660, 0.660);
const GROUND:  vec3f = vec3f(0.055, 0.060, 0.075);

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // Reconstruct the view ray. Depth 1.0 is the near plane under reversed-Z.
    let p = cam.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
    let dir = normalize(p.xyz / p.w - cam.eye.xyz);

    let t = clamp(dir.y, -1.0, 1.0);
    let sun = normalize(SUN_DIR);

    // Two-stage gradient: the sharp near-horizon band is what reads as
    // atmosphere. A single mix from zenith to horizon looks like a backdrop.
    var color = mix(HORIZON, MID, smoothstep(0.0, 0.22, t));
    color = mix(color, ZENITH, smoothstep(0.18, 0.85, t));

    // Warm scattering toward the sun, strongest near the horizon.
    let sd = clamp(dot(dir, sun), 0.0, 1.0);
    let haze = pow(sd, 6.0) * (1.0 - smoothstep(0.0, 0.55, t));
    color += vec3f(0.42, 0.26, 0.12) * haze;

    // Sun disc plus bloom.
    color += vec3f(1.0, 0.94, 0.82) * pow(sd, 900.0) * 1.4;
    color += vec3f(0.55, 0.44, 0.30) * pow(sd, 48.0) * 0.55;

    // Below the horizon, fade to the ground haze the terrain fogs into.
    color = mix(color, GROUND, smoothstep(0.0, -0.10, t));

    // Vignette. Cheap, and it is most of what makes UI over a 3D scene
    // readable without dimming the whole viewport.
    let v = 1.0 - 0.30 * dot(in.ndc, in.ndc);
    color *= clamp(v, 0.0, 1.0);

    return vec4f(linear_to_srgb(color), 1.0);
}

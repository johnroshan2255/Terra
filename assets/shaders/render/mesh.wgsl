// Instanced solid meshes -- the car body and its wheels.
//
// The first geometry in this renderer to use an actual vertex buffer; terrain
// and sky are both generated from indices alone. Instances carry their own
// model matrix and colour, so one draw covers a chassis and four wheels.

struct Camera {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    eye: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    // Model matrix, one column per location.
    @location(2) m0: vec4f,
    @location(3) m1: vec4f,
    @location(4) m2: vec4f,
    @location(5) m3: vec4f,
    @location(6) color: vec4f,
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4f(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4f(in.position, 1.0);

    // Rotation only for the normal. These are rigid transforms with uniform
    // scale, so the upper 3x3 is safe to use directly -- no inverse-transpose.
    let rot = mat3x3f(in.m0.xyz, in.m1.xyz, in.m2.xyz);

    var out: VsOut;
    out.world = world.xyz;
    out.normal = normalize(rot * in.normal);
    out.color = in.color.rgb;
    out.clip = cam.view_proj * world;
    return out;
}

const SUN: vec3f = vec3f(0.42, 0.34, 0.60);
const SKY: vec3f = vec3f(0.235, 0.360, 0.560);

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let ndl = clamp(dot(n, normalize(SUN)), 0.0, 1.0);
    let ambient = mix(vec3f(0.055, 0.065, 0.080), SKY * 0.35, clamp(n.y, 0.0, 1.0));

    var color = in.color * (ndl * 1.6 + 0.25) + ambient * in.color;

    // A tight specular lobe reads as painted metal rather than matte plastic.
    let h = normalize(normalize(SUN) + normalize(cam.eye.xyz - in.world));
    color += vec3f(0.9) * pow(clamp(dot(n, h), 0.0, 1.0), 64.0) * 0.25;

    // Same aerial perspective as the terrain, or the car detaches from the
    // scene as it drives away.
    let dist = length(in.world - cam.eye.xyz);
    let fog = 1.0 - exp(-dist * 0.00035);
    color = mix(color, SKY, clamp(fog, 0.0, 1.0));

    return vec4f(linear_to_srgb(color), 1.0);
}

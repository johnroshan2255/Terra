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

// Per-mesh albedo. Meshes with no map get a 1x1 white texture, so there is one
// pipeline rather than two and no branch in the shader.
@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;
// Fragments below x are discarded; x < 0 disables the test.
@group(1) @binding(2) var<uniform> alpha_cutoff: vec4f;

@group(2) @binding(0) var<uniform> light: Light;
@group(2) @binding(1) var shadow_map: texture_depth_2d_array;
@group(2) @binding(2) var shadow_samp: sampler_comparison;
@group(2) @binding(3) var fog_grid: texture_3d<f32>;
@group(2) @binding(4) var fog_samp: sampler;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    // Model matrix, one column per location.
    @location(3) m0: vec4f,
    @location(4) m1: vec4f,
    @location(5) m2: vec4f,
    @location(6) m3: vec4f,
    @location(7) color: vec4f,
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
    @location(3) uv: vec2f,
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
    out.uv = in.uv;
    out.clip = cam.view_proj * world;
    return out;
}


fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let tex = textureSample(albedo, albedo_samp, in.uv);
    if (alpha_cutoff.x >= 0.0 && tex.a < alpha_cutoff.x) {
        discard;
    }
    // Leaf cards are drawn from both sides; the back of one faces away from
    // its own normal, and without this every other leaf is lit as if in shadow.
    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    let sun_dir = normalize(light.sun_direction.xyz);
    let ndl = clamp(dot(n, sun_dir), 0.0, 1.0);
    let dist = length(in.world - cam.eye.xyz);
    let shadow = sun_visibility(in.world, dist, ndl);
    let ambient = mix(light.ambient.rgb * 0.55, light.ambient.rgb, clamp(n.y, 0.0, 1.0));

    let base = in.color * tex.rgb;
    var color = base * light.sun_color.rgb * ndl * shadow + base * ambient;

    // A tight specular lobe reads as painted metal rather than matte plastic.
    let h = normalize(sun_dir + normalize(cam.eye.xyz - in.world));
    color += light.sun_color.rgb * pow(clamp(dot(n, h), 0.0, 1.0), 64.0) * 0.25 * shadow;

    // Same aerial perspective as the terrain, or the car detaches from the
    // scene as it drives away.
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
// Alpha-tested, because a cut-out fern casting the shadow of its bounding
// quads is worse than no shadow at all.
@vertex
fn vs_shadow(in: VsIn) -> VsOut {
    let model = mat4x4f(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4f(in.position, 1.0);
    var out: VsOut;
    out.world = world.xyz;
    out.normal = in.normal;
    out.color = in.color.rgb;
    out.uv = in.uv;
    out.clip = cam.view_proj * world;
    return out;
}

@fragment
fn fs_shadow(in: VsOut) {
    let tex = textureSample(albedo, albedo_samp, in.uv);
    if (alpha_cutoff.x >= 0.0 && tex.a < alpha_cutoff.x) {
        discard;
    }
}

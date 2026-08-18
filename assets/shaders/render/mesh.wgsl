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
    // The instance, 32 bytes. Not a model matrix: every instance drawn here is a
    // rigid transform with uniform scale, so a quaternion, a scalar and a
    // position say the same thing in a fifth of the space -- which is what makes
    // one instance buffer per LOD affordable. See `mesh::Instance`.
    @location(3) i_pos: vec3f,
    // Snorm16x4, so this arrives already decoded to -1..1.
    @location(4) i_rot: vec4f,
    // Float16x2 over the f16 scale and its padding; only x is meaningful.
    @location(5) i_scale: vec2f,
    // Unorm8x4, already 0..1.
    @location(6) i_color: vec4f,
    @location(7) i_seed: u32,
};

/// Rotation matrix from a unit quaternion.
///
/// The instance stores i16 snorm components, which round-trip to very slightly
/// off unit length; normalising here rather than trusting the encoding is one
/// `inverseSqrt` and removes the question of whether the scale is exactly what
/// was asked for.
fn quat_to_mat3(q_in: vec4f) -> mat3x3f {
    let q = normalize(q_in);
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    return mat3x3f(
        vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        vec3f(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
        vec3f(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)),
    );
}

/// The instance's world transform, as a rotation basis and a translation.
///
/// Returned as a 3x3 plus a vec3 rather than a mat4x4 because the normal needs
/// the rotation on its own, and building a 4x4 only to pull the corner back out
/// is what the old layout did.
struct InstanceXform {
    rot: mat3x3f,
    scale: f32,
    pos: vec3f,
};

fn instance_xform(in: VsIn) -> InstanceXform {
    var x: InstanceXform;
    x.rot = quat_to_mat3(in.i_rot);
    x.scale = in.i_scale.x;
    x.pos = in.i_pos;
    return x;
}

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
    @location(3) uv: vec2f,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let x = instance_xform(in);
    let world = x.pos + x.rot * (in.position * x.scale);

    var out: VsOut;
    out.world = world;
    // Rotation only for the normal. The scale is uniform, so it drops out under
    // the normalize and there is no inverse-transpose to take.
    out.normal = normalize(x.rot * in.normal);
    out.color = in.i_color.rgb;
    out.uv = in.uv;
    out.clip = cam.view_proj * vec4f(world, 1.0);
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
    let x = instance_xform(in);
    let world = x.pos + x.rot * (in.position * x.scale);
    var out: VsOut;
    out.world = world;
    out.normal = in.normal;
    out.color = in.i_color.rgb;
    out.uv = in.uv;
    out.clip = cam.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn fs_shadow(in: VsOut) {
    let tex = textureSample(albedo, albedo_samp, in.uv);
    if (alpha_cutoff.x >= 0.0 && tex.a < alpha_cutoff.x) {
        discard;
    }
}

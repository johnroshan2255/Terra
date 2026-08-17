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

@group(1) @binding(0) var<uniform> light: Light;
@group(1) @binding(1) var shadow_map: texture_depth_2d_array;
@group(1) @binding(2) var shadow_samp: sampler_comparison;
@group(1) @binding(3) var fog_grid: texture_3d<f32>;
@group(1) @binding(4) var fog_samp: sampler;

// The half-res cloud buffer: rgb scattered radiance, a transmittance. Sampled
// rather than marched here -- see `clouds.wgsl` for why the march moved out.
@group(3) @binding(0) var cloud_tex: texture_2d<f32>;
@group(3) @binding(1) var cloud_samp: sampler;

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

    // Single-scattered atmosphere from the mixer's coefficients, not a
    // gradient: see `atmosphere.wgsl`. `atmosphere_enabled` falls back to the
    // old two-stage gradient so switching Sky Atmosphere off is still a sky.
    let sun = normalize(env.sun_direction.xyz);
    var color: vec3f;
    if (env.mie.w > 0.5) {
        color = atmosphere(cam.eye.xyz, dir, sun);
        color += sun_disc(dir, sun);
    } else {
        color = sky_color(dir);
    }

    // Clouds, composited so they dim what is behind them rather than being
    // painted over it. Bilinear from the half-res buffer: the layer is soft and
    // 2 km out, so the upsample costs nothing visible.
    let cuv = vec2f(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    let cl = textureSampleLevel(cloud_tex, cloud_samp, cuv, 0.0);
    color = color * cl.a + cl.rgb;

    // Below the horizon, fade to the ground haze the terrain fogs into.
    // Widened from -0.10: a tenth of a unit in `dir.y` is under six degrees, and
    // the transition read as a drawn line across the frame rather than as the
    // ground haze the terrain fogs into.
    color = mix(color, env.ambient_ground.rgb + env.ambient_horizon.rgb * 0.35,
        smoothstep(0.0, -0.28, dir.y));

    // The sky is behind everything, so it takes the fog of the entire column.
    if (fog_enabled()) {
        let uvw = fog_lookup(cam.eye.xyz + dir * light.fog.y, cam.eye.xyz, in.clip.xy);
        color = apply_fog(color, textureSampleLevel(fog_grid, fog_samp, uvw, 0.0));
    }

    // Vignette. Cheap, and it is most of what makes UI over a 3D scene
    // readable without dimming the whole viewport.
    let v = 1.0 - 0.30 * dot(in.ndc, in.ndc);
    color *= clamp(v, 0.0, 1.0);

    // Alpha 1: nothing occludes the sun here, so this is where god rays draw their light from.
    // Linear out: exposure and the sRGB transfer happen once, in the post pass.
    return vec4f(color, 1.0);
}

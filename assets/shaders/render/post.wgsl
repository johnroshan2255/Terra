// God rays and the final resolve.
//
// Volumetric scattering as a post-process: march from each pixel toward the
// sun's screen position, accumulating whatever light the scene left unoccluded.
// The occlusion test is the scene's own alpha -- 1 where sky, 0 on geometry --
// so a ridge or a tree trunk carves the shaft without any extra pass.
//
// The march runs at half resolution because a light shaft has no high-frequency
// detail; running it at full res costs four times as much to produce an image
// that blurs to the same thing.

struct Post {
    // Sun position in UV space, and how much of the effect to apply.
    sun_uv: vec2f,
    strength: f32,
    // 0 when the sun is behind the camera or below the horizon.
    // Named `enabled`: `active` is a reserved WGSL keyword.
    enabled: f32,
    // Exposure, and the sRGB switch for the resolve.
    exposure: f32,
    density: f32,
    decay: f32,
    _pad: f32,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;
@group(0) @binding(2) var<uniform> post: Post;
// Half-resolution copy of the scene, marched instead of the full one. Stepping
// a 32-tap ray across a full-res texture reads one bilinear tap per step and
// skips most of what is between them, which shimmers on every high-contrast
// edge -- and sits after the temporal resolve, so nothing downstream can fix it.
@group(2) @binding(0) var source: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Same oversized triangle as the sky: no vertex buffer, no seam.
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.clip = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    // Clip y is up, texture v is down.
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    return out;
}

/// Box-filter the scene down to the march's resolution.
///
/// Sampling at the centre of each 2x2 block makes one bilinear tap return the
/// exact average of the four texels.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4f {
    return textureSampleLevel(scene, scene_samp, in.uv, 0.0);
}

/// Accumulate scattered light along the ray to the sun.
@fragment
fn fs_rays(in: VsOut) -> @location(0) vec4f {
    if (post.enabled < 0.5) {
        return vec4f(0.0);
    }
    const STEPS: i32 = 32;

    // Step toward the sun, shortening as we go so the near end is dense.
    var uv = in.uv;
    let delta = (in.uv - post.sun_uv) * (post.density / f32(STEPS));
    var illumination = 1.0;
    var accum = vec3f(0.0);

    for (var i = 0; i < STEPS; i++) {
        uv -= delta;
        let s = textureSampleLevel(source, scene_samp, uv, 0.0);
        // Alpha is the sky mask: geometry contributes nothing, which is what
        // makes the shafts stop at the silhouette rather than glowing through.
        accum += s.rgb * s.a * illumination;
        illumination *= post.decay;
    }
    return vec4f(accum / f32(STEPS), 1.0);
}

/// Narkowicz's fit to the ACES filmic curve.
fn aces(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@group(1) @binding(0) var rays: texture_2d<f32>;
@group(1) @binding(1) var rays_samp: sampler;

/// Composite, expose and encode. The one place the sRGB transfer is applied,
/// so no pass can double-convert.
@fragment
fn fs_resolve(in: VsOut) -> @location(0) vec4f {
    var color = textureSampleLevel(scene, scene_samp, in.uv, 0.0).rgb;

    if (post.enabled >= 0.5) {
        // Bilinear upsample from the half-res march. A shaft has no edge to
        // preserve, so nothing more careful is warranted.
        let shafts = textureSampleLevel(rays, rays_samp, in.uv, 0.0).rgb;
        color += shafts * post.strength;
    }

    color *= post.exposure;

    // ACES filmic, rather than Reinhard.
    //
    // Plain Reinhard maps 1.0 to 0.5: a correctly exposed white comes out mid
    // grey, and everything below it is compressed toward the same place. The
    // result is a scene that is technically correct and looks flat at every
    // time of day, which is exactly what it was doing. This curve keeps its
    // toe and shoulder but leaves the midtones alone -- 0.18 in comes out at
    // 0.27, and white lands at 0.80 rather than half way.
    color = aces(color);

    return vec4f(linear_to_srgb(color), 1.0);
}

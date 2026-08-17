// Volumetric clouds, marched at half resolution and accumulated over frames.
//
// The march itself lives in `atmosphere.wgsl`; this is the pass that makes it
// affordable. Measured at 1280x720, marching every pixel every frame cost 10.8 ms
// on Medium against a 5 ms budget for the whole renderer. Two standard fixes,
// together worth about 8x:
//
//   half resolution      4x fewer pixels, and a cloud layer 2 km away carries
//                        almost no detail a full-res pixel could show
//   temporal accumulation blends with the reprojected previous result, which
//                        smooths the march's noise and carries detail forward
//
// # Reprojection
//
// History is looked up by reprojecting the *slab entry point* -- where the view
// ray first enters the cloud layer -- through the previous frame's view-projection.
// Reprojecting by direction alone is simpler and handles rotation, but the layer
// sits 1.5-4 km out and the camera moves at up to 120 m/s, so translation
// parallax is large enough to smear. Using a real world position costs one
// matrix multiply and fixes it.
//
// History is rejected when the reprojected point falls outside the previous frame,
// which is what stops a camera turn from dragging stale cloud in from the edges.

struct Camera {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    eye: vec4f,
};

struct Reproject {
    prev_view_proj: mat4x4f,
    /// x = frame index, y = 1 when history is valid, zw = target size in pixels.
    params: vec4f,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<uniform> reproj: Reproject;
@group(1) @binding(1) var history: texture_2d<f32>;
@group(1) @binding(2) var history_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.clip = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y: clip space is +Y up, texture space is +Y down.
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    return out;
}

// There is deliberately no sub-pixel jitter here.
//
// A 2x2 rotated grid was tried, to recover the detail half resolution gives up.
// It shook: the accumulated buffer still carried a quarter-pixel of the newest
// offset, the sky samples that buffer through its own TAA jitter, and the layer
// slid a fraction of a pixel every frame. Cloud edges are soft and the TAA
// resolve already antialiases them, so a stable buffer is worth more than the
// detail the jitter bought.
//
// The temporal blend below is kept for what it is actually good at -- smoothing
// the march's noise and carrying detail forward as the camera moves.

/// Where the view ray first enters the cloud slab, for reprojection.
///
/// Returns the world position, or the eye when the ray misses the layer -- a
/// miss contributes no colour, so its history lookup does not matter.
fn slab_entry(eye: vec3f, rd: vec3f) -> vec3f {
    let base = env.cloud_params.y;
    let top = base + max(env.cloud_params.z, 1.0);
    if (abs(rd.y) < 1e-5) {
        return eye;
    }
    var t0 = (base - eye.y) / rd.y;
    var t1 = (top - eye.y) / rd.y;
    if (t0 > t1) {
        let tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    t0 = max(t0, 0.0);
    if (t1 <= t0) {
        return eye;
    }
    return eye + rd * t0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let uv = in.uv;

    // Reconstruct the ray. Depth 1.0 is the near plane under reversed-Z.
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = cam.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let rd = normalize(p.xyz / p.w - cam.eye.xyz);
    let sun = normalize(env.sun_direction.xyz);

    // Per-pixel, per-frame march offset. The temporal blend below is what turns
    // the resulting noise back into a smooth image.
    let jit = dither(in.clip.xy, reproj.params.x);

    // rgb scattered radiance, a transmittance.
    let current = clouds(cam.eye.xyz, rd, sun, jit);

    if (reproj.params.y < 0.5) {
        return current;
    }

    // Reproject the slab entry point into the previous frame.
    let world = slab_entry(cam.eye.xyz, rd);
    let prev_clip = reproj.prev_view_proj * vec4f(world, 1.0);
    if (prev_clip.w <= 0.0) {
        return current;
    }
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = vec2f(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    // Off-screen last frame: there is no history to reuse, and reusing the
    // clamped edge is what drags stale cloud in from the border on a camera turn.
    if (any(prev_uv < vec2f(0.0)) || any(prev_uv > vec2f(1.0))) {
        return current;
    }

    let prev = textureSampleLevel(history, history_samp, prev_uv, 0.0);

    // 1/4 new per frame. Slower converges smoother but lags visibly when the
    // time of day is running; faster stops the accumulation from denoising the
    // march at all.
    var blend = 0.25;

    // Reject history that disagrees sharply on transmittance. That is a
    // disocclusion -- a cloud edge has moved across this pixel -- and blending
    // through it is what leaves a comet trail behind every cloud.
    if (abs(prev.a - current.a) > 0.35) {
        blend = 1.0;
    }

    return mix(prev, current, blend);
}

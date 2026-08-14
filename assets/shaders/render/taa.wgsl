// Temporal anti-aliasing.
//
// Each frame is rendered with the projection nudged by a sub-pixel offset, so
// the samples land in different places within each pixel. Accumulating those
// over several frames is what resolves the sub-pixel geometry this renderer is
// now full of -- hundreds of thousands of grass blades about a pixel wide, which
// no amount of density fixes and which crawls under any static sampling.
//
// It is also the other half of the grass dissolve: a dithered fade is designed
// to be integrated over time, and without this it is a stipple.
//
// Reprojection is camera-only. There are no motion vectors, so anything that
// moves on its own -- the car, blades bending in the wind -- reprojects as if it
// were static and would smear. Neighbourhood clamping is what stops that: the
// history is pulled back into the range of colours actually present around the
// pixel this frame, so a stale sample cannot survive as a ghost.

struct Taa {
    // Previous frame's unjittered view-projection.
    prev_view_proj: mat4x4f,
    // This frame's inverse, for rebuilding world position from depth.
    inv_view_proj: mat4x4f,
    // xy = this frame's jitter in NDC, z = history weight, w = enabled.
    params: vec4f,
};

@group(0) @binding(0) var current: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> t: Taa;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.clip = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy);
    let cur = textureLoad(current, px, 0);
    if (t.params.w < 0.5) {
        return cur;
    }

    let size = vec2f(textureDimensions(current));

    // Rebuild this pixel's world position, then ask where it was last frame.
    let depth = textureLoad(depth_tex, px, 0);
    // Reversed-Z: 0 is infinitely far, and there is nothing there to track.
    if (depth <= 0.0) {
        return cur;
    }
    let ndc = vec3f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, depth);
    let world = t.inv_view_proj * vec4f(ndc, 1.0);
    let prev_clip = t.prev_view_proj * vec4f(world.xyz / world.w, 1.0);
    if (prev_clip.w <= 0.0) {
        return cur;
    }
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = vec2f(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    // Off-screen last frame means there is no history to blend.
    if (any(prev_uv < vec2f(0.0)) || any(prev_uv > vec2f(1.0))) {
        return cur;
    }

    // Neighbourhood of the current frame, as an axis-aligned colour box.
    var lo = cur;
    var hi = cur;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = textureLoad(current, clamp(px + vec2i(x, y), vec2i(0), vec2i(size) - 1), 0);
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }

    // Clamping rather than rejecting: a history sample that has drifted outside
    // what is plausible here is pulled to the nearest plausible value, which
    // keeps the accumulated detail instead of throwing the pixel away.
    let hist = clamp(textureSampleLevel(history, samp, prev_uv, 0.0), lo, hi);
    return mix(cur, hist, t.params.z);
}

// Hierarchical depth pyramid.
//
// Each level holds the *minimum* reversed-Z of the region below it. In
// reversed-Z a larger value is nearer, so the minimum over a region is the
// farthest surface drawn there -- which is exactly the test an occlusion query
// needs: anything farther than that is behind everything in its footprint.
//
// Built from last frame's depth buffer, so the answer is one frame stale. That
// is the standard trade and the reason the test carries a margin: an object
// that has just come around a corner may be culled for a single frame, which is
// invisible, whereas reading back this frame's depth would cost a stall.

// Separate layouts bind these: the copy takes only the first, the reduce only
// the second. One group holding both would have the copy pass reading level
// zero while writing it.
@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var src_color: texture_2d<f32>;

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

/// Level zero: the depth buffer, halved.
///
/// The halving is the whole reason this pass exists rather than a plain copy.
/// Reading one texel per output would sample only the top-left quarter of the
/// screen and stretch it over the whole pyramid, so every occlusion test would
/// then be answered by the depth of somewhere else entirely -- which reads as
/// working until the wrong somewhere else happens to be near, and then culls
/// the scene away.
@fragment
fn fs_copy(in: VsOut) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy) * 2;
    let dims = vec2i(textureDimensions(src_depth)) - 1;
    let a = textureLoad(src_depth, min(px, dims), 0);
    let b = textureLoad(src_depth, min(px + vec2i(1, 0), dims), 0);
    let c = textureLoad(src_depth, min(px + vec2i(0, 1), dims), 0);
    let d = textureLoad(src_depth, min(px + vec2i(1, 1), dims), 0);
    return vec4f(min(min(a, b), min(c, d)), 0.0, 0.0, 1.0);
}

/// Every level after: the minimum of the block below.
///
/// Three wide instead of two wherever the level below has an odd dimension.
/// Halving an odd size rounds down, so a plain 2x2 would drop the last row or
/// column entirely -- and dropping a *farther* sample raises the minimum, which
/// is the direction that culls things that are actually visible.
@fragment
fn fs_reduce(in: VsOut) -> @location(0) vec4f {
    let size = vec2i(textureDimensions(src_color, 0));
    let dims = size - 1;
    let extra = size % 2;
    let px = vec2i(in.clip.xy) * 2;

    var m = 1.0e30;
    for (var y = 0; y <= 1 + extra.y; y++) {
        for (var x = 0; x <= 1 + extra.x; x++) {
            m = min(m, textureLoad(src_color, min(px + vec2i(x, y), dims), 0).r);
        }
    }
    return vec4f(m, 0.0, 0.0, 1.0);
}

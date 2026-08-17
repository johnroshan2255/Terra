// Cloud shadows on the ground.
//
// A top-down map of sun transmittance through the cloud layer, sampled by the
// terrain shader by world XZ. Without it, clouds are scenery: they change the sky
// and nothing else, and a heavily overcast day still has hard sunlit ground.
//
// # Why a 2D map and not a shadow cascade
//
// The layer is a slab 1.5-4 km up and the sun ray through it barely changes over
// the tens of metres of terrain relief beneath, so the transmittance at ground
// level is a function of XZ alone to well within what the eye can see. That makes
// it one small texture rather than another cascade, and it costs one texture
// fetch in the terrain shader instead of a second shadow lookup.

struct ShadowRegion {
    /// xy = centre in world XZ, z = side length in metres, w = texels per side.
    params: vec4f,
};

@group(0) @binding(0) var<uniform> region: ShadowRegion;

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
    // Fully lit when there is no layer to shadow with, so switching clouds off
    // does not leave the ground dark.
    if (env.flags.z == 0u || env.cloud_params.x <= 0.0) {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }

    let half = region.params.z * 0.5;
    let world_xz = region.params.xy + (in.uv - 0.5) * region.params.z * vec2f(1.0, 1.0);
    let sun = normalize(env.sun_direction.xyz);

    // A sun on the horizon makes the ray graze the slab for tens of kilometres,
    // which is both unaffordable to march and invisible -- there is no direct
    // light left to shadow at that elevation anyway.
    if (sun.y < 0.05) {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }

    // Start at ground level under this texel and march up the sun ray through
    // the slab. Ground *height* is not known in this pass and does not matter:
    // the slab is kilometres above any terrain, so the segment of sun ray inside
    // it is the same to within a fraction of a texel.
    let base = env.cloud_params.y;
    let top = base + max(env.cloud_params.z, 1.0);
    let ground = vec3f(world_xz.x, 0.0, world_xz.y);
    let t_in = base / sun.y;
    let t_out = top / sun.y;
    let span = t_out - t_in;

    // Eight steps across the slab. This is a soft, low-frequency quantity --
    // a cloud shadow has no sharp edge by the time it reaches the ground -- so
    // it is the cheapest march in the renderer and the one that tolerates the
    // fewest samples.
    let steps = 8;
    let dt = span / f32(steps);
    var depth = 0.0;
    for (var i = 0; i < steps; i = i + 1) {
        let p = ground + sun * (t_in + dt * (f32(i) + 0.5));
        // Shape, not full density: shadow is low-frequency, and the detail
        // octaves cost three times as much for a difference the ground cannot
        // show.
        depth += cloud_shape(p) * env.cloud_params.w * dt;
    }

    // Never fully dark. A cloud shadow on real ground is still lit by the sky,
    // and the sky light term handles that separately -- but clamping here stops
    // a thick layer from producing pure black direct light, which reads as a
    // hole rather than a shadow.
    let transmittance = clamp(exp(-depth), 0.15, 1.0);
    return vec4f(transmittance, 0.0, 0.0, 1.0);
}

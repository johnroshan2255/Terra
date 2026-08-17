// CDLOD patch vertex placement.
//
// The GPU half of `terra-render/src/cdlod.rs`. Its own chunk, prepended to whatever
// pass draws patches, for two reasons:
//
//   * both the colour pass and the shadow pass have to place vertices *identically*
//     -- a caster that disagreed with the shaded surface would put a band of shadow
//     acne along every LOD boundary -- and sharing one function is stronger than
//     two that look alike;
//   * it takes no uniforms and no bindings, so a compute shader can call it with
//     made-up inputs and compare against `Patch::vertex_xz` on the CPU. That test
//     is the only thing standing between this arithmetic and a class of bug whose
//     symptom is a hairline of sky flickering through the ground.

// Must match `cdlod::Patch` byte for byte: 32 bytes, no vec3 anywhere.
struct CdlodPatch {
    origin: vec2f,
    size: f32,
    level: u32,
    morph_start: f32,
    morph_end: f32,
    _pad: vec2f,
};

// Distance metric shared with CPU patch selection.
//
// `eye` is xz = the camera's world XZ, z = its vertical distance to the slab that
// contains all terrain. Horizontal distance with that gap folded in -- exactly what
// `cdlod::Node::distance_to` computes, because a patch selected for one level and
// morphed as if it were at another is a crack, not an approximation.
fn cdlod_dist(world_xz: vec2f, eye: vec3f) -> f32 {
    let d = world_xz - eye.xy;
    return sqrt(dot(d, d) + eye.z * eye.z);
}

// World XZ of vertex `vi` of `p`, morphed toward the parent level.
//
// Odd grid indices slide onto the even ones; the even ones are the parent level's
// own vertices, because a patch is one quadrant of its parent with the same vertex
// count. At `k == 1` the patch is geometrically the parent's quadrant, which is what
// makes a level change invisible instead of a pop.
//
// The factor comes from the *unmorphed* position on purpose. Two patches that share
// an edge must agree about the vertices they share; the unmorphed position is the
// same in both, the morphed one is not until the factors already agree.
fn cdlod_vertex_xz(p: CdlodPatch, vi: u32, quads: u32, eye: vec3f) -> vec2f {
    let verts = quads + 1u;
    let g = vec2f(f32(vi % verts), f32(vi / verts));
    let step = p.size / f32(quads);

    let base = p.origin + g * step;
    // A zero-width band would divide by zero; clamping the span keeps a degenerate
    // patch snapped rather than NaN, and a NaN position takes the whole patch off
    // screen.
    let span = max(p.morph_end - p.morph_start, 1e-3);
    let k = clamp((cdlod_dist(base, eye) - p.morph_start) / span, 0.0, 1.0);

    // fract(g/2)*2 is 1 on odd indices and 0 on even ones.
    let odd = fract(g * 0.5) * 2.0;
    return p.origin + (g - odd * k) * step;
}

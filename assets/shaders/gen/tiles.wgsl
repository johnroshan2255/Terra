// Tier-0 -> tier-1 bake. 2 m/texel out.
//
//   h = bicubic(tier0, uv)                        the eroded global map
//     + ridged_multifractal(p, high_freq) * w     detail below the sim's reach
//
// The detail layer is ridged multifractal at high frequency, matching the base
// basis -- mixing a different basis in here would make the seam between eroded
// and non-eroded scales visible as a change in character.
//
// `w` must fall off with local slope-independent flatness: adding full-strength
// detail to a river bed the sim carefully flattened undoes the erosion result.
//
// TODO: implement.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    // placeholder
}

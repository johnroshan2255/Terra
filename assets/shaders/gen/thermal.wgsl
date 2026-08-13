// Thermal erosion: move material from any texel steeper than the angle of
// repose to its lower neighbours. Pure gather over the 8-neighbourhood.
//
// TODO: implement.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    // placeholder
}

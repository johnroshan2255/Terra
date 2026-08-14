// Volumetric fog, as a camera-aligned froxel grid.
//
// Three passes over a 3D texture whose x and y are screen space and whose z is
// distance from the camera, distributed exponentially so the near slices are
// thin where the detail is:
//
//   inject      density and in-scattered light per cell, the light tested
//               against the sun's shadow map so a ridge casts a real shaft
//               through the air rather than only onto the ground
//   accumulate  march front to back along z, integrating scattering against
//               the transmittance already accumulated
//   sample      every shading pass reads the result by distance
//
// This replaces the analytic `1 - exp(-d)` fog those passes used to apply.
// That fog could only ever be uniform: it had no way to know the sun was
// blocked, so it lit the air inside a shadow exactly as brightly as the air
// outside it.

struct Fog {
    // x = near, y = far, z = slices, w = density scale.
    range: vec4f,
    // x = height falloff, y = mist base height, z = mist strength, w = time.
    mist: vec4f,
    // rgb = albedo of the medium, w = anisotropy g.
    medium: vec4f,
    // xy = 1/viewport, z = ambient scale, w = unused.
    screen: vec4f,
};

/// Distance to the centre of a slice. Exponential, so the first few metres get
/// as many cells as the last few hundred.
fn froxel_distance(slice: f32, f: Fog) -> f32 {
    return f.range.x * pow(f.range.y / f.range.x, slice / f.range.z);
}

/// Inverse, for looking the grid up from a world distance.
fn froxel_slice(dist: f32, f: Fog) -> f32 {
    return f.range.z * log(max(dist, f.range.x) / f.range.x) / log(f.range.y / f.range.x);
}

/// Henyey-Greenstein. Real fog scatters forward far more than backward, which
/// is why looking toward the sun through mist is bright and looking away is
/// flat -- isotropic scattering misses that entirely.
fn phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let d = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (12.566371 * max(d * sqrt(max(d, 1e-4)), 1e-4));
}

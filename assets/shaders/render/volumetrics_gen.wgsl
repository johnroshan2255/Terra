// Injection and accumulation for the froxel grid.

@group(0) @binding(0) var<uniform> light: Light;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
@group(1) @binding(0) var<uniform> fog: Fog;
@group(1) @binding(1) var<uniform> cam: Camera;
@group(1) @binding(2) var injected_out: texture_storage_3d<rgba16float, write>;
// Read back as a sampled texture in the second pass. Storage textures only
// support read-write access at 32 bits per channel, and the grid is half float
// because in-scattering is HDR by nature -- a sunlit cell is far brighter than
// a shadowed one, which is the entire point of the buffer.
@group(2) @binding(0) var injected_in: texture_3d<f32>;
@group(2) @binding(1) var scattered_out: texture_storage_3d<rgba16float, write>;

/// World position at the centre of a froxel.
fn froxel_world(coord: vec3u, dims: vec3u) -> vec3f {
    let uv = (vec2f(coord.xy) + 0.5) / vec2f(dims.xy);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Reversed-Z: depth 1 is the near plane.
    let p = cam.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let dir = normalize(p.xyz / p.w - cam.eye.xyz);
    return cam.eye.xyz + dir * froxel_distance(f32(coord.z) + 0.5, fog);
}

@compute @workgroup_size(8, 8, 1)
fn inject(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(injected_out);
    if (any(id >= dims)) {
        return;
    }
    let world = froxel_world(id, dims);

    // Density: a uniform haze plus mist that pools in the low ground. The
    // exponential term is what makes valleys fill and ridges stay clear, and it
    // is most of what reads as weather rather than as a grey filter.
    let height = max(world.y - fog.mist.y, 0.0);
    let mist = fog.mist.z * exp(-height * fog.mist.x);
    // Slow 3D-ish noise so the medium is not perfectly even. Sampled in world
    // space and scrolled, so it drifts past rather than swimming with the view.
    let drift = vec2f(fog.mist.w * 0.6, fog.mist.w * 0.25);
    let n = warp_basis(world.xz * 0.004 + drift) * 0.5
        + warp_basis(vec2f(world.y, world.x) * 0.009 - drift) * 0.5;
    let density = max(fog.range.w + mist, 0.0) * (0.55 + 0.9 * (n * 0.5 + 0.5));

    // In-scattering. The shadow test is the whole point: air inside a shadow
    // receives no sun, so the lit air beside it becomes a visible shaft.
    let to_eye = normalize(cam.eye.xyz - world);
    let sun = normalize(light.sun_direction.xyz);
    let visibility = sun_visibility(world, length(world - cam.eye.xyz), 1.0);
    let p = phase(dot(-to_eye, sun), fog.medium.w);

    // No 4*pi here. `phase` is already normalised to integrate to one over the
    // sphere -- it carries the 1/4*pi itself -- so multiplying by 4*pi again
    // made every cell scatter twelve times too much light. Integrated across a
    // 450 m column that added ~0.2 of flat grey to every pixel in the frame,
    // which is what made the whole scene look washed out at every time of day.
    var scatter = light.sun_color.rgb * visibility * p;
    // Ambient from the sky, so shadowed fog is dim rather than black.
    scatter += light.ambient.rgb * fog.screen.z;

    textureStore(injected_out, id, vec4f(scatter * fog.medium.rgb * density, density));
}

@compute @workgroup_size(8, 8, 1)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(injected_in);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    var accum = vec3f(0.0);
    var transmittance = 1.0;
    var previous = fog.range.x;

    // Front to back, so each slice is attenuated by everything already between
    // it and the camera. Marching the other way would need the whole column
    // resident before any of it could be weighted.
    for (var z = 0u; z < dims.z; z++) {
        let cell = textureLoad(injected_in, vec3i(vec3u(id.xy, z)), 0);
        let next = froxel_distance(f32(z) + 1.0, fog);
        let thickness = max(next - previous, 0.0);
        previous = next;

        let extinction = max(cell.a, 1e-6);
        let slice_t = exp(-extinction * thickness);
        // Analytic integration of in-scattering across the slice, rather than a
        // point sample times its length: at these slice thicknesses the two
        // differ visibly in the first few metres.
        let integrated = (cell.rgb - cell.rgb * slice_t) / extinction;

        accum += integrated * transmittance;
        transmittance *= slice_t;
        textureStore(scattered_out, vec3u(id.xy, z), vec4f(accum, transmittance));
    }
}

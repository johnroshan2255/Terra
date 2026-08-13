// Grid ("pipe model") hydraulic erosion -- Mei et al. 2007,
// "Fast Hydraulic Erosion Simulation and Visualization on GPU".
//
// Six dispatches per iteration. Every pass is a pure gather -- one thread per
// texel, reading neighbours and writing only its own cell -- so no atomics are
// needed. That matters: WebGPU has no f32 atomics at all, which rules out the
// particle/droplet formulation entirely.
//
// State is split into separate per-field buffers rather than one interleaved
// struct. At 4096^2 an interleaved buffer would exceed WebGPU's default 128 MB
// maxStorageBufferBindingSize; split, the largest single field (flux, 16 B per
// texel) stays under it.
//
// Sediment is the one field that needs ping-ponging: advection reads at a
// back-traced position (i.e. a neighbour) and would otherwise race with its own
// writes. The host alternates bindings 5 and 6 each iteration.

struct Params {
    res: u32,
    dt: f32,
    rain_rate: f32,
    evaporation: f32,
    capacity: f32,
    dissolve: f32,
    deposit: f32,
    min_slope: f32,
    pipe_area: f32,
    gravity: f32,
    cell_size: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> height: array<f32>;
@group(0) @binding(2) var<storage, read_write> water: array<f32>;
// Outflow per cell, in the order (left, right, up, down).
@group(0) @binding(3) var<storage, read_write> flux: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> sed_src: array<f32>;
@group(0) @binding(6) var<storage, read_write> sed_dst: array<f32>;
// Accumulated stream power: speed times depth, integrated over the run.
//
// Not accumulated outflow volume -- under uniform rainfall every cell passes
// roughly its own rain, so that metric has almost no dynamic range and marks
// the entire map as a channel. Speed times depth concentrates sharply where
// water is both deep and fast, which is exactly what a channel is, and is the
// same quantity that does the carving.
@group(0) @binding(7) var<storage, read_write> flow: array<f32>;

fn idx(x: i32, y: i32) -> u32 {
    let n = i32(P.res);
    return u32(clamp(y, 0, n - 1) * n + clamp(x, 0, n - 1));
}

fn inside(x: i32, y: i32) -> bool {
    let n = i32(P.res);
    return x >= 0 && y >= 0 && x < n && y < n;
}

// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn rain(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.res || id.y >= P.res) { return; }
    let i = idx(i32(id.x), i32(id.y));
    water[i] += P.rain_rate * P.dt;
}

// Outflow through four virtual pipes, driven by the difference in total
// surface height (terrain + water) with each neighbour.
@compute @workgroup_size(8, 8)
fn flux_pass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.res || id.y >= P.res) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let i = idx(x, y);

    let here = height[i] + water[i];
    let k = P.dt * P.pipe_area * P.gravity / P.cell_size;

    var f = flux[i];
    let dirs = array<vec2i, 4>(vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
    var out = vec4f(0.0);

    for (var d = 0; d < 4; d++) {
        let nx = x + dirs[d].x;
        let ny = y + dirs[d].y;
        // Zero flux across the map edge. Letting water leave would carve a
        // trench around the whole border; evaporation removes the surplus.
        if (!inside(nx, ny)) {
            continue;
        }
        let j = idx(nx, ny);
        let dh = here - (height[j] + water[j]);
        out[d] = max(0.0, f[d] + k * dh);
    }

    // Never move more water than the cell holds, or the solver diverges.
    let total = out.x + out.y + out.z + out.w;
    if (total > 0.0) {
        let avail = water[i] * P.cell_size * P.cell_size;
        let scale = min(1.0, avail / (total * P.dt));
        out *= scale;
    }
    flux[i] = out;
}

// Apply net flux, then derive the velocity field the sediment model needs.
@compute @workgroup_size(8, 8)
fn water_update(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.res || id.y >= P.res) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let i = idx(x, y);

    // Inflow is each neighbour's outflow aimed back at us.
    let fl = select(0.0, flux[idx(x - 1, y)].y, inside(x - 1, y));
    let fr = select(0.0, flux[idx(x + 1, y)].x, inside(x + 1, y));
    let fu = select(0.0, flux[idx(x, y - 1)].w, inside(x, y - 1));
    let fd = select(0.0, flux[idx(x, y + 1)].z, inside(x, y + 1));

    let out = flux[i];
    let inflow = fl + fr + fu + fd;
    let outflow = out.x + out.y + out.z + out.w;

    let w0 = water[i];
    let dv = P.dt * (inflow - outflow);
    let w1 = max(0.0, w0 + dv / (P.cell_size * P.cell_size));
    water[i] = w1;

    // Velocity from the average through-flow, normalized by mean depth.
    let wx = (fl - out.x + out.y - fr) * 0.5;
    let wy = (fu - out.z + out.w - fd) * 0.5;
    let mean = max((w0 + w1) * 0.5, 1e-5);
    vel[i] = vec2f(wx, wy) / (P.cell_size * mean);

    flow[i] += length(vel[i]) * w1 * P.dt;
}

@compute @workgroup_size(8, 8)
fn erode_deposit(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.res || id.y >= P.res) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let i = idx(x, y);

    let dhdx = (height[idx(x + 1, y)] - height[idx(x - 1, y)]) / (2.0 * P.cell_size);
    let dhdy = (height[idx(x, y + 1)] - height[idx(x, y - 1)]) / (2.0 * P.cell_size);
    let grad = sqrt(dhdx * dhdx + dhdy * dhdy);
    // sin of the tilt angle, floored: with a true zero, flat ground has no
    // capacity, rivers stop cutting, and valley floors never form.
    let sin_tilt = max(grad / sqrt(1.0 + grad * grad), P.min_slope);

    let speed = length(vel[i]);
    // Shallow water carries little; without this every damp cell erodes.
    let depth = clamp(water[i] * 40.0, 0.0, 1.0);
    let capacity = P.capacity * sin_tilt * speed * depth;

    let s = sed_src[i];
    if (capacity > s) {
        let amount = P.dissolve * (capacity - s) * P.dt;
        height[i] -= amount;
        sed_src[i] = s + amount;
    } else {
        let amount = P.deposit * (s - capacity) * P.dt;
        height[i] += amount;
        sed_src[i] = s - amount;
    }
}

// Semi-Lagrangian transport: sample where this parcel came from.
@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.res || id.y >= P.res) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let i = idx(x, y);

    let v = vel[i];
    let src_pos = vec2f(f32(x), f32(y)) - v * P.dt / P.cell_size;

    let fx = floor(src_pos.x);
    let fy = floor(src_pos.y);
    let t = src_pos - vec2f(fx, fy);
    let bx = i32(fx);
    let by = i32(fy);

    let a = sed_src[idx(bx, by)];
    let b = sed_src[idx(bx + 1, by)];
    let c = sed_src[idx(bx, by + 1)];
    let d = sed_src[idx(bx + 1, by + 1)];

    sed_dst[i] = mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

@compute @workgroup_size(8, 8)
fn evaporate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.res || id.y >= P.res) { return; }
    let i = idx(i32(id.x), i32(id.y));
    water[i] *= max(0.0, 1.0 - P.evaporation * P.dt);
}

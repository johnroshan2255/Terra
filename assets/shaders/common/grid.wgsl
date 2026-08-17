// --- the default surface, when no material has been imported ---
//
// Nothing ships prebuilt, so a fresh project has an empty palette. Flat grey was
// the first answer and it reads as unfinished: there is no sense of scale, no way
// to see where you are, and a sculpt stroke on it is nearly invisible until it
// catches a highlight.
//
// Unreal's answer is `WorldGridMaterial` -- a metric checkerboard with heavier
// lines at the decade marks -- and it is the right one. It is a ruler: a glance
// tells you how big a hill is and how far away it is, which is exactly what is
// missing from an untextured surface. It also needs no texture asset, so it does
// not reintroduce the prebuilt content this project deliberately has none of.
//
// # Antialiasing is not optional here
//
// A grid drawn naively is far worse than flat grey. Past a couple of pixels per
// cell the lines merge into a solid wash, and with TAA jitter on top it crawls.
// Both helpers below therefore measure their own screen footprint with `fwidth`
// and fade themselves out entirely once a cell is too small to resolve -- so each
// decade of the grid appears as you approach it and dissolves as you leave.

/// How much of one decade of the grid to show, given its cells-per-pixel.
///
/// Measured rather than guessed. The first thresholds were 0.25 to 0.5 -- fading out
/// only once cells reached two pixels -- and `grid_gpu.rs` showed 1.3-pixel lines
/// still covering most of the surface there: a flat wash, and one that crawls under
/// TAA. Fading out by three pixels a cell fixes it, and costs nothing, because
/// decades are a factor of ten apart: when one is at 0.25 cells a pixel the next is
/// at 0.025 and fully present, so there is never a gap.
fn grid_visibility(cells_per_pixel: f32) -> f32 {
    return 1.0 - smoothstep(0.10, 0.25, cells_per_pixel);
}

/// Lines on a unit lattice in `coord` space, `width_px` pixels wide, faded out
/// once the cells are too small to resolve.
fn grid_lines(coord: vec2f, width_px: f32) -> f32 {
    // Cells per pixel. `fwidth` is the right measure rather than distance: it
    // accounts for grazing angles, where one axis is compressed and the other is
    // not.
    let fw = fwidth(coord);
    // Distance to the nearest line, in pixels.
    let d = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(1e-6));
    let line = 1.0 - clamp(min(d.x, d.y) / width_px, 0.0, 1.0);
    return line * grid_visibility(max(fw.x, fw.y));
}

/// A checkerboard on the same lattice, fading to its own average when the squares
/// fall below a couple of pixels.
fn grid_checker(coord: vec2f) -> f32 {
    let c = floor(coord);
    // fract((x+y)/2)*2 is 1 on odd squares and 0 on even ones -- no integer
    // modulo, which would need the cell index to stay exact far from the origin.
    let odd = fract((c.x + c.y) * 0.5) * 2.0;
    let fw = max(fwidth(coord.x), fwidth(coord.y));
    return mix(0.5, odd, grid_visibility(fw));
}

/// Albedo of the default world grid at a world XZ position, in linear space.
fn world_grid(xz: vec2f) -> vec3f {
    // Anchored to the world origin and metric, so the lines mean something: 1 m
    // squares, 10 m minor lines, 100 m major ones.
    var c = mix(vec3f(0.240, 0.242, 0.248), vec3f(0.295, 0.297, 0.303), grid_checker(xz));
    c = mix(c, vec3f(0.150, 0.152, 0.158), grid_lines(xz * 0.1, 1.1));
    c = mix(c, vec3f(0.400, 0.404, 0.415), grid_lines(xz * 0.01, 1.3));
    return c;
}

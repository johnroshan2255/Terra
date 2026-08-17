// The sculpt brush cursor, drawn on the surface.
//
// Drawn on the ground rather than as a screen-space circle because the brush acts
// on the terrain: a ring that follows the surface tells you which slope you are
// about to cut, and one drawn flat on the screen does not.
//
// # Why the width is in pixels
//
// It used to be in metres -- `radius * 0.02 + 0.5` -- and that is what made the
// ring disappear. Measured at the editor's default camera, 380 m above the ground
// at about 1.9 m per pixel:
//
//     brush radius    ring half-width    in pixels
//              8 m             0.66 m         0.34
//             20 m             0.90 m         0.46
//             60 m             1.70 m         0.88
//            120 m             2.90 m         1.50
//            800 m            16.50 m         8.52
//
// So a small brush drew a ring a third of a pixel wide -- nothing, or a flicker on
// one pixel row that TAA then averages away -- while a large one drew a fat band.
// Since `[` and `]` size the brush from 8 m up, the sizes that vanished are the
// ones actually in use. The same thing happened to *any* brush once the camera
// pulled back, because metres per pixel grows with distance.
//
// Taking the width from the screen footprint instead makes the ring a constant
// weight at every brush size, camera distance and viewing angle.
//
// # And why there is a dark halo
//
// A single bright ring is legible on mid grey and much less so on pale ground,
// snow, or the bright lines of the default world grid. Pairing the bright core with
// a dark outer edge means one of the two always contrasts, whatever is underneath.

/// Pixels from the ring on which the core is still fully opaque.
///
/// Split from the feather below rather than ramping straight down from zero. A
/// single `smoothstep(0.0, w, edge)` puts the half-intensity point at `w/2`, so the
/// first attempt at this -- one ramp out to 1.6 px -- measured as a *one pixel* ring
/// and `brush_gpu.rs` failed on it. An opaque core plus a feather is what gives a
/// ring you can actually see.
const BRUSH_CORE_PX: f32 = 0.9;

/// Pixels from the ring at which the core has faded to nothing.
///
/// Together with the core this makes the ring about three pixels wide at half
/// intensity, which is the thinnest that stays solid under TAA jitter rather than
/// shimmering.
const BRUSH_FEATHER_PX: f32 = 2.1;

/// Pixels from the ring at which the dark halo has faded to nothing.
const BRUSH_HALO_PX: f32 = 5.0;

/// Overlay weights for the brush cursor at one pixel.
///
/// `x` is the bright core, `y` the dark halo just outside it, `z` a faint wash over
/// the whole disc so the affected area is legible without hiding the ground.
///
/// `metres_per_pixel` is the radial screen footprint -- `fwidth` of the distance --
/// supplied by the caller because derivatives are a fragment-stage builtin and
/// keeping them out makes this function pure, and therefore testable off a render
/// pass.
fn brush_ring_weights(dist_m: f32, radius_m: f32, metres_per_pixel: f32) -> vec3f {
    // Guard: a perfectly flat, perfectly axis-aligned surface can hand us a zero
    // derivative, and dividing by it would make the ring cover the screen.
    let px = max(metres_per_pixel, 1e-6);
    let edge_px = abs(dist_m - radius_m) / px;

    let core = 1.0 - smoothstep(BRUSH_CORE_PX, BRUSH_FEATHER_PX, edge_px);

    // The halo ramps in only once the core is nearly gone. `(1.0 - core)` looks like
    // the obvious weight and is wrong: it is already a third of the way up while the
    // core is still at two thirds, so the halo would darken the bright ring it is
    // supposed to frame.
    let halo = smoothstep(BRUSH_CORE_PX * 1.6, BRUSH_FEATHER_PX, edge_px)
        * (1.0 - smoothstep(BRUSH_FEATHER_PX, BRUSH_HALO_PX, edge_px));

    // The disc itself. In metres, not pixels: this one *should* scale with the
    // brush, because it is showing the area the brush covers.
    let fill = 1.0 - smoothstep(0.0, max(radius_m, 1e-6), dist_m);

    return vec3f(core, halo, fill);
}

/// Lay the brush cursor over an already-shaded colour.
fn brush_overlay(color: vec3f, w: vec3f) -> vec3f {
    // Halo first, then the core on top of it.
    var c = mix(color, vec3f(0.015, 0.015, 0.02), w.y * 0.55);
    c = mix(c, vec3f(1.0, 0.86, 0.38), w.x);
    return mix(c, vec3f(1.0, 0.86, 0.38), w.z * 0.05);
}

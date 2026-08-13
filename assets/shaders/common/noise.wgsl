// Shared noise basis.
//
// POLICY -- the terrain heightfield is ALWAYS ridged multifractal, never fBm.
//
//   ridged_multifractal()  the heightfield. Tier-0 base, and the tier-1 detail
//                          layer added on top of the eroded map.
//   warp_offset()          NOT terrain. A smooth 2D coordinate offset fed into
//                          ridged_multifractal()'s input position.
//
// Erosion features come from the compute-shader hydraulic solver in
// gen/erosion.wgsl, never from a noise function shaped to imitate them.

const ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);

fn hash21(p: vec2f) -> f32 {
    var h = fract(p * vec2f(0.1031, 0.1030));
    h += dot(h, h.yx + 33.33);
    return fract((h.x + h.y) * h.x) * 2.0 - 1.0;
}

// Value noise returning (value, d/dx, d/dy).
//
// The analytic derivative is kept because the render pass needs cheap normals
// for the sub-2 m detail layer -- finite-differencing would cost three extra
// samples per octave. Nothing in the generation path damps octaves by gradient;
// that trick approximates erosion, and we simulate it instead.
fn noised(p: vec2f) -> vec3f {
    let i = floor(p);
    let f = fract(p);

    // Quintic interpolant -- C2 continuous, so lighting has no facet seams at
    // cell boundaries the way it does with the cubic smoothstep.
    let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let a = hash21(i + vec2f(0.0, 0.0));
    let b = hash21(i + vec2f(1.0, 0.0));
    let c = hash21(i + vec2f(0.0, 1.0));
    let d = hash21(i + vec2f(1.0, 1.0));

    let k1 = b - a;
    let k2 = c - a;
    let k3 = a - b - c + d;

    return vec3f(
        a + k1 * u.x + k2 * u.y + k3 * u.x * u.y,
        du * vec2f(k1 + k3 * u.y, k2 + k3 * u.x)
    );
}

fn noise2(p: vec2f) -> f32 {
    return noised(p).x;
}

// ---------------------------------------------------------------------------
// The terrain basis.
// ---------------------------------------------------------------------------

// Ridged multifractal.
//
// `prev` is what makes this multifractal rather than plain ridged fBm: each
// octave is scaled by the previous octave's value, so detail concentrates on
// ridges and lowlands stay smooth. With sharpness = 0 it degenerates to ridged
// fBm -- uniform crumpled foil, the classic "procedural terrain" tell.
//
// This is the only function permitted to produce height.
fn ridged_multifractal(
    p_in: vec2f,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    offset: f32,
    sharpness: f32,
) -> f32 {
    var p = p_in;
    var sum = 0.0;
    var amp = 0.5;
    var prev = 1.0;
    var norm = 0.0;

    for (var i = 0; i < octaves; i++) {
        var n = offset - abs(noise2(p));
        n = n * n;
        n = n * mix(1.0, prev, sharpness);

        sum  += n * amp;
        norm += amp;
        prev  = clamp(n, 0.0, 1.0);   // unclamped, the feedback term explodes

        amp *= gain;
        p = ROT * p * lacunarity;     // rotate each octave to break axis alignment
    }
    return sum / max(norm, 1e-6);
}

// ---------------------------------------------------------------------------
// Domain warp -- a coordinate offset, not terrain.
// ---------------------------------------------------------------------------

// Smooth low-frequency vector field added to the input position before the
// ridged basis is evaluated. This is what makes ridgelines meander like real
// ranges instead of running in straight statistical lines.
//
// The basis here is deliberately smooth (plain summed value noise), NOT ridged.
// Ridged noise is C0 but not C1 -- it has a gradient discontinuity at every
// ridge -- and warping coordinates with a creased field creases the terrain.
// Four octaves is plenty; the warp only needs to be low-frequency.
//
// Callers can disable warping entirely by passing strength = 0.
fn warp_basis(p_in: vec2f) -> f32 {
    var p = p_in;
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;

    for (var i = 0; i < 4; i++) {
        sum  += amp * noise2(p);
        norm += amp;
        amp *= 0.5;
        p = ROT * p * 2.0;
    }
    return sum / max(norm, 1e-6);
}

fn warp_offset(p: vec2f, strength: f32, scale: f32) -> vec2f {
    if (strength <= 0.0) {
        return vec2f(0.0);
    }
    let q = vec2f(
        warp_basis(p / scale),
        warp_basis(p / scale + vec2f(5.2, 1.3))
    );
    return q * strength;
}

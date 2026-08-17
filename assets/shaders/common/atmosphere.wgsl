// Atmospheric single scattering, and the volumetric cloud layer that sits in it.
//
// The block below must match `environment::EnvironmentUniform` field for field:
// 16 vec4s, 256 bytes. Every member is a vector because WGSL rounds uniform
// members up to 16-byte alignment, so a lone f32 between two vectors inserts
// padding Rust does not and every field after it reads shifted.
//
// # What is physical here and what is not
//
// The sky is a real single-scattering integral: march the view ray through an
// exponential atmosphere, accumulate Rayleigh and Mie in-scattering weighted by
// their phase functions, attenuate by transmittance to the sun and back. The
// coefficients are the measured per-metre values for air, which is why the sky
// is this blue rather than a gradient someone chose.
//
// Multiple scattering is *approximated*, not integrated -- a cheap isotropic
// term rather than the second-order integral. Without any approximation the sky
// is too dark near the horizon and shadowed ground goes black; with the full
// integral this would be a precomputed LUT pass rather than an inline march.
//
// The clouds are a standard raymarch: density from noise shaped by a height
// profile, lit by a short march toward the sun under Beer-Lambert. Coverage,
// altitude, thickness and wind come from the mixer.

struct Env {
    sun_direction: vec4f,
    sun_radiance: vec4f,
    rayleigh: vec4f,
    mie: vec4f,
    ozone: vec4f,
    ambient_zenith: vec4f,
    ambient_horizon: vec4f,
    ambient_ground: vec4f,
    fog_params: vec4f,
    fog_albedo: vec4f,
    fog_extra: vec4f,
    cloud_params: vec4f,
    cloud_wind: vec4f,
    tone: vec4f,
    flags: vec4u,
    frame: vec4f,
};

@group(2) @binding(0) var<uniform> env: Env;

// Earth-scale shells, in metres. The atmosphere is thin relative to the planet
// -- 100 km against 6371 km -- and that ratio is what makes the horizon a sharp
// line rather than a soft fade.
const PLANET_R: f32 = 6371000.0;
const ATMOS_R: f32 = 6471000.0;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// Distance to the far intersection of a ray with a sphere centred at the origin,
// or -1 if it misses. `ro` is relative to the centre.
fn ray_sphere_far(ro: vec3f, rd: vec3f, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let d = b * b - c;
    if (d < 0.0) {
        return -1.0;
    }
    return -b + sqrt(d);
}

// Near intersection, or -1 if the ray misses or the hit is behind.
fn ray_sphere_near(ro: vec3f, rd: vec3f, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let d = b * b - c;
    if (d < 0.0) {
        return -1.0;
    }
    let t = -b - sqrt(d);
    return select(-1.0, t, t >= 0.0);
}

// World position to a planet-centred one. The camera sits on the surface, so the
// planet centre is one radius below the origin.
fn to_planet(p: vec3f) -> vec3f {
    return p + vec3f(0.0, PLANET_R, 0.0);
}

// ---------------------------------------------------------------------------
// Dithering
// ---------------------------------------------------------------------------

// Interleaved gradient noise, for offsetting a march's first step.
//
// A ray march that always starts a whole step in samples the medium on a lattice
// aligned to distance-from-camera, and that lattice shows as bands along the
// contours of constant distance -- horizontal stripes across a sky, which is
// exactly what this fixes. Offsetting the start by a per-pixel fraction of a
// step turns the banding into noise, and the temporal accumulation then averages
// the noise away over a few frames.
//
// IGN rather than a plain hash: it is designed to look uniform under a 3x3 box
// filter, which is roughly what the half-res upsample plus TAA amounts to. A
// white-noise hash needs more frames to resolve and looks grainier in the
// meantime.
fn ign(pixel: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
}

// Cycled over frames so the accumulation sees a different offset each time. The
// golden-ratio increment is what keeps successive frames maximally spread rather
// than repeating with a short period.
fn dither(pixel: vec2f, frame: f32) -> f32 {
    return fract(ign(pixel) + frame * 0.61803399);
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

// Rayleigh: nearly isotropic, slightly stronger forward and back. This mild
// shape is why the whole sky glows rather than only the part near the sun.
fn phase_rayleigh(cos_t: f32) -> f32 {
    return 3.0 / (16.0 * 3.14159265) * (1.0 + cos_t * cos_t);
}

// Henyey-Greenstein, for Mie and for cloud droplets. `g` toward 1 throws light
// sharply forward, which is what puts a halo around a low sun and a bright rim
// on a cloud the sun is behind.
fn phase_hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_t;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(denom, 1e-4), 1.5));
}

// ---------------------------------------------------------------------------
// Medium
// ---------------------------------------------------------------------------

// Rayleigh and Mie density at an altitude, as fractions of their sea-level
// values. Both are exponential, with very different scale heights: air reaches
// far higher than aerosol, which is why haze sits in the lower sky.
fn air_density(altitude_m: f32) -> vec2f {
    let h = max(altitude_m, 0.0);
    return vec2f(exp(-h / max(env.rayleigh.w, 1.0)), exp(-h / max(env.mie.y, 1.0)));
}

// Ozone sits in a band around 25 km rather than falling off exponentially, so it
// is a tent function. It is the reason a clear zenith trends violet at dusk
// instead of simply darkening.
fn ozone_density(altitude_m: f32) -> f32 {
    return max(0.0, 1.0 - abs(altitude_m - 25000.0) / 15000.0);
}

// Optical depth from a point to the edge of the atmosphere along `rd`.
//
// Five steps -- see the note inside. This runs per view-ray sample, so it is
// the inner loop of an already-nested march.

fn transmittance(origin: vec3f, rd: vec3f) -> vec3f {
    let far = ray_sphere_far(origin, rd, ATMOS_R);
    if (far <= 0.0) {
        return vec3f(1.0);
    }
    // A ray that re-enters the planet is fully shadowed: this is the horizon.
    if (ray_sphere_near(origin, rd, PLANET_R) > 0.0) {
        return vec3f(0.0);
    }

    // Five steps. The integrand is smooth and monotonic, and this runs once per
    // view-ray sample -- it is the inner loop of an already-nested march, so it
    // is the cheapest place to buy back time and the hardest place to see it.
    let steps = 5;
    let dt = far / f32(steps);
    var depth = vec3f(0.0);
    for (var i = 0; i < steps; i = i + 1) {
        let p = origin + rd * (dt * (f32(i) + 0.5));
        let alt = length(p) - PLANET_R;
        let d = air_density(alt);
        depth += (env.rayleigh.rgb * d.x + vec3f(env.mie.x) * d.y * 1.11
            + env.ozone.rgb * ozone_density(alt)) * dt;
    }
    return exp(-depth);
}

// ---------------------------------------------------------------------------
// Sky
// ---------------------------------------------------------------------------

// Single-scattered sky radiance along `rd`.
//
// Deliberately *not* dithered. Its 16 steps were measured for banding and have
// essentially none -- a row-mean second-difference score of 0.066 against a
// smooth gradient, which is the gradient's own curvature. Offsetting the start
// per pixel raised that to 0.26 for a single frame and 0.12 even after averaging
// eight, so the dither was adding noise to fix nothing. The integrand here is
// smooth and exponential; it is the cloud march that has the hard lattice.
fn atmosphere(eye: vec3f, rd: vec3f, sun: vec3f) -> vec3f {
    let ro = to_planet(eye);
    var far = ray_sphere_far(ro, rd, ATMOS_R);
    if (far <= 0.0) {
        return vec3f(0.0);
    }
    // Looking down: stop at the ground rather than integrating through rock.
    let ground = ray_sphere_near(ro, rd, PLANET_R);
    if (ground > 0.0) {
        far = min(far, ground);
    }

    let cos_t = dot(rd, sun);
    let ph_r = phase_rayleigh(cos_t);
    let ph_m = phase_hg(cos_t, env.mie.z);

    let steps = 16;
    let dt = far / f32(steps);
    var sum_r = vec3f(0.0);
    var sum_m = vec3f(0.0);
    var optical = vec3f(0.0);

    for (var i = 0; i < steps; i = i + 1) {
        let p = ro + rd * (dt * (f32(i) + 0.5));
        let alt = length(p) - PLANET_R;
        let d = air_density(alt);

        // Extinction accumulated from the eye to here.
        let step_ext = (env.rayleigh.rgb * d.x + vec3f(env.mie.x) * d.y * 1.11
            + env.ozone.rgb * ozone_density(alt)) * dt;
        optical += step_ext;
        let view_t = exp(-optical);

        // ...times the transmittance from here to the sun. The product is what
        // reddens a low sun: both legs are long, and both lose blue.
        let sun_t = transmittance(p, sun);
        let vis = view_t * sun_t;

        sum_r += vis * d.x * dt;
        sum_m += vis * d.y * dt;
    }

    // The source term of the scattering integral is the sun's *irradiance*, but
    // `sun_radiance` is normalized so 1.0 is the radiance of a sunlit white
    // Lambertian surface -- which is E/pi. So E = pi * sun_radiance.
    //
    // Leaving the pi out made the sky about three times too dark: a zenith
    // luminance of 0.034 against a sunlit white surface at 1.0, where the real
    // ratio is 0.1 to 0.25.
    let irradiance = env.sun_radiance.rgb * 3.14159265;
    var color =
        irradiance * (sum_r * env.rayleigh.rgb * ph_r + sum_m * vec3f(env.mie.x) * ph_m);

    // Cheap stand-in for multiple scattering. Without it the horizon is far too
    // dark and a shadowed slope reads black; the real term is a second-order
    // integral and belongs in a precomputed LUT, not in this loop.
    //
    // 0.6 rather than a token amount: for a clear sky, multiple scattering is
    // roughly half of zenith radiance again, and more of it toward the horizon.
    let ms = (sum_r * env.rayleigh.rgb + sum_m * vec3f(env.mie.x)) * irradiance * 0.6;
    color += ms;

    return color;
}

// The sun disc, with a soft edge sized by its angular diameter.
//
// Drawn as a term of the sky rather than as geometry, so it is behind
// everything, needs no depth handling, and is what the god-ray pass finds when
// it looks for a bright source.
fn sun_disc(rd: vec3f, sun: vec3f) -> vec3f {
    let cos_t = dot(rd, sun);
    // Half-angle in radians. Widening the disc softens every shadow in the
    // scene, so it is the same dial that drives shadow softness.
    let half_angle = radians(max(env.sun_radiance.w, 0.05)) * 0.5;
    let edge = cos(half_angle);
    // A degree of feathering, or the limb aliases into a stair-stepped circle.
    let soft = 1.0 - cos(half_angle * 1.35);
    let disc = smoothstep(edge - soft, edge, cos_t);
    // Limb darkening, which is what stops it reading as a flat sticker.
    let limb = mix(0.72, 1.0, sqrt(max(disc, 0.0)));
    return env.sun_radiance.rgb * disc * limb * 22.0;
}

// ---------------------------------------------------------------------------
// Clouds
// ---------------------------------------------------------------------------

fn hash31(p: vec3f) -> f32 {
    var q = fract(p * 0.3183099 + vec3f(0.1, 0.2, 0.3));
    q += vec3f(dot(q, q.yzx + 19.19));
    return fract((q.x + q.y) * q.z);
}

// Trilinear value noise, smoothstepped so the lattice does not show as a grid.
fn value3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash31(i + vec3f(0.0, 0.0, 0.0));
    let c100 = hash31(i + vec3f(1.0, 0.0, 0.0));
    let c010 = hash31(i + vec3f(0.0, 1.0, 0.0));
    let c110 = hash31(i + vec3f(1.0, 1.0, 0.0));
    let c001 = hash31(i + vec3f(0.0, 0.0, 1.0));
    let c101 = hash31(i + vec3f(1.0, 0.0, 1.0));
    let c011 = hash31(i + vec3f(0.0, 1.0, 1.0));
    let c111 = hash31(i + vec3f(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// Summed octaves, for the cloud shape.
fn fbm3(p: vec3f, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i = i + 1) {
        sum += amp * value3(q);
        q = q * 2.02;
        amp *= 0.5;
    }
    return sum;
}

// Vertical density profile of a cumulus: flat-bottomed, billowing in the middle,
// wispy on top. `h` is 0 at the layer base and 1 at its top.
//
// A slab of uniform density reads as fog with a hard edge; almost all of what
// makes a cloud look like a cloud from the ground is this curve.
fn cloud_profile(h: f32) -> f32 {
    let base = smoothstep(0.0, 0.12, h);
    let top = 1.0 - smoothstep(0.45, 1.0, h);
    return base * top;
}

// Low-frequency cloud shape, before detail erosion. Four octaves.
//
// Split out from [`cloud_density`] because it is the cheap test that decides
// whether the expensive one is worth running: most samples along a view ray are
// in clear air, and evaluating three more octaves of noise to confirm that a
// point is empty is where a naive cloud march spends most of its budget.
fn cloud_shape(p: vec3f) -> f32 {
    let base = env.cloud_params.y;
    let thickness = max(env.cloud_params.z, 1.0);
    let h = (p.y - base) / thickness;
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }

    // Wind advects the whole field.
    let wind = env.cloud_wind.xyz * env.frame.x;
    let scale = max(env.cloud_wind.w, 1.0);
    let q = (p + wind) / scale;

    // Coverage remaps the noise rather than scaling it: subtracting a threshold
    // and rescaling erodes cloud edges inward, where multiplying would just fade
    // the whole layer uniformly and never open holes in it.
    let coverage = clamp(env.cloud_params.x, 0.0, 1.0);
    var d = fbm3(q * 2.0, 4) - (1.0 - coverage);
    if (d <= 0.0) {
        return 0.0;
    }
    return (d / max(coverage, 1e-3)) * cloud_profile(h);
}

// Full cloud density, detail included.
fn cloud_density(p: vec3f) -> f32 {
    let shape = cloud_shape(p);
    if (shape <= 0.0) {
        return 0.0;
    }
    let wind = env.cloud_wind.xyz * env.frame.x;
    let scale = max(env.cloud_wind.w, 1.0);
    let q = (p + wind) / scale;

    // Detail erosion at the boundary only: applying it everywhere costs octaves
    // in the cloud interior, which is opaque and cannot show them.
    var d = shape;
    d = d - fbm3(q * 12.0, 3) * 0.35 * (1.0 - smoothstep(0.0, 0.6, d));
    return clamp(d, 0.0, 1.0) * env.cloud_params.w;
}

// Optical depth from a point toward the sun through the cloud layer.
//
// Depth rather than transmittance, because the multiple-scattering
// approximation below needs to re-attenuate it several times at different rates
// and `exp` does not compose that way.
//
// Five steps with a widening stride: the near samples decide whether a pixel is
// lit at all, and the far ones only soften it. A uniform stride spends the same
// budget resolving detail 2 km away that nothing can see.
fn cloud_light_depth(p: vec3f, sun: vec3f) -> f32 {
    let thickness = max(env.cloud_params.z, 1.0);
    var depth = 0.0;
    var dist = thickness * 0.03;
    for (var i = 0; i < 5; i = i + 1) {
        // Shape, not full density: this march decides how *shadowed* a point is,
        // and shadow is a low-frequency quantity. Sampling detail octaves five
        // more times per view sample was a third of the whole cost and changed
        // the result by less than the dithering hides.
        depth += cloud_shape(p + sun * dist) * env.cloud_params.w * dist * 0.6;
        dist *= 1.9;
    }
    return depth;
}

// Sun luminance reaching a point inside a cloud, as a multiple of the sun's own
// radiance.
//
// # Why this is not just `exp(-depth) * phase`
//
// Two things were wrong with that, and together they made clouds about twelve
// times too dark and the wrong colour.
//
// `phase_hg` carries the 1/4pi normalization a radiative-transfer integral
// needs, which is only correct when the source term is the sun's true
// irradiance. Here `sun_radiance` is normalized so 1.0 is a sunlit white
// surface, so the 1/4pi has to come back out.
//
// And a single forward-scattering lobe makes a cloud with the sun *behind the
// camera* almost black, when it should be the brightest white in the frame. Real
// clouds are bright from every angle because light bounces many times inside
// them before leaving. This is the standard cheap stand-in for that: a few
// octaves with geometrically weaker extinction, weaker contribution and a more
// isotropic phase each time, so deep and back-lit cloud still receives light.
fn cloud_sun_luminance(depth: f32, cos_t: f32) -> f32 {
    var sum = 0.0;
    var attenuation = 1.0;
    var contribution = 1.0;
    var eccentricity = 1.0;
    for (var n = 0; n < 3; n = n + 1) {
        // Two lobes: forward for the silver lining, a weak backward one for the
        // body. 4pi undoes the normalization -- see above.
        let ph = mix(
            phase_hg(cos_t, 0.8 * eccentricity),
            phase_hg(cos_t, -0.15 * eccentricity),
            0.4,
        ) * 4.0 * 3.14159265;
        sum += contribution * ph * exp(-depth * attenuation);
        attenuation *= 0.5;
        contribution *= 0.6;
        eccentricity *= 0.5;
    }
    // Beer-Powder on top: plain Beer's law makes thin edges too dark, because it
    // cannot represent light scattered *into* an edge from around it.
    let powder = 1.0 - exp(-depth * 2.0);
    return sum * mix(1.0, powder * 2.0, 0.25);
}

// March the cloud layer along the view ray.
//
// Returns rgb scattered radiance and, in `a`, the transmittance left -- so the
// caller composites `sky * a + rgb` and clouds correctly dim what is behind
// them instead of being drawn on top.
fn clouds(eye: vec3f, rd: vec3f, sun: vec3f, jit: f32) -> vec4f {
    if (env.flags.z == 0u || env.cloud_params.x <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    // Looking down, or at a layer entirely behind the camera: nothing to march.
    let base = env.cloud_params.y;
    let top = base + max(env.cloud_params.z, 1.0);
    if (rd.y <= 0.001 && eye.y < base) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Slab entry and exit. Solved on the plane pair rather than against the
    // planet shells: at 1-3 km the curvature is under a metre across the whole
    // marched span, and flat planes make the interval exact and branch-free.
    var t0 = (base - eye.y) / rd.y;
    var t1 = (top - eye.y) / rd.y;
    if (t0 > t1) {
        let tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    t0 = max(t0, 0.0);
    if (t1 <= t0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    // Cap the span. A ray a few degrees above the horizon crosses hundreds of
    // kilometres of slab, and marching it at a useful step count is not
    // affordable -- so the layer fades out instead of being resolved.
    let span = min(t1 - t0, 60000.0);

    // Adaptive stride: stride long through clear air, drop to the fine step as
    // soon as the cheap shape test finds cloud, and go back to long strides after
    // leaving it. A uniform 48-step march spends most of its samples confirming
    // that empty sky is empty.
    let base_steps = 48.0 * clamp(env.frame.y, 0.25, 4.0);
    let fine = span / base_steps;
    // 2.5x, not 4x. The coarse stride is what a cloud's leading edge gets snapped
    // to, so it is the size of the quantization that shows as banding; 4x was
    // visible as lines across the layer. The dithered start scatters the lattice
    // per pixel and the accumulation averages it, but a smaller step is the part
    // of the fix that does not depend on either.
    let coarse = fine * 2.5;
    let cos_t = dot(rd, sun);

    // Ambient from the sky above and the ground below, so a cloud base is not
    // uniformly grey.
    let amb_top = env.ambient_zenith.rgb;
    let amb_bot = env.ambient_ground.rgb;

    var scattered = vec3f(0.0);
    var transmit = 1.0;
    // Start a fraction of a step in, per pixel. Without this the fine steps land
    // on the same distances for every pixel at a given elevation and the layer is
    // crossed by horizontal bands.
    var travelled = fine * jit;
    var inside = false;

    // Bounded so a degenerate stride cannot spin: 48 fine steps plus the coarse
    // ones it takes to cross the gaps between clouds.
    let max_iters = i32(base_steps * 2.0) + 8;
    for (var i = 0; i < max_iters; i = i + 1) {
        if (travelled >= span) {
            break;
        }
        let dt = select(coarse, fine, inside);
        let p = eye + rd * (t0 + travelled + dt * 0.5);

        // Cheap test first. In clear air this is all that runs.
        if (cloud_shape(p) <= 0.0) {
            inside = false;
            travelled += dt;
            continue;
        }
        // Entering cloud on a coarse stride: step back and re-walk it finely, or
        // the leading edge of every cloud is chopped to the coarse stride.
        if (!inside) {
            inside = true;
            continue;
        }

        let density = cloud_density(p);
        if (density <= 0.001) {
            travelled += dt;
            continue;
        }

        let extinction = density * dt;
        let sun_depth = cloud_light_depth(p, sun);
        let h = clamp((p.y - base) / max(env.cloud_params.z, 1.0), 0.0, 1.0);

        let sunlit = env.sun_radiance.rgb * cloud_sun_luminance(sun_depth, cos_t);
        // Sky fill, which is what keeps a cloud base from going black. Scaled up
        // from 0.6: the ambient tints are per-steradian sky radiance and a cloud
        // base sees most of the hemisphere.
        let ambient = mix(amb_bot, amb_top, h) * 2.0;
        // Integrate in-scattering over the step analytically rather than
        // sampling it at a point, which visibly bands at this step count.
        let integral = (1.0 - exp(-extinction)) * transmit;
        scattered += (sunlit + ambient) * integral;

        transmit *= exp(-extinction);
        travelled += dt;
        if (transmit < 0.02) {
            break;
        }
    }

    // Fade the layer out toward the horizon, where the marched span was capped
    // and the result is unreliable.
    //
    // Widened from 0.055: a fade that tight ends the cloud layer within three
    // degrees of the horizon and draws a hard horizontal line where it stops.
    // 0.16 spreads it over about nine degrees, which reads as haze swallowing the
    // distant layer instead of a cut.
    let horizon_fade = smoothstep(0.0, 0.16, rd.y);
    scattered *= horizon_fade;
    transmit = mix(1.0, transmit, horizon_fade);

    return vec4f(scattered, clamp(transmit, 0.0, 1.0));
}

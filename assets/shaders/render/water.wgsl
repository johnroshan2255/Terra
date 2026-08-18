// Water surface.
//
// One blended pass over CDLOD patches, drawn after everything opaque. Modelled on
// what Unreal's Water plugin and Crest do, minus the parts that need a second scene
// capture:
//
//   * Gerstner waves, summed, displacing XZ as well as Y so crests sharpen
//   * Beer-Lambert absorption by *depth*, which is what gives shallows their colour
//   * Fresnel between a sky reflection and the absorbed body colour
//   * shoreline foam where the water is shallow
//
// # Depth without a depth buffer
//
// The usual way to know how deep water is over the ground is to sample the scene
// depth, which cannot be done in the same pass that writes it -- so engines either
// copy the depth buffer or render a top-down "water info" texture first. Unreal does
// the latter.
//
// Neither is needed here, because the terrain heightfield is already a storage buffer
// this pass can read. Depth is `level - terrain_height` evaluated exactly, at any
// point, with no copy, no extra pass and no resolution loss. That is the one place
// this renderer's design makes water *easier* rather than harder.
//
// # No refraction
//
// Refraction needs the scene colour behind the surface, which is the copy this
// deliberately avoids. Instead the absorption drives the blend alpha: shallow water is
// transparent and the ground shows through by ordinary blending, deep water goes opaque
// and reaches the body colour. That is the same physics, resolved by the blender rather
// than by a texture fetch -- it just cannot bend what it shows.

/// One rectangular body with its own level and waves. Must match
/// `water::RegionUniform` byte for byte -- three vec4s, so every member lands on the
/// 16-byte boundary WGSL rounds up to and no padding is inserted on one side only.
struct WaterRegion {
    // xy = min corner, zw = max corner, in world XZ.
    bounds: vec4f,
    // x = level, y = wave height, z = wave length, w = wave speed.
    params: vec4f,
    // xy = wind direction, zw unused.
    wind: vec4f,
};

/// Must match `water::MAX_REGIONS`.
const MAX_REGIONS: u32 = 8u;

struct WaterUniform {
    // xyz = shallow colour, w = water level in metres.
    shallow: vec4f,
    // xyz = deep colour, w = absorption per metre.
    deep: vec4f,
    // x = wave height, y = wave length, z = wave speed, w = time.
    wave: vec4f,
    // xy = wind direction, z = foam width in metres, w = roughness.
    surface: vec4f,
    // x = world extent, y = height texels per side, z = patch quads, w = unused.
    grid: vec4f,
    // xyz = eye, w = unused. Separate from the camera block because the morph needs
    // it in the shadow pass too, exactly as the terrain's does.
    eye: vec4f,
};

// Groups follow `sky.wgsl` exactly, and not by preference: `atmosphere.wgsl` declares
// its own `env` uniform at **group 2**, so any module reflecting the sky has to leave
// that group alone. Camera 0, lighting 1, env 2 -- water takes 3.
@group(0) @binding(0) var<uniform> cam: Camera;
// All five lighting entries are named even though only the uniform and the shadow map
// are read: `lighting.wgsl` refers to the others by name, so they have to resolve for
// the module to parse at all.
@group(1) @binding(0) var<uniform> light: Light;
@group(1) @binding(1) var shadow_map: texture_depth_2d_array;
@group(1) @binding(2) var shadow_samp: sampler_comparison;
@group(1) @binding(3) var fog_grid: texture_3d<f32>;
@group(1) @binding(4) var fog_samp: sampler;
@group(3) @binding(0) var<uniform> w: WaterUniform;
@group(3) @binding(1) var<storage, read> heights: array<f32>;
@group(3) @binding(2) var<storage, read> patches: array<CdlodPatch>;
@group(3) @binding(3) var<uniform> regions: array<WaterRegion, MAX_REGIONS>;

/// The water at a world XZ: its level, its wave parameters and whether there is any.
///
/// Regions win over the global surface, and the *first* containing region wins over
/// later ones -- overlapping rectangles resolve by placement order rather than by
/// blending, which would need a signed distance per region and produce a seam anyway.
struct Body {
    level: f32,
    height: f32,
    length: f32,
    speed: f32,
    wind: vec2f,
    present: bool,
};

fn body_at(xz: vec2f) -> Body {
    var b: Body;
    let count = u32(w.grid.w);
    for (var i = 0u; i < MAX_REGIONS; i = i + 1u) {
        if (i >= count) {
            break;
        }
        let r = regions[i];
        if (xz.x >= r.bounds.x && xz.x <= r.bounds.z
            && xz.y >= r.bounds.y && xz.y <= r.bounds.w) {
            b.level = r.params.x;
            b.height = r.params.y;
            b.length = r.params.z;
            b.speed = r.params.w;
            b.wind = r.wind.xy;
            b.present = true;
            return b;
        }
    }
    // Outside every region, the global surface -- which may itself be switched off, in
    // which case there is no water here and the fragment is discarded.
    b.level = w.shallow.w;
    b.height = w.wave.x;
    b.length = w.wave.y;
    b.speed = w.wave.z;
    b.wind = w.surface.xy;
    // `grid.w` carries the region count; the sign of `eye.w` carries whether the global
    // surface is on, so both fit without growing the uniform.
    b.present = w.eye.w > 0.5;
    return b;
}

/// Terrain height at a normalised position, bilinear. The same filtering the terrain
/// itself uses, so the shoreline the water computes agrees with the ground drawn under
/// it to the texel.
fn terrain_h(uv: vec2f) -> f32 {
    let res = u32(w.grid.y);
    let p = clamp(uv, vec2f(0.0), vec2f(1.0)) * f32(res - 1u);
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let z = i32(i.y);
    let mx = i32(res) - 1;
    let x0 = clamp(x, 0, mx);
    let z0 = clamp(z, 0, mx);
    let x1 = clamp(x + 1, 0, mx);
    let z1 = clamp(z + 1, 0, mx);
    let r = i32(res);
    let h00 = heights[z0 * r + x0];
    let h10 = heights[z0 * r + x1];
    let h01 = heights[z1 * r + x0];
    let h11 = heights[z1 * r + x1];
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

fn to_uv(xz: vec2f) -> vec2f {
    return xz / w.grid.x + 0.5;
}

/// Depth of water over the ground at a world XZ, in metres. Negative means the ground
/// is above the surface, i.e. dry land.
fn depth_below(level: f32, xz: vec2f) -> f32 {
    return level - terrain_h(to_uv(xz));
}

// ---------------------------------------------------------------------------
// Gerstner waves
// ---------------------------------------------------------------------------
//
// Four, at descending amplitude and rotated headings. Gerstner rather than a sum of
// vertical sines because the horizontal term is what sharpens a crest and flattens a
// trough -- a plain sine surface reads as cloth. Steepness is shared and kept well
// under the value that would make the surface self-intersect.
//
// FFT would give a richer spectrum for the same shader cost at high wave counts, and
// `rustfft` exists for the CPU half, but it needs a precomputed spectrum texture and a
// per-frame IFFT. Four Gerstners need nothing but a clock, and at the scale a terrain
// editor looks at water they are indistinguishable.
const WAVE_COUNT: i32 = 4;
const STEEPNESS: f32 = 0.55;

/// One wave's contribution, and its analytic tangent and binormal derivatives.
///
/// Returning the derivatives rather than differencing the surface later is what keeps
/// the normal exact at any vertex density: the mesh under this is CDLOD, so its spacing
/// changes with distance and a finite difference would change the lighting with it.
struct Wave {
    offset: vec3f,
    dtan: vec3f,
    dbin: vec3f,
};

fn gerstner(xz: vec2f, dir: vec2f, amplitude: f32, wavelength: f32, speed: f32, t: f32) -> Wave {
    var out: Wave;
    let k = 6.28318530718 / max(wavelength, 0.01);
    let d = normalize(dir);
    // Deep-water dispersion: long waves travel faster, which is what stops a
    // multi-wave sum looking like one rigid pattern sliding across the surface.
    let c = sqrt(9.81 / k) * speed;
    let f = k * (dot(d, xz) - c * t);
    let a = amplitude;
    // Steepness folded into the horizontal term only. `q/k` is the classic form; the
    // division keeps the horizontal displacement in proportion to the wavelength.
    let q = STEEPNESS / (k * a * f32(WAVE_COUNT) + 1e-5);
    let qa = min(q, 1.0) * a;

    let sf = sin(f);
    let cf = cos(f);
    out.offset = vec3f(d.x * qa * cf, a * sf, d.y * qa * cf);
    out.dtan = vec3f(
        -d.x * d.x * qa * k * sf,
        d.x * a * k * cf,
        -d.x * d.y * qa * k * sf,
    );
    out.dbin = vec3f(
        -d.x * d.y * qa * k * sf,
        d.y * a * k * cf,
        -d.y * d.y * qa * k * sf,
    );
    return out;
}

/// The whole wave stack at a point: displacement plus the surface normal.
struct Surface {
    offset: vec3f,
    normal: vec3f,
};

fn wave_surface(xz: vec2f, fade: f32, b: Body) -> Surface {
    let base = normalize(select(vec2f(1.0, 0.0), b.wind, length(b.wind) > 1e-4));
    var offset = vec3f(0.0);
    var tan = vec3f(1.0, 0.0, 0.0);
    var bin = vec3f(0.0, 0.0, 1.0);

    var amp = b.height * fade;
    var len = b.length;
    for (var i = 0; i < WAVE_COUNT; i = i + 1) {
        // Each wave turned off the last, so the stack is not one heading repeated.
        // 0.7 rad is wide enough to break up the pattern and narrow enough that the
        // set still reads as driven by one wind.
        let a = f32(i) * 0.7 - 1.05;
        let ca = cos(a);
        let sa = sin(a);
        let dir = vec2f(base.x * ca - base.y * sa, base.x * sa + base.y * ca);
        let wv = gerstner(xz, dir, amp, len, b.speed, w.wave.w);
        offset += wv.offset;
        tan += wv.dtan;
        bin += wv.dbin;
        // Descending amplitude and wavelength: the large waves carry the shape, the
        // small ones the glitter.
        amp *= 0.55;
        len *= 0.62;
    }

    var out: Surface;
    out.offset = offset;
    out.normal = normalize(cross(bin, tan));
    return out;
}

// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    // Depth at the *undisplaced* position, so the shoreline does not wobble with the
    // waves -- it is a property of the ground, not of the surface above it. Negative
    // means dry land or no water body here.
    @location(2) depth: f32,
    // The level this fragment's body sits at, for the foam's crest test. Interpolated,
    // which is exact: it is constant across any one region.
    @location(3) level: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let xz = cdlod_vertex_xz(patches[ii], vi, u32(w.grid.z), w.eye.xyz);
    let b = body_at(xz);
    let depth = depth_below(b.level, xz);

    // Waves flattened into the shallows. A full-amplitude crest in 10 cm of water
    // punches the surface through the beach, and the vertex would end up below the
    // ground it is supposed to be covering.
    let fade = clamp(depth / max(b.height * 2.5, 0.1), 0.0, 1.0);
    let s = wave_surface(xz, fade, b);

    var out: VsOut;
    out.world = vec3f(xz.x, b.level, xz.y) + s.offset;
    out.normal = s.normal;
    // Negative where there is no water at all, so the fragment stage discards it on the
    // same test that rejects dry land.
    out.depth = select(-1.0, depth, b.present);
    out.level = b.level;
    out.clip = cam.view_proj * vec4f(out.world, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // Dry land, or outside every water body. Discarded rather than clipped by geometry,
    // because the shoreline is wherever the heightfield crosses the level and no mesh
    // boundary could follow that without being rebuilt on every sculpt stroke -- and
    // because a region's rectangle is only a lookup bound, not the shape of its shore.
    if (in.depth <= 0.0) {
        discard;
    }

    let n = normalize(in.normal);
    let view = normalize(cam.eye.xyz - in.world);
    let sun = normalize(light.sun_direction.xyz);

    // --- absorption ---
    //
    // Beer-Lambert along the path light takes: down through the water and back up to
    // the eye. Doubling the vertical depth is the cheap standing-in for that, and it
    // is why a lake edge is green and its middle is nearly black.
    let path = in.depth * (1.0 + 1.0 / max(dot(view, n), 0.25));
    let absorb = 1.0 - exp(-path * w.deep.w);
    let body = mix(w.shallow.xyz, w.deep.xyz, absorb);

    // --- reflection ---
    //
    // The sky, through the same single-scattering integral the sky pass uses, so the
    // water reflects the atmosphere actually being drawn rather than a guess at it. A
    // ray that reflects downward is clamped to the horizon: below it there is no sky
    // to sample and the integral would march out of the atmosphere shell.
    var r = reflect(-view, n);
    r.y = max(r.y, 0.02);
    let sky = atmosphere(cam.eye.xyz, normalize(r), sun);

    // Schlick, with water's 0.02 normal-incidence reflectance. Grazing angles go to
    // pure mirror, which is most of what makes a still surface read as water.
    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(n, view), 0.0, 1.0), 5.0);

    // --- specular ---
    //
    // A GGX-ish lobe on the same normal. Roughness is exposed because a mirror-smooth
    // sheet and a wind-chopped one are the same geometry with different highlights.
    let h = normalize(view + sun);
    let rough = clamp(w.surface.w, 0.02, 1.0);
    let a2 = rough * rough * rough * rough;
    let ndh = max(dot(n, h), 0.0);
    let d = a2 / max(3.14159265 * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0), 1e-4);
    let ndl = max(dot(n, sun), 0.0);

    // Cascade shadows, on the same terms the ground gets them. A lake in a mountain's
    // shadow has to go dark, and the sun glitter has to stop with it -- a highlight
    // burning on water that is provably in shade is the single most obvious way for a
    // surface to read as pasted on.
    let view_depth = length(in.world - cam.eye.xyz);
    let shade = sun_visibility(in.world, view_depth, ndl);
    let spec = light.sun_color.rgb * d * ndl * 0.25 * shade;

    var color =
        mix(body * light.sun_color.rgb * max(ndl * shade, 0.25), sky, fresnel) + spec;

    // --- foam ---
    //
    // A band at the shoreline, widened where a wave crest is riding high. Cheap, and
    // it is what stops the water meeting the beach as a hard line.
    let crest = clamp((in.world.y - in.level) / max(w.wave.x, 0.01), 0.0, 1.0);
    let band = max(w.surface.z, 0.01);
    let foam = (1.0 - smoothstep(0.0, band, in.depth)) * (0.55 + 0.45 * crest);
    color = mix(color, vec3f(0.92, 0.95, 0.97) * light.sun_color.rgb, clamp(foam, 0.0, 0.9));

    // Alpha is the absorption, floored by the Fresnel and the foam: a shallow edge is
    // nearly clear, deep water is opaque, and a grazing view is reflective whatever
    // the depth. This is the blend standing in for a refraction fetch.
    let alpha = clamp(max(max(absorb, fresnel), foam), 0.0, 1.0);
    return vec4f(color, alpha);
}

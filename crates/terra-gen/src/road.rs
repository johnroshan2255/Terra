//! Stamping road splines into a heightfield.
//!
//! Three stages, and the middle one is what makes a road look built rather
//! than painted on:
//!
//! 1. **Resample** the control points to a Catmull-Rom centreline at fixed
//!    arc-length steps.
//! 2. **Grade-limit** the elevation profile. Draping a road over terrain gives
//!    it every bump the ground has, and grades no vehicle could climb. Real
//!    roads climb at a bounded rate, which is why they cut and fill.
//! 3. **Stamp** a cross-section: cambered carriageway with wheel ruts,
//!    shoulders, then batter slopes at the angle of repose out to wherever they
//!    meet the terrain.

use terra_project::roads::{Road, RoadNetwork};

/// Spacing between centreline samples. Finer than the 2-4 m texel spacing, so
/// the cross-section never steps between samples.
const SAMPLE_M: f32 = 1.0;

/// Smoothing window applied to a freehand stroke before simplification.
const SMOOTH_WINDOW: usize = 3;

/// Relaxation passes used to *shape* the profile before it is made legal.
/// These only smooth; the guarantee comes from the sweep afterwards.
const GRADE_PASSES: usize = 64;

/// One point along the finished centreline.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub x: f32,
    pub z: f32,
    /// Road surface elevation after grade limiting and cut/fill clamping.
    pub y: f32,
}

/// What a stamp produced, alongside the modified heights.
#[derive(Debug, Default, Clone)]
pub struct RoadSurface {
    /// 1 on the carriageway, fading across the shoulder, 0 beyond.
    pub mask: Vec<f32>,
    /// 1 in the bottom of a wheel rut, 0 elsewhere. Drives puddles and wetness.
    pub rut: Vec<f32>,
}

/// Moving-average smoothing of a freehand stroke.
///
/// A dragged cursor carries hand tremor and pixel quantisation, both of which
/// survive simplification as visible kinks. Smoothing first means the retained
/// control points sit on the line the user meant to draw.
pub fn smooth_path(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let w = SMOOTH_WINDOW as isize;
    (0..points.len() as isize)
        .map(|i| {
            let mut acc = [0.0f32; 2];
            let mut n = 0.0;
            for k in -w..=w {
                let j = (i + k).clamp(0, points.len() as isize - 1) as usize;
                acc[0] += points[j][0];
                acc[1] += points[j][1];
                n += 1.0;
            }
            [acc[0] / n, acc[1] / n]
        })
        .collect()
}

/// Ramer-Douglas-Peucker: keep only the points that carry the shape.
///
/// A freehand drag produces hundreds of samples. Feeding those to the spline
/// directly makes every one a control point, so the curve reproduces the noise
/// instead of smoothing it -- and the road becomes uneditable, because nobody
/// can drag three hundred handles.
pub fn simplify(points: &[[f32; 2]], epsilon_m: f32) -> Vec<[f32; 2]> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let (first, last) = (points[0], points[points.len() - 1]);

    let mut worst = 0.0f32;
    let mut split = 0usize;
    for (i, p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let (d, _) = point_segment(*p, first, last);
        if d > worst {
            worst = d;
            split = i;
        }
    }

    if worst <= epsilon_m {
        return vec![first, last];
    }
    let mut left = simplify(&points[..=split], epsilon_m);
    let right = simplify(&points[split..], epsilon_m);
    left.pop(); // shared point
    left.extend(right);
    left
}

/// Deterministic smooth 1D noise, for lateral wander.
fn wander_noise(s: f32) -> f32 {
    // Two incommensurate sines: no table, no seed plumbing, and the period is
    // long enough that a road never visibly repeats.
    (s * 0.0173).sin() * 0.65 + (s * 0.0416 + 1.7).sin() * 0.35
}

/// Catmull-Rom through `p1`->`p2`, with `p0`/`p3` as tangent neighbours.
fn catmull(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], t: f32) -> [f32; 2] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0; 2];
    for i in 0..2 {
        out[i] = 0.5
            * ((2.0 * p1[i])
                + (-p0[i] + p2[i]) * t
                + (2.0 * p0[i] - 5.0 * p1[i] + 4.0 * p2[i] - p3[i]) * t2
                + (-p0[i] + 3.0 * p1[i] - 3.0 * p2[i] + p3[i]) * t3);
    }
    out
}

/// Resample control points into a centreline at uniform `SAMPLE_M` spacing.
/// Y is left at zero; [`grade_profile`] fills it.
///
/// Two stages, because Catmull-Rom is **not** arc-length parameterised:
/// stepping its parameter uniformly bunches samples where the curve slows.
/// With duplicated end tangents the spacing varies by 2.5x, which silently
/// corrupts any grade computed as rise-per-sample. So: oversample the spline,
/// then walk that polyline at constant distance.
pub fn centreline(points: &[[f32; 2]]) -> Vec<Sample> {
    if points.len() < 2 {
        return Vec::new();
    }
    let at = |i: isize| -> [f32; 2] {
        let n = points.len() as isize;
        points[i.clamp(0, n - 1) as usize]
    };

    // Stage 1: dense, non-uniform.
    let mut dense: Vec<[f32; 2]> = Vec::new();
    for i in 0..points.len() as isize - 1 {
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        let chord = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        // 4x oversampled so the polyline is a close approximation of the curve.
        let steps = ((chord / SAMPLE_M * 4.0).ceil() as usize).max(4);
        for st in 0..steps {
            dense.push(catmull(p0, p1, p2, p3, st as f32 / steps as f32));
        }
    }
    dense.push(points[points.len() - 1]);

    // Stage 2: walk it at constant arc length.
    let mut out = vec![Sample { x: dense[0][0], z: dense[0][1], y: 0.0 }];
    let mut carry = 0.0f32;
    for w in dense.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if seg <= 1e-6 {
            continue;
        }
        let mut travelled = SAMPLE_M - carry;
        while travelled <= seg {
            let t = travelled / seg;
            out.push(Sample { x: a[0] + (b[0] - a[0]) * t, z: a[1] + (b[1] - a[1]) * t, y: 0.0 });
            travelled += SAMPLE_M;
        }
        carry = seg - (travelled - SAMPLE_M);
    }

    let last = points[points.len() - 1];
    let tail = out.last().unwrap();
    // Only append the true end point if the walk did not already land on it.
    if ((tail.x - last[0]).powi(2) + (tail.z - last[1]).powi(2)).sqrt() > SAMPLE_M * 0.25 {
        out.push(Sample { x: last[0], z: last[1], y: 0.0 });
    }
    out
}

/// Nudge a centreline sideways along its length.
///
/// Real tracks are not surveyed -- they follow whatever was easiest at the
/// time and drift by a metre or two over a hundred. A perfectly true dirt road
/// reads as engineered, which is exactly wrong for a mud track. Ends are pinned
/// so the road still meets whatever the user pointed at.
pub fn apply_wander(line: &mut [Sample], amount_m: f32) {
    if amount_m <= 0.0 || line.len() < 3 {
        return;
    }
    let n = line.len();
    let original: Vec<Sample> = line.to_vec();
    for i in 1..n - 1 {
        let (a, b) = (original[i - 1], original[i + 1]);
        let (dx, dz) = (b.x - a.x, b.z - a.z);
        let len = (dx * dx + dz * dz).sqrt().max(1e-4);
        // Perpendicular in the ground plane.
        let (nx, nz) = (-dz / len, dx / len);

        let s = i as f32 * SAMPLE_M;
        // Taper to zero at both ends so the road still hits its end points.
        let taper = ((i as f32 / n as f32) * std::f32::consts::PI).sin();
        let off = wander_noise(s) * amount_m * taper;
        line[i].x = original[i].x + nx * off;
        line[i].z = original[i].z + nz * off;
    }
}

/// Fill in road elevations: sample the terrain, smooth the profile, hold it
/// near the ground, then make the grade legal.
///
/// **Grade is the hard constraint and cut/fill is soft.** Where the two
/// conflict -- a steep hillside with a tight excavation budget -- the road
/// digs deeper rather than becoming undriveable. A road nobody can climb is
/// useless; a deeper cut is merely expensive. Real surveyors would add a
/// switchback, which this tool cannot invent on the user's behalf.
pub fn grade_profile(
    line: &mut [Sample],
    terrain: impl Fn(f32, f32) -> f32,
    max_grade: f32,
    cut_fill_limit_m: f32,
) {
    if line.is_empty() {
        return;
    }
    let ground: Vec<f32> = line.iter().map(|s| terrain(s.x, s.z)).collect();
    for (s, g) in line.iter_mut().zip(&ground) {
        s.y = *g;
    }

    // Measured, never assumed. Resampling makes these ~uniform, but the grade
    // guarantee should not depend on that holding exactly.
    let run: Vec<f32> = line
        .windows(2)
        .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].z - w[0].z).powi(2)).sqrt().max(1e-4))
        .collect();
    let grade = max_grade.max(0.005);

    // Symmetric relaxation: each pass sweeps forward then backward so the
    // profile does not drift toward whichever end is processed first. This
    // shapes the profile but does not guarantee anything -- diffusing a change
    // across n samples by halves needs O(n^2) passes, far more than is worth
    // running.
    for _ in 0..GRADE_PASSES {
        for i in 1..line.len() {
            let max_step = run[i - 1] * grade;
            let d = line[i].y - line[i - 1].y;
            if d.abs() > max_step {
                let fix = (d.abs() - max_step) * 0.5 * d.signum();
                line[i].y -= fix;
                line[i - 1].y += fix;
            }
        }
        for i in (1..line.len()).rev() {
            let max_step = run[i - 1] * grade;
            let d = line[i].y - line[i - 1].y;
            if d.abs() > max_step {
                let fix = (d.abs() - max_step) * 0.5 * d.signum();
                line[i].y -= fix;
                line[i - 1].y += fix;
            }
        }
        // Never excavate or embank more than allowed. Applied inside the loop
        // so the grade relaxation works against the clamp rather than being
        // undone by it at the end.
        for (s, g) in line.iter_mut().zip(&ground) {
            s.y = s.y.clamp(g - cut_fill_limit_m, g + cut_fill_limit_m);
        }
    }

    // Guarantee. Each sample is pinned within one legal step of the sample
    // before it, which are already final -- so a single forward sweep makes
    // every segment legal by construction, no iteration needed.
    for i in 1..line.len() {
        let max_step = run[i - 1] * grade;
        let prev = line[i - 1].y;
        line[i].y = line[i].y.clamp(prev - max_step, prev + max_step);
    }
}

/// Distance from `p` to segment `a`-`b`, plus the parameter along it.
fn point_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> (f32, f32) {
    let (dx, dz) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dz * dz;
    if len2 <= 1e-9 {
        return (((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt(), 0.0);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / len2).clamp(0.0, 1.0);
    let (cx, cz) = (a[0] + dx * t, a[1] + dz * t);
    (((p[0] - cx).powi(2) + (p[1] - cz).powi(2)).sqrt(), t)
}

fn smoothstep(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Stamp one road into `heights`, accumulating into `surface`.
///
/// `heights` is row-major `res * res` in meters, covering `extent_m` centered
/// on the origin.
///
/// Two passes. The first finds, for every affected texel, the nearest point on
/// the whole centreline; the second applies the cross-section once using that
/// distance.
///
/// It cannot be done in one pass. The influence radius spans many one-metre
/// segments, so a texel is in range of dozens of them at different distances --
/// evaluating the profile per segment and combining the results lets each texel
/// pick whichever segment happens to sit at rut distance, which flattens the
/// carriageway into a uniformly rutted band with no camber at all.
pub fn stamp(heights: &mut [f32], surface: &mut RoadSurface, res: u32, extent_m: f32, road: &Road) {
    let line = {
        let mut l = centreline(&road.points);
        if l.len() < 2 {
            return;
        }
        apply_wander(&mut l, road.wander_m);
        let sample = |x: f32, z: f32| sample_height(heights, res, extent_m, x, z);
        grade_profile(&mut l, sample, road.max_grade, road.cut_fill_limit_m);
        l
    };

    let n = res as i32;
    let step = extent_m / (res - 1) as f32;
    let half_extent = extent_m * 0.5;
    let influence = road.influence_m();
    let half_w = road.width_m * 0.5;
    let verge = half_w + road.shoulder_m;
    let tan_batter = road.batter_angle_deg.to_radians().tan().max(0.05);
    let rut_sigma = (road.rut_spacing_m * 0.22).max(0.15);

    if surface.mask.len() != heights.len() {
        surface.mask = vec![0.0; heights.len()];
        surface.rut = vec![0.0; heights.len()];
    }

    // Texel box covering the whole road plus its batter slopes.
    let to_idx = |v: f32| (((v + half_extent) / step).floor() as i32).clamp(0, n - 1);
    let (mut bx0, mut bx1, mut bz0, mut bz1) = (n - 1, 0, n - 1, 0);
    for s in &line {
        bx0 = bx0.min(to_idx(s.x - influence));
        bx1 = bx1.max(to_idx(s.x + influence));
        bz0 = bz0.min(to_idx(s.z - influence));
        bz1 = bz1.max(to_idx(s.z + influence));
    }
    if bx0 > bx1 || bz0 > bz1 {
        return;
    }
    let bw = (bx1 - bx0 + 1) as usize;
    let bh = (bz1 - bz0 + 1) as usize;

    // Pass 1: nearest centreline point per texel.
    let mut best_dist = vec![f32::INFINITY; bw * bh];
    let mut best_y = vec![0.0f32; bw * bh];
    for w in line.windows(2) {
        let (a, b) = (w[0], w[1]);
        let sx0 = to_idx(a.x.min(b.x) - influence).max(bx0);
        let sx1 = to_idx(a.x.max(b.x) + influence).min(bx1);
        let sz0 = to_idx(a.z.min(b.z) - influence).max(bz0);
        let sz1 = to_idx(a.z.max(b.z) + influence).min(bz1);

        for zi in sz0..=sz1 {
            let wz = zi as f32 * step - half_extent;
            for xi in sx0..=sx1 {
                let wx = xi as f32 * step - half_extent;
                let (dist, t) = point_segment([wx, wz], [a.x, a.z], [b.x, b.z]);
                let k = (zi - bz0) as usize * bw + (xi - bx0) as usize;
                if dist < best_dist[k] {
                    best_dist[k] = dist;
                    best_y[k] = a.y + (b.y - a.y) * t;
                }
            }
        }
    }

    // Pass 2: one cross-section per texel.
    for zi in bz0..=bz1 {
        for xi in bx0..=bx1 {
            let k = (zi - bz0) as usize * bw + (xi - bx0) as usize;
            let dist = best_dist[k];
            if dist > influence {
                continue;
            }
            let i = (zi * n + xi) as usize;
            let terrain = heights[i];
            let road_y = best_y[k];

            // Crown: centre high, edges low, so water sheds sideways.
            let across = (dist / half_w).min(1.0);
            let mut target = road_y - road.camber * half_w * across * across;

            // Two wheel ruts, gaussian in cross-section.
            //
            // At tier-0 resolution (2-4 m/texel) ruts 1.8 m apart are sub-texel,
            // so this displacement will not be visible -- the `rut` mask is what
            // actually renders them, as a shader detail and a wetness cue. The
            // geometry is kept because it is correct wherever the field is fine
            // enough to resolve it.
            let rut = if dist <= verge {
                let d = (dist - road.rut_spacing_m * 0.5) / rut_sigma;
                (-0.5 * d * d).exp()
            } else {
                0.0
            };
            target -= road.rut_depth_m * rut;

            let (new_h, mask) = if dist <= verge {
                (target, 1.0 - smoothstep((dist - half_w) / road.shoulder_m.max(0.01)) * 0.35)
            } else {
                // Batter: rise to meet terrain on a cut, fall away on a fill.
                // Taking min/max against the terrain means the slope simply
                // stops where it reaches ground.
                let over = dist - verge;
                if target < terrain {
                    (terrain.min(target + over * tan_batter), 0.0)
                } else {
                    (terrain.max(target - over * tan_batter), 0.0)
                }
            };

            heights[i] = new_h;
            surface.mask[i] = surface.mask[i].max(mask);
            surface.rut[i] = surface.rut[i].max(rut);
        }
    }
}

/// Stamp a whole network in order.
pub fn stamp_network(
    heights: &mut [f32],
    res: u32,
    extent_m: f32,
    network: &RoadNetwork,
) -> RoadSurface {
    let mut surface = RoadSurface { mask: vec![0.0; heights.len()], rut: vec![0.0; heights.len()] };
    for road in network.drawable() {
        stamp(heights, &mut surface, res, extent_m, road);
    }
    surface
}

/// Bilinear terrain height at a world position.
pub fn sample_height(heights: &[f32], res: u32, extent_m: f32, x: f32, z: f32) -> f32 {
    let n = res as i32;
    let fx = ((x / extent_m + 0.5) * (res - 1) as f32).clamp(0.0, (res - 1) as f32);
    let fz = ((z / extent_m + 0.5) * (res - 1) as f32).clamp(0.0, (res - 1) as f32);
    let (x0, z0) = (fx.floor() as i32, fz.floor() as i32);
    let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
    let at = |ix: i32, iz: i32| -> f32 {
        heights[(iz.clamp(0, n - 1) * n + ix.clamp(0, n - 1)) as usize]
    };
    let top = at(x0, z0) + (at(x0 + 1, z0) - at(x0, z0)) * tx;
    let bot = at(x0, z0 + 1) + (at(x0 + 1, z0 + 1) - at(x0, z0 + 1)) * tx;
    top + (bot - top) * tz
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_project::roads::Road;

    const RES: u32 = 128;
    const EXTENT: f32 = 512.0;

    fn flat(h: f32) -> Vec<f32> {
        vec![h; (RES * RES) as usize]
    }

    /// Terrain rising steeply along +X.
    fn ramp(grade: f32) -> Vec<f32> {
        let step = EXTENT / (RES - 1) as f32;
        (0..RES * RES)
            .map(|i| {
                let x = (i % RES) as f32 * step - EXTENT * 0.5;
                100.0 + x * grade
            })
            .collect()
    }

    /// Wander is deliberately off here: these tests assert the cross-section,
    /// and a road that drifts sideways no longer passes through the texel they
    /// sample. Wander itself is covered by its own test.
    fn straight_road() -> Road {
        Road { points: vec![[-200.0, 0.0], [200.0, 0.0]], wander_m: 0.0, ..Default::default() }
    }

    #[test]
    fn simplify_keeps_shape_and_drops_noise() {
        // A gentle arc sampled densely, with jitter on top.
        let raw: Vec<[f32; 2]> = (0..300)
            .map(|i| {
                let t = i as f32;
                let jitter = if i % 2 == 0 { 0.05 } else { -0.05 };
                [t, (t * 0.01).sin() * 40.0 + jitter]
            })
            .collect();

        let smoothed = smooth_path(&raw);
        let out = simplify(&smoothed, 1.0);
        assert!(out.len() < 40, "should collapse 300 samples, kept {}", out.len());
        assert!(out.len() >= 3, "an arc needs more than a straight line");
        // Endpoints of the *simplified* path, which is the smoothed one --
        // smoothing shifts the ends slightly, and that is fine.
        assert_eq!(out[0], smoothed[0], "must keep the first point");
        assert_eq!(*out.last().unwrap(), smoothed[smoothed.len() - 1], "must keep the last point");
    }

    #[test]
    fn simplify_collapses_a_straight_line_to_two_points() {
        let raw: Vec<[f32; 2]> = (0..100).map(|i| [i as f32, 0.0]).collect();
        assert_eq!(simplify(&raw, 0.5).len(), 2);
    }

    #[test]
    fn a_default_road_wanders() {
        // A dirt track that runs perfectly true looks surveyed. This is the
        // difference between a road and a ruler, so it is on by default.
        assert!(Road::default().wander_m > 0.0);
    }

    #[test]
    fn wander_moves_the_middle_but_pins_the_ends() {
        let mut line = centreline(&[[0.0, 0.0], [200.0, 0.0]]);
        let before = line.clone();
        apply_wander(&mut line, 3.0);

        assert!((line[0].x - before[0].x).abs() < 1e-4);
        assert!((line[0].z - before[0].z).abs() < 1e-4);
        let n = line.len() - 1;
        assert!((line[n].z - before[n].z).abs() < 1e-3, "the far end must stay put");

        let mid = line.len() / 2;
        assert!(
            (line[mid].z - before[mid].z).abs() > 0.1,
            "the middle should have drifted off the straight line"
        );
    }

    #[test]
    fn centreline_passes_through_the_end_points() {
        let line = centreline(&[[0.0, 0.0], [100.0, 0.0], [200.0, 50.0]]);
        assert!(line.len() > 100, "should resample to ~1 m spacing");
        assert!((line[0].x).abs() < 1e-3 && (line[0].z).abs() < 1e-3);
        let last = line.last().unwrap();
        assert!((last.x - 200.0).abs() < 1e-3 && (last.z - 50.0).abs() < 1e-3);
    }

    #[test]
    fn grade_limiting_respects_the_maximum() {
        let h = ramp(0.5); // 50% terrain, far steeper than any road
        let mut line = centreline(&[[-200.0, 0.0], [200.0, 0.0]]);
        let max_grade = 0.1;
        grade_profile(
            &mut line,
            |x, z| sample_height(&h, RES, EXTENT, x, z),
            max_grade,
            1000.0, // no cut/fill limit, so grade is the only constraint
        );

        for w in line.windows(2) {
            let run = ((w[1].x - w[0].x).powi(2) + (w[1].z - w[0].z).powi(2)).sqrt();
            let grade = (w[1].y - w[0].y).abs() / run.max(1e-4);
            assert!(grade <= max_grade * 1.05, "segment grade {grade} exceeds {max_grade}");
        }
    }

    #[test]
    fn cut_fill_limit_holds_when_it_does_not_fight_the_grade() {
        // Terrain at 8%, road allowed 12% -- no conflict, so the soft
        // excavation limit should be respected everywhere.
        let h = ramp(0.08);
        let mut line = centreline(&[[-200.0, 0.0], [200.0, 0.0]]);
        let limit = 3.0;
        grade_profile(&mut line, |x, z| sample_height(&h, RES, EXTENT, x, z), 0.12, limit);

        for s in &line {
            let ground = sample_height(&h, RES, EXTENT, s.x, s.z);
            assert!(
                (s.y - ground).abs() <= limit + 0.05,
                "road sits {} m from ground, limit {limit}",
                (s.y - ground).abs()
            );
        }
    }

    #[test]
    fn grade_wins_when_it_conflicts_with_the_excavation_budget() {
        // 50% terrain, 10% road, only 3 m of cut allowed. Those cannot both
        // hold; the road must stay driveable and dig deeper than budgeted.
        let h = ramp(0.5);
        let mut line = centreline(&[[-200.0, 0.0], [200.0, 0.0]]);
        grade_profile(&mut line, |x, z| sample_height(&h, RES, EXTENT, x, z), 0.1, 3.0);

        for w in line.windows(2) {
            let run = ((w[1].x - w[0].x).powi(2) + (w[1].z - w[0].z).powi(2)).sqrt();
            assert!((w[1].y - w[0].y).abs() / run.max(1e-4) <= 0.1 * 1.05);
        }
        let worst = line
            .iter()
            .map(|s| (s.y - sample_height(&h, RES, EXTENT, s.x, s.z)).abs())
            .fold(0.0f32, f32::max);
        assert!(worst > 3.0, "the budget should have yielded, deviation was only {worst}");
    }

    #[test]
    fn a_road_on_flat_ground_barely_moves_it() {
        let mut h = flat(100.0);
        let before = h.clone();
        let mut s = RoadSurface::default();
        stamp(&mut h, &mut s, RES, EXTENT, &straight_road());

        // Camber and ruts are centimetres; nothing should move by a metre.
        let worst = h.iter().zip(&before).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.5, "flat ground moved by {worst} m");
    }

    #[test]
    fn the_mask_marks_the_carriageway_and_nothing_far_away() {
        let mut h = flat(100.0);
        let mut s = RoadSurface::default();
        let road = straight_road();
        stamp(&mut h, &mut s, RES, EXTENT, &road);

        let idx = |x: f32, z: f32| {
            let step = EXTENT / (RES - 1) as f32;
            let xi = ((x + EXTENT * 0.5) / step).round() as usize;
            let zi = ((z + EXTENT * 0.5) / step).round() as usize;
            zi * RES as usize + xi
        };
        assert!(s.mask[idx(0.0, 0.0)] > 0.9, "centreline should be fully road");
        assert_eq!(s.mask[idx(0.0, 200.0)], 0.0, "far from the road must be untouched");
    }

    #[test]
    fn ruts_are_cut_beside_the_centre_not_on_it() {
        // A fine grid on purpose: at the 2-4 m/texel the editor actually uses,
        // ruts 1.8 m apart fall inside a single texel and cannot be resolved
        // as geometry at all. This checks the cross-section is right where the
        // field is fine enough to show it.
        const FINE: u32 = 128;
        const SMALL: f32 = 64.0; // 0.5 m/texel
        let mut h = vec![100.0f32; (FINE * FINE) as usize];
        let mut s = RoadSurface::default();
        let road =
            Road { points: vec![[-30.0, 0.0], [30.0, 0.0]], wander_m: 0.0, ..Default::default() };
        stamp(&mut h, &mut s, FINE, SMALL, &road);

        let step = SMALL / (FINE - 1) as f32;
        let at = |z: f32| {
            let zi = ((z + SMALL * 0.5) / step).round() as usize;
            let xi = ((SMALL * 0.5) / step).round() as usize;
            h[zi * FINE as usize + xi]
        };
        assert!(
            at(0.0) > at(road.rut_spacing_m * 0.5),
            "centre crown {} should sit above the wheel track {}",
            at(0.0),
            at(road.rut_spacing_m * 0.5)
        );
    }

    #[test]
    fn cutting_through_a_hill_lowers_it() {
        let mut h = ramp(0.4);
        let before = h.clone();
        let mut s = RoadSurface::default();
        // Cross the slope, so the road must cut on one side.
        stamp(&mut h, &mut s, RES, EXTENT, &straight_road());

        let cut: f32 = before.iter().zip(&h).map(|(b, a)| (b - a).max(0.0)).sum();
        assert!(cut > 0.0, "a road across a slope must excavate something");
    }

    #[test]
    fn an_empty_or_single_point_road_is_a_no_op() {
        let mut h = flat(100.0);
        let before = h.clone();
        let mut s = RoadSurface::default();
        stamp(&mut h, &mut s, RES, EXTENT, &Road::default());
        stamp(
            &mut h,
            &mut s,
            RES,
            EXTENT,
            &Road { points: vec![[0.0, 0.0]], ..Default::default() },
        );
        assert_eq!(h, before);
    }
}

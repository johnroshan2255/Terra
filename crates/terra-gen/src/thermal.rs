//! Thermal erosion -- material sliding down slopes steeper than the angle of
//! repose.
//!
//! Run briefly *before* hydraulic erosion to relax raw noise artifacts (the
//! solver otherwise spends its budget fighting them), and longer *after* to
//! build the talus/scree aprons at cliff bases.
//!
//! On the CPU: unlike the hydraulic solver this is a handful of passes rather
//! than thousands, so a rayon loop finishes in well under a second and avoids
//! another set of GPU buffers and bind groups.

use rayon::prelude::*;
use terra_project::params::ThermalParams;

const NEIGHBOURS: [(i32, i32); 8] =
    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)];

/// One pass over the field. Returns the new heights.
///
/// Two parallel sweeps rather than one parallel and one serial. The obvious
/// formulation -- remove material, then push it onto downhill neighbours -- is
/// a scatter, so it cannot be parallelized without contention. Recasting the
/// second sweep as a gather (each cell pulls what its uphill neighbours shed)
/// gives identical results and scales across cores; at 2048^2 that was the
/// difference between 46 ms and 4 ms per iteration.
fn pass(src: &[f32], res: u32, cell_size_m: f32, talus_deg: f32, rate: f32) -> Vec<f32> {
    let n = res as i32;
    // Maximum height difference a neighbour can hold before material slides.
    let max_drop = talus_deg.to_radians().tan() * cell_size_m;

    let at = |x: i32, y: i32| -> f32 { src[(y.clamp(0, n - 1) * n + x.clamp(0, n - 1)) as usize] };

    // Sweep 1: how much each cell sheds, and the total excess it is spread over.
    let mut shed = vec![0.0f32; src.len()];
    let mut excess = vec![0.0f32; src.len()];
    shed.par_chunks_mut(res as usize)
        .zip(excess.par_chunks_mut(res as usize))
        .enumerate()
        .for_each(|(y, (shed_row, excess_row))| {
            let y = y as i32;
            for x in 0..n {
                let h = at(x, y);
                let mut total = 0.0f32;
                let mut steepest = 0.0f32;
                for (dx, dy) in NEIGHBOURS {
                    let (nx, ny) = (x + dx, y + dy);
                    // Skip rather than clamp. Clamping makes an out-of-bounds
                    // diagonal alias onto a real neighbour, inflating the
                    // denominator here while the gather sweep counts that
                    // neighbour once -- and the difference is material lost.
                    if nx < 0 || ny < 0 || nx >= n || ny >= n {
                        continue;
                    }
                    let d = h - at(nx, ny) - max_drop;
                    if d > 0.0 {
                        total += d;
                        steepest = steepest.max(d);
                    }
                }
                excess_row[x as usize] = total;
                // At most half the steepest excess, so a cell and its neighbour
                // cannot swap places and oscillate.
                shed_row[x as usize] = if total > 0.0 { rate * steepest * 0.5 } else { 0.0 };
            }
        });

    // Sweep 2: gather. Each cell pulls its share of what every uphill
    // neighbour shed, weighted by the drop between them.
    let get = |v: &[f32], x: i32, y: i32| -> f32 {
        v[(y.clamp(0, n - 1) * n + x.clamp(0, n - 1)) as usize]
    };
    let mut dst = vec![0.0f32; src.len()];
    dst.par_chunks_mut(res as usize).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        for x in 0..n {
            let i = (y * n + x) as usize;
            let mut h = src[i] - shed[i];
            for (dx, dy) in NEIGHBOURS {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= n || ny >= n {
                    continue;
                }
                let total = get(&excess, nx, ny);
                if total <= 0.0 {
                    continue;
                }
                let d = at(nx, ny) - src[i] - max_drop;
                if d > 0.0 {
                    h += get(&shed, nx, ny) * (d / total);
                }
            }
            row[x as usize] = h;
        }
    });
    dst
}

/// Apply `iterations` thermal passes.
pub fn run(
    heights: &[f32],
    res: u32,
    cell_size_m: f32,
    p: &ThermalParams,
    iterations: u32,
) -> Vec<f32> {
    let mut cur = heights.to_vec();
    for _ in 0..iterations {
        cur = pass(&cur, res, cell_size_m, p.talus_angle_deg, p.rate);
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> ThermalParams {
        ThermalParams::default()
    }

    #[test]
    fn a_spike_collapses_toward_the_angle_of_repose() {
        let res = 9u32;
        let mut h = vec![0.0f32; (res * res) as usize];
        h[(4 * res + 4) as usize] = 100.0;

        let out = run(&h, res, 4.0, &params(), 40);
        let peak = out[(4 * res + 4) as usize];
        assert!(peak < 100.0, "spike must shed material, got {peak}");
        assert!(out.iter().any(|v| *v > 0.0 && *v < peak), "material must land nearby");
    }

    #[test]
    fn conserves_material() {
        let res = 16u32;
        let h: Vec<f32> = (0..res * res).map(|i| ((i * 37) % 101) as f32).collect();
        let before: f32 = h.iter().sum();
        let after: f32 = run(&h, res, 4.0, &params(), 5).iter().sum();
        // Thermal erosion moves material, it does not create or destroy it.
        assert!((before - after).abs() / before < 0.02, "{before} -> {after}");
    }

    #[test]
    fn flat_ground_is_untouched() {
        let h = vec![50.0f32; 64];
        assert_eq!(run(&h, 8, 4.0, &params(), 10), h);
    }
}

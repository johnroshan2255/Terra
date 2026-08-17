//! Splitting a vehicle glTF into a body and four wheels, and measuring it.
//!
//! A scattered rock is one transform and is rightly flattened into a single buffer by
//! [`crate::mesh::load_gltf`]. A car is not: its wheels steer and spin independently of
//! the body, so they have to arrive as separate meshes with known centres, and the
//! suspension has to know where those centres are.
//!
//! # Why the wheels are found by geometry rather than by name
//!
//! Naming is not something an asset can be relied on for. The Hummer this was built
//! against has no node names at all -- nineteen unnamed nodes, eight meshes -- which is
//! entirely normal for a downloaded or exported model, and a loader keyed on the string
//! `"wheel"` would simply have found nothing.
//!
//! Geometry is reliable, because a road wheel is a very specific object: four parts of
//! near-identical size, mirrored in pairs about the centreline, round in profile and
//! narrower across the axle than they are tall. That description picks the four wheels
//! out of this file without reading a single string, and it would pick them out of
//! another four-wheeled vehicle too.
//!
//! # Why every dimension comes from here
//!
//! The collision box, the suspension mount points and the drawn mesh all have to agree.
//! Before this they did not even try: the collider was a hand-typed 3.6 m "small
//! hatchback" half-extent and the renderer drew a matching box, neither connected to any
//! vehicle model. Dropping a 5.2 m Hummer into that would have left the wheels outside
//! the arches and the body grounding out on nothing, with no way to tell from the code
//! that anything was wrong. Measuring the rig makes the collider the shape you can see,
//! and keeps it that way if the mesh is replaced.

use crate::mesh::{MeshData, MeshPart, load_gltf_parts};
use anyhow::Result;
use std::path::Path;
use terra_core::VehicleDims;

/// A vehicle mesh split into its moving parts, with the rig measured from it.
pub struct VehicleRig {
    /// Everything that is not a wheel, one entry per source part, all drawn at the chassis
    /// transform.
    ///
    /// Kept separate rather than merged because the renderer binds one texture per draw and
    /// this vehicle has five materials. Merging kept the first albedo and drew the glass,
    /// the interior and the underbody with the body shell's map. Four extra draws for one
    /// object is nothing next to getting the materials right.
    pub body: Vec<MeshData>,
    /// The four wheels, each recentred on its own axle so it can be rotated, in the
    /// order front-left, front-right, rear-left, rear-right.
    ///
    /// Four meshes rather than one reused four times, because the left and right wheels
    /// are mirrored copies in the file. Reusing one would need a negative scale on the
    /// far side, which inverts triangle winding and lets back-face culling eat the
    /// visible faces -- the tyre turns inside out on one side of the car.
    pub wheels: [MeshData; 4],
    pub dims: VehicleDims,
}

/// Corner order. The physics controller adds its wheels in the same order, so a wheel
/// index means one thing from the mesh through to the suspension.
const CORNERS: [(&str, f32, f32); 4] = [
    ("front-left", -1.0, 1.0),
    ("front-right", 1.0, 1.0),
    ("rear-left", -1.0, -1.0),
    ("rear-right", 1.0, -1.0),
];

/// Fraction of the body's height, from its underside up, that the collision box covers.
///
/// Not the full height. A box reaching the roof puts the centre of mass 1.2 m up on a
/// 2.0 m track, which tips the vehicle in any corner taken at speed -- and the roof and
/// aerial should not stop it against a rock face either. Four fifths keeps the box over
/// the real mass while leaving the greenhouse out of it.
const BODY_HEIGHT_FRACTION: f32 = 0.80;

/// Metres taken off each side of the collision box.
///
/// Mirrors and arch flares set the visual width, and a collider that wide catches on
/// scenery the vehicle looks like it should brush past.
const WIDTH_INSET_M: f32 = 0.10;

impl VehicleRig {
    /// Load and split a vehicle model.
    ///
    /// `mass_kg` is the one figure that cannot be measured: a mesh has no density.
    pub fn from_gltf(path: &Path, mass_kg: f32) -> Result<Self> {
        let mut parts = load_gltf_parts(path)?;
        anyhow::ensure!(
            parts.len() >= 5,
            "{}: {} parts is not enough for a body and four wheels",
            path.display(),
            parts.len()
        );

        // Turn the vehicle to face +Z before measuring anything.
        //
        // glTF fixes no facing for objects, and this Hummer is modelled nose-first along
        // -Z. Every other part of the engine treats +Z as forward -- the vehicle
        // controller's `index_forward_axis`, the chase camera, the chassis collider -- so
        // normalising here means exactly one place has to know, instead of every consumer
        // carrying a sign.
        //
        // It is not a cosmetic detail. Facing this the wrong way makes positive throttle
        // drive the vehicle in reverse and puts the chase camera in front of the bonnet.
        if nose_direction(&parts) < 0.0 {
            log::info!("{}: modelled nose-first along -Z, turning it round", path.display());
            for part in &mut parts {
                flip_about_y(part);
            }
        }

        // Score every part, then take the best four. Scoring rather than filtering means
        // an extra round part -- a spare on the tailgate, a steering wheel -- loses to
        // the real wheels instead of displacing one of them.
        let mut scored: Vec<(usize, f32)> =
            parts.iter().enumerate().map(|(i, p)| (i, wheel_score(p))).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        anyhow::ensure!(
            scored[3].1 > 0.0,
            "{}: only {} parts look like wheels",
            path.display(),
            scored.iter().filter(|(_, s)| *s > 0.0).count()
        );

        let mut wheel_idx: Vec<usize> = scored.iter().take(4).map(|(i, _)| *i).collect();
        // Into the fixed corner order, so wheel 0 is always front-left.
        wheel_idx.sort_by_key(|&i| {
            let c = parts[i].centre();
            (if c[2] >= 0.0 { 0 } else { 1 }, if c[0] < 0.0 { 0 } else { 1 })
        });

        let centres: Vec<[f32; 3]> = wheel_idx.iter().map(|&i| parts[i].centre()).collect();

        // They must actually form two mirrored pairs. If they do not, the geometric guess
        // has picked up something that is not a wheel, and continuing would mount the
        // suspension in the wrong places -- which looks like a physics bug rather than an
        // import one.
        for (n, (name, sx, sz)) in CORNERS.iter().enumerate() {
            let c = centres[n];
            anyhow::ensure!(
                c[0] * sx > 0.0 && c[2] * sz > 0.0,
                "{}: the {name} wheel came out at {c:?}, which is not that corner",
                path.display()
            );
        }

        // Radius from the two round axes; the third is the tyre's width.
        let radius = wheel_idx
            .iter()
            .map(|&i| {
                let s = parts[i].size();
                (s[1] + s[2]) * 0.25
            })
            .sum::<f32>()
            / 4.0;
        let width = wheel_idx.iter().map(|&i| parts[i].size()[0]).sum::<f32>() / 4.0;

        // Averaged across each axle rather than taken from one wheel, so a model that is
        // a millimetre asymmetric does not give the car crab steer.
        let axle_half_width =
            (centres[0][0].abs() + centres[1][0].abs() + centres[2][0].abs() + centres[3][0].abs())
                / 4.0;
        let front_axle_z = (centres[0][2] + centres[1][2]) * 0.5;
        let rear_axle_z = (centres[2][2] + centres[3][2]) * 0.5;

        let body_parts: Vec<usize> = (0..parts.len()).filter(|i| !wheel_idx.contains(i)).collect();
        anyhow::ensure!(
            !body_parts.is_empty(),
            "{}: nothing left for a body after the wheels",
            path.display()
        );

        let mut body_min = [f32::MAX; 3];
        let mut body_max = [f32::MIN; 3];
        for &i in &body_parts {
            for a in 0..3 {
                body_min[a] = body_min[a].min(parts[i].min[a]);
                body_max[a] = body_max[a].max(parts[i].max[a]);
            }
        }

        // The collision box's floor sits at three quarters of a wheel radius, which is
        // about where a real chassis rail runs: above the axles, below the sills. Taking
        // it from the wheel rather than from the body's own underside keeps stray
        // underbody geometry -- differentials, exhaust -- out of the collider.
        let floor = radius * 0.75;
        let roof = body_min[1] + (body_max[1] - body_min[1]) * BODY_HEIGHT_FRACTION;
        let chassis_half = [
            ((body_max[0] - body_min[0]) * 0.5 - WIDTH_INSET_M).max(0.1),
            ((roof - floor) * 0.5).max(0.1),
            ((body_max[2] - body_min[2]) * 0.5).max(0.1),
        ];

        let dims = VehicleDims {
            chassis_half,
            chassis_centre_y: floor + chassis_half[1],
            wheel_radius: radius,
            wheel_width: width,
            axle_half_width,
            front_axle_z,
            rear_axle_z,
            mass_kg,
        };

        // Recentre each wheel on its own axle, so rotating it spins the tyre rather than
        // swinging it around the vehicle's origin.
        let mut wheels: Vec<MeshData> = Vec::with_capacity(4);
        for (n, &i) in wheel_idx.iter().enumerate() {
            let mut w = parts[i].data.clone();
            // Vertical centre from the measured radius, not the part's bounding-box
            // centre: a tyre modelled with a flattened contact patch has its box centre
            // slightly above the true axle, and using it would show as the wheel sinking
            // and rising as it spins.
            let c = [centres[n][0], radius, centres[n][2]];
            for p in &mut w.positions {
                for a in 0..3 {
                    p[a] -= c[a];
                }
            }
            wheels.push(w);
        }
        let wheels: [MeshData; 4] =
            wheels.try_into().map_err(|_| anyhow::anyhow!("wheel count changed"))?;

        // Largest part first, so the biggest piece of the body is drawn before the trim.
        let mut order = body_parts.clone();
        order.sort_by_key(|&i| std::cmp::Reverse(parts[i].data.positions.len()));
        let body: Vec<MeshData> = order.into_iter().map(|i| parts[i].data.clone()).collect();

        Ok(Self { body, wheels, dims })
    }
}

/// Which way the vehicle's nose points along Z: `+1.0` or `-1.0`.
///
/// From the roofline. A vehicle's bonnet is lower than its cabin, so the end of the body
/// with the lower silhouette is the front -- on the measured Hummer the body rises from
/// 1.68 m at one end to 2.47 m over the cabin, which is a sloping bonnet followed by a
/// windscreen and is not ambiguous.
///
/// Measured on the largest part, which is the body shell: wheels are symmetric and trim is
/// too small to read. When the two ends are within a tenth of the body's height of each
/// other the shape says nothing useful, so this assumes the engine's own convention and
/// says so rather than guessing from noise.
fn nose_direction(parts: &[MeshPart]) -> f32 {
    let Some(shell) = parts.iter().max_by_key(|p| p.data.positions.len()) else {
        return 1.0;
    };
    let (zlo, zhi) = (shell.min[2], shell.max[2]);
    let span = zhi - zlo;
    let height = shell.max[1] - shell.min[1];
    if span <= 0.0 || height <= 0.0 {
        return 1.0;
    }

    // The outer eighth at each end, which is bonnet or tailgate and nothing else.
    let band = span / 8.0;
    let mut low_end = f32::MIN;
    let mut high_end = f32::MIN;
    for v in &shell.data.positions {
        if v[2] < zlo + band {
            low_end = low_end.max(v[1]);
        } else if v[2] > zhi - band {
            high_end = high_end.max(v[1]);
        }
    }
    if low_end == f32::MIN || high_end == f32::MIN {
        return 1.0;
    }

    if (low_end - high_end).abs() < height * 0.1 {
        log::warn!("vehicle roofline is symmetric; assuming it faces +Z");
        return 1.0;
    }
    // The lower end is the bonnet.
    if low_end < high_end { -1.0 } else { 1.0 }
}

/// Rotate a part 180 degrees about the Y axis, in place.
///
/// Negating X and Z is that rotation. Winding is unaffected: a 180-degree rotation is a
/// proper rotation, not a mirror, so triangles keep their orientation and back-face culling
/// still works -- which is the trap with fixing a facing by scaling an axis by -1.
fn flip_about_y(part: &mut MeshPart) {
    for p in &mut part.data.positions {
        p[0] = -p[0];
        p[2] = -p[2];
    }
    for n in &mut part.data.normals {
        n[0] = -n[0];
        n[2] = -n[2];
    }
    for a in [0usize, 2] {
        let (lo, hi) = (part.min[a], part.max[a]);
        part.min[a] = -hi;
        part.max[a] = -lo;
    }
}

/// How much one part looks like a road wheel, `0.0` for "not at all".
///
/// Three properties, all of which a wheel has and little else on a vehicle does: its two
/// non-axle axes are near equal, it is narrower along the axle than it is tall, and it is
/// offset from the centreline.
///
/// Deliberately no size threshold in metres. That would only work for vehicles authored
/// at one scale, and glTF files arrive at every scale there is.
fn wheel_score(p: &MeshPart) -> f32 {
    let s = p.size();
    let c = p.centre();
    if s[0] <= 0.0 || s[1] <= 0.0 || s[2] <= 0.0 {
        return 0.0;
    }
    // Round: the two non-axle axes agree, because a wheel is a disc.
    let round = 1.0 - ((s[1] - s[2]).abs() / s[1].max(s[2]));
    // Narrow: a tyre is thinner than its diameter. A body panel is not.
    let narrow = 1.0 - (s[0] / s[1].max(s[2])).min(1.0);
    // Off the centreline. A spare wheel on the tailgate scores well on the first two,
    // which is exactly why this third term exists.
    let offset = if c[0].abs() > s[0] * 0.5 { 1.0 } else { 0.0 };
    if round < 0.85 || narrow < 0.3 {
        return 0.0;
    }
    round + narrow + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hummer() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/game/hummer.glb")
    }

    fn rig() -> Option<VehicleRig> {
        let p = hummer();
        if !p.exists() {
            return None;
        }
        Some(VehicleRig::from_gltf(&p, 2900.0).expect("the hummer should rig"))
    }

    #[test]
    fn the_hummer_splits_into_a_body_and_four_wheels() {
        let Some(r) = rig() else { return };
        assert!(!r.body.is_empty(), "no body parts");
        assert!(r.body.iter().all(|b| !b.positions.is_empty()), "an empty body part");
        for (n, w) in r.wheels.iter().enumerate() {
            assert!(!w.positions.is_empty(), "wheel {n} is empty");
            assert!(!w.indices.is_empty(), "wheel {n} has no triangles");
        }
        // The body must be the bulk of it. If a wheel search went wrong and swallowed the
        // shell, the body would come out small instead of missing.
        let wheel_verts: usize = r.wheels.iter().map(|w| w.positions.len()).sum();
        let body_verts: usize = r.body.iter().map(|b| b.positions.len()).sum();
        assert!(
            body_verts > wheel_verts,
            "body has {body_verts} vertices against {wheel_verts} of wheels"
        );
        // Every part keeps its own material, which is the point of not merging them.
        assert!(
            r.body.iter().all(|b| b.albedo.is_some()),
            "a body part lost its texture in the split"
        );
    }

    #[test]
    fn the_measured_rig_is_a_hummer_and_not_a_hatchback() {
        // The numbers the collider and suspension are built from. Checked against the
        // real vehicle, because the failure this guards against -- a mesh and a collider
        // of different sizes -- is invisible in code and obvious on screen.
        let Some(r) = rig() else { return };
        let d = r.dims;
        // H1: 3.30 m wheelbase, 4.69 m long, 2.19 m wide, 0.41 m clearance, 0.94 m tyres.
        assert!((3.1..3.7).contains(&d.wheelbase()), "wheelbase {}", d.wheelbase());
        assert!((4.4..5.6).contains(&d.length()), "length {}", d.length());
        assert!((1.7..2.4).contains(&d.track()), "track {}", d.track());
        assert!((0.40..0.65).contains(&d.wheel_radius), "wheel radius {}", d.wheel_radius);
        assert!((0.20..0.65).contains(&d.wheel_width), "tyre width {}", d.wheel_width);
        assert!(d.ground_clearance() > 0.3, "clearance {}", d.ground_clearance());
        assert_eq!(d.mass_kg, 2900.0);
    }

    #[test]
    fn the_vehicle_is_turned_to_face_plus_z() {
        // The thing I got wrong twice. This Hummer is modelled nose-first along -Z, so
        // measuring it as it arrives puts the front axle at the back and makes positive
        // throttle drive it in reverse -- which is invisible in the numbers and obvious the
        // moment you press W.
        //
        // After normalising, the bonnet must be the +Z end: lower silhouette at the front,
        // cabin behind it.
        let Some(r) = rig() else { return };
        let shell = r.body.iter().max_by_key(|b| b.positions.len()).expect("a body shell");
        let (mut zlo, mut zhi) = (f32::MAX, f32::MIN);
        let (mut ylo, mut yhi) = (f32::MAX, f32::MIN);
        for p in &shell.positions {
            zlo = zlo.min(p[2]);
            zhi = zhi.max(p[2]);
            ylo = ylo.min(p[1]);
            yhi = yhi.max(p[1]);
        }
        let band = (zhi - zlo) / 8.0;
        let mut front_roof = f32::MIN;
        let mut rear_roof = f32::MIN;
        for p in &shell.positions {
            if p[2] > zhi - band {
                front_roof = front_roof.max(p[1]);
            } else if p[2] < zlo + band {
                rear_roof = rear_roof.max(p[1]);
            }
        }
        assert!(
            front_roof < rear_roof - (yhi - ylo) * 0.1,
            "the +Z end stands {front_roof} m tall against {rear_roof} at the other, \
             so the bonnet is not at the front"
        );

        // And the front axle sits under that bonnet, ahead of the rear one.
        assert!(
            r.dims.front_axle_z > r.dims.rear_axle_z,
            "front axle {} is not ahead of the rear {}",
            r.dims.front_axle_z,
            r.dims.rear_axle_z
        );
    }

    #[test]
    fn turning_the_vehicle_round_does_not_invert_its_triangles() {
        // A 180 degree rotation is a proper rotation, so winding survives it. Fixing a
        // facing by scaling Z by -1 would look identical in the bounds and turn the whole
        // body inside out under back-face culling.
        let mut part = MeshPart {
            data: MeshData {
                positions: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                normals: vec![[0.0, 1.0, 0.0]; 3],
                indices: vec![0, 1, 2],
                ..Default::default()
            },
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let normal_before = {
            let p = &part.data.positions;
            let (a, b, c) =
                (glam::Vec3::from(p[0]), glam::Vec3::from(p[1]), glam::Vec3::from(p[2]));
            (b - a).cross(c - a).normalize()
        };
        flip_about_y(&mut part);
        let normal_after = {
            let p = &part.data.positions;
            let (a, b, c) =
                (glam::Vec3::from(p[0]), glam::Vec3::from(p[1]), glam::Vec3::from(p[2]));
            (b - a).cross(c - a).normalize()
        };
        // The face normal rotates with the geometry rather than flipping to its opposite.
        let expected = glam::Vec3::new(-normal_before.x, normal_before.y, -normal_before.z);
        assert!(
            normal_after.distance(expected) < 1e-5,
            "winding changed: {normal_before} -> {normal_after}, expected {expected}"
        );
        // Bounds follow the geometry.
        assert_eq!(part.min, [-1.0, 0.0, -1.0]);
        assert_eq!(part.max, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_symmetric_body_falls_back_to_the_engine_convention() {
        // A box has no nose. Guessing from noise would flip the vehicle at random between
        // runs, so the fallback has to be deterministic.
        let cube = MeshPart {
            data: MeshData {
                positions: vec![
                    [-1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                ..Default::default()
            },
            min: [-1.0, 0.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        assert_eq!(nose_direction(&[cube]), 1.0);
        assert_eq!(nose_direction(&[]), 1.0, "no parts at all must not panic");
    }

    #[test]
    fn the_wheels_land_in_their_corners() {
        // The order is a contract with the physics controller: wheel 0 is front-left in
        // both, and the front pair is what steers. Getting it wrong steers from the rear.
        let Some(r) = rig() else { return };
        let d = r.dims;
        assert!(d.front_axle_z > 0.0, "the front axle is behind the origin");
        assert!(d.rear_axle_z < 0.0, "the rear axle is ahead of the origin");
        assert!(d.axle_half_width > 0.5, "the wheels are on the centreline");
    }

    #[test]
    fn each_wheel_is_centred_on_its_own_axle() {
        // If a wheel keeps the vehicle's origin, rotating it swings the tyre in a
        // three-metre arc around the car instead of spinning it.
        let Some(r) = rig() else { return };
        for (n, w) in r.wheels.iter().enumerate() {
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for p in &w.positions {
                for a in 0..3 {
                    min[a] = min[a].min(p[a]);
                    max[a] = max[a].max(p[a]);
                }
            }
            for a in 0..3 {
                let centre = (min[a] + max[a]) * 0.5;
                assert!(
                    centre.abs() < r.dims.wheel_radius * 0.25,
                    "wheel {n} axis {a} is centred at {centre}, not on its axle"
                );
            }
        }
    }

    #[test]
    fn the_collision_box_is_not_the_full_visual_height() {
        // The centre of mass comes from this box. Reaching the roof would put it high
        // enough to tip the vehicle in any corner, which reads as broken physics.
        let Some(r) = rig() else { return };
        let g = r.dims.rollover_threshold_g();
        assert!(g > 0.75, "would tip at {g} g");
        assert!(g < 1.6, "rollover threshold {g} g is implausible for a Hummer");
    }

    #[test]
    fn a_wheel_scores_and_a_body_panel_does_not() {
        // The classifier in isolation, so a failure says which half is wrong.
        let wheel =
            MeshPart { data: MeshData::default(), min: [0.8, 0.0, 1.2], max: [1.2, 1.1, 2.3] };
        assert!(wheel_score(&wheel) > 0.0, "a 0.4 x 1.1 x 1.1 disc off-centre is a wheel");

        // A door: wide, tall, thin, but not round.
        let door =
            MeshPart { data: MeshData::default(), min: [1.0, 0.5, -1.0], max: [1.1, 1.8, 1.0] };
        assert_eq!(wheel_score(&door), 0.0, "a door is not a wheel");

        // A roof panel: broad and flat.
        let roof =
            MeshPart { data: MeshData::default(), min: [-1.4, 2.3, -2.5], max: [1.4, 2.5, 2.5] };
        assert_eq!(wheel_score(&roof), 0.0, "a roof is not a wheel");

        // A spare on the centreline: round and narrow, but not offset. It still scores,
        // just below a real wheel, which is what keeps it from displacing one.
        let spare =
            MeshPart { data: MeshData::default(), min: [-0.2, 0.9, -2.7], max: [0.2, 2.0, -1.6] };
        assert!(wheel_score(&spare) < wheel_score(&wheel), "a centreline spare must lose");
    }

    #[test]
    fn a_file_that_is_not_a_vehicle_is_refused() {
        // Better a clear error than a car with its suspension bolted to a tree.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models");
        let Some(other) = crate::mesh::discover(&dir).into_iter().next() else { return };
        if other == hummer() {
            return;
        }
        assert!(
            VehicleRig::from_gltf(&other, 1000.0).is_err(),
            "{} was accepted as a vehicle",
            other.display()
        );
    }
}

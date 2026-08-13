//! Road networks: authored splines, stored in `world/edits/roads.ron`.
//!
//! Roads are geometry, never baked pixels. The heightfield on disk stays the
//! terrain as generated; roads are stamped onto a copy at load time. That is
//! what lets a road be moved, re-shaped, or deleted after the fact, and what
//! stops a terrain regenerate from destroying every road in the world.

use crate::error::{ProjectError, Result};
use crate::layout::ProjectPaths;
use crate::version::FORMAT_VERSION;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Surface {
    /// Bare earth. Rutted, holds water, dark when wet.
    Mud,
    /// Compacted gravel. Lighter, drains, few ruts.
    Gravel,
}

impl Surface {
    pub const ALL: [Surface; 2] = [Surface::Mud, Surface::Gravel];

    pub fn label(self) -> &'static str {
        match self {
            Surface::Mud => "Mud",
            Surface::Gravel => "Gravel",
        }
    }
}

/// One road: a centreline plus the cross-section to cut along it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Road {
    /// Control points in world XZ meters. The centreline is a Catmull-Rom
    /// spline through these.
    pub points: Vec<[f32; 2]>,
    /// Carriageway width.
    pub width_m: f32,
    /// Graded verge either side, easing into the batter slopes.
    pub shoulder_m: f32,
    /// Crown height as a fraction of half-width. Sheds water sideways; without
    /// it the surface holds a continuous sheet of water and reads as a canal.
    pub camber: f32,
    /// Maximum climb as a fraction. Loaded vehicles struggle past ~0.15 on
    /// dirt, and a road that ignores this looks draped rather than built.
    pub max_grade: f32,
    /// Refuse to move more earth than this vertically. Keeps a road from
    /// tunnelling through a mountain when the control points demand it.
    pub cut_fill_limit_m: f32,
    /// Angle of repose for cut banks and fill embankments, in degrees.
    pub batter_angle_deg: f32,
    /// Depth of the two wheel ruts.
    pub rut_depth_m: f32,
    /// Distance between rut centres.
    pub rut_spacing_m: f32,
    /// How far the track drifts sideways from its drawn line, in meters. Real
    /// tracks were never surveyed; a perfectly true one reads as engineered.
    pub wander_m: f32,
    pub surface: Surface,
}

impl Default for Road {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            width_m: 4.5,
            shoulder_m: 1.2,
            camber: 0.035,
            max_grade: 0.14,
            cut_fill_limit_m: 8.0,
            batter_angle_deg: 34.0,
            rut_depth_m: 0.07,
            rut_spacing_m: 1.8,
            wander_m: 1.6,
            surface: Surface::Mud,
        }
    }
}

impl Road {
    /// Half the full disturbed width: carriageway, shoulder, and the batter
    /// slope needed to reach terrain at the steepest allowed cut or fill.
    pub fn influence_m(&self) -> f32 {
        let batter_run = self.cut_fill_limit_m / self.batter_angle_deg.to_radians().tan().max(0.05);
        self.width_m * 0.5 + self.shoulder_m + batter_run
    }

    /// A road needs two points to have a direction and three for the spline to
    /// curve. One point is a marker, not a road.
    pub fn is_drawable(&self) -> bool {
        self.points.len() >= 2
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoadNetwork {
    pub version: u32,
    pub roads: Vec<Road>,
}

impl Default for RoadNetwork {
    fn default() -> Self {
        Self { version: FORMAT_VERSION, roads: Vec::new() }
    }
}

impl RoadNetwork {
    /// Load, or an empty network if the world has no roads yet.
    pub fn load(paths: &ProjectPaths) -> Self {
        let path = paths.roads();
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        match ron::from_str::<Self>(&text) {
            Ok(n) => n,
            Err(e) => {
                // Roads are authored work; refusing to open the world over a
                // parse error would be worse than starting without them, but
                // it must be loud.
                log::error!("{}: unreadable road network: {e}", path.display());
                Self::default()
            }
        }
    }

    pub fn save(&self, paths: &ProjectPaths) -> Result<()> {
        let path = paths.roads();
        if self.roads.is_empty() {
            // Don't leave an empty file behind after the last road is deleted.
            let _ = std::fs::remove_file(&path);
            return Ok(());
        }
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(self, cfg)
            .map_err(|e| ProjectError::Serialize { path: path.clone(), source: Box::new(e) })?;
        std::fs::write(&path, text).map_err(|e| ProjectError::Io { path: path.clone(), source: e })
    }

    pub fn drawable(&self) -> impl Iterator<Item = &Road> {
        self.roads.iter().filter(|r| r.is_drawable())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;
    use terra_core::WorldSize;

    fn temp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn round_trips_through_disk() {
        let root = temp("terra-roads-round");
        let project = Project::create(&root, "Roads", WorldSize::Small, 1).unwrap();

        let net = RoadNetwork {
            version: FORMAT_VERSION,
            roads: vec![Road {
                points: vec![[0.0, 0.0], [100.0, 20.0], [250.0, -40.0]],
                surface: Surface::Gravel,
                ..Default::default()
            }],
        };
        net.save(&project.paths).unwrap();
        assert_eq!(RoadNetwork::load(&project.paths), net);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn missing_file_is_an_empty_network() {
        let root = temp("terra-roads-none");
        let project = Project::create(&root, "None", WorldSize::Small, 1).unwrap();
        assert!(RoadNetwork::load(&project.paths).roads.is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn deleting_the_last_road_removes_the_file() {
        let root = temp("terra-roads-empty");
        let project = Project::create(&root, "Empty", WorldSize::Small, 1).unwrap();

        RoadNetwork {
            version: FORMAT_VERSION,
            roads: vec![Road { points: vec![[0.0, 0.0], [50.0, 0.0]], ..Default::default() }],
        }
        .save(&project.paths)
        .unwrap();
        assert!(project.paths.roads().is_file());

        RoadNetwork::default().save(&project.paths).unwrap();
        assert!(!project.paths.roads().exists());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn influence_covers_the_full_batter_run() {
        let r = Road { cut_fill_limit_m: 8.0, batter_angle_deg: 34.0, ..Default::default() };
        // 8 m of fill at 34 degrees needs ~11.9 m of run, plus half-width and
        // shoulder. Anything less and the stamp would clip the embankment.
        assert!(r.influence_m() > 11.0 + r.width_m * 0.5 + r.shoulder_m);
    }

    #[test]
    fn a_single_point_is_not_a_road() {
        let mut r = Road::default();
        assert!(!r.is_drawable());
        r.points.push([0.0, 0.0]);
        assert!(!r.is_drawable());
        r.points.push([10.0, 0.0]);
        assert!(r.is_drawable());
    }
}

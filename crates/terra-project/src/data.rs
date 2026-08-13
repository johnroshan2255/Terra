//! The heavy world data: heightfield and erosion masks.
//!
//! Separate from [`crate::world::WorldManifest`], which holds only parameters.
//! Manifests are kilobytes of RON that a human might edit; this is megabytes of
//! binary that only the tools touch. Keeping them apart means a parameter
//! change never rewrites a 67 MB heightmap.
//!
//! Every field shares the `.r16` container -- little-endian `u16`, no header --
//! but they carry different meanings, so each has its own conversion.

use crate::error::{ProjectError, Result};
use crate::layout::ProjectPaths;
use std::path::Path;
use terra_core::{BASE_ELEVATION_M, HeightGrid, WorldSize};

/// Everything about a world that is too large for the manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldData {
    /// Terrain height in meters, row-major, `res * res`.
    pub heights: Vec<f32>,
    /// Channel mask from erosion, `0..=1`. Empty if never generated.
    pub flow: Vec<f32>,
    /// Deposition map, `0..=1`, where 0.5 means unchanged. Empty if never
    /// generated.
    pub deposition: Vec<f32>,
}

impl WorldData {
    /// A freshly created world: flat, at [`BASE_ELEVATION_M`], with neutral
    /// masks.
    pub fn flat(size: WorldSize) -> Self {
        let res = size.tier0_res() as usize;
        let n = res * res;
        Self { heights: vec![BASE_ELEVATION_M; n], flow: vec![0.0; n], deposition: vec![0.5; n] }
    }

    /// Load from disk, falling back to flat defaults for anything missing.
    ///
    /// A world that has been created but never generated or saved has no files
    /// at all, which is the normal first-open case rather than an error.
    pub fn load(paths: &ProjectPaths, size: WorldSize) -> Self {
        let res = size.tier0_res();
        let n = (res as usize) * (res as usize);

        let heights = match read_grid(&paths.global_height(), res) {
            Some(g) => g.to_meters(),
            None => vec![BASE_ELEVATION_M; n],
        };
        let flow = match read_grid(&paths.global_flow(), res) {
            Some(g) => g.to_unit(),
            None => vec![0.0; n],
        };
        let deposition = match read_grid(&paths.global_sediment(), res) {
            Some(g) => g.to_unit(),
            None => vec![0.5; n],
        };

        Self { heights, flow, deposition }
    }

    /// Write every field that has data. Masks are skipped when empty, so a
    /// world that has only ever been sculpted does not gain meaningless
    /// all-zero mask files.
    pub fn save(&self, paths: &ProjectPaths, size: WorldSize) -> Result<()> {
        let res = size.tier0_res();

        let grid = HeightGrid::from_meters(res, &self.heights).map_err(|e| ProjectError::Io {
            path: paths.global_height(),
            source: std::io::Error::other(e.to_string()),
        })?;
        write_bytes(&paths.global_height(), &grid.to_r16_bytes())?;

        if !self.flow.is_empty() {
            write_unit(&paths.global_flow(), res, &self.flow)?;
        }
        if !self.deposition.is_empty() {
            write_unit(&paths.global_sediment(), res, &self.deposition)?;
        }
        Ok(())
    }

    /// Bytes this world occupies on disk once saved.
    pub fn disk_size(size: WorldSize) -> u64 {
        // Height plus two masks, all the same shape.
        size.tier0_bytes() * 3
    }
}

fn read_grid(path: &Path, res: u32) -> Option<HeightGrid> {
    let bytes = std::fs::read(path).ok()?;
    match HeightGrid::from_r16_bytes(res, &bytes) {
        Ok(g) => Some(g),
        Err(e) => {
            // Wrong size means the file belongs to a different world. Refusing
            // to guess is better than loading terrain at the wrong scale.
            log::warn!("{}: {e}; using defaults", path.display());
            None
        }
    }
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes)
        .map_err(|e| ProjectError::Io { path: path.to_path_buf(), source: e })
}

fn write_unit(path: &Path, res: u32, values: &[f32]) -> Result<()> {
    let grid = HeightGrid::from_unit(res, values).map_err(|e| ProjectError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;
    write_bytes(path, &grid.to_r16_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;

    fn temp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn a_new_world_loads_flat_without_any_files() {
        let root = temp("terra-data-fresh");
        let project = Project::create(&root, "Fresh", WorldSize::Small, 1).unwrap();

        let data = WorldData::load(&project.paths, WorldSize::Small);
        assert!(data.heights.iter().all(|h| *h == BASE_ELEVATION_M));
        assert!(data.flow.iter().all(|f| *f == 0.0));
        assert!(data.deposition.iter().all(|d| *d == 0.5));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn round_trips_through_disk() {
        let root = temp("terra-data-round");
        let project = Project::create(&root, "Round", WorldSize::Small, 1).unwrap();
        let size = WorldSize::Small;
        let n = (size.tier0_res() as usize).pow(2);

        let written = WorldData {
            heights: (0..n).map(|i| 200.0 + (i % 900) as f32).collect(),
            flow: (0..n).map(|i| (i % 100) as f32 / 99.0).collect(),
            deposition: (0..n).map(|i| (i % 50) as f32 / 49.0).collect(),
        };
        written.save(&project.paths, size).unwrap();

        let read = WorldData::load(&project.paths, size);
        for (a, b) in written.heights.iter().zip(&read.heights) {
            assert!((a - b).abs() < 0.05, "height {a} vs {b}");
        }
        for (a, b) in written.flow.iter().zip(&read.flow) {
            assert!((a - b).abs() < 1e-3, "flow {a} vs {b}");
        }

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn empty_masks_do_not_create_files() {
        let root = temp("terra-data-nomask");
        let project = Project::create(&root, "NoMask", WorldSize::Small, 1).unwrap();
        let size = WorldSize::Small;

        let sculpted = WorldData {
            heights: vec![300.0; (size.tier0_res() as usize).pow(2)],
            flow: vec![],
            deposition: vec![],
        };
        sculpted.save(&project.paths, size).unwrap();

        assert!(project.paths.global_height().is_file());
        assert!(!project.paths.global_flow().exists(), "no erosion run, so no flow map");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn disk_size_matches_what_save_produces() {
        let root = temp("terra-data-size");
        let project = Project::create(&root, "Size", WorldSize::Small, 1).unwrap();
        let size = WorldSize::Small;

        WorldData::flat(size).save(&project.paths, size).unwrap();

        let actual: u64 = [
            project.paths.global_height(),
            project.paths.global_flow(),
            project.paths.global_sediment(),
        ]
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

        assert_eq!(actual, WorldData::disk_size(size));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

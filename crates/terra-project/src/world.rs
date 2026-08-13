//! `world/world.ron` -- world size, seed, and generation parameters.

use crate::error::{ProjectError, Result};
use crate::layout::ProjectPaths;
use crate::params::TerrainParams;
use crate::version::FORMAT_VERSION;
use serde::{Deserialize, Serialize};
use terra_core::{TileBounds, WorldSize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldManifest {
    pub version: u32,
    /// Chosen at creation and immutable thereafter. Changing it would orphan
    /// every source map, so the editor offers no path to edit it.
    pub size: WorldSize,
    pub seed: u64,
    pub terrain: TerrainParams,
}

impl WorldManifest {
    pub fn new(size: WorldSize, seed: u64) -> Self {
        Self { version: FORMAT_VERSION, size, seed, terrain: TerrainParams::default() }
    }

    pub fn bounds(&self) -> TileBounds {
        TileBounds::centered(self.size)
    }

    pub fn load(paths: &ProjectPaths) -> Result<Self> {
        let path = paths.world_manifest();
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ProjectError::Io { path: path.clone(), source: e })?;
        let manifest: Self = ron::from_str(&text)
            .map_err(|e| ProjectError::Parse { path: path.clone(), source: Box::new(e) })?;

        crate::version::check(&path, manifest.version)?;
        manifest.verify_against_disk(paths)?;
        Ok(manifest)
    }

    pub fn save(&self, paths: &ProjectPaths) -> Result<()> {
        let path = paths.world_manifest();
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(self, cfg)
            .map_err(|e| ProjectError::Serialize { path: path.clone(), source: Box::new(e) })?;
        std::fs::write(&path, text).map_err(|e| ProjectError::Io { path: path.clone(), source: e })
    }

    /// Confirm the tier-0 heightmap on disk matches the declared world size.
    ///
    /// A mismatch means the manifest was hand-edited or the file came from a
    /// different project. Baking tiles from it would silently overwrite the
    /// user's terrain, so this is a hard error rather than a warning.
    fn verify_against_disk(&self, paths: &ProjectPaths) -> Result<()> {
        let height = paths.global_height();
        let Ok(meta) = std::fs::metadata(&height) else {
            return Ok(()); // not generated yet -- legal for a fresh project
        };

        let expected = self.size.tier0_bytes();
        if meta.len() != expected {
            let actual = ((meta.len() / 2) as f64).sqrt().round() as u32;
            return Err(ProjectError::ResolutionMismatch {
                path: height,
                expected: self.size.tier0_res(),
                actual,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_ron() {
        let w = WorldManifest::new(WorldSize::Medium, 0x5EED_1234);
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(&w, cfg).unwrap();
        assert_eq!(ron::from_str::<WorldManifest>(&text).unwrap(), w);
    }

    #[test]
    fn bounds_match_declared_size() {
        let w = WorldManifest::new(WorldSize::Large, 1);
        assert_eq!(w.bounds().tiles_per_side(), 8);
    }
}

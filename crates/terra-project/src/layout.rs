//! The on-disk shape of a project, in one place.
//!
//! ```text
//! MyGame/
//! |- project.ron            manifest: name, engine version, world size, seed
//! |- thumbnail.png          top-down render, refreshed on save
//! |- world/
//! |  |- world.ron           terrain + erosion parameters
//! |  |- source/             AUTHORITATIVE -- back this up
//! |  |  |- global_height.r16    tier 0, eroded. The one irreplaceable file.
//! |  |  |- global_flow.r16      flow accumulation, free from erosion
//! |  |  |- global_sediment.r16  deposition map, free from erosion
//! |  |  \- masks/               hand-painted R8 PNGs (roads, density, biomes)
//! |  |- edits/              USER WORK -- never regenerated
//! |  |  |- sculpt/              sparse per-tile height deltas
//! |  |  |- props.ron            hand-placed objects
//! |  |  \- roads.ron            authored road splines
//! |  \- cache/              DELETE ANYTIME -- rebuilt from source + params
//! |     \- tiles/               tier 1, 2 m/texel baked tiles
//! |- assets/                project-local meshes and textures
//! \- game/                  spawns, gameplay config
//! ```
//!
//! The split that matters is `source` / `edits` / `cache`. Anything in `cache`
//! is reproducible from `source` plus the parameters in `world.ron`, so it is
//! gitignored and never backed up. Anything in `edits` represents human effort
//! and is never touched by a regenerate.

use crate::error::{ProjectError, Result};
use std::path::{Path, PathBuf};
use terra_core::TileCoord;

/// Every path within one project, derived from its root directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPaths {
    root: PathBuf,
}

impl ProjectPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    // --- manifests ---

    pub fn project_manifest(&self) -> PathBuf {
        self.root.join("project.ron")
    }

    pub fn thumbnail(&self) -> PathBuf {
        self.root.join("thumbnail.png")
    }

    pub fn world_manifest(&self) -> PathBuf {
        self.world_dir().join("world.ron")
    }

    // --- directories ---

    pub fn world_dir(&self) -> PathBuf {
        self.root.join("world")
    }

    pub fn source_dir(&self) -> PathBuf {
        self.world_dir().join("source")
    }

    pub fn masks_dir(&self) -> PathBuf {
        self.source_dir().join("masks")
    }

    pub fn edits_dir(&self) -> PathBuf {
        self.world_dir().join("edits")
    }

    pub fn sculpt_dir(&self) -> PathBuf {
        self.edits_dir().join("sculpt")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.world_dir().join("cache")
    }

    pub fn tiles_dir(&self) -> PathBuf {
        self.cache_dir().join("tiles")
    }

    pub fn assets_dir(&self) -> PathBuf {
        self.root.join("assets")
    }

    pub fn game_dir(&self) -> PathBuf {
        self.root.join("game")
    }

    // --- tier-0 source maps ---

    pub fn global_height(&self) -> PathBuf {
        self.source_dir().join("global_height.r16")
    }

    pub fn global_flow(&self) -> PathBuf {
        self.source_dir().join("global_flow.r16")
    }

    pub fn global_sediment(&self) -> PathBuf {
        self.source_dir().join("global_sediment.r16")
    }

    pub fn mask(&self, name: &str) -> PathBuf {
        self.masks_dir().join(format!("{name}.png"))
    }

    // --- per-tile files ---

    /// Baked tier-1 heightmap for one tile. Regenerable.
    pub fn tile_height(&self, c: TileCoord) -> PathBuf {
        self.tiles_dir().join(format!("{}.r16", c.stem("h")))
    }

    /// Sparse sculpt deltas for one tile. User work, never regenerated.
    pub fn tile_sculpt(&self, c: TileCoord) -> PathBuf {
        self.sculpt_dir().join(format!("{}.delta", c.stem("s")))
    }

    pub fn props(&self) -> PathBuf {
        self.edits_dir().join("props.ron")
    }

    /// Authored road splines. Under `edits/` because they are human work and
    /// must survive a terrain regenerate.
    /// Painted material weights. Beside the other masks: it is authored data,
    /// but it is a raster, not a document.
    pub fn splat(&self) -> PathBuf {
        self.masks_dir().join("splat.bin")
    }

    /// Foliage rules and painted density. Rules are a document; the density
    /// masks are rasters appended after them.
    pub fn foliage(&self) -> PathBuf {
        self.edits_dir().join("foliage.bin")
    }

    pub fn roads(&self) -> PathBuf {
        self.edits_dir().join("roads.ron")
    }

    pub fn game_config(&self) -> PathBuf {
        self.game_dir().join("config.ron")
    }

    pub fn spawns(&self) -> PathBuf {
        self.game_dir().join("spawns.ron")
    }

    /// Create the full directory tree. Safe to re-run on an existing project;
    /// it only fills in missing directories.
    pub fn scaffold(&self) -> Result<()> {
        for dir in [
            self.world_dir(),
            self.source_dir(),
            self.masks_dir(),
            self.edits_dir(),
            self.sculpt_dir(),
            self.cache_dir(),
            self.tiles_dir(),
            self.assets_dir(),
            self.game_dir(),
        ] {
            std::fs::create_dir_all(&dir)
                .map_err(|e| ProjectError::Io { path: dir.clone(), source: e })?;
        }
        Ok(())
    }

    /// True if this directory looks like a project we can open.
    pub fn looks_like_project(root: &Path) -> bool {
        root.join("project.ron").is_file()
    }

    /// Drop every regenerable byte. Called when terrain parameters change, and
    /// exposed in the editor as "Clear cache".
    pub fn clear_cache(&self) -> Result<()> {
        let dir = self.cache_dir();
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .map_err(|e| ProjectError::Io { path: dir.clone(), source: e })?;
        }
        std::fs::create_dir_all(self.tiles_dir())
            .map_err(|e| ProjectError::Io { path: self.tiles_dir(), source: e })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_lives_under_world_not_root() {
        // The gitignore rule is `**/cache/`; keeping the cache nested under
        // world/ means a stray root-level cache dir cannot be silently ignored.
        let p = ProjectPaths::new("/games/Rally");
        assert!(p.tiles_dir().starts_with(p.world_dir()));
    }

    #[test]
    fn negative_tiles_produce_valid_filenames() {
        let p = ProjectPaths::new("/games/Rally");
        let f = p.tile_height(TileCoord::new(-2, 1));
        assert_eq!(f.file_name().unwrap(), "h_-2_1.r16");
    }

    #[test]
    fn scaffold_is_idempotent() {
        let tmp = std::env::temp_dir().join("terra-scaffold-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let p = ProjectPaths::new(&tmp);
        p.scaffold().unwrap();
        p.scaffold().unwrap();
        assert!(p.tiles_dir().is_dir());
        assert!(p.masks_dir().is_dir());
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}

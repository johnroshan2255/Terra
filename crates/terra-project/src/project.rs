//! `project.ron` -- one game.
//!
//! A project is entirely self-contained and movable: nothing inside it refers
//! to an absolute path. Copying the folder to another machine, or into a git
//! repo, must just work.

use crate::error::{ProjectError, Result};
use crate::layout::ProjectPaths;
use crate::version::FORMAT_VERSION;
use crate::world::WorldManifest;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use terra_core::WorldSize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub version: u32,
    pub name: String,
    /// Build that last wrote this project. Shown in the browser so a user can
    /// tell why an old project behaves oddly.
    pub engine_version: String,
    /// Seconds since the Unix epoch. Stored as an integer rather than a date
    /// string to avoid a calendar dependency in the data layer.
    pub created_unix: u64,
}

impl ProjectManifest {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: FORMAT_VERSION,
            name: name.into(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            created_unix: now_unix(),
        }
    }
}

/// An open project: its manifests plus the paths they were loaded from.
#[derive(Debug, Clone)]
pub struct Project {
    pub paths: ProjectPaths,
    pub manifest: ProjectManifest,
    pub world: WorldManifest,
}

impl Project {
    /// Create a new project on disk: scaffold directories, write both
    /// manifests. Does not generate terrain -- that is the editor's first job
    /// once the project is open.
    pub fn create(
        root: impl Into<PathBuf>,
        name: &str,
        size: WorldSize,
        seed: u64,
    ) -> Result<Self> {
        let paths = ProjectPaths::new(root);
        paths.scaffold()?;

        let manifest = ProjectManifest::new(name);
        let world = WorldManifest::new(size, seed);

        let project = Self { paths, manifest, world };
        project.save()?;
        Ok(project)
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !ProjectPaths::looks_like_project(root) {
            return Err(ProjectError::NotAProject { path: root.to_path_buf() });
        }
        let paths = ProjectPaths::new(root);

        let path = paths.project_manifest();
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ProjectError::Io { path: path.clone(), source: e })?;
        let manifest: ProjectManifest = ron::from_str(&text)
            .map_err(|e| ProjectError::Parse { path: path.clone(), source: Box::new(e) })?;
        crate::version::check(&path, manifest.version)?;

        // Directories may be missing if the project came from a zip that
        // dropped empty folders, or from a repo where cache/ was gitignored.
        paths.scaffold()?;

        let world = WorldManifest::load(&paths)?;
        Ok(Self { paths, manifest, world })
    }

    pub fn save(&self) -> Result<()> {
        let path = self.paths.project_manifest();
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(&self.manifest, cfg)
            .map_err(|e| ProjectError::Serialize { path: path.clone(), source: Box::new(e) })?;
        std::fs::write(&path, text)
            .map_err(|e| ProjectError::Io { path: path.clone(), source: e })?;

        self.world.save(&self.paths)
    }

    pub fn size(&self) -> WorldSize {
        self.world.size
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_then_open_round_trips() {
        let tmp = std::env::temp_dir().join("terra-project-test");
        let _ = std::fs::remove_dir_all(&tmp);

        let made = Project::create(&tmp, "Desert Rally", WorldSize::Medium, 42).unwrap();
        let opened = Project::open(&tmp).unwrap();

        assert_eq!(opened.manifest, made.manifest);
        assert_eq!(opened.world, made.world);
        assert_eq!(opened.size(), WorldSize::Medium);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn opening_a_plain_directory_fails_clearly() {
        let tmp = std::env::temp_dir().join("terra-not-a-project");
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(matches!(Project::open(&tmp), Err(ProjectError::NotAProject { .. })));
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}

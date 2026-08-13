//! The project library -- the index behind the "which game?" browser.
//!
//! The library stores **paths, not contents**. Projects live wherever the user
//! wants: Documents, an external drive, a git checkout. A path that no longer
//! resolves is shown greyed-out in the browser, never treated as an error --
//! an unplugged drive must not lose someone's entry.

use crate::error::{ProjectError, Result};
use crate::layout::ProjectPaths;
use crate::version::FORMAT_VERSION;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One row in the browser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectEntry {
    pub name: String,
    pub path: PathBuf,
    pub last_opened_unix: u64,
}

impl ProjectEntry {
    /// Whether the project is currently reachable. Drives the greyed-out state.
    pub fn is_available(&self) -> bool {
        ProjectPaths::looks_like_project(&self.path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Library {
    pub version: u32,
    pub projects: Vec<ProjectEntry>,
}

impl Default for Library {
    fn default() -> Self {
        Self { version: FORMAT_VERSION, projects: Vec::new() }
    }
}

impl Library {
    /// Platform data directory holding `library.ron`, the shared asset library,
    /// and the default location for new projects.
    ///
    /// macOS: `~/Library/Application Support/in.synctric.Terra/`
    pub fn data_dir() -> Result<PathBuf> {
        directories::ProjectDirs::from("in", "synctric", "Terra")
            .map(|d| d.data_dir().to_path_buf())
            .ok_or(ProjectError::NoDataDir)
    }

    pub fn index_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("library.ron"))
    }

    /// Default parent directory for newly created projects. Users may pick
    /// anywhere else; this is only the pre-filled suggestion.
    pub fn default_projects_dir() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("projects"))
    }

    /// Shared asset library, visible to every project. Keeping meshes here
    /// rather than inside each project means ten games referencing the same
    /// boulder store it once.
    pub fn shared_assets_dir() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("assets"))
    }

    /// Load the index, or return an empty one on first run. A corrupt index is
    /// logged and replaced rather than propagated: it is a convenience cache,
    /// and refusing to launch over it would be worse than losing the recents.
    pub fn load() -> Result<Self> {
        let path = Self::index_path()?;
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(Self::default());
        };
        match ron::from_str::<Self>(&text) {
            Ok(lib) => Ok(lib),
            Err(e) => {
                log::warn!("{}: unreadable project index, starting empty: {e}", path.display());
                Ok(Self::default())
            }
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::index_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ProjectError::Io { path: parent.to_path_buf(), source: e })?;
        }
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(self, cfg)
            .map_err(|e| ProjectError::Serialize { path: path.clone(), source: Box::new(e) })?;
        std::fs::write(&path, text).map_err(|e| ProjectError::Io { path: path.clone(), source: e })
    }

    /// Record a project as most-recently-opened, de-duplicating by path.
    pub fn touch(&mut self, name: &str, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        self.projects.retain(|p| p.path != path);
        self.projects.insert(
            0,
            ProjectEntry {
                name: name.to_string(),
                path,
                last_opened_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            },
        );
    }

    /// Remove from the index only. Never deletes files from disk -- "remove
    /// from list" and "delete this game" must stay separate actions.
    pub fn forget(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        self.projects.retain(|p| p.path != path);
    }

    /// Most recent first.
    pub fn sorted(&self) -> Vec<&ProjectEntry> {
        let mut v: Vec<_> = self.projects.iter().collect();
        v.sort_by_key(|p| std::cmp::Reverse(p.last_opened_unix));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_deduplicates_and_promotes() {
        let mut lib = Library::default();
        lib.touch("A", "/games/a");
        lib.touch("B", "/games/b");
        lib.touch("A", "/games/a");

        assert_eq!(lib.projects.len(), 2);
        assert_eq!(lib.projects[0].path, PathBuf::from("/games/a"));
    }

    #[test]
    fn forget_only_touches_the_index() {
        let mut lib = Library::default();
        lib.touch("A", "/games/a");
        lib.forget("/games/a");
        assert!(lib.projects.is_empty());
    }
}

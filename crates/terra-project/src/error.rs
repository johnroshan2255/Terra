//! Errors raised while reading, writing, or validating a project.

use std::path::PathBuf;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ProjectError>;

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    // The ron error types are large enough that inlining them here pushes
    // ProjectError past clippy's `result_large_err` threshold, making every
    // Result in the crate expensive to move. Box them.
    #[error("{path}: malformed RON: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: Box<ron::error::SpannedError>,
    },

    #[error("could not serialize {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: Box<ron::Error>,
    },

    #[error("{path} is not a project directory (no project.ron)")]
    NotAProject { path: PathBuf },

    /// Written by a newer build. Refuse rather than guess at the schema.
    #[error(
        "{path} was saved by format version {found}, but this build understands \
         up to {supported}. Update the editor to open it."
    )]
    FormatTooNew { path: PathBuf, found: u32, supported: u32 },

    /// The manifest and the files on disk disagree. Always a hard error: the
    /// alternative is baking tiles over somebody's work.
    #[error(
        "{path}: manifest says the tier-0 heightmap is {expected}x{expected}, \
         but the file on disk is {actual}x{actual}"
    )]
    ResolutionMismatch { path: PathBuf, expected: u32, actual: u32 },

    #[error("{tiles} tiles per side is not a supported world size")]
    UnknownWorldSize { tiles: u32 },

    #[error("could not locate a platform data directory for the project library")]
    NoDataDir,
}

//! Format versioning.
//!
//! The version field and this check exist from the first commit on purpose. It
//! costs a few lines now; retrofitting versioning onto files that users already
//! have on disk does not.

use crate::error::{ProjectError, Result};
use std::path::Path;

/// Bump whenever a manifest schema changes in a way older builds cannot read.
/// Add a `migrate` arm in the same commit.
pub const FORMAT_VERSION: u32 = 1;

/// Reject files from the future; accept and migrate files from the past.
pub fn check(path: &Path, found: u32) -> Result<()> {
    if found > FORMAT_VERSION {
        return Err(ProjectError::FormatTooNew {
            path: path.to_path_buf(),
            found,
            supported: FORMAT_VERSION,
        });
    }
    Ok(())
}

/// Upgrade a parsed manifest in place.
///
/// Called after a successful parse, before the manifest is handed to the rest
/// of the app. Version 1 is the first format, so there is nothing to do yet --
/// the shape is here so the first real migration is a one-line addition rather
/// than a refactor.
pub fn migrate(from: u32) -> u32 {
    let mut v = from;
    // match v {
    //     1 => { /* 1 -> 2: field added, default applied by serde */ v = 2; }
    //     _ => {}
    // }
    if v == 0 {
        v = 1; // pre-versioning files, if any escaped
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn future_versions_are_rejected() {
        let p = PathBuf::from("world.ron");
        assert!(check(&p, FORMAT_VERSION).is_ok());
        assert!(check(&p, FORMAT_VERSION + 1).is_err());
    }
}

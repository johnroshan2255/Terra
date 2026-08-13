//! `assets.ron` -- id -> file, with content hashes for dedup.

use serde::{Deserialize, Serialize};

/// Stable identity for an asset. Survives renames and moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub uuid::Uuid);

// TODO: AssetDb { entries: HashMap<AssetId, AssetEntry> } with load/save,
//       resolving project-local assets ahead of the shared library.
pub struct AssetDb;

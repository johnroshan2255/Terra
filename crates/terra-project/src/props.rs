//! Hand-placed objects.
//!
//! The exception to the scatter rule. Everything else is placed by rules plus a
//! seed and never stored per-object; a prop is placed because someone put it
//! exactly there, so its transform is the data and there is nothing to derive.
//!
//! Species are referenced by name rather than by index. A palette reordered by
//! adding a model folder would otherwise turn every placed rock into a tree.

use crate::error::{ProjectError, Result};
use crate::layout::ProjectPaths;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prop {
    pub species: String,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub scale: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropSet {
    pub props: Vec<Prop>,
}

impl PropSet {
    pub fn load(paths: &ProjectPaths) -> Self {
        let path = paths.props();
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        match ron::from_str::<Self>(&text) {
            Ok(p) => p,
            Err(e) => {
                // Same call as roads: placed objects are authored work, but
                // refusing to open the world over a parse error is worse than
                // opening it without them. It must be loud either way.
                log::error!("{}: unreadable props: {e}", path.display());
                Self::default()
            }
        }
    }

    pub fn save(&self, paths: &ProjectPaths) -> Result<()> {
        let path = paths.props();
        if self.props.is_empty() {
            let _ = std::fs::remove_file(&path);
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ProjectError::Io { path: path.clone(), source: e })?;
        }
        let cfg = ron::ser::PrettyConfig::new().struct_names(true);
        let text = ron::ser::to_string_pretty(self, cfg)
            .map_err(|e| ProjectError::Serialize { path: path.clone(), source: Box::new(e) })?;
        std::fs::write(&path, text).map_err(|e| ProjectError::Io { path: path.clone(), source: e })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_ron() {
        let set = PropSet {
            props: vec![Prop {
                species: "Rock".into(),
                pos: [12.5, 3.0, -400.25],
                yaw: 1.25,
                scale: 2.0,
            }],
        };
        let text = ron::ser::to_string(&set).unwrap();
        let back: PropSet = ron::from_str(&text).unwrap();
        assert_eq!(back.props.len(), 1);
        assert_eq!(back.props[0].species, "Rock");
        assert_eq!(back.props[0].pos, [12.5, 3.0, -400.25]);
    }
}

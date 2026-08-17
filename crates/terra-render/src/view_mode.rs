//! Viewport visualization modes, on Unreal's `Alt+` hotkeys.
//!
//! Each mode answers a different question, and the reason there are five rather
//! than a "debug" toggle is that they isolate the inputs to shading from each
//! other:
//!
//! | Mode | Hotkey | Answers |
//! |---|---|---|
//! | Wireframe | `Alt+2` | Is the topology what I think it is? |
//! | Unlit | `Alt+3` | Is the albedo right, independent of lighting? |
//! | Lit | `Alt+4` | The real thing. |
//! | Detail Lighting | `Alt+5` | Are the normal maps doing anything? |
//! | Lighting Only | `Alt+6` | Is the *lighting* right, independent of materials? |
//!
//! Detail Lighting and Lighting Only differ in exactly one respect and it is
//! easy to get backwards: both replace albedo with neutral grey, but Detail
//! Lighting keeps the material's normal maps while Lighting Only discards them
//! and shades from the geometric normal alone. So a bumpy surface that looks
//! flat in Detail Lighting has a broken normal map, and one that looks flat in
//! Lighting Only is genuinely flat.

/// What the viewport is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    /// Everything, as shipped.
    #[default]
    Lit,
    /// Albedo straight out, with no lighting applied at all.
    Unlit,
    /// Triangle edges only.
    Wireframe,
    /// Neutral 50% grey albedo, material normals kept.
    DetailLighting,
    /// Neutral 50% grey albedo, geometric normals only.
    LightingOnly,
}

impl ViewMode {
    /// In hotkey order, which is also the order Unreal lists them.
    pub const ALL: [ViewMode; 5] = [
        ViewMode::Wireframe,
        ViewMode::Unlit,
        ViewMode::Lit,
        ViewMode::DetailLighting,
        ViewMode::LightingOnly,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Lit => "Lit",
            ViewMode::Unlit => "Unlit",
            ViewMode::Wireframe => "Wireframe",
            ViewMode::DetailLighting => "Detail Lighting",
            ViewMode::LightingOnly => "Lighting Only",
        }
    }

    /// The digit this mode sits on, with Alt held.
    ///
    /// Matching Unreal's numbering exactly, including that it starts at 2 --
    /// `Alt+1` is Brush Wireframe there, which has no meaning here.
    pub fn hotkey_digit(self) -> u32 {
        match self {
            ViewMode::Wireframe => 2,
            ViewMode::Unlit => 3,
            ViewMode::Lit => 4,
            ViewMode::DetailLighting => 5,
            ViewMode::LightingOnly => 6,
        }
    }

    /// Mode for a digit, or `None` if nothing is bound to it.
    pub fn from_digit(digit: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.hotkey_digit() == digit)
    }

    /// Value the shader branches on. Part of the CPU/GPU contract, so these are
    /// pinned rather than derived from declaration order.
    pub fn shader_index(self) -> u32 {
        match self {
            ViewMode::Lit => 0,
            ViewMode::Unlit => 1,
            ViewMode::Wireframe => 2,
            ViewMode::DetailLighting => 3,
            ViewMode::LightingOnly => 4,
        }
    }

    /// Whether this mode draws edges instead of filled triangles.
    pub fn is_wireframe(self) -> bool {
        self == ViewMode::Wireframe
    }

    /// Whether the mode replaces albedo with neutral grey.
    ///
    /// The point of both grey modes: with albedo held constant, everything left
    /// in the image is lighting, so a dark patch is a shadow rather than a dark
    /// texture.
    pub fn neutral_albedo(self) -> bool {
        matches!(self, ViewMode::DetailLighting | ViewMode::LightingOnly)
    }

    /// Whether material normal maps still perturb the surface.
    pub fn uses_material_normals(self) -> bool {
        !matches!(self, ViewMode::LightingOnly)
    }

    /// Whether the scene's own lighting is applied.
    pub fn is_lit(self) -> bool {
        !matches!(self, ViewMode::Unlit | ViewMode::Wireframe)
    }

    /// Whether atmospheric effects belong in the picture.
    ///
    /// Fog and god rays are switched off in every debug mode. They are
    /// view-dependent haze over the whole frame, and the entire purpose of these
    /// modes is to look at one term without anything else on top of it.
    pub fn shows_atmosphere(self) -> bool {
        self == ViewMode::Lit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotkeys_match_unreals_numbering() {
        assert_eq!(ViewMode::Wireframe.hotkey_digit(), 2);
        assert_eq!(ViewMode::Unlit.hotkey_digit(), 3);
        assert_eq!(ViewMode::Lit.hotkey_digit(), 4);
        assert_eq!(ViewMode::DetailLighting.hotkey_digit(), 5);
        assert_eq!(ViewMode::LightingOnly.hotkey_digit(), 6);
    }

    #[test]
    fn digits_round_trip_and_nothing_else_is_bound() {
        for m in ViewMode::ALL {
            assert_eq!(ViewMode::from_digit(m.hotkey_digit()), Some(m));
        }
        // Alt+1 is Brush Wireframe in Unreal and has no meaning here; 7 and up
        // are unbound. Neither may fall through to some other mode.
        for d in [0, 1, 7, 8, 9] {
            assert_eq!(ViewMode::from_digit(d), None, "digit {d} should be unbound");
        }
    }

    #[test]
    fn shader_indices_are_unique_and_stable() {
        // The shader switches on these, so they are an ABI. Reordering the enum
        // must not renumber them.
        let mut seen = Vec::new();
        for m in ViewMode::ALL {
            let i = m.shader_index();
            assert!(!seen.contains(&i), "{} reuses shader index {i}", m.label());
            seen.push(i);
        }
        assert_eq!(ViewMode::Lit.shader_index(), 0, "Lit must be 0 -- it is the default");
    }

    #[test]
    fn the_default_is_lit() {
        assert_eq!(ViewMode::default(), ViewMode::Lit);
        assert!(ViewMode::default().is_lit());
        assert!(ViewMode::default().shows_atmosphere());
    }

    #[test]
    fn detail_lighting_keeps_normals_and_lighting_only_does_not() {
        // The one difference between the two, and the easiest thing here to get
        // backwards. A surface that looks flat in Detail Lighting has a broken
        // normal map; one that looks flat in Lighting Only is genuinely flat.
        assert!(ViewMode::DetailLighting.neutral_albedo());
        assert!(ViewMode::LightingOnly.neutral_albedo());
        assert!(ViewMode::DetailLighting.uses_material_normals());
        assert!(!ViewMode::LightingOnly.uses_material_normals());
    }

    #[test]
    fn unlit_is_the_only_mode_that_drops_lighting_without_dropping_albedo() {
        assert!(!ViewMode::Unlit.is_lit());
        assert!(!ViewMode::Unlit.neutral_albedo(), "Unlit shows the real albedo");
        assert!(ViewMode::Lit.is_lit());
        assert!(ViewMode::DetailLighting.is_lit());
        assert!(ViewMode::LightingOnly.is_lit());
    }

    #[test]
    fn only_lit_shows_the_atmosphere() {
        // Fog over a debug view defeats the purpose: the mode exists to isolate
        // one term, and haze is a second one laid over the whole frame.
        for m in ViewMode::ALL {
            assert_eq!(
                m.shows_atmosphere(),
                m == ViewMode::Lit,
                "{} disagrees about atmosphere",
                m.label()
            );
        }
    }

    #[test]
    fn wireframe_is_the_only_edge_mode() {
        assert!(ViewMode::Wireframe.is_wireframe());
        for m in ViewMode::ALL.into_iter().filter(|m| *m != ViewMode::Wireframe) {
            assert!(!m.is_wireframe(), "{} should draw filled triangles", m.label());
        }
    }

    #[test]
    fn every_mode_is_labelled() {
        for m in ViewMode::ALL {
            assert!(!m.label().is_empty());
        }
        assert_eq!(ViewMode::ALL.len(), 5);
    }
}

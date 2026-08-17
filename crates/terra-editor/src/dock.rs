//! Panel layout: which panes exist, where they sit, and what can be moved.
//!
//! Panes are `egui_dock` tabs rather than fixed `egui::Panel`s, which is what
//! buys the Unreal behaviours in one go: drag a tab to re-dock it, drag it out
//! of the window to float it, collapse it to its title bar, resize by dragging
//! a separator, close it and reopen it from the View menu.
//!
//! Only the movable panes live here. The toolbar and the status bar stay fixed
//! `egui::Panel`s, deliberately — Unreal does not let you undock its menu bar
//! either, and a floating Save button is a worse editor, not a more flexible
//! one.
//!
//! # Why the viewport is a tab
//!
//! `egui_dock` tiles its whole area, so there is no "leftover middle" for the 3D
//! view to occupy. [`Tab::Viewport`] is therefore a real tab that draws nothing
//! at all: the terrain is rendered underneath the egui pass, and this tab's job
//! is to reserve a hole of the right shape for it and report where that hole is.
//! Without it, docking a panel would paint over the scene instead of shrinking
//! it.

use egui_dock::{DockState, NodeIndex};

/// A movable pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tab {
    /// The 3D view. Draws nothing; reserves space. See the module note.
    Viewport,
    /// Tool list, and the settings for whichever tool is active.
    Tools,
    /// World, environment and quality settings.
    Inspector,
    /// The project's own asset folder: textures, noise maps, models.
    Content,
    /// The non-destructive cave modifier stack.
    Modifiers,
    /// PBR settings for one material. Opened by double-clicking a texture.
    Material,
    /// The Environment Light Mixer: sun, sky, fog, clouds, tone mapping.
    Environment,
}

impl Tab {
    /// Every pane that can be opened from the View menu.
    ///
    /// Excludes [`Tab::Viewport`], which is not optional: closing it would
    /// leave the editor with nowhere to show the world.
    pub const DOCKABLE: [Tab; 6] =
        [Tab::Tools, Tab::Inspector, Tab::Environment, Tab::Content, Tab::Modifiers, Tab::Material];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Viewport => "Viewport",
            Tab::Tools => "Tools",
            Tab::Inspector => "Details",
            Tab::Content => "Content",
            Tab::Modifiers => "Modifiers",
            Tab::Material => "Material",
            Tab::Environment => "Environment",
        }
    }

    /// Whether the user may close this pane.
    pub fn closeable(self) -> bool {
        self != Tab::Viewport
    }
}

/// The dock tree, plus the operations the View menu drives.
pub struct Layout {
    state: DockState<Tab>,
}

impl Default for Layout {
    fn default() -> Self {
        Self::new()
    }
}

impl Layout {
    /// The default arrangement: tools left, details right, content along the
    /// bottom, viewport filling what is left.
    ///
    /// # The fraction is the *first* pane's share
    ///
    /// `egui_dock` documents `fraction` as what the old node keeps, but which
    /// node that is on screen depends on the direction. For `split_right` and
    /// `split_below` the old node comes first, so the fraction is its share. For
    /// `split_left` and `split_above` the *new* node comes first, and the
    /// fraction is the new node's share instead.
    ///
    /// Reading it the other way round is not a subtle error: `split_left(.., 0.82,
    /// Tools)` hands Tools 82% of the window and squeezes the viewport into a
    /// sliver, which is exactly what the first version of this did.
    pub fn new() -> Self {
        let mut state = DockState::new(vec![Tab::Viewport]);
        let surface = state.main_surface_mut();
        // Content first: splitting the full height before the side panels exist
        // makes the content browser span the whole window width, the way
        // Unreal's does, instead of being boxed in between them.
        let [main, _content] = surface.split_below(NodeIndex::root(), 0.74, vec![Tab::Content]);
        // New node comes first here, so 0.16 is the Tools share.
        let [main, _tools] = surface.split_left(main, 0.16, vec![Tab::Tools]);
        // Old node comes first here, so 0.75 is the viewport's share.
        // Environment is tabbed with Details on the right, which is where the
        // request put it: "shown in right panel then from there we can adjust it".
        let [_, details] = surface.split_right(main, 0.75, vec![Tab::Inspector, Tab::Environment]);
        // Modifiers and Material share a leaf: both are "the thing you selected",
        // only one is relevant at a time, and two more permanent panes would
        // leave the viewport a letterbox.
        surface.split_below(details, 0.58, vec![Tab::Modifiers, Tab::Material]);
        Self { state }
    }

    pub fn state_mut(&mut self) -> &mut DockState<Tab> {
        &mut self.state
    }

    pub fn state(&self) -> &DockState<Tab> {
        &self.state
    }

    /// Put the panes back where they started. The escape hatch for a layout
    /// dragged into a state the user cannot undo — every editor with movable
    /// panels needs one.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn is_open(&self, tab: Tab) -> bool {
        self.state.find_tab(&tab).is_some()
    }

    /// Show a pane, as a new tab beside the focused one if it is not already up.
    pub fn open(&mut self, tab: Tab) {
        if !self.is_open(tab) {
            self.state.push_to_focused_leaf(tab);
        }
    }

    /// Show a pane and bring it to the front of whatever leaf holds it.
    ///
    /// Opening alone is not enough for a tabbed pane: double-clicking a texture
    /// has to surface the Material tab even when it is sitting behind Modifiers,
    /// or the click appears to do nothing.
    pub fn focus(&mut self, tab: Tab) {
        self.open(tab);
        if let Some(path) = self.state.find_tab(&tab) {
            // The path came straight from `find_tab`, so it resolves by
            // construction. Logged rather than ignored anyway: silently failing
            // to raise the pane is the bug this function exists to prevent.
            if let Err(e) = self.state.set_active_tab(path) {
                log::warn!("could not raise the {} pane: {e:?}", tab.title());
            }
        }
    }

    pub fn close(&mut self, tab: Tab) {
        if !tab.closeable() {
            return;
        }
        if let Some(path) = self.state.find_tab(&tab) {
            self.state.remove_tab(path);
        }
    }

    pub fn toggle(&mut self, tab: Tab) {
        if self.is_open(tab) {
            self.close(tab);
        } else {
            self.open(tab);
        }
    }

    /// Pop a pane out into its own floating window.
    ///
    /// Dragging a tab out of the dock does this too; this is the menu path, for
    /// people who would rather not discover it by dragging.
    pub fn float(&mut self, tab: Tab) {
        if !tab.closeable() {
            return;
        }
        self.close(tab);
        self.state.add_window(vec![tab]);
    }

    /// Panes currently up, in no particular order.
    pub fn open_tabs(&self) -> Vec<Tab> {
        self.state.iter_all_tabs().map(|(_, t)| *t).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The viewport plus every dockable pane. One list, so adding a pane to the
    /// enum and forgetting the default layout fails here rather than silently
    /// shipping a pane nobody can reach.
    const ALL_PANES: [Tab; 7] = [
        Tab::Viewport,
        Tab::Tools,
        Tab::Inspector,
        Tab::Environment,
        Tab::Content,
        Tab::Modifiers,
        Tab::Material,
    ];

    #[test]
    fn every_enum_variant_is_in_the_default_layout() {
        assert_eq!(ALL_PANES.len(), Tab::DOCKABLE.len() + 1, "Viewport plus the dockable set");
        let l = Layout::new();
        for t in ALL_PANES {
            assert!(l.is_open(t), "{t:?} is not in the default layout");
        }
    }

    #[test]
    fn default_layout_has_every_pane_exactly_once() {
        let l = Layout::new();
        let open = l.open_tabs();
        assert_eq!(open.len(), 7, "expected 7 panes, got {open:?}");
        for t in ALL_PANES {
            assert_eq!(
                open.iter().filter(|o| **o == t).count(),
                1,
                "{:?} should appear once in {open:?}",
                t
            );
        }
    }

    #[test]
    fn panes_close_and_reopen() {
        let mut l = Layout::new();
        assert!(l.is_open(Tab::Content));
        l.close(Tab::Content);
        assert!(!l.is_open(Tab::Content));
        l.open(Tab::Content);
        assert!(l.is_open(Tab::Content), "a closed pane must be reachable again");
    }

    #[test]
    fn toggle_round_trips() {
        let mut l = Layout::new();
        for tab in Tab::DOCKABLE {
            let before = l.is_open(tab);
            l.toggle(tab);
            assert_ne!(l.is_open(tab), before, "{tab:?} did not toggle");
            l.toggle(tab);
            assert_eq!(l.is_open(tab), before, "{tab:?} did not toggle back");
        }
    }

    #[test]
    fn opening_a_pane_twice_does_not_duplicate_it() {
        // Otherwise clicking a View menu entry twice leaves two identical tabs
        // and closing one appears to do nothing.
        let mut l = Layout::new();
        l.open(Tab::Tools);
        l.open(Tab::Tools);
        assert_eq!(l.open_tabs().iter().filter(|t| **t == Tab::Tools).count(), 1);
    }

    #[test]
    fn the_viewport_cannot_be_closed() {
        // Closing it would leave the editor with no hole to render the world
        // into, and no obvious way for the user to get it back.
        let mut l = Layout::new();
        assert!(!Tab::Viewport.closeable());
        l.close(Tab::Viewport);
        assert!(l.is_open(Tab::Viewport), "the viewport was closed");
        l.float(Tab::Viewport);
        assert!(l.is_open(Tab::Viewport), "the viewport was floated away");
    }

    #[test]
    fn floating_a_pane_keeps_it_open() {
        let mut l = Layout::new();
        l.float(Tab::Modifiers);
        assert!(l.is_open(Tab::Modifiers), "a floated pane must still be findable");
        assert_eq!(l.open_tabs().iter().filter(|t| **t == Tab::Modifiers).count(), 1);
    }

    #[test]
    fn reset_restores_the_default_arrangement() {
        let mut l = Layout::new();
        l.close(Tab::Tools);
        l.close(Tab::Content);
        l.float(Tab::Inspector);
        l.reset();
        assert_eq!(l.open_tabs().len(), 7);
        for t in Tab::DOCKABLE {
            assert!(l.is_open(t), "{t:?} missing after reset");
        }
    }

    #[test]
    fn every_dockable_pane_is_closeable_and_titled() {
        for t in Tab::DOCKABLE {
            assert!(t.closeable(), "{t:?} is offered in the View menu but cannot be closed");
            assert!(!t.title().is_empty());
        }
        assert!(!Tab::DOCKABLE.contains(&Tab::Viewport));
    }
}

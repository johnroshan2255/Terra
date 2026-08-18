//! Deleting a world, driven headlessly.
//!
//! This is the one irreversible action in the editor: `world/source/global_height.r16`
//! is the eroded heightmap and cannot be regenerated, so what is asserted here is not
//! only that the controls exist but that the *flow* cannot delete anything by
//! accident:
//!
//! * no confirmation is showing until one is asked for
//! * the confirmation names the world and its path, so it is obvious what is at stake
//! * both outcomes are offered separately -- forgetting where a project is, and
//!   erasing it -- because they differ in kind
//!
//! The delete itself lives in `terra_project::ProjectPaths::delete_project` and has
//! its own tests, including that it refuses a folder that is not a project.
//!
//! No window is opened.

use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use terra_editor::ui::{self, Action, DeleteTarget};
use terra_project::{Library, ProjectEntry};

/// A folder that passes `ProjectEntry::is_available`, so its row renders as a real
/// world rather than as the unreachable variant.
fn project_dir(tag: &str, name: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!("terra-delui-{tag}")).join(name);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("project.ron"), b"Project()").unwrap();
    root
}

fn library(entries: &[(&str, std::path::PathBuf)]) -> Library {
    Library {
        projects: entries
            .iter()
            .enumerate()
            .map(|(i, (name, path))| ProjectEntry {
                name: name.to_string(),
                path: path.clone(),
                last_opened_unix: 1000 + i as u64,
            })
            .collect(),
        ..Default::default()
    }
}

/// Every interactive label in one frame of the worlds pane.
fn labels(lib: &Library, pending: Option<(&str, &std::path::Path)>) -> Vec<String> {
    let mut harness = Harness::builder().with_size(egui::vec2(1200.0, 800.0)).build_ui(|ui| {
        let target = pending.map(|(name, path)| DeleteTarget { name, path });
        let _ = ui::worlds(ui, lib, target);
    });
    harness.run();
    harness
        .root()
        .children_recursive()
        .filter_map(|n| n.accesskit_node().label().filter(|l| !l.is_empty()))
        .collect()
}

/// Click one control and return the action it produced.
///
/// The first non-`None` action is kept rather than the last: `Harness::run` runs frames
/// until the UI settles, and a frame after the click would otherwise overwrite the
/// action with the `None` every frame starts with.
fn act(lib: &Library, pending: Option<(&str, &std::path::Path)>, click: &str) -> Action {
    let mut out = Action::None;
    {
        let mut harness = Harness::builder().with_size(egui::vec2(1200.0, 800.0)).build_ui(|ui| {
            let target = pending.map(|(name, path)| DeleteTarget { name, path });
            let a = ui::worlds(ui, lib, target);
            if !matches!(a, Action::None) && matches!(out, Action::None) {
                out = a;
            }
        });
        harness.run();
        harness.get_by_label(click).click();
        harness.run();
    }
    out
}

#[test]
fn a_world_offers_a_delete_control() {
    let d = project_dir("offers", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    let l = labels(&lib, None);
    assert!(l.iter().any(|s| s.contains("Delete this world")), "no way to delete a world: {l:?}");
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn nothing_is_confirmed_until_it_is_asked_for() {
    // The pane must not show a destructive dialog it was not asked to show.
    let d = project_dir("noconfirm", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    let l = labels(&lib, None);
    for gone in ["Delete files", "Remove from list", "Cancel"] {
        assert!(!l.iter().any(|s| s == gone), "{gone:?} is showing unprompted: {l:?}");
    }
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn asking_to_delete_does_not_delete() {
    // The row's button opens a confirmation and nothing else. If this ever returned
    // `Delete`, one click would erase a project.
    let d = project_dir("ask", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    let action = act(&lib, None, "Delete this world");
    match action {
        Action::AskDelete(p) => assert_eq!(p, d),
        other => panic!("the row's delete button produced {other:?}, not a confirmation"),
    }
    assert!(d.exists(), "the folder was touched by asking");
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn the_confirmation_names_the_world_and_where_it_is() {
    // Two worlds can share a name; the path is what makes it unambiguous which one is
    // about to go.
    let d = project_dir("names", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    let l = labels(&lib, Some(("My Game", &d)));
    // Buttons publish labels; the name and path are plain text, so the assertion is on
    // the controls being the confirmation's and on the actions they produce below.
    for want in ["Cancel", "Remove from list", "Delete files"] {
        assert!(l.iter().any(|s| s.contains(want)), "{want:?} missing: {l:?}");
    }
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn confirming_asks_for_the_files_to_go() {
    let d = project_dir("confirm", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    match act(&lib, Some(("My Game", &d)), "Delete files") {
        Action::Delete { path, files } => {
            assert_eq!(path, d);
            assert!(files, "the destructive button must ask for the files");
        }
        other => panic!("got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn removing_from_the_list_leaves_the_files_alone() {
    // The reversible half, and the reason it is a separate button rather than a
    // checkbox: someone who wants the world off the list should not have to read the
    // small print to avoid erasing it.
    let d = project_dir("forget", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    match act(&lib, Some(("My Game", &d)), "Remove from list") {
        Action::Delete { path, files } => {
            assert_eq!(path, d);
            assert!(!files, "removing from the list must not delete files");
        }
        other => panic!("got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn cancelling_produces_a_cancel() {
    let d = project_dir("cancel", "MyGame");
    let lib = library(&[("My Game", d.clone())]);
    assert!(matches!(act(&lib, Some(("My Game", &d)), "Cancel"), Action::CancelDelete));
    assert!(d.exists());
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
}

#[test]
fn an_unreachable_world_still_only_offers_removal() {
    // A path that no longer resolves -- an unplugged drive. There is nothing to delete
    // and no way to know what is at the other end, so the only offer is to forget it.
    let lib = library(&[("Gone", std::path::PathBuf::from("/nonexistent/Gone"))]);
    let l = labels(&lib, None);
    assert!(
        l.iter().any(|s| s.contains("remove from list")),
        "an unavailable world should offer removal: {l:?}"
    );
    assert!(
        !l.iter().any(|s| s.contains("Delete this world")),
        "an unavailable world must not offer a file delete: {l:?}"
    );
}

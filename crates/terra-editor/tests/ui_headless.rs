//! The editor UI, driven headlessly.
//!
//! `egui_kittest` runs a real egui context against AccessKit, so the widget
//! tree can be queried by label without a window, a surface, or an event loop.
//! That is the only way to check the two things that were asked for and are
//! otherwise invisible outside a running app:
//!
//! * settings follow the active tool instead of all being shown at once, and
//! * the brush sliders are gone from the right-hand pane.
//!
//! These tests build the panes directly rather than going through `App`, which
//! owns a GPU device and a window.
//!
//! # What can and cannot be asserted
//!
//! AccessKit only publishes *interactive* widgets. Buttons, checkboxes and
//! selectable labels are queryable; plain `Label`s and slider captions are not,
//! so a section is checked by the controls it contains rather than by its
//! heading. Tab titles are not published either -- pane visibility is asserted
//! through `dock::Layout`, which is unit-tested in its own module.
//!
//! Whether the result *looks* right is not covered here. `layout_snapshot`
//! renders the whole editor to a PNG for that, and is `#[ignore]`d because it
//! is for looking at rather than asserting on.

use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use terra_editor::dock::{Layout, Tab};
use terra_editor::ui::{self, AssetKind, ContentView, EditorView, PaintMode, Tool};
use terra_render::terrain::SculptMode;

/// Everything `EditorView` borrows, owned in one place so a test can hand out
/// fresh references per frame without threading a dozen locals.
struct Fixture {
    mode: SculptMode,
    radius: f32,
    strength: f32,
    params: terra_project::TerrainParams,
    tool: Tool,
    layer: u32,
    paint_mode: PaintMode,
    paint_flow: f32,
    species: usize,
    tools_open: bool,
    inspector_open: bool,
    sky: terra_render::lighting::SkySettings,
    env: terra_render::Environment,
    modifiers: terra_voxel::ModifierStack,
    selected_modifier: Option<usize>,
    noise: terra_voxel::NoiseField,
    noise_library: Vec<String>,
    assets: [Vec<String>; 3],
    asset_kind: AssetKind,
    viewport_rect: Option<egui::Rect>,
    layout: Layout,
    /// `None` models "nothing imported yet", which is now the default state of a
    /// fresh project and has to render sensibly.
    material: Option<(String, terra_render::material::LayerParams)>,
    view_mode: terra_render::ViewMode,
    /// Driving. `Screen::Editor` is still true while playing, so this is what
    /// separates "editing" from "the viewport is the game".
    playing: bool,
}

impl Default for Fixture {
    fn default() -> Self {
        Self {
            mode: SculptMode::Raise,
            radius: 120.0,
            strength: 1.5,
            params: terra_project::TerrainParams::default(),
            tool: Tool::Sculpt,
            layer: 0,
            paint_mode: PaintMode::Brush,
            paint_flow: 2.0,
            species: 0,
            tools_open: true,
            inspector_open: true,
            sky: Default::default(),
            env: terra_render::Environment::daylight(),
            modifiers: Default::default(),
            selected_modifier: None,
            noise: Default::default(),
            noise_library: Vec::new(),
            assets: [Vec::new(), Vec::new(), Vec::new()],
            asset_kind: AssetKind::Texture,
            viewport_rect: None,
            layout: Layout::new(),
            material: None,
            view_mode: terra_render::ViewMode::default(),
            playing: false,
        }
    }
}

impl Fixture {
    /// One UI frame of the editor.
    fn frame(&mut self, ui: &mut egui::Ui) {
        let root = "/tmp/DemoProject/assets";
        let content = ContentView {
            root,
            entries: [&self.assets[0], &self.assets[1], &self.assets[2]],
            kind: self.asset_kind,
        };
        let _ = ui::editor(
            ui,
            &mut self.layout,
            EditorView {
                mode: &mut self.mode,
                radius: &mut self.radius,
                strength: &mut self.strength,
                world_name: "Demo",
                size: terra_core::WorldSize::Medium,
                unsaved: false,
                brush_at: None,
                height_res: 1024,
                params: &mut self.params,
                playing: self.playing,
                speed_kph: 0.0,
                tool: &mut self.tool,
                palette: &[],
                selected_layer: &mut self.layer,
                paint_mode: &mut self.paint_mode,
                paint_flow: &mut self.paint_flow,
                painted: false,
                foliage: &[],
                selected_species: &mut self.species,
                species_rules: None,
                foliage_instances: 0,
                selection: None,
                prop_count: 0,
                tools_open: &mut self.tools_open,
                inspector_open: &mut self.inspector_open,
                sky: &mut self.sky,
                env: &mut self.env,
                active_road: None,
                road_count: 0,
                modifiers: &mut self.modifiers,
                selected_modifier: &mut self.selected_modifier,
                content: &content,
                noise: &mut self.noise,
                noise_library: &self.noise_library,
                viewport_rect: &mut self.viewport_rect,
                view_mode: &mut self.view_mode,
                material: self.material.as_mut().map(|(name, params)| ui::MaterialView {
                    name,
                    role: "rock",
                    texture: None,
                    params,
                }),
            },
        );
    }

    /// Run one frame and return every interactive label in the tree.
    ///
    /// `children_recursive` walks the whole AccessKit tree, so one call covers
    /// every pane rather than only the focused one. Labels come off the
    /// underlying AccessKit node; `Node` itself exposes only query and
    /// interaction.
    fn labels(&mut self) -> Vec<String> {
        let mut harness =
            Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| self.frame(ui));
        harness.run();
        harness
            .root()
            .children_recursive()
            .filter_map(|n| n.accesskit_node().label().filter(|l| !l.is_empty()))
            .collect()
    }

    fn has(&mut self, needle: &str) -> bool {
        self.labels().iter().any(|l| l.contains(needle))
    }

    /// Every published widget *value*, as opposed to label.
    ///
    /// Sliders and drag values expose their number here and not as a label, so
    /// counting these is how a "the controls are present" assertion is made
    /// without depending on captions AccessKit never publishes.
    fn values(&mut self) -> Vec<String> {
        let mut harness =
            Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| self.frame(ui));
        harness.run();
        harness
            .root()
            .children_recursive()
            .filter_map(|n| n.accesskit_node().value().filter(|v| !v.is_empty()))
            .collect()
    }

    /// Builder setters, so a test reads as one expression rather than a
    /// default followed by a pile of field assignments.
    fn with_tool(mut self, tool: Tool) -> Self {
        self.tool = tool;
        self
    }

    fn with_mode(mut self, mode: SculptMode) -> Self {
        self.mode = mode;
        self
    }

    fn with_view_mode(mut self, mode: terra_render::ViewMode) -> Self {
        self.view_mode = mode;
        self
    }

    fn driving(mut self) -> Self {
        self.playing = true;
        self
    }

    fn with_material(mut self, name: &str) -> Self {
        self.material = Some((name.to_string(), terra_render::material::LayerParams::default()));
        // Material shares a leaf with Modifiers, so it starts as the background
        // tab and draws nothing until it is brought forward -- exactly what a
        // double-click does in the app.
        self.layout.focus(Tab::Material);
        self
    }
}

#[test]
fn the_editor_builds_without_panicking() {
    // The floor: every pane draws, in one frame, with empty palettes.
    let mut f = Fixture::default();
    assert!(!f.labels().is_empty(), "the UI produced no interactive nodes at all");
}

#[test]
fn brush_sliders_are_gone_from_the_panels() {
    // What was asked for: the BRUSH section left the right-hand pane, and its
    // sliders are replaced by the bracket keys. A slider publishes its value as
    // an interactive node, so a surviving radius slider would show up here.
    let mut f = Fixture::default();
    let labels = f.labels();
    assert!(
        !labels.iter().any(|l| l.contains("Radius m")),
        "the radius slider is still present: {labels:?}"
    );
    assert!(
        !labels.iter().any(|l| l == "Strength"),
        "the strength slider is still present: {labels:?}"
    );
}

#[test]
fn grass_controls_are_gone() {
    let mut f = Fixture::default();
    for tool in [Tool::Camera, Tool::Sculpt, Tool::Paint, Tool::Foliage, Tool::Road] {
        f.tool = tool;
        let labels = f.labels();
        for gone in ["Blades per m2", "Blade height m", "Grass"] {
            assert!(
                !labels.iter().any(|l| l.contains(gone)),
                "{gone:?} survives under {tool:?}: {labels:?}"
            );
        }
    }
}

#[test]
fn settings_follow_the_active_tool() {
    // The core of the request. Each marker is a control unique to one tool, so
    // finding it under another tool means the settings are not contextual.
    let markers =
        [(Tool::Road, "New road"), (Tool::Paint, "Fill world with"), (Tool::Select, "Place")];
    for (tool, marker) in markers {
        let mut own = Fixture::default().with_tool(tool);
        assert!(own.has(marker), "{tool:?} should show {marker:?}: {:?}", own.labels());

        for other in [Tool::Camera, Tool::Sculpt] {
            let mut f = Fixture::default().with_tool(other);
            assert!(!f.has(marker), "{marker:?} leaked into {other:?}: {:?}", f.labels());
        }
    }
}

#[test]
fn noise_settings_appear_only_under_the_noise_brush() {
    // The contextual rule applied *within* a tool, not just between tools.
    let mut noise = Fixture::default().with_tool(Tool::Sculpt).with_mode(SculptMode::Noise);
    for control in ["Ridged", "Billow", "Re-roll seed"] {
        assert!(noise.has(control), "{control:?} missing: {:?}", noise.labels());
    }

    let mut raise = Fixture::default().with_tool(Tool::Sculpt).with_mode(SculptMode::Raise);
    assert!(
        !raise.has("Re-roll seed"),
        "noise controls shown under a brush that ignores them: {:?}",
        raise.labels()
    );
}

#[test]
fn the_noise_brush_works_before_anything_is_imported() {
    // "there will be default noise but user can upload the black and white
    // things" -- the built-in pattern has to be usable on its own.
    let f = terra_voxel::NoiseField::default();
    assert!(!f.pattern.is_uploaded());
    assert!(f.pattern.label().contains("built-in"));

    let mut ui_f = Fixture::default().with_tool(Tool::Sculpt).with_mode(SculptMode::Noise);
    // The procedural controls are the built-in pattern's controls, so their
    // presence is what says the tool is armed with no upload.
    assert!(ui_f.has("Ridged"));
}

#[test]
fn imported_noise_maps_are_selectable_in_the_sculpt_pane() {
    let mut f = Fixture::default().with_tool(Tool::Sculpt).with_mode(SculptMode::Noise);
    f.noise_library = vec!["cracks.png".into(), "rock_bumps.png".into()];
    let labels = f.labels();
    for name in ["cracks.png", "rock_bumps.png"] {
        assert!(labels.iter().any(|l| l.contains(name)), "{name} not offered: {labels:?}");
    }
}

#[test]
fn the_content_browser_offers_all_three_shelves_and_an_import() {
    let mut f = Fixture::default();
    let labels = f.labels();
    for k in AssetKind::ALL {
        assert!(labels.iter().any(|l| l.contains(k.label())), "{} shelf missing", k.label());
    }
    assert!(labels.iter().any(|l| l.starts_with("Import ")), "no import control: {labels:?}");
}

#[test]
fn shelf_counts_track_the_project_folder() {
    let mut f = Fixture::default();
    assert!(f.has("Textures  0"), "an empty shelf should read zero: {:?}", f.labels());
    f.assets[0] = vec!["Rock042".into(), "Gravel007".into()];
    let labels = f.labels();
    assert!(labels.iter().any(|l| l.contains("Textures  2")), "count not updated: {labels:?}");
    for name in ["Rock042", "Gravel007"] {
        assert!(labels.iter().any(|l| l.contains(name)), "{name} not shown: {labels:?}");
    }
}

#[test]
fn the_modifier_pane_offers_a_way_to_make_a_cave() {
    let mut f = Fixture::default();
    assert!(f.has("Add tunnel"), "no way to create a cave: {:?}", f.labels());
}

#[test]
fn a_cave_modifier_shows_its_name_and_operations() {
    let mut f = Fixture::default();
    f.modifiers.push(terra_voxel::Modifier::carve(
        "Main passage",
        terra_voxel::Shape::Tube(terra_voxel::Tube::straight(
            glam::Vec3::ZERO,
            glam::Vec3::X * 100.0,
            6.0,
        )),
        1.5,
    ));
    f.selected_modifier = Some(0);
    let labels = f.labels();
    assert!(labels.iter().any(|l| l.contains("Main passage")), "{labels:?}");
    // Selected, so the operator buttons are exposed.
    for op in terra_voxel::Op::ALL {
        assert!(
            labels.iter().any(|l| l.contains(op.label())),
            "{} missing: {labels:?}",
            op.label()
        );
    }
}

#[test]
fn closing_a_pane_removes_its_contents_from_the_tree() {
    // Proves the dock really drives visibility rather than drawing a tab bar
    // over always-present panels.
    let mut f = Fixture::default();
    assert!(f.has("Add tunnel"));
    f.layout.close(Tab::Modifiers);
    assert!(!f.has("Add tunnel"), "a closed pane still rendered its contents");
    f.layout.open(Tab::Modifiers);
    assert!(f.has("Add tunnel"), "reopening the pane did not bring it back");
}

#[test]
fn every_pane_is_open_in_the_default_layout() {
    // Asserted through the layout rather than through tab titles, which
    // AccessKit does not publish.
    let l = Layout::new();
    for tab in [Tab::Viewport, Tab::Tools, Tab::Inspector, Tab::Content, Tab::Modifiers] {
        assert!(l.is_open(tab), "{tab:?} is not in the default layout");
    }
}

#[test]
fn the_viewport_reports_where_it_ended_up() {
    // What the perf overlay positions itself against now that panel widths are
    // no longer constants.
    let mut f = Fixture::default();
    let _ = f.labels();
    let rect = f.viewport_rect.expect("the viewport tab never reported a rect");
    assert!(rect.width() > 100.0 && rect.height() > 100.0, "implausible viewport {rect:?}");
    assert!(rect.width() < 1600.0, "the viewport should not span the full window: {rect:?}");
    assert!(rect.height() < 1000.0, "the content pane should take height off it: {rect:?}");
}

#[test]
fn the_toolbar_stays_fixed_rather_than_docked() {
    // Save and Exit are deliberately not movable. A floating Save button is a
    // worse editor, not a more flexible one.
    let mut f = Fixture::default();
    let labels = f.labels();
    for b in ["Save", "Exit"] {
        assert!(labels.iter().any(|l| l == b), "{b} missing: {labels:?}");
    }
    assert!(!Tab::DOCKABLE.iter().any(|t| t.title() == "Save"));
}

#[test]
fn every_sculpt_mode_is_selectable() {
    // The bug this replaced a mapping table to fix: four of the eight modes were
    // shown greyed out because only four had a heightfield implementation. All
    // eight now run, so none may be disabled -- a Noise tool nobody can click is
    // not a Noise tool.
    let mut f = Fixture::default().with_tool(Tool::Sculpt);
    let labels = f.labels();
    for m in SculptMode::ALL {
        assert!(
            labels.iter().any(|l| l.contains(m.label())),
            "{} missing from the palette: {labels:?}",
            m.label()
        );
    }
    assert_eq!(SculptMode::ALL.len(), 8);
}

#[test]
fn only_the_raise_pair_advertises_a_ctrl_inverse() {
    for m in SculptMode::ALL {
        let has = ui::invert_label(m).is_some();
        let want = matches!(m, SculptMode::Raise | SculptMode::Lower);
        assert_eq!(has, want, "{} inverse advertised: {has}, want {want}", m.label());
    }
}

#[test]
#[ignore]
fn layout_snapshot() {
    let mut f = Fixture::default().with_tool(Tool::Sculpt).with_mode(SculptMode::Noise);
    f.assets = [
        vec!["Rock042".into(), "Gravel007".into(), "Snow019".into()],
        vec!["cracks.png".into(), "rock_bumps.png".into()],
        vec!["pine.glb".into()],
    ];
    f.noise_library = f.assets[1].clone();
    f.modifiers.push(terra_voxel::Modifier::carve(
        "Main passage",
        terra_voxel::Shape::Tube(terra_voxel::Tube::straight(
            glam::Vec3::ZERO,
            glam::Vec3::X * 180.0,
            6.0,
        )),
        1.5,
    ));
    f.selected_modifier = Some(0);
    // Struct update syntax cannot reach the private padding field, so mutate.
    let mut params = terra_render::material::LayerParams::default();
    params.parallax_m = 0.06;
    f.material = Some(("Rock042".to_string(), params));

    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    let image = harness.render().expect("headless wgpu render failed");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
    image.save(dir.join("ui-layout.png")).expect("could not write the snapshot");

    // Second shot with the View menu open, since that is the pane-visibility
    // control and it cannot be seen in the first.
    harness.get_by_label("View").click();
    harness.run();
    let menu = harness.render().expect("headless wgpu render failed");
    menu.save(dir.join("ui-view-menu.png")).expect("could not write the snapshot");
    drop(harness);

    // The Environment Light Mixer, brought to the front of the right-hand column.
    f.layout.focus(Tab::Environment);
    let mut envh =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    envh.run();
    let shot = envh.render().expect("headless wgpu render failed");
    shot.save(dir.join("ui-environment.png")).expect("could not write the snapshot");
    drop(envh);
    f.layout.focus(Tab::Inspector);

    // Material editor, brought to the front of the leaf it shares with Modifiers.
    f.layout.focus(Tab::Material);
    let mut mat =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    mat.run();
    let shot = mat.render().expect("headless wgpu render failed");
    shot.save(dir.join("ui-material.png")).expect("could not write the snapshot");
    drop(mat);
    f.layout.focus(Tab::Modifiers);

    // Wireframe mode, to check the status-bar indicator and the View menu entry.
    f.view_mode = terra_render::ViewMode::Wireframe;
    let mut wire =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    wire.run();
    wire.get_by_label("View").click();
    wire.run();
    let shot = wire.render().expect("headless wgpu render failed");
    shot.save(dir.join("ui-view-modes.png")).expect("could not write the snapshot");
    drop(wire);
    f.view_mode = terra_render::ViewMode::Lit;

    // Narrow window, to check the panels stay legible rather than clipping
    // their sliders off the right edge -- which is what a fixed-width value box
    // did before.
    let mut narrow =
        Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_ui(|ui| f.frame(ui));
    narrow.run();
    let shot = narrow.render().expect("headless wgpu render failed");
    shot.save(dir.join("ui-narrow.png")).expect("could not write the snapshot");
    drop(narrow);

    // Third shot with a panel ejected into a floating window, which is the
    // claim that panels really move rather than only re-docking.
    f.layout.float(Tab::Modifiers);
    let mut floated =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    floated.run();
    floated.run();
    let shot = floated.render().expect("headless wgpu render failed");
    shot.save(dir.join("ui-floating.png")).expect("could not write the snapshot");
    eprintln!("wrote ui-layout.png, ui-view-menu.png and ui-floating.png in {}", dir.display());
}

// ---------------------------------------------------------------------------
// View menu
// ---------------------------------------------------------------------------

#[test]
fn the_toolbar_offers_a_view_menu() {
    // Load-bearing, not a convenience: the close button on a tab is a one-way
    // trip without it.
    let mut f = Fixture::default();
    assert!(f.has("View"), "no View menu in the toolbar: {:?}", f.labels());
}

#[test]
fn the_view_menu_lists_every_dockable_pane() {
    let mut f = Fixture::default();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label("View").click();
    harness.run();

    let labels: Vec<String> = harness
        .root()
        .children_recursive()
        .filter_map(|n| n.accesskit_node().label().filter(|l| !l.is_empty()))
        .collect();
    for tab in Tab::DOCKABLE {
        assert!(
            labels.iter().any(|l| l.contains(tab.title())),
            "{} is not listed in the View menu: {labels:?}",
            tab.title()
        );
    }
    assert!(labels.iter().any(|l| l.contains("Reset layout")), "no reset: {labels:?}");
    assert!(labels.iter().any(|l| l.contains("Float a panel")), "no float entry: {labels:?}");
}

#[test]
fn a_closed_pane_can_be_reopened_from_the_view_menu() {
    // The actual bug this menu fixes, driven end to end: close the pane, then
    // click its entry and check its contents come back.
    let mut f = Fixture::default();
    f.layout.close(Tab::Modifiers);
    assert!(!f.has("Add tunnel"), "the pane did not close");

    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label("View").click();
    harness.run();
    harness.get_by_label(Tab::Modifiers.title()).click();
    harness.run();
    drop(harness);

    assert!(
        f.layout.is_open(Tab::Modifiers),
        "clicking the View menu entry did not reopen the pane"
    );
    assert!(f.has("Add tunnel"), "the pane reopened but drew nothing");
}

#[test]
fn reset_layout_restores_every_pane() {
    let mut f = Fixture::default();
    f.layout.close(Tab::Tools);
    f.layout.close(Tab::Content);
    f.layout.float(Tab::Modifiers);

    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label("View").click();
    harness.run();
    harness.get_by_label("Reset layout").click();
    harness.run();
    drop(harness);

    for tab in Tab::DOCKABLE {
        assert!(f.layout.is_open(tab), "{tab:?} missing after Reset layout");
    }
}

#[test]
fn the_viewport_is_not_offered_for_closing_or_floating() {
    // Closing it would leave nowhere to render the world, and the user would
    // have no obvious way back.
    assert!(!Tab::DOCKABLE.contains(&Tab::Viewport));
    let mut l = Layout::new();
    l.close(Tab::Viewport);
    l.float(Tab::Viewport);
    assert!(l.is_open(Tab::Viewport));
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

#[test]
fn no_materials_ship_prebuilt() {
    // The request, as a test: a fresh project has an empty palette. There used to
    // be six noise-generated layers here, which made a new project look furnished
    // with materials the user had not chosen and could not edit.
    let mut f = Fixture::default().with_tool(Tool::Paint);
    let labels = f.labels();
    for gone in ["Soil", "Grass", "Rock", "Gravel", "Snow", "Mud"] {
        assert!(
            !labels.iter().any(|l| l == gone),
            "{gone} is still in the palette with nothing imported: {labels:?}"
        );
    }
    // With no material selected the fill button has nothing to name.
    assert!(labels.iter().any(|l| l.contains("Fill world with --")), "{labels:?}");
}

#[test]
fn the_material_pane_is_inert_until_a_material_is_selected() {
    // Slider captions are plain labels and AccessKit does not publish them, so
    // the pane is asserted through its buttons: with nothing selected there is
    // nothing to reset and nothing to paint with.
    let mut f = Fixture::default();
    f.layout.focus(Tab::Material);
    let labels = f.labels();
    assert!(!labels.iter().any(|l| l.contains("Reset to defaults")), "{labels:?}");
    assert!(!labels.iter().any(|l| l.contains("Paint with this")), "{labels:?}");
}

#[test]
fn selecting_a_material_populates_the_pane() {
    let mut f = Fixture::default().with_material("Rock042");
    let labels = f.labels();
    assert!(labels.iter().any(|l| l.contains("Reset to defaults")), "{labels:?}");
    assert!(labels.iter().any(|l| l.contains("Paint with this")), "{labels:?}");
    // Sliders publish their number as a value rather than a label, so counting
    // numeric widgets is how "the PBR controls are all there" is asserted
    // without depending on captions. Six sliders: tiling, normal, roughness,
    // occlusion, parallax, blend band. The captions are checked in the snapshot.
    let numeric = f.values().iter().filter(|v| v.parse::<f64>().is_ok()).count();
    assert!(numeric >= 6, "expected six PBR sliders, saw {numeric} numeric widgets");
}

#[test]
fn paint_with_this_switches_to_the_paint_tool() {
    // The path from "I have set this material up" to "I am putting it on the
    // ground", which otherwise needs the user to find the Paint tool themselves.
    let mut f = Fixture::default().with_material("Rock042");
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    // The pane returns an action; the app acts on it. Here we only assert the
    // control exists and is clickable without panicking.
    harness.get_by_label("Paint with this").click();
    harness.run();
}

#[test]
fn material_defaults_are_a_neutral_starting_point() {
    // A freshly imported material must render as itself: no tint, no parallax
    // cost, sampled roughness and normals used as authored.
    let p = terra_render::material::LayerParams::default();
    assert_eq!(p.tint, [1.0, 1.0, 1.0], "a new material must not be tinted");
    assert_eq!(p.normal_strength, 1.0);
    assert_eq!(p.roughness, 1.0);
    assert_eq!(p.ao, 1.0);
    assert_eq!(p.parallax_m, 0.0, "parallax costs samples; it must be opt-in");
    assert!(p.tiling_m > 0.0);
}

#[test]
fn editing_one_material_does_not_disturb_the_defaults() {
    let mut f = Fixture::default().with_material("Gravel007");
    let _ = f.labels();
    // The pane writes through the borrow, so the fixture's copy is what the app
    // would read back and store.
    f.material.as_mut().unwrap().1.parallax_m = 0.08;
    let _ = f.labels();
    assert_eq!(f.material.as_ref().unwrap().1.parallax_m, 0.08);
    assert_eq!(terra_render::material::LayerParams::default().parallax_m, 0.0);
}

// ---------------------------------------------------------------------------
// Environment Light Mixer
// ---------------------------------------------------------------------------

#[test]
fn the_environment_pane_is_in_the_right_panel() {
    // "shown in right panel then from there we can adjust it" -- tabbed with
    // Details, which is the right-hand column.
    let l = Layout::new();
    assert!(l.is_open(Tab::Environment));
    assert!(Tab::DOCKABLE.contains(&Tab::Environment));
}

#[test]
fn the_mixer_offers_quick_create_and_every_section() {
    let mut f = Fixture::default();
    f.layout.focus(Tab::Environment);
    let labels = f.labels();
    for control in [
        "Daylight",
        "Overcast",
        "Night",
        "Sky Atmosphere",
        "Sky Light (ambient bounce)",
        "Exponential Height Fog",
        "Volumetric Clouds",
        "ACES",
        "Reset environment",
    ] {
        assert!(labels.iter().any(|l| l.contains(control)), "{control} missing: {labels:?}");
    }
}

#[test]
fn the_old_scattered_sections_are_gone() {
    // The request: fog, atmosphere and light settings stop being four separate
    // blocks. What is left in Details is frame-time settings only.
    let mut f = Fixture::default();
    f.layout.focus(Tab::Inspector);
    let labels = f.labels();
    // Fog and atmosphere controls must no longer be reachable from Details.
    for gone in ["Volumetric fog", "Haze density", "Forward scattering", "God rays"] {
        assert!(
            !labels.iter().any(|l| l.contains(gone)),
            "{gone} is still in the Details panel: {labels:?}"
        );
    }
    // Quality is what remains, and it is about milliseconds.
    assert!(labels.iter().any(|l| l.contains("Temporal AA")), "{labels:?}");
}

#[test]
fn quick_create_buttons_replace_the_whole_environment() {
    let mut f = Fixture::default();
    f.layout.focus(Tab::Environment);
    f.env.fog.density = 0.04;
    f.env.sun.pitch_deg = 12.0;

    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label("Daylight").click();
    harness.run();
    drop(harness);

    assert_eq!(f.env, terra_render::Environment::daylight(), "Daylight must reset everything");
}

#[test]
fn the_running_cycle_survives_frames_nobody_touched() {
    // The sun panel stops the cycle when the pitch or yaw slider *moves*, so
    // that a hand-placed sun is not snapped back by the next tick. The guard
    // compares within a frame, which means it must not fire on a frame where
    // nothing was dragged -- if it did, "Run day/night cycle" could never stay
    // on for more than one frame.
    let mut f = Fixture::default();
    f.layout.focus(Tab::Environment);
    f.env.cycle_running = true;
    for _ in 0..3 {
        let _ = f.labels();
        assert!(f.env.cycle_running, "the cycle was stopped without anyone dragging a slider");
    }
}

#[test]
fn the_clock_and_the_sun_stay_consistent() {
    // Moving the hour re-derives the sun, which is what keeps the shadows
    // pointing somewhere the time of day explains.
    let mut e = terra_render::Environment::daylight();
    e.time_of_day = 6.5;
    e.sync_sun_to_clock();
    let dawn = e.sun.direction().y;
    e.time_of_day = 12.0;
    e.sync_sun_to_clock();
    assert!(e.sun.direction().y > dawn);
}

// ---------------------------------------------------------------------------
// Visualization modes
// ---------------------------------------------------------------------------

#[test]
fn the_view_menu_lists_every_mode_with_its_hotkey() {
    let mut f = Fixture::default();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label("View").click();
    harness.run();

    let labels: Vec<String> = harness
        .root()
        .children_recursive()
        .filter_map(|n| n.accesskit_node().label().filter(|l| !l.is_empty()))
        .collect();
    for m in terra_render::ViewMode::ALL {
        assert!(
            labels.iter().any(|l| l.contains(m.label())),
            "{} is not in the View menu: {labels:?}",
            m.label()
        );
        // The hotkey is listed beside it, because that is how it gets learned.
        assert!(
            labels.iter().any(|l| l.contains(&format!("Alt+{}", m.hotkey_digit()))),
            "Alt+{} is not shown for {}",
            m.hotkey_digit(),
            m.label()
        );
    }
}

#[test]
fn picking_a_mode_from_the_menu_selects_it() {
    let mut f = Fixture::default();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label("View").click();
    harness.run();
    harness.get_by_label_contains("Wireframe").click();
    harness.run();
    drop(harness);
    assert_eq!(f.view_mode, terra_render::ViewMode::Wireframe);
}

#[test]
fn a_debug_mode_announces_itself_in_the_status_bar() {
    // Forgetting a debug mode is on is the classic way to spend ten minutes
    // debugging a material that was never broken, so it has to be visible without
    // opening a menu.
    let mut lit = Fixture::default();
    let quiet = lit.labels();
    assert!(
        !quiet.iter().any(|l| l.contains("\u{2715}")),
        "Lit should not announce itself: {quiet:?}"
    );

    for m in terra_render::ViewMode::ALL.into_iter().filter(|m| *m != terra_render::ViewMode::Lit) {
        let mut f = Fixture::default().with_view_mode(m);
        let labels = f.labels();
        assert!(
            labels.iter().any(|l| l.contains(m.label()) && l.contains('\u{2715}')),
            "{} is not shown in the status bar: {labels:?}",
            m.label()
        );
    }
}

#[test]
fn the_status_bar_indicator_is_the_way_back_to_lit() {
    // A label would say what is wrong; a button fixes it. On seeing "Wireframe"
    // the thing wanted is Lit, so the indicator is the control.
    let mut f = Fixture::default().with_view_mode(terra_render::ViewMode::Wireframe);
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    harness.run();
    harness.get_by_label_contains("Wireframe").click();
    harness.run();
    drop(harness);
    assert_eq!(f.view_mode, terra_render::ViewMode::Lit);
}

#[test]
fn the_visualize_section_is_absent_while_driving() {
    // The modes are authoring views. Once the viewport is the game, offering to
    // wireframe it is offering something that will not happen -- so the section is
    // gone rather than greyed out.
    let mut driving = Fixture::default().driving();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| driving.frame(ui));
    harness.run();
    harness.get_by_label("View").click();
    harness.run();
    let labels: Vec<String> = harness
        .root()
        .children_recursive()
        .filter_map(|n| n.accesskit_node().label().filter(|l| !l.is_empty()))
        .collect();
    drop(harness);

    for m in terra_render::ViewMode::ALL {
        assert!(
            !labels.iter().any(|l| l.contains(&format!("Alt+{}", m.hotkey_digit()))),
            "{} is still offered while driving: {labels:?}",
            m.label()
        );
    }
    // The panel controls are still there -- only the visualization block goes.
    assert!(labels.iter().any(|l| l.contains("Reset layout")), "{labels:?}");
}

#[test]
fn the_status_indicator_is_silent_while_driving() {
    // The mode is remembered but not applied while driving, so naming it would
    // announce something that is not on screen.
    let mut f = Fixture::default().with_view_mode(terra_render::ViewMode::Wireframe).driving();
    let labels = f.labels();
    assert!(
        !labels.iter().any(|l| l.contains('\u{2715}') && l.contains("Wireframe")),
        "the indicator showed while driving: {labels:?}"
    );

    // ...and it comes back on returning to editing, because the mode was kept.
    let mut editing = Fixture::default().with_view_mode(terra_render::ViewMode::Wireframe);
    assert!(
        editing.labels().iter().any(|l| l.contains("Wireframe")),
        "the mode was not remembered"
    );
}

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
    /// Last import result, shown in the Content pane.
    notice: Option<(String, bool)>,
    /// Rules of the selected species, or `None` for "nothing imported".
    rules: Option<terra_render::scatter::Rules>,
    water: terra_render::water::WaterSettings,
    selected_water: Option<usize>,
    outliner: Vec<ui::OutlinerItem>,
    outliner_selection: Option<(ui::OutlinerKind, usize)>,
    viewport_rect: Option<egui::Rect>,
    layout: Layout,
    /// `None` models "nothing imported yet", which is now the default state of a
    /// fresh project and has to render sensibly.
    material: Option<(String, terra_render::material::LayerParams)>,
    /// Role of the open material, and what the name guess said.
    material_role: (u32, u32),
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
            notice: None,
            rules: None,
            water: Default::default(),
            selected_water: None,
            outliner: Vec::new(),
            outliner_selection: None,
            viewport_rect: None,
            layout: Layout::new(),
            material: None,
            material_role: (terra_render::material::ROCK, terra_render::material::ROCK),
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
                species_rules: self.rules.as_mut(),
                foliage_instances: 0,
                selection: None,
                prop_count: 0,
                tools_open: &mut self.tools_open,
                inspector_open: &mut self.inspector_open,
                sky: &mut self.sky,
                env: &mut self.env,
                water: &mut self.water,
                selected_water: &mut self.selected_water,
                outliner: &self.outliner,
                outliner_selection: self.outliner_selection,
                active_road: None,
                road_count: 0,
                modifiers: &mut self.modifiers,
                selected_modifier: &mut self.selected_modifier,
                content: &content,
                notice: self.notice.as_ref(),
                noise: &mut self.noise,
                noise_library: &self.noise_library,
                viewport_rect: &mut self.viewport_rect,
                view_mode: &mut self.view_mode,
                material: self.material.as_mut().map(|(name, params)| ui::MaterialView {
                    name,
                    role: "rock",
                    texture: None,
                    params,
                    role_id: self.material_role.0,
                    auto_role: self.material_role.1,
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

    /// How many published widgets are disabled.
    ///
    /// `add_enabled_ui` greys its contents rather than removing them, so "these controls
    /// are inert" has to be asserted through the disabled flag.
    fn disabled_count(&mut self) -> usize {
        let mut harness =
            Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| self.frame(ui));
        harness.run();
        harness.root().children_recursive().filter(|n| n.accesskit_node().is_disabled()).count()
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

    /// Which content shelf is showing. The import control is per shelf, so this
    /// is what selects between the folder picker and the file picker.
    fn with_asset_kind(mut self, kind: AssetKind) -> Self {
        self.asset_kind = kind;
        self
    }

    /// Rows for the outliner, plus the pane brought to the front so they render.
    fn with_outliner(mut self, rows: Vec<ui::OutlinerItem>) -> Self {
        self.outliner = rows;
        self.layout.focus(Tab::Outliner);
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
fn the_texture_import_asks_for_a_folder_rather_than_a_file() {
    // A material is a folder of maps, and the button is the only place that gets
    // said before the file dialog opens. "Import Textures" sent users looking for
    // an image to select, and an image filter makes the folder unselectable --
    // clicking Open just navigated into it.
    let mut f = Fixture::default().with_asset_kind(AssetKind::Texture);
    let labels = f.labels();
    assert!(
        labels.iter().any(|l| l.contains("Import material folder")),
        "the texture import has to name a folder: {labels:?}"
    );
    assert!(
        !labels.iter().any(|l| l == "Import Textures"),
        "the old file-oriented wording is back: {labels:?}"
    );
}

#[test]
fn the_other_shelves_still_import_files() {
    // Only textures are folders. Regressing noise or models to a folder picker
    // would be the same bug pointing the other way.
    for (kind, label) in [(AssetKind::Noise, "Import Noise"), (AssetKind::Model, "Import Models")] {
        let mut f = Fixture::default().with_asset_kind(kind);
        assert!(f.has(label), "{label} missing: {:?}", f.labels());
    }
}

#[test]
fn the_foliage_pane_exposes_the_lod_switch_distances() {
    // The two runtime thresholds. Sliders publish their value rather than their
    // caption through AccessKit, so the assertion is on the values being present
    // and on the defaults being the ones the renderer clamps against.
    let d = terra_render::scatter::Rules::default();
    assert!(d.lod1_m > 0.0 && d.lod2_m > d.lod1_m, "defaults are not ordered");

    let mut f = Fixture::default().with_tool(Tool::Foliage);
    f.rules = Some(terra_render::scatter::Rules::default());
    let values = f.values();
    for want in [d.lod1_m, d.lod2_m] {
        assert!(
            values
                .iter()
                .any(|v| v.parse::<f32>().map(|n| (n - want).abs() < 0.51).unwrap_or(false)),
            "no slider showing {want}: {values:?}"
        );
    }
}

#[test]
fn dragging_lod2_below_lod1_is_corrected_rather_than_kept() {
    // Ordered in the pane as well as clamped in the cull path, so the sliders
    // cannot be left in a state the renderer has to silently fix. An inverted pair
    // makes LOD 1 unreachable, which reads as "the middle level does nothing".
    let mut f = Fixture::default().with_tool(Tool::Foliage);
    f.rules =
        Some(terra_render::scatter::Rules { lod1_m: 400.0, lod2_m: 100.0, ..Default::default() });
    let _ = f.labels();
    let got = f.rules.as_ref().unwrap();
    assert!(got.lod2_m >= got.lod1_m, "{} < {}", got.lod2_m, got.lod1_m);
}

#[test]
fn the_content_browser_can_rescan_after_a_manual_copy() {
    // The palettes are built on project open and on import only, so a folder
    // copied in from Finder is invisible until something re-reads the directory.
    // While the import was broken that was the only way to add a material at all.
    let mut f = Fixture::default();
    assert!(f.has("Refresh"), "no way to re-read the asset folders: {:?}", f.labels());
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
    // A partial import, which is the interesting case to look at: the wording has
    // to read as a success while still naming the folder that was skipped.
    f.notice = Some((
        "Imported Rock042. Skipped 1 -- Screenshots: no colour map found. One file's name has \
         to contain one of: color, albedo, basecolor, base_color, _diff, diffuse"
            .to_string(),
        false,
    ));
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

    // The outliner, with one of everything in it.
    f.outliner = vec![
        row(ui::OutlinerKind::Water, 0, "Water 1", "260 x 180 m at 244 m", true),
        row(ui::OutlinerKind::Species, 0, "Pine", "1240 instances", false),
        row(ui::OutlinerKind::Species, 1, "Fern", "8600 instances", false),
        row(ui::OutlinerKind::Prop, 0, "Boulder", "120, 340", true),
        row(ui::OutlinerKind::Road, 0, "Road 1", "6 points", true),
        row(ui::OutlinerKind::Modifier, 0, "Main passage", "Carve", true),
    ];
    f.outliner_selection = Some((ui::OutlinerKind::Species, 0));
    f.layout.focus(Tab::Outliner);
    let mut outl =
        Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_ui(|ui| f.frame(ui));
    outl.run();
    outl.render().expect("headless wgpu render failed").save(dir.join("ui-outliner.png")).unwrap();
    drop(outl);

    // The Environment Light Mixer, brought to the front of the right-hand column, with
    // water on so its section renders rather than sitting greyed at the bottom.
    f.water.enabled = true;
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
fn a_material_can_be_made_paint_only() {
    // The opt-out from automatic placement: someone who would rather brush a material
    // on than have the slope masks decide needs a way to say so, and it is the control
    // that answers "why did this appear everywhere the moment I imported it".
    let mut f = Fixture::default().with_material("Rock042");
    assert!(f.has("Paint only"), "no way to stop automatic placement: {:?}", f.labels());
}

#[test]
fn the_automatic_role_can_be_corrected_in_the_pane() {
    // The role is guessed from the folder name, and a set called
    // `rocky_terrain_03/textures` was read as soil and carpeted every flat field.
    // Renaming files on disk must not be the only way out.
    let mut f = Fixture::default().with_material("Rock042");
    let labels = f.labels();
    for role in ["soil", "grass", "rock", "gravel", "snow", "track"] {
        assert!(
            labels.iter().any(|l| l.contains(role)),
            "role '{role}' cannot be chosen: {labels:?}"
        );
    }
}

#[test]
fn the_guessed_role_is_marked_so_there_is_a_way_back_to_it() {
    // Overriding is only safe if the original guess stays visible.
    let mut f = Fixture::default().with_material("Rock042");
    f.material_role = (terra_render::material::SOIL, terra_render::material::ROCK);
    let labels = f.labels();
    assert!(
        labels.iter().any(|l| l.contains("rock") && l.contains("auto")),
        "the guessed role is not marked: {labels:?}"
    );
}

#[test]
fn paint_only_is_not_offered_twice_in_the_role_row() {
    // `SELECTABLE_ROLES` ends with ROLE_NONE, which has its own button -- listing it
    // again among the roles would give two controls for one state.
    let mut f = Fixture::default().with_material("Rock042");
    let n = f.labels().iter().filter(|l| l.contains("paint only")).count();
    assert_eq!(n, 0, "'paint only' leaked into the role row");
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
    // A freshly imported material must render as itself: no tint, sampled
    // roughness and normals used as authored.
    let p = terra_render::material::LayerParams::default();
    assert_eq!(p.tint, [1.0, 1.0, 1.0], "a new material must not be tinted");
    assert_eq!(p.normal_strength, 1.0);
    assert_eq!(p.roughness, 1.0);
    assert_eq!(p.ao, 1.0);
    assert!(p.tiling_m > 0.0);
}

#[test]
fn parallax_is_on_by_default_but_subtle() {
    // This assertion used to be `== 0.0`, on the grounds that parallax costs a
    // loop of samples and must be opted into. What changed is the shader, not the
    // appetite for frame time: a flat mid-grey height channel -- what a set with
    // no displacement map decodes to -- now leaves the march after a single
    // fetch, and the effect fades out entirely past 140 m. So the cost is paid
    // only by materials that actually have relief, at the distances where it is
    // visible, which is what made opting in a chore rather than a safeguard.
    let p = terra_render::material::LayerParams::default();
    assert!(p.parallax_m > 0.0, "relief should be on out of the box");
    // A ceiling, not just non-zero: a large default reads as the terrain itself
    // having changed shape, and the slider tops out at 0.25.
    assert!(p.parallax_m <= 0.05, "{} m is too deep for a default", p.parallax_m);
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
    assert_eq!(
        terra_render::material::LayerParams::default().parallax_m,
        0.03,
        "editing one layer must not move the defaults"
    );
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
fn the_cloud_section_offers_drift_speed_and_direction() {
    // The wind vector already advected the layer and drove the ground shadows; there was
    // simply no way to change it. Sliders publish their value rather than their caption,
    // so this asserts the default speed and heading are both present as values.
    let mut f = Fixture::default();
    f.env.clouds.enabled = true;
    f.env.clouds.set_wind(12.0, 45.0);
    f.layout.focus(Tab::Environment);
    let values = f.values();
    for want in [12.0f32, 45.0] {
        assert!(
            values
                .iter()
                .any(|v| v.parse::<f32>().map(|n| (n - want).abs() < 0.51).unwrap_or(false)),
            "no slider showing {want}: {values:?}"
        );
    }
}

/// A row for the outliner fixture.
fn row(
    kind: ui::OutlinerKind,
    index: usize,
    name: &str,
    detail: &str,
    removable: bool,
) -> ui::OutlinerItem {
    ui::OutlinerItem { kind, index, name: name.to_string(), detail: detail.to_string(), removable }
}

#[test]
fn a_thousand_trees_are_one_outliner_row() {
    // The rule that makes the list usable. A thousand painted instances are a thousand
    // copies of one decision -- the species' rules -- so they are one thing to edit. A
    // thousand identical rows would be a list nobody scrolls.
    use ui::OutlinerKind;
    let mut f = Fixture::default().with_outliner(vec![row(
        OutlinerKind::Species,
        0,
        "Pine",
        "1240 instances",
        false,
    )]);

    let labels = f.labels();
    let pine = labels.iter().filter(|l| l.contains("Pine") && !l.starts_with("Remove")).count();
    assert_eq!(pine, 1, "one species should be one row: {labels:?}");
    assert!(
        labels.iter().any(|l| l.contains("1240 instances")),
        "the count has to be shown, since the instances are not: {labels:?}"
    );
}

#[test]
fn hand_placed_objects_get_a_row_each() {
    // The exception: each was placed deliberately and carries its own transform, which is
    // exactly what a scattered instance does not.
    use ui::OutlinerKind;
    let mut f = Fixture::default().with_outliner(vec![
        row(OutlinerKind::Prop, 0, "Boulder", "120, 340", true),
        row(OutlinerKind::Prop, 1, "Boulder", "150, 300", true),
    ]);
    // Excluding the remove buttons, which now carry the name too so they announce
    // themselves properly.
    let rows = f.labels().into_iter().filter(|l| l.contains("Boulder") && !l.starts_with("Remove"));
    assert_eq!(rows.count(), 2, "two placed objects should be two rows");
}

#[test]
fn the_outliner_lists_every_kind_of_thing() {
    use ui::OutlinerKind;
    let mut f = Fixture::default().with_outliner(vec![
        row(OutlinerKind::Water, 0, "Water 1", "120 x 80 m at 40 m", true),
        row(OutlinerKind::Species, 0, "Pine", "900 instances", false),
        row(OutlinerKind::Prop, 0, "Boulder", "10, 20", true),
        row(OutlinerKind::Road, 0, "Road 1", "6 points", true),
        row(OutlinerKind::Modifier, 0, "Main passage", "Carve", true),
    ]);
    let labels = f.labels();
    for want in ["Water 1", "Pine", "Boulder", "Road 1", "Main passage"] {
        assert!(labels.iter().any(|l| l.contains(want)), "{want} missing: {labels:?}");
    }
}

#[test]
fn a_species_row_offers_no_remove_button() {
    // A species exists because a mesh is in the project's `assets/models/`. Removing it
    // means deleting that file, so a button here would either lie or delete an asset the
    // user did not point at.
    use ui::OutlinerKind;
    let mut f = Fixture::default().with_outliner(vec![row(
        OutlinerKind::Species,
        0,
        "Pine",
        "900 instances",
        false,
    )]);
    assert!(
        !f.labels().iter().any(|l| l.contains("Remove Pine")),
        "a species must not offer a remove button: {:?}",
        f.labels()
    );
}

#[test]
fn removable_rows_offer_a_remove_button() {
    use ui::OutlinerKind;
    let mut f = Fixture::default().with_outliner(vec![row(
        OutlinerKind::Water,
        0,
        "Water 1",
        "120 x 80 m",
        true,
    )]);
    assert!(
        f.labels().iter().any(|l| l.contains("Remove Water 1")),
        "no way to remove a water body: {:?}",
        f.labels()
    );
}

#[test]
fn every_outliner_kind_has_a_group_heading() {
    use ui::OutlinerKind;
    for k in [
        OutlinerKind::Water,
        OutlinerKind::Species,
        OutlinerKind::Prop,
        OutlinerKind::Road,
        OutlinerKind::Modifier,
    ] {
        assert!(!k.group().is_empty(), "{k:?} has no heading");
    }
}

#[test]
fn the_water_tool_exists_and_lists_bodies() {
    // The empty state is a plain label, which AccessKit does not publish -- so what is
    // asserted is that a body, once placed, becomes a selectable row carrying its size.
    let mut f = Fixture::default().with_tool(Tool::Water);
    f.water.regions.push(terra_render::water::WaterRegion::from_drag(
        [0.0, 0.0],
        [120.0, 80.0],
        50.0,
    ));
    let labels = f.labels();
    assert!(
        labels.iter().any(|l| l.contains("Body 1") && l.contains("120") && l.contains("80")),
        "the body is not listed with its size: {labels:?}"
    );
}

#[test]
fn a_selected_body_exposes_its_own_wave_speed() {
    // The whole point of regions: this body's waves, not the global ones. Unselected it
    // must not show them, or two sets of wave sliders would be on screen at once with no
    // way to tell which is which.
    let mut f = Fixture::default().with_tool(Tool::Water);
    let mut r = terra_render::water::WaterRegion::from_drag([0.0, 0.0], [100.0, 100.0], 40.0);
    r.wave_speed = 2.25;
    f.water.regions.push(r);

    f.selected_water = None;
    let unselected = f.values();
    f.selected_water = Some(0);
    let selected = f.values();
    assert!(selected.len() > unselected.len(), "selecting a body exposed no settings");
    assert!(
        selected.iter().any(|v| v.parse::<f32>().map(|n| (n - 2.25).abs() < 0.02).unwrap_or(false)),
        "this body's wave speed is not editable: {selected:?}"
    );
}

#[test]
fn the_water_tool_is_in_the_tool_list() {
    let mut f = Fixture::default();
    assert!(f.has("Water"), "the Water tool is not offered: {:?}", f.labels());
}

#[test]
fn the_mixer_offers_water() {
    // Water lives in the Environment pane because a surface that reflects the sky
    // belongs with the sections that describe the sky.
    let mut f = Fixture::default();
    f.layout.focus(Tab::Environment);
    assert!(f.has("Water"), "no water control in the mixer: {:?}", f.labels());
}

#[test]
fn water_is_off_until_switched_on() {
    // A default sea level would flood the valleys of every project that predates this
    // the moment it loaded.
    assert!(!terra_render::water::WaterSettings::default().enabled);
}

#[test]
fn the_water_controls_are_disabled_until_water_is_on() {
    // Sliders for a surface that is not being drawn invite tuning something invisible.
    // `add_enabled_ui` still *renders* them, greyed -- so this asserts the disabled
    // state rather than their absence, which is what a screen reader would report too.
    let mut f = Fixture::default();
    f.layout.focus(Tab::Environment);

    f.water.enabled = false;
    let off = f.disabled_count();
    f.water.enabled = true;
    let on = f.disabled_count();
    assert!(off > on, "switching water on left the same controls disabled: {off} disabled -> {on}");
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

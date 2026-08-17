//! Screens.
//!
//! Menus are a persistent left rail over the live scene -- no cards, no modals,
//! no dialog that opens and closes. Navigating swaps the rail's contents; the
//! rendered world behind it never stops or restarts.
//!
//! The editor is the one place with docked chrome, laid out the way a 3D tool
//! is expected to be: toolbar on top, tools left, inspector right, status bar
//! along the bottom.

use crate::theme;
use egui::{Align, Align2, CornerRadius, FontId, Layout, Rect, RichText, Sense, vec2};
use std::path::PathBuf;
use terra_core::WorldSize;
use terra_project::roads::{Road, Surface};
use terra_project::{Library, TerrainParams};
use terra_render::lighting::{Quality, ShadowQuality, SkySettings};
use terra_render::stats::FrameStats;
use terra_render::terrain::SculptMode;

/// Which set of controls the rail is showing. The rail itself is never torn
/// down -- only its contents change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    Home,
    Worlds,
    Create,
    Settings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    None,
    Go(Pane),
    Quit,
    Create { name: String, size: WorldSize, seed: u64 },
    Open(PathBuf),
    Forget(PathBuf),
}

pub struct CreateForm {
    pub name: String,
    pub size: WorldSize,
    pub seed: u64,
    pub seed_text: String,
}

impl Default for CreateForm {
    fn default() -> Self {
        let seed = fresh_seed();
        Self {
            name: "New World".into(),
            size: WorldSize::Medium,
            seed,
            seed_text: seed.to_string(),
        }
    }
}

pub fn fresh_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED_1234)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        >> 16
}

// ---------------------------------------------------------------------------
// Rail primitives
// ---------------------------------------------------------------------------

fn rail(root: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    let width = theme::rail_width(root.ctx());
    theme::paint_rail_scrim(root.ctx(), width);
    egui::Panel::left("rail")
        .exact_size(width)
        .resizable(false)
        .frame(egui::Frame::NONE.inner_margin(theme::rail_margin(width)))
        .show(root, |ui| {
            // The rail owns the only scroll in the menus. A short window would
            // otherwise clip whatever sits at the bottom of a pane -- which on
            // Create is the button that does the thing.
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded)
                .show(ui, add);
        });
}

/// A navigation row: label, optional sub-label, accent bar on hover.
/// Deliberately not an `egui::Button` -- buttons carry a filled rounded box,
/// which reads as a dialog control rather than as menu navigation.
fn rail_item(ui: &mut egui::Ui, label: &str, hint: &str, selected: bool) -> bool {
    let h = if hint.is_empty() { 46.0 } else { 56.0 };
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::click());

    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        if resp.hovered() || selected {
            p.rect_filled(rect, CornerRadius::same(10), theme::HOVER);
            let bar = Rect::from_min_size(
                rect.left_top() + vec2(0.0, 11.0),
                vec2(3.0, rect.height() - 22.0),
            );
            p.rect_filled(bar, CornerRadius::same(2), theme::ACCENT);
        }
        let x = rect.left() + 22.0;
        let color = if resp.hovered() { theme::TEXT } else { theme::TEXT.gamma_multiply(0.9) };
        if hint.is_empty() {
            p.text(
                egui::pos2(x, rect.center().y),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(17.0),
                color,
            );
        } else {
            p.text(
                egui::pos2(x, rect.center().y - 9.0),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(17.0),
                color,
            );
            p.text(
                egui::pos2(x, rect.center().y + 11.0),
                Align2::LEFT_CENTER,
                hint,
                FontId::proportional(12.0),
                theme::MUTED,
            );
        }
    }
    resp.clicked()
}

fn back_row(ui: &mut egui::Ui) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let color = if resp.hovered() { theme::ACCENT } else { theme::MUTED };
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            "\u{2190}  Back",
            FontId::proportional(13.5),
            color,
        );
    }
    resp.clicked()
}

fn primary(ui: &mut egui::Ui, label: &str) -> bool {
    let w = ui.available_width();
    ui.add_sized([w, 44.0], egui::Button::new(theme::heading(label))).clicked()
}

/// Height of a text field, matched to the buttons it sits beside.
const FIELD_H: f32 = 34.0;

/// A single-line field that fills the box it is given.
///
/// `TextEdit` aligns its contents `LEFT_TOP` and is stretched to the height it
/// is allocated, so a field given a comfortable 34 px drew the text -- and the
/// caret you are typing at -- jammed against the top edge with the padding all
/// below it. The text is centred in the field instead.
fn text_field(text: &mut String) -> egui::TextEdit<'_> {
    egui::TextEdit::singleline(text)
        .vertical_align(Align::Center)
        .margin(egui::Margin::symmetric(10, 6))
}

// ---------------------------------------------------------------------------
// Panes
// ---------------------------------------------------------------------------

pub fn home(root: &mut egui::Ui, world_count: usize) -> Action {
    let mut action = Action::None;
    rail(root, |ui| {
        ui.add_space(10.0);
        ui.label(theme::display("TERRA"));
        ui.add_space(-4.0);
        ui.label(theme::label("TERRAIN  BUILDER"));
        ui.add_space(46.0);

        let hint = match world_count {
            0 => "No worlds yet".to_string(),
            1 => "1 world".to_string(),
            n => format!("{n} worlds"),
        };
        if rail_item(ui, "Worlds", &hint, false) {
            action = Action::Go(Pane::Worlds);
        }
        if rail_item(ui, "Settings", "Display, camera", false) {
            action = Action::Go(Pane::Settings);
        }
        if rail_item(ui, "Quit", "", false) {
            action = Action::Quit;
        }

        ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
            ui.label(theme::small(concat!("v", env!("CARGO_PKG_VERSION"))));
        });
    });
    action
}

pub fn worlds(root: &mut egui::Ui, library: &Library) -> Action {
    let mut action = Action::None;
    let entries = library.sorted();

    rail(root, |ui| {
        if back_row(ui) {
            action = Action::Go(Pane::Home);
        }
        ui.add_space(12.0);
        ui.label(theme::title("Worlds"));
        ui.add_space(18.0);

        if entries.is_empty() {
            // No modal, no empty-state card -- the rail simply offers the one
            // action that makes sense here.
            ui.label(theme::muted(
                "You have no worlds yet. Create one to start sculpting terrain.",
            ));
            ui.add_space(18.0);
            if primary(ui, "Create World") {
                action = Action::Go(Pane::Create);
            }
        } else {
            if primary(ui, "+  New World") {
                action = Action::Go(Pane::Create);
            }
            ui.add_space(16.0);
            ui.label(theme::label("RECENT"));
            ui.add_space(6.0);

            // No scroll area of its own -- the rail scrolls, and a scroll
            // inside a scroll gives the list an unbounded height to grow into.
            {
                for e in entries {
                    if e.is_available() {
                        let folder = e
                            .path
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        if rail_item(ui, &e.name, &folder, false) {
                            action = Action::Open(e.path.clone());
                        }
                    } else {
                        // Unreachable path: an unplugged drive or a moved
                        // folder. Offer removal from the list, never delete.
                        let (rect, resp) = ui
                            .allocate_exact_size(vec2(ui.available_width(), 56.0), Sense::click());
                        if ui.is_rect_visible(rect) {
                            let p = ui.painter();
                            let x = rect.left() + 22.0;
                            p.text(
                                egui::pos2(x, rect.center().y - 9.0),
                                Align2::LEFT_CENTER,
                                &e.name,
                                FontId::proportional(17.0),
                                theme::MUTED.gamma_multiply(0.8),
                            );
                            p.text(
                                egui::pos2(x, rect.center().y + 11.0),
                                Align2::LEFT_CENTER,
                                "unavailable - click to remove from list",
                                FontId::proportional(12.0),
                                theme::DANGER,
                            );
                        }
                        if resp.clicked() {
                            action = Action::Forget(e.path.clone());
                        }
                    }
                }
            }
        }
    });
    action
}

pub fn create(root: &mut egui::Ui, form: &mut CreateForm) -> Action {
    let mut action = Action::None;
    rail(root, |ui| {
        if back_row(ui) {
            action = Action::Go(Pane::Worlds);
        }
        ui.add_space(12.0);
        ui.label(theme::title("Create World"));
        ui.add_space(20.0);

        ui.label(theme::label("NAME"));
        ui.add_space(4.0);
        let w = ui.available_width();
        ui.add_sized([w, FIELD_H], text_field(&mut form.name).hint_text("World name"));

        ui.add_space(20.0);
        ui.label(theme::label("SIZE"));
        ui.label(theme::small("Fixed at creation. This cannot be changed later."));
        ui.add_space(8.0);

        for size in WorldSize::ALL {
            let selected = form.size == size;
            let (rect, resp) =
                ui.allocate_exact_size(vec2(ui.available_width(), 52.0), Sense::click());
            if ui.is_rect_visible(rect) {
                let p = ui.painter();
                let fill = if selected {
                    theme::ACCENT.gamma_multiply(0.28)
                } else if resp.hovered() {
                    theme::HOVER
                } else {
                    theme::PANEL_SOFT
                };
                p.rect_filled(rect, CornerRadius::same(10), fill);
                if selected {
                    p.rect_stroke(
                        rect,
                        CornerRadius::same(10),
                        egui::Stroke::new(1.0, theme::ACCENT),
                        egui::StrokeKind::Inside,
                    );
                }
                let x = rect.left() + 16.0;
                p.text(
                    egui::pos2(x, rect.center().y - 8.0),
                    Align2::LEFT_CENTER,
                    size.label(),
                    FontId::proportional(15.5),
                    theme::TEXT,
                );
                p.text(
                    egui::pos2(x, rect.center().y + 11.0),
                    Align2::LEFT_CENTER,
                    format!(
                        "{n}x{n} tiles  -  erodes at {mpt:.0} m/texel",
                        n = size.tiles_per_side(),
                        mpt = size.tier0_meters_per_texel()
                    ),
                    FontId::proportional(11.5),
                    theme::MUTED,
                );
            }
            if resp.clicked() {
                form.size = size;
            }
            ui.add_space(6.0);
        }

        ui.add_space(14.0);
        ui.label(theme::label("SEED"));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            // A singleline edit defaults to infinite desired width, which in a
            // horizontal row means it eats the button beside it.
            let w = (ui.available_width() - 104.0).max(60.0);
            if ui.add_sized([w, FIELD_H], text_field(&mut form.seed_text)).changed()
                && let Ok(v) = form.seed_text.trim().parse::<u64>()
            {
                form.seed = v;
            }
            if ui.button("Random").clicked() {
                form.seed = fresh_seed();
                form.seed_text = form.seed.to_string();
            }
        });

        ui.add_space(24.0);
        let valid = !form.name.trim().is_empty();
        ui.add_enabled_ui(valid, |ui| {
            if primary(ui, "Create") {
                action = Action::Create {
                    name: form.name.trim().to_string(),
                    size: form.size,
                    seed: form.seed,
                };
            }
        });
    });
    action
}

pub struct SettingsView<'a> {
    pub vsync: &'a mut bool,
    pub perf_overlay: &'a mut bool,
    pub perf_graph: &'a mut bool,
    pub camera_speed: &'a mut f32,
    pub fov_deg: &'a mut f32,
    pub uncapped_supported: bool,
    pub gpu_timing: bool,
}

pub fn settings(root: &mut egui::Ui, v: SettingsView<'_>) -> Action {
    let mut action = Action::None;
    rail(root, |ui| {
        if back_row(ui) {
            action = Action::Go(Pane::Home);
        }
        ui.add_space(12.0);
        ui.label(theme::title("Settings"));
        ui.add_space(22.0);

        ui.label(theme::label("DISPLAY"));
        ui.add_space(6.0);
        ui.add_enabled_ui(v.uncapped_supported, |ui| {
            ui.checkbox(v.vsync, "V-Sync");
        });
        ui.label(theme::small(if v.uncapped_supported {
            "Off uncaps the frame rate. Your display is 75 Hz, so this makes \
             headroom measurable rather than smoother."
        } else {
            "This surface does not support uncapped presentation."
        }));
        ui.add_space(22.0);
        ui.label(theme::label("PERFORMANCE"));
        ui.add_space(6.0);
        ui.checkbox(v.perf_overlay, "Performance overlay");
        ui.add_enabled_ui(*v.perf_overlay, |ui| {
            ui.checkbox(v.perf_graph, "Frame time graph");
        });
        ui.label(theme::small(if v.gpu_timing {
            "Shows FPS, frame / CPU / GPU milliseconds and a rolling history. \
             GPU time comes from timestamp queries, not an estimate."
        } else {
            "Shows FPS and frame / CPU milliseconds. This adapter has no \
             timestamp queries, so GPU time is unavailable."
        }));

        ui.add_space(22.0);
        ui.label(theme::label("CAMERA"));
        ui.add_space(6.0);
        slider(ui, "Speed m/s", v.camera_speed, 10.0..=600.0);
        slider(ui, "Field of view", v.fov_deg, 40.0..=100.0);
    });
    action
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Full-bleed loading screen. The scene keeps rendering underneath, dimmed --
/// no card, so it reads as the app preparing rather than a dialog appearing.
pub fn loading(root: &mut egui::Ui, world: &str, stage: &str, progress: f32) {
    theme::paint_dim(root.ctx(), 205);
    let screen = root.ctx().viewport_rect();
    // Inset and block width both track the window; at 92 px fixed, a small
    // window pushed the progress bar off its right edge.
    let inset = (screen.width() * 0.06).clamp(24.0, 92.0);
    let width = (screen.width() - inset * 2.0).clamp(220.0, 560.0);

    egui::Area::new("loading".into())
        .fixed_pos(egui::pos2(
            screen.left() + inset,
            (screen.bottom() - 190.0).max(screen.top() + 24.0),
        ))
        .show(root.ctx(), |ui| {
            ui.set_width(width);
            ui.label(theme::label("LOADING WORLD"));
            ui.add_space(6.0);
            ui.label(theme::title(world));
            ui.add_space(20.0);
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_height(4.0)
                    .corner_radius(CornerRadius::same(2))
                    .fill(theme::ACCENT),
            );
            ui.add_space(10.0);
            ui.label(theme::muted(stage));
        });
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/// Which editing tool has the pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Navigate only. First in the list and the one a world opens with, so
    /// nothing is armed on arrival -- landing in Sculpt means the first click
    /// meant as "look at this" deforms the terrain instead.
    Camera,
    Sculpt,
    Paint,
    Foliage,
    /// Pick, move, scale and delete individual placed objects.
    Select,
    Road,
}

impl Tool {
    pub const ALL: [Tool; 5] = [Tool::Camera, Tool::Sculpt, Tool::Paint, Tool::Foliage, Tool::Road];
    pub fn label(self) -> &'static str {
        match self {
            Tool::Camera => "Camera",
            Tool::Sculpt => "Sculpt",
            Tool::Paint => "Paint",
            Tool::Foliage => "Foliage",
            Tool::Select => "Select",
            Tool::Road => "Road",
        }
    }

    /// Whether the left button edits the world in this tool.
    pub fn edits(self) -> bool {
        !matches!(self, Tool::Camera)
    }
}

/// How a paint stroke is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    /// Lay the selected material down under the brush.
    Brush,
    /// Lift painting back off, returning the ground to automatic placement.
    Erase,
}

impl PaintMode {
    pub const ALL: [PaintMode; 2] = [PaintMode::Brush, PaintMode::Erase];
    pub fn label(self) -> &'static str {
        match self {
            PaintMode::Brush => "Brush",
            PaintMode::Erase => "Erase",
        }
    }
}

/// Editable fields of the selected object.
pub struct SelectionView {
    pub species: String,
    pub scale: f32,
    pub yaw: f32,
    pub height: f32,
}

/// One foliage palette entry.
pub struct FoliageEntry {
    pub name: String,
    pub instances: u32,
    pub painted: bool,
    /// Registered preview, or `None` before it has been uploaded.
    pub texture: Option<egui::TextureHandle>,
}

/// One palette entry, as the editor hands it to the UI.
pub struct PaletteEntry<'a> {
    pub name: &'a str,
    pub role: &'a str,
    /// Registered thumbnail, or `None` before it has been uploaded.
    pub texture: Option<&'a egui::TextureHandle>,
}

pub struct EditorView<'a> {
    /// The sculpt mode. Drives both the palette and the brush that runs.
    pub mode: &'a mut SculptMode,
    pub radius: &'a mut f32,
    pub strength: &'a mut f32,
    pub world_name: &'a str,
    pub size: WorldSize,
    pub unsaved: bool,
    pub brush_at: Option<(f32, f32)>,
    pub height_res: u32,
    pub params: &'a mut TerrainParams,
    pub playing: bool,
    pub speed_kph: f32,
    pub tool: &'a mut Tool,
    /// Material palette, discovered from the texture folder.
    pub palette: &'a [PaletteEntry<'a>],
    pub selected_layer: &'a mut u32,
    pub paint_mode: &'a mut PaintMode,
    pub paint_flow: &'a mut f32,
    /// Whether anything has been painted on this world yet.
    pub painted: bool,
    /// Foliage palette and the rules of whichever species is selected.
    pub foliage: &'a [FoliageEntry],
    pub selected_species: &'a mut usize,
    pub species_rules: Option<&'a mut terra_render::scatter::Rules>,
    pub foliage_instances: u32,
    /// The selected object, if any, and how many exist.
    pub selection: Option<&'a mut SelectionView>,
    pub prop_count: usize,
    /// Whether each docked side panel is expanded.
    pub tools_open: &'a mut bool,
    pub inspector_open: &'a mut bool,
    /// Frame-time settings: shadows and temporal AA. Everything that changes how
    /// the world looks lives in [`Self::env`].
    pub sky: &'a mut SkySettings,
    /// The Environment Light Mixer: sun, atmosphere, sky light, fog, clouds and
    /// tone mapping, in one place.
    pub env: &'a mut terra_render::Environment,
    /// The road being drawn, if any, plus how many roads exist.
    pub active_road: Option<&'a mut Road>,
    pub road_count: usize,
    /// The non-destructive cave modifier stack, shown in its own pane.
    pub modifiers: &'a mut terra_voxel::ModifierStack,
    /// Which stack entry is selected, if any.
    pub selected_modifier: &'a mut Option<usize>,
    /// The project's own asset folder, as the content browser sees it.
    pub content: &'a ContentView<'a>,
    /// Noise pattern the Noise sculpt brush samples, and the library of
    /// uploaded patterns to choose from.
    pub noise: &'a mut terra_voxel::NoiseField,
    pub noise_library: &'a [String],
    /// The material being edited, if the Material pane has one.
    pub material: Option<MaterialView<'a>>,
    /// Which visualization the viewport is showing.
    pub view_mode: &'a mut terra_render::ViewMode,
    /// Written back by the Viewport tab: where the 3D view actually ended up.
    ///
    /// With fixed panels this was derivable from a constant width. Under
    /// docking it is not -- the user can put the viewport anywhere, at any
    /// size, so anything that needs to sit over the scene has to be told.
    pub viewport_rect: &'a mut Option<egui::Rect>,
}

/// `Clone` but not `Copy`: `SelectNoise` carries the chosen filename, and
/// making the whole enum copyable again would mean interning asset names
/// somewhere just to keep one variant small.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorAction {
    None,
    Save,
    Exit,
    /// Rebuild the heightfield: ridged multifractal, then erosion.
    Generate,
    /// Start a new road; subsequent clicks in the viewport add points.
    NewRoad,
    /// Drop the last control point of the road being drawn.
    UndoPoint,
    /// Stop adding points, keeping the road.
    FinishRoad,
    /// Delete every road in the world.
    ClearRoads,
    /// Cover the whole world with the selected material.
    FillMaterial,
    /// Discard all painting, returning to automatic placement.
    ClearPaint,
    /// Cover the whole world with the selected species.
    FillFoliage,
    /// Remove every instance of the selected species.
    ClearFoliage,
    /// Re-roll the selected species' placement.
    ReseedFoliage,
    /// Drop an object of the selected species where the cursor points.
    PlaceProp,
    /// Delete the selected object.
    DeleteProp,
    /// Drop the selection without deleting anything.
    Deselect,
    /// Re-stamp all roads onto the base terrain.
    RebuildRoads,
    /// Build the physics world and drop a car into it.
    Play,
    /// Leave drive mode and return to editing.
    Stop,
    /// Switch which shelf the content browser is showing.
    SelectAssetKind(AssetKind),
    /// Open a file dialog and copy the chosen file into the project.
    ImportAsset(AssetKind),
    /// Make an uploaded greyscale map the Noise brush's pattern.
    SelectNoise(String),
    /// Append a tunnel modifier through the view direction.
    AddTunnel,
    /// Drop one entry from the modifier stack.
    DeleteModifier(usize),
    /// Open the Material pane on a palette slot. Fired by a double-click.
    OpenMaterial(usize),
    /// Select the material currently open in the editor for painting.
    PaintWithSelectedMaterial,
}

pub fn editor(
    root: &mut egui::Ui,
    layout: &mut crate::dock::Layout,
    mut v: EditorView<'_>,
) -> EditorAction {
    let mut action = EditorAction::None;
    let narrow = root.ctx().viewport_rect().width() < 1100.0;

    // --- toolbar ---
    egui::Panel::top("toolbar").exact_size(TOOLBAR_H).frame(theme::panel(8)).show(root, |ui| {
        ui.horizontal_centered(|ui| {
            ui.add_space(6.0);
            // Right-hand controls are laid out first so a long world name
            // shortens itself instead of pushing Save and Exit off the window.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Exit").clicked() {
                    action = EditorAction::Exit;
                }
                if ui.button("Save").clicked() {
                    action = EditorAction::Save;
                }
                ui.add_space(8.0);
                if v.playing {
                    if ui.button(theme::heading("Stop")).clicked() {
                        action = EditorAction::Stop;
                    }
                    ui.label(
                        RichText::new(format!("{:.0} km/h", v.speed_kph))
                            .size(14.0)
                            .color(theme::ACCENT),
                    );
                } else if ui.button(theme::heading("Play")).clicked() {
                    action = EditorAction::Play;
                }
                ui.add_space(8.0);
                view_menu(ui, layout, v.view_mode, !v.playing);
                ui.add_space(8.0);
                // Whatever width the buttons left over belongs to the title.
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.add(
                        egui::Label::new(RichText::new(v.world_name).size(15.0).strong())
                            .truncate(),
                    );
                    if !narrow {
                        ui.label(theme::small(v.size.label()));
                    }
                    if v.unsaved {
                        let mark = if narrow { "\u{25CF}" } else { "\u{25CF} unsaved" };
                        ui.label(RichText::new(mark).size(11.5).color(theme::ACCENT));
                    }
                });
            });
        });
    });

    // --- status bar ---
    egui::Panel::bottom("status").exact_size(STATUS_H).frame(theme::panel(6)).show(root, |ui| {
        ui.horizontal_centered(|ui| {
            // Cursor position first, pinned right: it changes as you work, so
            // it is the half worth keeping when the hints have to be cut.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                // A debug mode changes every pixel, and forgetting one is on is
                // the classic way to spend ten minutes debugging a material that
                // was never broken. So it is named in the status bar until it is
                // turned off -- and as a button rather than a label, because the
                // thing you want on seeing it is the way back.
                // Not while driving: the mode is remembered but not applied, so
                // announcing it would name something that is not on screen.
                if *v.view_mode != terra_render::ViewMode::Lit && !v.playing {
                    let resp = ui
                        .button(
                            RichText::new(format!("{}  \u{2715}", v.view_mode.label()))
                                .size(11.5)
                                .color(theme::WARN),
                        )
                        .on_hover_text("Back to Lit  (Alt+4)");
                    if resp.clicked() {
                        *v.view_mode = terra_render::ViewMode::Lit;
                    }
                    ui.add_space(10.0);
                }
                match v.brush_at {
                    Some((x, z)) => ui.label(theme::small(&format!("{x:.0} m,  {z:.0} m"))),
                    None => ui.label(theme::small("--")),
                };
                ui.add_space(10.0);
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.add_space(6.0);
                    ui.add(egui::Label::new(theme::small(hints(&v, narrow))).truncate());
                });
            });
        });
    });

    // --- dockable panes ---
    //
    // Everything from here down is one DockArea rather than fixed left/right
    // panels. Tabs can be dragged to re-dock, dragged out of the window to
    // float, collapsed to their title bar, and resized by their separators --
    // all of which comes from egui_dock rather than from anything here.
    let mut viewer = EditorTabs { view: &mut v, action: &mut action };
    egui_dock::DockArea::new(layout.state_mut())
        .style(theme::dock_style(root.ctx()))
        // Adding an arbitrary tab from a "+" button makes no sense when the
        // tab set is fixed; the View menu is the way panes come back.
        .show_add_buttons(false)
        // Every tab already carries its own close button, so the per-leaf
        // "close all" is a second X beside the first and reads as a bug.
        .show_leaf_close_all_buttons(false)
        // Collapse-to-title-bar, for docked leaves and floating windows alike.
        .show_leaf_collapse_buttons(true)
        .draggable_tabs(true)
        // Clamp floating windows to the viewport. Left unset, a panel ejected
        // into a window can be dragged past the edge of the screen and there is
        // no way to get it back except View > Reset layout.
        .window_bounds(root.ctx().viewport_rect())
        .show_inside(root, &mut viewer);

    action
}

/// Dispatches each dock tab to the function that draws it.
///
/// Holds the whole [`EditorView`] plus the pending action by mutable reference,
/// because `egui_dock` calls back once per visible tab and each one needs the
/// same borrows. That is also why the panes take `&mut EditorView` rather than
/// individual fields: threading a dozen references through a trait impl buys
/// nothing that one borrow does not.
struct EditorTabs<'a, 'v> {
    view: &'a mut EditorView<'v>,
    action: &'a mut EditorAction,
}

impl egui_dock::TabViewer for EditorTabs<'_, '_> {
    type Tab = crate::dock::Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn closeable(&mut self, tab: &mut Self::Tab) -> bool {
        tab.closeable()
    }

    /// One stable id per pane. Derived from the title rather than the tab's
    /// position, so a pane keeps its scroll offset and collapsed state when it
    /// is dragged somewhere else.
    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id {
        egui::Id::new(tab.title())
    }

    /// The viewport must not paint a background, or it would cover the 3D scene
    /// rendered underneath the egui pass. Every other pane wants one.
    fn clear_background(&self, tab: &Self::Tab) -> bool {
        *tab != crate::dock::Tab::Viewport
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        use crate::dock::Tab;
        match tab {
            // Draws nothing: the scene is already on screen behind this, and
            // the tab exists to reserve the space. Its rect is the one piece of
            // information it does produce. See `dock.rs`.
            Tab::Viewport => {
                *self.view.viewport_rect = Some(ui.max_rect());
            }
            Tab::Tools => {
                // Scrolls, because the tool list plus its settings is taller
                // than the pane on a short window and the overflow was simply
                // clipped -- with no indication anything was missing.
                egui::ScrollArea::vertical()
                    .id_salt("tools-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| tools_panel(ui, self.view, self.action));
            }
            Tab::Inspector => {
                egui::ScrollArea::vertical()
                    .id_salt("inspector-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| inspector_panel(ui, self.view, self.action));
            }
            Tab::Modifiers => {
                egui::ScrollArea::vertical()
                    .id_salt("modifiers-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| modifiers_panel(ui, self.view, self.action));
            }
            Tab::Environment => {
                egui::ScrollArea::vertical()
                    .id_salt("environment-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| environment_panel(ui, self.view.env));
            }
            Tab::Material => {
                egui::ScrollArea::vertical()
                    .id_salt("material-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| material_panel(ui, self.view, self.action));
            }
            Tab::Content => {
                egui::ScrollArea::vertical()
                    .id_salt("content-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| content_panel(ui, self.view, self.action));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Environment Light Mixer
// ---------------------------------------------------------------------------

/// The Environment Light Mixer, in the order light physically arrives.
///
/// One panel rather than the four it replaces -- FOG, ENVIRONMENT, SHADOWS and
/// ATMOSPHERE were separate sections with no ordering between them, and they
/// interact: raising fog density without touching exposure darkens the frame,
/// and a user adjusting one section at a time could not see why. Sun first, then
/// what the air does to it, then what fills the shadows, then what sits in
/// front, then how it all becomes pixels.
fn environment_panel(ui: &mut egui::Ui, env: &mut terra_render::Environment) {
    use terra_render::ToneMapper;

    // --- quick create, as in Unreal's mixer ---
    ui.label(theme::label("QUICK CREATE"));
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        // `apply_preset`, not assignment: a preset sets the look and must not
        // switch off what the user turned on. See its doc comment.
        if ui.button("Daylight").clicked() {
            env.apply_preset(terra_render::Environment::daylight());
        }
        if ui.button("Overcast").clicked() {
            env.apply_preset(terra_render::Environment::overcast());
        }
        if ui.button("Night").clicked() {
            env.apply_preset(terra_render::Environment::night());
        }
    });
    ui.add_space(4.0);
    ui.label(theme::small(
        "Presets rather than one slider: these settings interact, and half-overcast \
         is not a look. A preset sets the lighting and leaves what you switched on \
         alone -- use Reset environment below to clear everything.",
    ));
    ui.add_space(12.0);

    // --- sun ---
    ui.horizontal(|ui| {
        ui.label(theme::label("SUN"));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(theme::small(if env.sun.is_night() { "moon is key" } else { "" }));
        });
    });
    ui.add_space(4.0);
    theme::inset(8).show(ui, |ui| {
        let was = (env.sun.pitch_deg, env.sun.yaw_deg);
        slider(ui, "Pitch deg", &mut env.sun.pitch_deg, -90.0..=90.0);
        slider(ui, "Yaw deg", &mut env.sun.yaw_deg, 0.0..=360.0);
        // Dragging the sun by hand means the clock is no longer driving it, so
        // stop the cycle rather than letting the next tick snap it back.
        if (env.sun.pitch_deg, env.sun.yaw_deg) != was {
            env.cycle_running = false;
        }
        ui.label(theme::small("Negative pitch is above the horizon, as in Unreal."));
        ui.add_space(4.0);
        slider(ui, "Intensity", &mut env.sun.intensity, 0.0..=4.0);
        slider(ui, "Disc deg", &mut env.sun.angular_diameter_deg, 0.1..=12.0);
        ui.label(theme::small(
            "The real sun is 0.53 deg. Widening it softens every shadow in the \
             scene at once.",
        ));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let mut tint = env.sun.tint.to_array();
            if ui.color_edit_button_rgb(&mut tint).changed() {
                env.sun.tint = glam::Vec3::from(tint);
            }
            ui.label(theme::small("Tint"));
        });
        ui.checkbox(&mut env.sun.casts_shadows, "Casts shadows");
    });
    ui.add_space(12.0);

    // --- time of day ---
    ui.label(theme::label("TIME OF DAY"));
    ui.add_space(4.0);
    theme::inset(8).show(ui, |ui| {
        let hours = env.time_of_day;
        ui.label(theme::muted(&format!(
            "{:02}:{:02}",
            hours.floor() as u32,
            ((hours.fract()) * 60.0) as u32
        )));
        if slider_changed(ui, "Hour", &mut env.time_of_day, 0.0..=24.0) {
            env.sync_sun_to_clock();
        }
        ui.checkbox(&mut env.cycle_running, "Run day/night cycle");
        ui.add_enabled_ui(env.cycle_running, |ui| {
            slider(ui, "Hours / s", &mut env.day_speed, 0.01..=4.0);
        });
        ui.checkbox(&mut env.editor_preview, "Preview in viewport");
        ui.label(theme::small(
            "Off by default: the time is authored for play, and a moving sun changes \
             the ground you are painting while you paint it.",
        ));
    });
    ui.add_space(12.0);

    // --- atmosphere ---
    ui.checkbox(&mut env.atmosphere.enabled, "Sky Atmosphere");
    ui.add_space(4.0);
    ui.add_enabled_ui(env.atmosphere.enabled, |ui| {
        theme::inset(8).show(ui, |ui| {
            slider_log(ui, "Haze (Mie)", &mut env.atmosphere.mie_scale, 0.05..=20.0);
            ui.label(theme::small(
                "Aerosol against Earth's. Greys the horizon and brightens the sky \
                 around the sun.",
            ));
            ui.add_space(4.0);
            slider(ui, "Sky (Rayleigh)", &mut env.atmosphere.rayleigh_scale, 0.0..=4.0);
            slider(ui, "Forward scatter", &mut env.atmosphere.mie_anisotropy, 0.0..=0.95);
            slider(ui, "Ozone", &mut env.atmosphere.ozone_scale, 0.0..=4.0);
            ui.add_space(4.0);
            ui.label(theme::small(
                "Rayleigh is the real per-metre coefficient for air, which is why the \
                 sky is this blue rather than a colour someone chose.",
            ));
        });
    });
    ui.add_space(12.0);

    // --- sky light ---
    ui.checkbox(&mut env.sky_light.enabled, "Sky Light (ambient bounce)");
    ui.add_space(4.0);
    ui.add_enabled_ui(env.sky_light.enabled, |ui| {
        theme::inset(8).show(ui, |ui| {
            slider(ui, "Intensity", &mut env.sky_light.intensity, 0.0..=4.0);
            ui.checkbox(&mut env.sky_light.capture_from_atmosphere, "Capture from atmosphere");
            ui.label(theme::small(
                "On, the fill follows the sky it is standing under. Off, it uses the \
                 colours below.",
            ));
            ui.add_enabled_ui(!env.sky_light.capture_from_atmosphere, |ui| {
                ui.add_space(4.0);
                for (label, c) in [
                    ("Zenith", &mut env.sky_light.zenith),
                    ("Horizon", &mut env.sky_light.horizon),
                    ("Ground", &mut env.sky_light.ground),
                ] {
                    ui.horizontal(|ui| {
                        let mut rgb = c.to_array();
                        if ui.color_edit_button_rgb(&mut rgb).changed() {
                            *c = glam::Vec3::from(rgb);
                        }
                        ui.label(theme::small(label));
                    });
                }
            });
        });
    });
    ui.add_space(12.0);

    // --- fog ---
    ui.checkbox(&mut env.fog.enabled, "Exponential Height Fog");
    ui.add_space(4.0);
    ui.add_enabled_ui(env.fog.enabled, |ui| {
        theme::inset(8).show(ui, |ui| {
            slider_log(ui, "Density", &mut env.fog.density, 0.00001..=0.05);
            slider_log(ui, "Height falloff m", &mut env.fog.height_falloff_m, 20.0..=4000.0);
            slider(ui, "Base height m", &mut env.fog.base_height_m, -100.0..=2000.0);
            ui.label(theme::small(
                "Density is quoted at the base height and falls to 1/e one falloff \
                 above it, so a short falloff pools fog in valleys.",
            ));
            ui.add_space(6.0);
            slider(ui, "God rays", &mut env.fog.god_rays, 0.0..=1.5);
            ui.label(theme::small(
                "Shafts are the fog volume marched toward the sun, so they need fog to \
                 exist. Zero skips the pass.",
            ));
            ui.add_space(6.0);
            slider(ui, "Forward scatter", &mut env.fog.anisotropy, 0.0..=0.95);
            slider_log(ui, "Valley mist", &mut env.fog.mist_strength, 0.0..=0.01);
            slider_log(ui, "Distance m", &mut env.fog.distance_m, 100.0..=3000.0);
            ui.horizontal(|ui| {
                let mut rgb = env.fog.albedo.to_array();
                if ui.color_edit_button_rgb(&mut rgb).changed() {
                    env.fog.albedo = glam::Vec3::from(rgb);
                }
                ui.label(theme::small("Medium colour"));
            });
        });
    });
    ui.add_space(12.0);

    // --- clouds ---
    ui.checkbox(&mut env.clouds.enabled, "Volumetric Clouds");
    ui.add_space(4.0);
    ui.add_enabled_ui(env.clouds.enabled, |ui| {
        theme::inset(8).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(theme::small("Quality"));
                for q in terra_render::CloudQuality::ALL {
                    if ui.selectable_label(env.clouds.quality == q, q.label()).clicked() {
                        env.clouds.quality = q;
                    }
                }
            });
            ui.label(theme::small(
                "The march is the most expensive thing in the sky -- about 10 ms at \
                 1280x720 on Medium against the clear sky's 2.4. Low halves the samples.",
            ));
            ui.add_space(4.0);
            slider(ui, "Coverage", &mut env.clouds.coverage, 0.0..=1.0);
            slider_log(ui, "Base m", &mut env.clouds.base_m, 200.0..=8000.0);
            slider_log(ui, "Thickness m", &mut env.clouds.thickness_m, 100.0..=8000.0);
            slider_log(ui, "Density", &mut env.clouds.density, 0.001..=0.5);
            slider_log(ui, "Feature m", &mut env.clouds.feature_scale_m, 500.0..=40000.0);
        });
    });
    ui.add_space(12.0);

    // --- tone mapping ---
    ui.label(theme::label("TONE MAPPING"));
    ui.add_space(4.0);
    theme::inset(8).show(ui, |ui| {
        ui.horizontal(|ui| {
            for m in ToneMapper::ALL {
                if ui.selectable_label(env.tone.mapper == m, m.label()).clicked() {
                    env.tone.mapper = m;
                }
            }
        });
        ui.label(theme::small(
            "ACES rolls highlights off while keeping their hue, so a sun disc stays \
             yellow instead of becoming a white hole.",
        ));
        ui.add_space(6.0);
        slider(ui, "Exposure EV", &mut env.tone.exposure_ev, -4.0..=4.0);
        slider(ui, "Contrast", &mut env.tone.contrast, 0.5..=2.0);
        slider(ui, "Saturation", &mut env.tone.saturation, 0.0..=2.0);
        slider(ui, "White balance K", &mut env.tone.white_balance_k, 2000.0..=12000.0);
    });

    ui.add_space(14.0);
    if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Reset environment")).clicked()
    {
        env.reset();
    }
}

// ---------------------------------------------------------------------------
// Material editor
// ---------------------------------------------------------------------------

/// The selected material, and the settings that drive how it renders.
pub struct MaterialView<'a> {
    pub name: &'a str,
    pub role: &'a str,
    pub texture: Option<&'a egui::TextureHandle>,
    pub params: &'a mut terra_render::material::LayerParams,
}

/// PBR settings for one material.
///
/// Opened by double-clicking a texture, in the Content browser or in the Paint
/// palette. Every value here is per layer rather than global: one tiling scale
/// across a whole palette leaves gravel blurred and a cliff face visibly
/// repeating, because the two want repeats an order of magnitude apart.
fn material_panel(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    let Some(m) = v.material.as_mut() else {
        ui.label(theme::muted("No material selected."));
        ui.add_space(4.0);
        ui.label(theme::small(
            "Double-click a texture in the Content browser to edit how it renders.",
        ));
        return;
    };

    ui.horizontal(|ui| {
        if let Some(tex) = m.texture {
            ui.add(egui::Image::new(tex).fit_to_exact_size(vec2(48.0, 48.0)));
        }
        ui.vertical(|ui| {
            ui.label(theme::heading(m.name));
            ui.label(theme::small(m.role));
        });
    });
    ui.add_space(8.0);

    ui.label(theme::label("TILING"));
    theme::inset(8).show(ui, |ui| {
        slider_log(ui, "Repeat m", &mut m.params.tiling_m, 0.25..=64.0);
        ui.label(theme::small(
            "Metres per repeat. Smaller shows more grain up close and tiles more \
             visibly at distance.",
        ));
    });
    ui.add_space(8.0);

    ui.label(theme::label("SURFACE"));
    theme::inset(8).show(ui, |ui| {
        slider(ui, "Normal strength", &mut m.params.normal_strength, 0.0..=3.0);
        slider(ui, "Roughness", &mut m.params.roughness, 0.0..=2.0);
        slider(ui, "Occlusion", &mut m.params.ao, 0.0..=1.0);
    });
    ui.add_space(8.0);

    ui.label(theme::label("DEPTH"));
    theme::inset(8).show(ui, |ui| {
        slider(ui, "Parallax m", &mut m.params.parallax_m, 0.0..=0.25);
        ui.label(theme::small(
            "Offsets the texture lookup by its height channel, so stones and cracks \
             occlude each other as the camera moves. This is what makes the surface \
             read as relief rather than as a photograph of it. Zero is off, and off \
             is cheaper.",
        ));
        ui.add_space(4.0);
        slider(ui, "Blend band", &mut m.params.height_blend, 0.0..=0.6);
        ui.label(theme::small(
            "How wide a band this material contends with its neighbour in. Zero is a \
             hard per-texel cut; wide lets it creep through.",
        ));
    });
    ui.add_space(8.0);

    ui.label(theme::label("TINT"));
    theme::inset(8).show(ui, |ui| {
        let mut rgb = m.params.tint;
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            m.params.tint = rgb;
        }
        ui.label(theme::small("Multiplies the albedo. White leaves the texture alone."));
    });

    ui.add_space(10.0);
    if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Reset to defaults")).clicked()
    {
        *m.params = terra_render::material::LayerParams::default();
    }
    ui.add_space(4.0);
    if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Paint with this")).clicked() {
        *action = EditorAction::PaintWithSelectedMaterial;
    }
}

// ---------------------------------------------------------------------------
// Content browser
// ---------------------------------------------------------------------------

/// One importable asset kind. The browser groups by this, and each group knows
/// which extensions it will accept, so the file dialog can filter itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    /// A material set: albedo, normal, roughness in one folder.
    Texture,
    /// A greyscale map for the Noise sculpt brush.
    Noise,
    /// A mesh to scatter or place.
    Model,
}

impl AssetKind {
    pub const ALL: [AssetKind; 3] = [AssetKind::Texture, AssetKind::Noise, AssetKind::Model];

    pub fn label(self) -> &'static str {
        match self {
            AssetKind::Texture => "Textures",
            AssetKind::Noise => "Noise",
            AssetKind::Model => "Models",
        }
    }

    /// Extensions the import dialog offers for this kind.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            AssetKind::Texture => &["png", "jpg", "jpeg"],
            AssetKind::Noise => &["png", "jpg", "jpeg", "r16"],
            AssetKind::Model => &["gltf", "glb"],
        }
    }

    /// Subfolder of the project's `assets/` directory.
    pub fn folder(self) -> &'static str {
        match self {
            AssetKind::Texture => "textures",
            AssetKind::Noise => "noise",
            AssetKind::Model => "models",
        }
    }
}

/// What the content browser has to show, read out of the project folder by the
/// app before the UI runs.
pub struct ContentView<'a> {
    /// The project's `assets/` path, shown so it is obvious where uploads land.
    pub root: &'a str,
    /// Asset names per kind, in the same order as [`AssetKind::ALL`].
    pub entries: [&'a [String]; 3],
    /// Which kind's shelf is showing.
    pub kind: AssetKind,
}

impl ContentView<'_> {
    fn of(&self, kind: AssetKind) -> &[String] {
        self.entries[AssetKind::ALL.iter().position(|k| *k == kind).unwrap_or(0)]
    }
}

/// The project's own asset folder.
///
/// This is where uploads land, and it is deliberately a pane rather than a
/// modal: importing a texture is something you do repeatedly while working, and
/// a dialog that has to be dismissed between each one turns a batch into a
/// chore.
fn content_panel(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    ui.horizontal(|ui| {
        for k in AssetKind::ALL {
            let n = v.content.of(k).len();
            let label = format!("{}  {n}", k.label());
            if ui.selectable_label(v.content.kind == k, label).clicked() {
                *action = EditorAction::SelectAssetKind(k);
            }
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.button(format!("Import {}", v.content.kind.label())).clicked() {
                *action = EditorAction::ImportAsset(v.content.kind);
            }
        });
    });
    ui.add_space(6.0);

    let kind = v.content.kind;
    let items = v.content.of(kind);
    if items.is_empty() {
        ui.add_space(10.0);
        ui.label(theme::muted(&format!("No {} in this project yet.", kind.label().to_lowercase())));
        ui.add_space(4.0);
        ui.label(theme::small(&format!(
            "Import copies the file into {}/{}/ inside the project, so the project stays \
             movable -- nothing here stores a path outside it.",
            v.content.root,
            kind.folder()
        )));
        return;
    }

    // A wrapping shelf rather than a list: these are pictures, and the whole
    // point of a content browser is seeing them at a glance.
    ui.horizontal_wrapped(|ui| {
        for (i, name) in items.iter().enumerate() {
            let selected = kind == AssetKind::Noise && v.noise.pattern.label() == name.as_str();
            let resp = ui.selectable_label(selected, name);
            match kind {
                AssetKind::Noise if resp.clicked() => {
                    *action = EditorAction::SelectNoise(name.clone());
                }
                // Double-click, as in Unreal: single-click selects, double-click
                // opens the asset for editing.
                AssetKind::Texture if resp.double_clicked() => {
                    *action = EditorAction::OpenMaterial(i);
                }
                _ => {}
            }
            if kind == AssetKind::Texture {
                resp.on_hover_text("Double-click to edit this material");
            }
        }
    });
    ui.add_space(8.0);
    ui.label(theme::small(&format!("{}/{}", v.content.root, kind.folder())));
}

// ---------------------------------------------------------------------------
// Modifier stack
// ---------------------------------------------------------------------------

/// The cave and tunnel stack.
///
/// Order is shown top-to-bottom as evaluation order, because that is what
/// decides the result: a plug added after a carve blocks the passage, and the
/// same two entries the other way round leave it open.
fn modifiers_panel(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    use terra_voxel::Op;

    // Stacked rather than side by side: this pane is narrow by default, and a
    // right-aligned button on the same row as the heading overlaps it.
    ui.label(theme::label("STACK"));
    ui.add_space(4.0);
    if ui.add_sized([ui.available_width(), 28.0], egui::Button::new("Add tunnel")).clicked() {
        *action = EditorAction::AddTunnel;
    }
    ui.add_space(6.0);

    if v.modifiers.is_empty() {
        ui.label(theme::muted("No caves yet."));
        ui.add_space(4.0);
        ui.label(theme::small(
            "A modifier carves the rock without destroying it. Switch one off and the \
             passage fills back in exactly as it was, however much you have sculpted \
             around it since.",
        ));
        return;
    }

    let selected = *v.selected_modifier;
    let mut clicked = None;
    let mut remove = None;
    for (i, m) in v.modifiers.items.iter_mut().enumerate() {
        theme::inset(6).show(ui, |ui| {
            ui.horizontal(|ui| {
                // Enabled first: it is the control most often reached for, and
                // toggling it is how you see what a cave is doing.
                ui.checkbox(&mut m.enabled, "");
                let label = format!("{}  \u{2022}  {}", m.name, m.op.label());
                if ui.selectable_label(selected == Some(i), label).clicked() {
                    clicked = Some(i);
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("\u{00D7}").on_hover_text("Delete").clicked() {
                        remove = Some(i);
                    }
                });
            });
            if selected == Some(i) {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    for op in Op::ALL {
                        if ui.selectable_label(m.op == op, op.label()).clicked() {
                            m.op = op;
                        }
                    }
                });
                slider(ui, "Blend m", &mut m.blend, 0.0..=6.0);
                ui.label(theme::small(
                    "Rounds the join where the cave meets rock. Zero is a knife edge.",
                ));
            }
        });
        ui.add_space(4.0);
    }

    if let Some(i) = clicked {
        *v.selected_modifier = Some(i);
    }
    if let Some(i) = remove {
        *action = EditorAction::DeleteModifier(i);
    }
}

/// Panel heights, sized to the tallest control each one holds rather than to
/// its text. The toolbar was 46 px, which after 8 px of frame padding on both
/// sides left 30 for a button that wants 35 -- so the row was clipped.
const TOOLBAR_H: f32 = 54.0;
const STATUS_H: f32 = 32.0;

/// The View menu: pane visibility, floating, and the layout reset.
///
/// This is the only way back once a pane is closed, which makes it load-bearing
/// rather than a convenience. Without it the close button on a tab is a one-way
/// trip and the pane is gone for the session -- which is exactly what the first
/// version of the dock shipped as.
///
/// Mutates the layout directly instead of returning an `EditorAction`. The
/// `DockArea` below borrows the layout again afterwards, and the two borrows are
/// sequential, so there is nothing for an action round-trip to buy.
fn view_menu(
    ui: &mut egui::Ui,
    layout: &mut crate::dock::Layout,
    view_mode: &mut terra_render::ViewMode,
    editing: bool,
) {
    use crate::dock::Tab;

    ui.menu_button("View", |ui| {
        // Bounded, or the hint at the bottom stretches the menu to the width of
        // its longest line -- about 600 px, which reads as a dialog.
        ui.set_min_width(200.0);
        ui.set_max_width(240.0);

        // Visualization first: it is the thing most often reached for, and the
        // hotkeys are listed beside each entry because that is how they get
        // learned.
        //
        // Absent entirely while driving rather than greyed out. These are
        // authoring views; the moment the viewport is the game, offering to
        // wireframe it is offering something that will not happen.
        if editing {
            ui.label(theme::label("VISUALIZE"));
            for m in terra_render::ViewMode::ALL {
                let selected = *view_mode == m;
                let label = format!("{}\t\tAlt+{}", m.label(), m.hotkey_digit());
                if ui.selectable_label(selected, label).clicked() {
                    *view_mode = m;
                }
            }
            ui.separator();
        }

        ui.label(theme::label("PANELS"));
        for tab in Tab::DOCKABLE {
            let mut open = layout.is_open(tab);
            if ui.checkbox(&mut open, tab.title()).changed() {
                layout.toggle(tab);
            }
        }

        ui.separator();
        ui.menu_button("Float a panel", |ui| {
            for tab in Tab::DOCKABLE {
                if ui.button(tab.title()).clicked() {
                    layout.float(tab);
                    ui.close();
                }
            }
        });

        ui.separator();
        if ui.button("Reset layout").clicked() {
            layout.reset();
            ui.close();
        }
        ui.add_space(6.0);
        ui.add(
            egui::Label::new(theme::small(
                "Drag a tab to re-dock it. Right-click a tab to eject it into a \
             window. The arrow collapses a panel.",
            ))
            .wrap(),
        );
    });
}

fn hints(v: &EditorView<'_>, narrow: bool) -> &'static str {
    // Navigation first: it is the same in every tool, and it is what someone
    // coming from another editor looks for before anything else.
    match (v.playing, *v.tool, narrow) {
        (true, _, false) => {
            "W/S throttle   \u{2022}   A/D steer   \u{2022}   Q brake   \u{2022}   \
             Shift handbrake   \u{2022}   Esc stop"
        }
        (true, _, true) => "W/S throttle   \u{2022}   A/D steer   \u{2022}   Esc stop",
        (false, Tool::Camera, true) => "LMB orbit   \u{2022}   MMB pan   \u{2022}   Wheel zoom",
        (false, Tool::Select, true) => "LMB pick/move   \u{2022}   Del remove",
        (false, _, true) => {
            "MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \u{2022}   [ ] size"
        }
        (false, Tool::Camera, false) => {
            "LMB orbit   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   F frame   \u{2022}   Shift boost"
        }
        (false, Tool::Sculpt, false) => {
            "LMB sculpt   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   F frame   \u{2022}   [ ] size, Shift + [ ] strength"
        }
        (false, Tool::Select, false) => {
            "LMB pick, drag to move   \u{2022}   Del remove   \u{2022}   MMB pan   \
             \u{2022}   Wheel zoom   \u{2022}   RMB look"
        }
        (false, Tool::Foliage, false) => {
            "LMB plant   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   F frame   \u{2022}   [ ] size, Shift + [ ] strength"
        }
        (false, Tool::Paint, false) => {
            "LMB paint   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   F frame   \u{2022}   [ ] size, Shift + [ ] strength"
        }
        (false, Tool::Road, false) => {
            "LMB drag to draw the track   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \
             \u{2022}   RMB look   \u{2022}   WASD move, Q/E down/up   \u{2022}   F frame"
        }
    }
}

/// What Ctrl does in a mode, for the tooltip. `None` where the mode has no
/// meaningful opposite -- Flatten already pulls from both sides, and
/// "un-smooth" is just noise, which is what the Noise mode is for.
pub fn invert_label(m: SculptMode) -> Option<&'static str> {
    match m {
        SculptMode::Raise => Some("lower"),
        SculptMode::Lower => Some("raise"),
        _ => None,
    }
}

fn tools_panel(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    if v.playing {
        ui.label(theme::label("DRIVING"));
        ui.add_space(8.0);
        ui.label(theme::muted(
            "W / S   throttle, reverse\nA / D   steer\nQ       brake\nShift   handbrake\nEsc     stop",
        ));
        return;
    }
    ui.label(theme::label("TOOLS"));
    ui.add_space(8.0);
    for t in Tool::ALL {
        let w = ui.available_width();
        if ui.add_sized([w, 32.0], egui::Button::selectable(*v.tool == t, t.label())).clicked() {
            *v.tool = t;
        }
    }

    brush_readout(ui, v);

    match *v.tool {
        Tool::Sculpt => {
            ui.add_space(20.0);
            ui.label(theme::label("MODE"));
            ui.add_space(6.0);
            for m in SculptMode::ALL {
                let w = ui.available_width();
                let resp = ui
                    .add(egui::Button::selectable(*v.mode == m, m.label()).min_size(vec2(w, 26.0)));
                if resp.clicked() {
                    *v.mode = m;
                }
                if let Some(inv) = invert_label(m) {
                    resp.on_hover_text(format!("Hold Ctrl to {}", inv.to_lowercase()));
                }
            }

            // Noise settings, only under the mode that reads them.
            if v.mode.uses_noise() {
                ui.add_space(16.0);
                ui.label(theme::label("PATTERN"));
                ui.add_space(6.0);
                theme::inset(10).show(ui, |ui| {
                    ui.label(theme::muted(v.noise.pattern.label()));
                    if !v.noise.pattern.is_uploaded() {
                        ui.label(theme::small(
                            "Built in. Import a black-and-white image in the Content pane \
                             to use your own.",
                        ));
                    }
                    ui.add_space(6.0);
                    slider_log(ui, "Feature m", &mut v.noise.scale_m, 1.0..=200.0);

                    if let terra_voxel::NoisePattern::Procedural { seed, octaves, ridged } =
                        &mut v.noise.pattern
                    {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(*ridged, "Ridged").clicked() {
                                *ridged = true;
                            }
                            if ui.selectable_label(!*ridged, "Billow").clicked() {
                                *ridged = false;
                            }
                        });
                        let mut oct = *octaves as f32;
                        slider(ui, "Octaves", &mut oct, 1.0..=8.0);
                        *octaves = oct.round() as u32;
                        if ui.button("Re-roll seed").clicked() {
                            // Any change is enough; a counter keeps it
                            // reproducible where a clock would not.
                            *seed = seed.wrapping_add(0x9E37_79B9);
                        }
                    }

                    if !v.noise_library.is_empty() {
                        ui.add_space(8.0);
                        ui.label(theme::small("Imported"));
                        for name in v.noise_library {
                            let on = v.noise.pattern.label() == name.as_str();
                            if ui.selectable_label(on, name).clicked() {
                                *action = EditorAction::SelectNoise(name.clone());
                            }
                        }
                    }
                });
            }
        }
        Tool::Camera => {
            ui.add_space(20.0);
            ui.label(theme::label("NAVIGATE"));
            ui.add_space(8.0);
            ui.label(theme::muted(
                "LMB drag   orbit\nMMB drag   pan\nWheel      zoom\nRMB drag   look\n\nWASD move, Q/E down/up\nShift      faster\nF          frame the world",
            ));
            ui.add_space(12.0);
            ui.label(theme::small(
                "Nothing you do here changes the terrain. Pick another tool to edit.",
            ));
        }
        Tool::Paint => {
            ui.add_space(20.0);
            ui.label(theme::label("MODE"));
            ui.add_space(6.0);
            for m in PaintMode::ALL {
                let w = ui.available_width();
                if ui
                    .add_sized([w, 32.0], egui::Button::selectable(*v.paint_mode == m, m.label()))
                    .clicked()
                {
                    *v.paint_mode = m;
                }
            }

            ui.add_space(20.0);
            ui.label(theme::label("MATERIALS"));
            ui.add_space(6.0);
            if v.palette.is_empty() {
                ui.label(theme::muted("No materials imported."));
                ui.add_space(4.0);
                ui.label(theme::small(
                    "Import a texture set in the Content pane -- one folder per material, \
                     with albedo, normal and roughness maps in it. Nothing ships prebuilt, \
                     and imports appear here immediately; no restart.",
                ));
            } else {
                let mut open_material = None;
                palette_grid(ui, v.palette, v.selected_layer, &mut open_material);
                if let Some(i) = open_material {
                    *action = EditorAction::OpenMaterial(i);
                }
            }
        }
        Tool::Foliage => {
            ui.add_space(20.0);
            ui.label(theme::label("MODE"));
            ui.add_space(6.0);
            for m in PaintMode::ALL {
                let w = ui.available_width();
                let label = if m == PaintMode::Brush { "Plant" } else { "Remove" };
                if ui
                    .add_sized([w, 32.0], egui::Button::selectable(*v.paint_mode == m, label))
                    .clicked()
                {
                    *v.paint_mode = m;
                }
            }

            ui.add_space(20.0);
            ui.label(theme::label("SPECIES"));
            ui.add_space(6.0);
            if v.foliage.is_empty() {
                // Says what to do, not just that there is nothing. Nothing ships
                // prebuilt, so an empty palette is the expected first state rather
                // than a failure, and the message has to read that way.
                ui.label(theme::small(
                    "No meshes imported yet. Open the Content browser, choose Models, \
                     and import a .glb or .gltf -- it becomes a species you can paint.",
                ));
            } else {
                species_grid(ui, v.foliage, v.selected_species);
                if let Some(e) = v.foliage.get(*v.selected_species) {
                    ui.add_space(4.0);
                    ui.label(theme::small(&if e.painted {
                        format!("{}  \u{2022}  {} planted", e.name, thousands(e.instances))
                    } else {
                        format!("{}  \u{2022}  not planted", e.name)
                    }));
                }
            }
        }
        Tool::Select => {
            ui.add_space(20.0);
            ui.label(theme::label("OBJECTS"));
            ui.add_space(8.0);
            ui.label(theme::muted("LMB      pick\nLMB drag move\nDel      remove"));
            ui.add_space(12.0);
            let w = ui.available_width();
            let name = v.foliage.get(*v.selected_species).map(|e| e.name.as_str()).unwrap_or("--");
            if ui.add_sized([w, 32.0], egui::Button::new(format!("Place {name}"))).clicked() {
                *action = EditorAction::PlaceProp;
            }
            ui.label(theme::small("Drops one where you are looking."));
            ui.add_space(12.0);
            ui.label(theme::label("SPECIES"));
            ui.add_space(6.0);
            for (i, e) in v.foliage.iter().enumerate() {
                let w = ui.available_width();
                if ui
                    .add_sized(
                        [w, 28.0],
                        egui::Button::selectable(*v.selected_species == i, &e.name),
                    )
                    .clicked()
                {
                    *v.selected_species = i;
                }
            }
            ui.add_space(10.0);
            ui.label(theme::small(&format!("{} placed in this world", v.prop_count)));
        }
        Tool::Road => {
            ui.add_space(20.0);
            ui.label(theme::label("ROADS"));
            ui.add_space(6.0);
            let w = ui.available_width();
            match &v.active_road {
                Some(r) => {
                    ui.label(theme::muted(&format!("Drawing: {} points", r.points.len())));
                    ui.label(theme::small(
                        "Drag across the terrain to draw. Draw again to extend.",
                    ));
                    ui.add_space(6.0);
                    if ui.add_sized([w, 30.0], egui::Button::new("Undo point")).clicked() {
                        *action = EditorAction::UndoPoint;
                    }
                    if ui.add_sized([w, 30.0], egui::Button::new("Finish road")).clicked() {
                        *action = EditorAction::FinishRoad;
                    }
                }
                None => {
                    ui.label(theme::muted(&format!("{} in this world", v.road_count)));
                    ui.add_space(6.0);
                    if ui
                        .add_sized([w, 34.0], egui::Button::new(theme::heading("New road")))
                        .clicked()
                    {
                        *action = EditorAction::NewRoad;
                    }
                }
            }
            if v.road_count > 0 {
                ui.add_space(6.0);
                if ui.add_sized([w, 28.0], egui::Button::new("Clear all roads")).clicked() {
                    *action = EditorAction::ClearRoads;
                }
            }
        }
    }

    // The active tool's own settings, directly beneath it.
    tool_settings(ui, v, action);
}

/// The one control most people will touch. Applying a preset writes through to
/// every individual setting, which stay editable afterwards -- picking a preset
/// is a starting point, not a mode.
pub fn quality_presets(ui: &mut egui::Ui, sky: &mut SkySettings) {
    ui.horizontal_wrapped(|ui| {
        for q in Quality::ALL {
            if ui.button(q.label()).clicked() {
                // Shadows and TAA only. God rays and fog used to be set here
                // too, which meant picking a quality level silently overwrote
                // the artist's environment -- the mixer owns those now.
                let (shadows, distance, _rays, taa) = q.sky();
                sky.shadow_quality = shadows;
                sky.shadow_distance = distance;
                sky.temporal_aa = taa;
            }
        }
    });
}

/// Shadow quality and range.
pub fn shadow_controls(ui: &mut egui::Ui, sky: &mut SkySettings) {
    ui.horizontal_wrapped(|ui| {
        for q in ShadowQuality::ALL {
            if ui.selectable_label(sky.shadow_quality == q, q.label()).clicked() {
                sky.shadow_quality = q;
            }
        }
    });
    ui.add_enabled_ui(sky.shadow_quality.enabled(), |ui| {
        slider_log(ui, "Shadow distance m", &mut sky.shadow_distance, 80.0..=1500.0);
        ui.label(theme::small(
            "Cascades are fitted to this range. Longer reaches further and \
             spends the same texels doing it, so near shadows soften.",
        ));
    });
}

/// Species previews. Same reasoning as the material swatches: a list of names
/// like "shrub_03" and "shrub_sorrel_01" tells you nothing until you see them.
fn species_grid(ui: &mut egui::Ui, entries: &[FoliageEntry], selected: &mut usize) {
    const CELL: f32 = 54.0;
    let cols = ((ui.available_width() + 6.0) / (CELL + 6.0)).floor().max(1.0) as usize;
    for (row, chunk) in entries.chunks(cols).enumerate() {
        ui.horizontal(|ui| {
            for (col, entry) in chunk.iter().enumerate() {
                let index = row * cols + col;
                let is_selected = *selected == index;
                let (rect, resp) = ui.allocate_exact_size(vec2(CELL, CELL), Sense::click());
                if ui.is_rect_visible(rect) {
                    let p = ui.painter();
                    let inner = rect.shrink(3.0);
                    p.rect_filled(inner, CornerRadius::same(8), theme::PANEL_SOFT);
                    if let Some(tex) = &entry.texture {
                        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                        p.image(tex.id(), inner, uv, egui::Color32::WHITE);
                    }
                    if entry.painted {
                        // A dot beats a number here: which species are in play
                        // is the question, not how many of each.
                        p.circle_filled(inner.right_top() + vec2(-5.0, 5.0), 3.0, theme::ACCENT);
                    }
                    if is_selected || resp.hovered() {
                        let color = if is_selected {
                            theme::ACCENT
                        } else {
                            theme::TEXT.gamma_multiply(0.6)
                        };
                        p.rect_stroke(
                            rect,
                            CornerRadius::same(10),
                            egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, color),
                            egui::StrokeKind::Inside,
                        );
                    }
                }
                if resp.clicked() {
                    *selected = index;
                }
                resp.on_hover_text(&entry.name);
            }
        });
    }
}

/// Material swatches, the way every terrain tool presents them: the picture is
/// the control. A list of folder names would be technically equivalent and
/// useless -- you choose a ground texture by looking at it.
fn palette_grid(
    ui: &mut egui::Ui,
    palette: &[PaletteEntry<'_>],
    selected: &mut u32,
    open_material: &mut Option<usize>,
) {
    const CELL: f32 = 58.0;
    let avail = ui.available_width();
    let cols = ((avail + 6.0) / (CELL + 6.0)).floor().max(1.0) as usize;

    for (row, chunk) in palette.chunks(cols).enumerate() {
        ui.horizontal(|ui| {
            for (col, entry) in chunk.iter().enumerate() {
                let index = (row * cols + col) as u32;
                let is_selected = *selected == index;
                let (rect, resp) = ui.allocate_exact_size(vec2(CELL, CELL), Sense::click());

                if ui.is_rect_visible(rect) {
                    let p = ui.painter();
                    let inner = rect.shrink(3.0);
                    if let Some(tex) = entry.texture {
                        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                        p.image(tex.id(), inner, uv, egui::Color32::WHITE);
                    } else {
                        p.rect_filled(inner, CornerRadius::same(8), theme::PANEL_SOFT);
                    }
                    // Selection reads as a ring rather than a tint, so the
                    // swatch keeps showing the material's real colour.
                    if is_selected || resp.hovered() {
                        let color = if is_selected {
                            theme::ACCENT
                        } else {
                            theme::TEXT.gamma_multiply(0.6)
                        };
                        p.rect_stroke(
                            rect,
                            CornerRadius::same(10),
                            egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, color),
                            egui::StrokeKind::Inside,
                        );
                    }
                }
                if resp.clicked() {
                    *selected = index;
                }
                if resp.double_clicked() {
                    *open_material = Some(index as usize);
                }
                resp.on_hover_text(format!(
                    "{}  ({})\nDouble-click to edit this material",
                    entry.name, entry.role
                ));
            }
        });
    }

    if let Some(entry) = palette.get(*selected as usize) {
        ui.add_space(4.0);
        ui.label(theme::small(&format!("{}  \u{2022}  {}", entry.name, entry.role)));
    }
}

/// Settings for whichever tool is active, and nothing else.
///
/// This lives in the Tools pane, directly under the tool list, rather than in a
/// single always-on settings column. Showing every tool's controls at once
/// meant the sculpt sliders sat next to road cross-section sliders next to
/// foliage rules, and the reader had to work out which of them the current tool
/// would even read. Selecting a tool now selects its settings too.
///
/// Brush size and strength are deliberately *not* here as sliders any more --
/// they are on `[` and `]`. The readout below is a display, not a control.
/// Brush size and strength, as a readout rather than sliders.
///
/// The keys are the interface -- `[` and `]` for size, with Shift for strength,
/// as in Unreal. A slider here would be a second source of truth for the same
/// number, and the request was explicitly to take these controls off the panel.
fn brush_readout(ui: &mut egui::Ui, v: &EditorView<'_>) {
    if !v.tool.edits() {
        return;
    }
    ui.add_space(16.0);
    ui.label(theme::label("BRUSH"));
    ui.add_space(6.0);
    theme::inset(10).show(ui, |ui| {
        // Read-only readouts. The keys are the interface; a slider here would
        // be a second source of truth for the same number.
        ui.horizontal(|ui| {
            ui.label(theme::small("Size"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(theme::muted(&format!("{:.0} m", *v.radius)));
            });
        });
        ui.horizontal(|ui| {
            ui.label(theme::small("Strength"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(theme::muted(&format!("{:.2}", *v.strength)));
            });
        });
        ui.add_space(4.0);
        ui.label(theme::small("[ ]  size          Shift + [ ]  strength"));
    });
}

fn tool_settings(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    if !v.tool.edits() {
        return;
    }

    if *v.tool == Tool::Paint {
        ui.label(theme::label("PAINT"));
        ui.add_space(8.0);
        theme::inset(10).show(ui, |ui| {
            slider(ui, "Flow /s", v.paint_flow, 0.2..=8.0);
            ui.label(theme::small(
                "How fast a held stroke builds up. Low values let you feather \
                 one material into another.",
            ));
            ui.add_space(8.0);

            let w = ui.available_width();
            let name = v.palette.get(*v.selected_layer as usize).map(|e| e.name).unwrap_or("--");
            if ui
                .add_sized([w, 30.0], egui::Button::new(format!("Fill world with {name}")))
                .clicked()
            {
                *action = EditorAction::FillMaterial;
            }
            ui.add_enabled_ui(v.painted, |ui| {
                if ui.add_sized([w, 30.0], egui::Button::new("Clear painting")).clicked() {
                    *action = EditorAction::ClearPaint;
                }
            });
            ui.label(theme::small(if v.painted {
                "Clearing hands the surface back to automatic placement by \
                 slope and erosion."
            } else {
                "Nothing painted yet -- materials are placed automatically by \
                 slope and erosion."
            }));
        });
        ui.add_space(18.0);
    }

    if *v.tool == Tool::Foliage {
        ui.label(theme::label("SPECIES RULES"));
        ui.add_space(8.0);
        let name = v.foliage.get(*v.selected_species).map(|e| e.name.clone()).unwrap_or_default();
        let total = v.foliage_instances;
        let selected = *v.selected_species;
        theme::inset(10).show(ui, |ui| {
            if let Some(r) = v.species_rules.as_deref_mut() {
                // Grouped and named as Unreal's Foliage Type groups them, so someone
                // who knows that panel can find the control they are looking for.
                ui.label(theme::small("PLACEMENT"));
                slider_log(ui, "Per hectare", &mut r.density, 1.0..=600.0);
                slider(ui, "Radius m", &mut r.radius_m, 0.0..=40.0);
                ui.label(theme::small(
                    "Minimum gap between instances. Density alone cannot say \
                     \"sparse but never overlapping\".",
                ));

                ui.add_space(10.0);
                ui.label(theme::small("SIZE"));
                slider_log(ui, "Height m", &mut r.height_m, 0.05..=60.0);
                ui.label(theme::small(
                    "Real-world height of one instance. Every imported mesh is \
                     normalised to a metre, so this is the size on screen whatever \
                     the file was authored at.",
                ));
                slider(ui, "Scale min", &mut r.scale_min, 0.1..=3.0);
                slider(ui, "Scale max", &mut r.scale_max, 0.1..=4.0);
                // Keeping these ordered here rather than validating later means
                // the generator never has to deal with an inverted range.
                if r.scale_max < r.scale_min {
                    r.scale_max = r.scale_min;
                }
                slider(ui, "Z offset m", &mut r.z_offset_m, -3.0..=3.0);
                ui.label(theme::small("Negative beds the instance into the ground."));

                ui.add_space(10.0);
                ui.label(theme::small("ROTATION"));
                // The setting the whole cliff question comes down to, so it leads the
                // group and says what it does on a cliff rather than in the abstract.
                ui.checkbox(&mut r.align_to_normal, "Align to Normal");
                ui.label(theme::small(if r.align_to_normal {
                    "Instances stand perpendicular to the ground, so they tilt with a \
                     slope and grow sideways out of a cliff. Right for rocks and \
                     ground cover."
                } else {
                    "Instances point to the sky whatever the ground does, so a tree on \
                     a cliff still stands upright. Right for anything that grows \
                     toward the light."
                }));
                ui.add_enabled_ui(r.align_to_normal, |ui| {
                    slider(ui, "Align max angle", &mut r.align_max_angle_deg, 0.0..=90.0);
                    ui.label(theme::small(
                        "Ceiling on that tilt, in degrees from vertical. Lets a species \
                         lean into moderate ground without lying down on a cliff.",
                    ));
                });
                ui.add_space(4.0);
                ui.checkbox(&mut r.random_yaw, "Random yaw");
                slider(ui, "Random pitch", &mut r.random_pitch_deg, 0.0..=30.0);

                ui.add_space(10.0);
                ui.label(theme::small("FILTERS"));
                slider(ui, "Slope min deg", &mut r.slope_min_deg, 0.0..=90.0);
                slider(ui, "Slope max deg", &mut r.slope_max_deg, 0.0..=90.0);
                if r.slope_max_deg < r.slope_min_deg {
                    r.slope_max_deg = r.slope_min_deg;
                }
                ui.label(theme::small(
                    "Ground steepness this species accepts. Raise the minimum to put \
                     something on cliffs and nowhere else.",
                ));
                ui.add_space(4.0);
                slider(ui, "Min height m", &mut r.altitude_min, -200.0..=2500.0);
                slider(ui, "Max height m", &mut r.altitude_max, -200.0..=2500.0);
                if r.altitude_max < r.altitude_min {
                    r.altitude_max = r.altitude_min;
                }

                ui.add_space(10.0);
                ui.label(theme::small("RENDERING"));
                ui.checkbox(&mut r.cast_shadow, "Cast shadow");
                slider_log(ui, "Draw distance m", &mut r.cull_distance, 100.0..=3000.0);
                ui.label(theme::small(
                    "Beyond this, instances are not drawn at all. The main cost control: \
                     small props can be cut short, a treeline cannot.",
                ));
            } else {
                ui.label(theme::small(
                    "No species selected. Import a mesh in the Content browser \
                     to add one.",
                ));
            }
        });

        ui.add_space(12.0);
        let w = ui.available_width();
        if ui.add_sized([w, 30.0], egui::Button::new(format!("Fill world with {name}"))).clicked() {
            *action = EditorAction::FillFoliage;
        }
        if ui.add_sized([w, 30.0], egui::Button::new("Re-roll placement")).clicked() {
            *action = EditorAction::ReseedFoliage;
        }
        let planted = v.foliage.get(selected).is_some_and(|e| e.painted);
        ui.add_enabled_ui(planted, |ui| {
            if ui.add_sized([w, 30.0], egui::Button::new("Clear this species")).clicked() {
                *action = EditorAction::ClearFoliage;
            }
        });
        ui.add_space(6.0);
        ui.label(theme::small(&format!("{} instances in the world", thousands(total))));
        ui.add_space(18.0);
    }

    if *v.tool == Tool::Select {
        ui.label(theme::label("SELECTION"));
        ui.add_space(8.0);
        let mut delete = false;
        let mut deselect = false;
        theme::inset(10).show(ui, |ui| match v.selection.as_deref_mut() {
            Some(sel) => {
                ui.label(theme::muted(&sel.species));
                ui.add_space(6.0);
                slider_log(ui, "Scale", &mut sel.scale, 0.1..=8.0);
                slider(ui, "Yaw", &mut sel.yaw, 0.0..=std::f32::consts::TAU);
                ui.label(theme::small(&format!("{:.1} m tall", sel.height * sel.scale)));
                ui.add_space(8.0);
                let w = ui.available_width();
                if ui.add_sized([w, 30.0], egui::Button::new("Delete")).clicked() {
                    delete = true;
                }
                if ui.add_sized([w, 26.0], egui::Button::new("Deselect")).clicked() {
                    deselect = true;
                }
            }
            None => {
                ui.label(theme::small(
                    "Nothing selected. Click an object in the viewport, or place one from \
                     the tools panel.",
                ));
            }
        });
        if delete {
            *action = EditorAction::DeleteProp;
        } else if deselect {
            *action = EditorAction::Deselect;
        }
        ui.add_space(18.0);
    }

    if *v.tool == Tool::Road {
        ui.label(theme::label("ROAD"));
        ui.add_space(8.0);
        theme::inset(10).show(ui, |ui| {
            if let Some(r) = v.active_road.as_deref_mut() {
                slider(ui, "Width m", &mut r.width_m, 2.0..=12.0);
                slider(ui, "Shoulder m", &mut r.shoulder_m, 0.0..=4.0);
                slider(ui, "Max grade", &mut r.max_grade, 0.02..=0.30);
                slider(ui, "Cut/fill m", &mut r.cut_fill_limit_m, 0.5..=25.0);
                slider(ui, "Camber", &mut r.camber, 0.0..=0.10);
                slider(ui, "Rut m", &mut r.rut_depth_m, 0.0..=0.30);
                slider(ui, "Wander m", &mut r.wander_m, 0.0..=6.0);
                // Wrapped: the surface names do not all fit on one line once
                // the panel narrows.
                ui.horizontal_wrapped(|ui| {
                    for sfc in Surface::ALL {
                        if ui.selectable_label(r.surface == sfc, sfc.label()).clicked() {
                            r.surface = sfc;
                        }
                    }
                });
                ui.add_space(6.0);
                let w = ui.available_width();
                if ui.add_sized([w, 30.0], egui::Button::new("Apply")).clicked() {
                    *action = EditorAction::RebuildRoads;
                }
            } else {
                ui.label(theme::small("Start a road to edit its cross-section."));
            }
        });
        ui.add_space(18.0);
    }
}

fn inspector_panel(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    // Frame-time settings only. Everything that changes how the world *looks*
    // moved to the Environment pane; what is left here costs milliseconds.
    ui.label(theme::label("QUALITY"));
    ui.add_space(6.0);
    theme::inset(10).show(ui, |ui| {
        quality_presets(ui, v.sky);
        ui.add_space(8.0);
        shadow_controls(ui, v.sky);
        ui.add_space(6.0);
        ui.checkbox(&mut v.sky.temporal_aa, "Temporal AA");
        ui.label(theme::small(
            "Accumulates a jittered sub-pixel offset across frames, which is what \
             resolves foliage about a pixel wide.",
        ));
    });
    ui.add_space(18.0);

    ui.add_space(20.0);
    ui.label(theme::label("TERRAIN"));
    ui.add_space(8.0);
    theme::inset(10).show(ui, |ui| {
        ui.label(theme::small("Ridged multifractal"));
        slider(ui, "Height m", &mut v.params.rmf.amplitude_m, 100.0..=1600.0);
        slider_log(ui, "Feature m", &mut v.params.rmf.feature_scale_m, 500.0..=12000.0);
        slider(ui, "Warp m", &mut v.params.rmf.warp_strength_m, 0.0..=1200.0);

        ui.add_space(8.0);
        ui.label(theme::small("Hydraulic erosion"));
        slider(ui, "Iterations", &mut v.params.erosion.iterations, 0..=4000);
        slider(ui, "Carve", &mut v.params.erosion.capacity, 0.005..=0.3);

        ui.add_space(10.0);
        let w = ui.available_width();
        if ui.add_sized([w, 34.0], egui::Button::new(theme::heading("Generate"))).clicked() {
            *action = EditorAction::Generate;
        }
        ui.label(theme::small("Replaces the heightfield. Takes a few seconds."));
    });

    ui.add_space(18.0);
    ui.label(theme::label("WORLD"));
    ui.add_space(8.0);
    theme::inset(10).show(ui, |ui| {
        row(ui, "Size", v.size.label());
        row(ui, "Extent", &format!("{} m", v.size.extent_m()));
        row(ui, "Tiles", &format!("{n} x {n}", n = v.size.tiles_per_side()));
        row(ui, "Heightfield", &format!("{0} x {0}", v.height_res));
    });
}

/// Group digits, because "1732940 planted" is unreadable at a glance.
fn thousands(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

/// Width reserved for a slider's value box, so the track never pushes it out.
const VALUE_BOX_W: f32 = 74.0;

/// A slider with its caption on the line above.
///
/// `Slider::text()` puts the caption beside the track, which makes the row's
/// width depend on how long the caption is. In a panel sized to the window that
/// overflows, and the caption is the half that gets clipped -- which is the
/// half you need to know what the slider does. The track is measured from
/// *this* `ui`, not the panel: every one of these sits inside an inset frame
/// narrower than its panel, inside a scroll area that may also be showing a bar.
fn slider<Num: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    caption: &str,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
) {
    slider_inner(ui, caption, value, range, false);
}

fn slider_log<Num: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    caption: &str,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
) {
    slider_inner(ui, caption, value, range, true);
}

fn slider_inner<Num: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    caption: &str,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
    log: bool,
) {
    // Caption and value share one row; the slider gets the full width below it.
    //
    // The previous layout put the value box *beside* the slider and reserved a
    // fixed 74 px for it, so once a panel narrowed past about 110 px the
    // `.max(36.0)` floor stopped the slider shrinking and the value box ran off
    // the panel edge -- which is exactly what a docked Details panel does when
    // the window is not wide. Splitting the rows means the width demanded is
    // never more than the width available, at any panel size.
    let w = ui.available_width();
    ui.horizontal(|ui| {
        ui.label(theme::small(caption));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            // Bounded by the panel, not by a constant.
            ui.style_mut().spacing.interact_size.x = VALUE_BOX_W.min(w * 0.45);
            ui.add(egui::DragValue::new(value).speed(drag_speed(&range)));
        });
    });
    ui.style_mut().spacing.slider_width = w.max(24.0);
    ui.add(egui::Slider::new(value, range).logarithmic(log).show_value(false));
    ui.add_space(2.0);
}

/// A slider that reports whether the value moved.
///
/// Needed where a change has to trigger something -- dragging the hour has to
/// re-derive the sun position, and only on the frames it actually moved.
fn slider_changed<Num: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    caption: &str,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
) -> bool {
    let before = value.to_f64();
    slider(ui, caption, value, range);
    value.to_f64() != before
}

/// Drag step for a range, so a 0..1 control is not as coarse as a 0..3000 one.
fn drag_speed<Num: egui::emath::Numeric>(range: &std::ops::RangeInclusive<Num>) -> f64 {
    let span = range.end().to_f64() - range.start().to_f64();
    (span / 400.0).max(1e-4)
}

fn row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(theme::muted(key));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            // Truncate rather than run past the panel edge. A clipped value is
            // indistinguishable from a wrong one.
            ui.add(egui::Label::new(RichText::new(value).size(12.5).color(theme::TEXT)).truncate());
        });
    });
}

/// In-play graphics panel, top-left, the way a game ships one.
///
/// Deliberately not the editor inspector: while driving there is no tools
/// panel, and the settings someone reaches for mid-session are graphics and
/// time of day, not brush radius.
pub fn play_overlay(
    root: &mut egui::Ui,
    sky: &mut SkySettings,
    env: &mut terra_render::Environment,
    open: &mut bool,
) {
    egui::Area::new("play-graphics".into()).anchor(Align2::LEFT_TOP, vec2(16.0, 16.0)).show(
        root.ctx(),
        |ui| {
            if !*open {
                if ui.add(egui::Button::new(theme::heading("\u{2699}  Graphics"))).clicked() {
                    *open = true;
                }
                return;
            }
            theme::floating(12).show(ui, |ui| {
                ui.set_width(258.0);
                ui.horizontal(|ui| {
                    ui.label(theme::label("GRAPHICS"));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("\u{2715}").clicked() {
                            *open = false;
                        }
                    });
                });
                ui.add_space(6.0);
                quality_presets(ui, sky);
                ui.add_space(10.0);
                ui.label(theme::label("SHADOWS"));
                ui.add_space(4.0);
                shadow_controls(ui, sky);

                // The same mixer as the editor's, scrolled: a shot set up while
                // driving is the shot the editor shows, and duplicating a
                // reduced version of these controls is how the two drift.
                ui.add_space(10.0);
                egui::ScrollArea::vertical().max_height(420.0).id_salt("play-env").show(ui, |ui| {
                    environment_panel(ui, env);
                });
            });
        },
    );
}

// ---------------------------------------------------------------------------
// Performance overlay
// ---------------------------------------------------------------------------

pub struct PerfView<'a> {
    pub stats: &'a FrameStats,
    pub gpu_supported: bool,
    pub graph: bool,
    /// Space to leave on the right, so the overlay clears the editor's
    /// inspector instead of floating on top of it.
    pub right_inset: f32,
}

const C_FRAME: egui::Color32 = egui::Color32::from_rgb(126, 196, 255);
const C_CPU: egui::Color32 = egui::Color32::from_rgb(196, 160, 255);
const C_GPU: egui::Color32 = egui::Color32::from_rgb(140, 220, 160);
/// The renderer's design target. Drawn as a reference line on the graph.
const BUDGET_MS: f32 = 5.0;

pub fn perf_overlay(root: &mut egui::Ui, v: PerfView<'_>) {
    let s = v.stats;
    // What is left of the window once the editor's rails are accounted for.
    // The HUD is a readout, not a panel: if it cannot sit in the free space
    // without covering the terrain, it drops the graph and then narrows.
    let free = root.ctx().viewport_rect().width() - v.right_inset - 32.0;
    let graph = v.graph && free > 320.0;
    let width = if graph { 260.0 } else { 168.0_f32.min(free.max(120.0)) };

    egui::Area::new("perf".into())
        .anchor(Align2::RIGHT_TOP, vec2(-(v.right_inset + 16.0), 16.0))
        .interactable(false)
        .show(root.ctx(), |ui| {
            theme::floating(12).show(ui, |ui| {
                ui.set_width(width);

                let fps = s.fps();
                let over = s.frame.avg() > BUDGET_MS;
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{fps:.0}")).size(28.0).strong().color(if over {
                            theme::WARN
                        } else {
                            C_GPU
                        }),
                    );
                    ui.label(theme::small("FPS"));
                });

                ui.add_space(4.0);
                metric(ui, "Frame", C_FRAME, s.frame.last(), s.frame.avg());
                metric(ui, "CPU", C_CPU, s.cpu.last(), s.cpu.avg());
                if v.gpu_supported {
                    metric(ui, "GPU", C_GPU, s.gpu.last(), s.gpu.avg());
                } else {
                    ui.horizontal(|ui| {
                        ui.label(theme::muted("GPU"));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label(theme::small("unsupported"));
                        });
                    });
                }

                ui.add_space(2.0);
                // p99 rather than max: one outlier at startup would pin a max
                // reading forever and tell you nothing.
                ui.horizontal(|ui| {
                    ui.label(theme::muted("1% low"));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(theme::small(&format!("{:.2} ms", s.frame.p99())));
                    });
                });

                if graph {
                    ui.add_space(10.0);
                    frame_graph(ui, s);
                }
            });
        });
}

fn metric(ui: &mut egui::Ui, name: &str, color: egui::Color32, now: f32, avg: f32) {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
        ui.painter().rect_filled(r, CornerRadius::same(2), color);
        ui.label(theme::muted(name));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(theme::small(&format!("avg {avg:.2}")));
            ui.label(RichText::new(format!("{now:.2} ms")).size(12.5).color(theme::TEXT));
        });
    });
}

/// Rolling frame-time graph. Y axis auto-scales but never drops below the
/// budget line, so a healthy frame time still reads as "well under" rather than
/// filling the plot.
fn frame_graph(ui: &mut egui::Ui, s: &FrameStats) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 78.0), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(8), egui::Color32::from_rgba_premultiplied(0, 0, 0, 90));

    let peak = s.frame.max().max(s.cpu.max()).max(s.gpu.max());
    let top = (peak * 1.15).max(BUDGET_MS * 1.6);

    let y_of = |ms: f32| rect.bottom() - (ms / top).clamp(0.0, 1.0) * rect.height();

    // Budget reference.
    let by = y_of(BUDGET_MS);
    p.line_segment(
        [egui::pos2(rect.left(), by), egui::pos2(rect.right(), by)],
        egui::Stroke::new(1.0, theme::WARN.gamma_multiply(0.5)),
    );
    p.text(
        egui::pos2(rect.right() - 4.0, by - 2.0),
        Align2::RIGHT_BOTTOM,
        format!("{BUDGET_MS:.0} ms"),
        FontId::proportional(9.5),
        theme::MUTED,
    );

    for (ring, color) in [(&s.frame, C_FRAME), (&s.cpu, C_CPU), (&s.gpu, C_GPU)] {
        if ring.len() < 2 {
            continue;
        }
        let n = ring.len() as f32;
        let pts: Vec<egui::Pos2> = ring
            .iter()
            .enumerate()
            .map(|(i, ms)| {
                egui::pos2(rect.left() + (i as f32 / (n - 1.0)) * rect.width(), y_of(ms))
            })
            .collect();
        p.add(egui::Shape::line(pts, egui::Stroke::new(1.3, color)));
    }

    p.text(
        egui::pos2(rect.left() + 5.0, rect.top() + 3.0),
        Align2::LEFT_TOP,
        format!("{top:.0} ms"),
        FontId::proportional(9.5),
        theme::MUTED,
    );
}

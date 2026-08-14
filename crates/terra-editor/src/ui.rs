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
use terra_render::grass::{GrassSettings, GrassStyle};
use terra_render::lighting::{Quality, ShadowQuality, SkySettings};
use terra_render::stats::FrameStats;
use terra_render::terrain::SculptMode;
use terra_render::volumetrics::FogSettings;

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
            if ui.add_sized([w, FIELD_H], text_field(&mut form.seed_text)).changed() {
                if let Ok(v) = form.seed_text.trim().parse::<u64>() {
                    form.seed = v;
                }
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
    /// Sun and graphics settings, edited in place.
    pub sky: &'a mut SkySettings,
    pub grass: &'a mut GrassSettings,
    pub fog: &'a mut FogSettings,
    /// The road being drawn, if any, plus how many roads exist.
    pub active_road: Option<&'a mut Road>,
    pub road_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

pub fn editor(root: &mut egui::Ui, mut v: EditorView<'_>) -> EditorAction {
    let mut action = EditorAction::None;
    let (tools_w, inspector_w) = theme::editor_panels(root.ctx());
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

    // --- tools ---
    if *v.tools_open {
        egui::Panel::left("tools")
            .exact_size(tools_w)
            .frame(theme::panel(if tools_w < 168.0 { 10 } else { 14 }))
            .show(root, |ui| {
                let collapse = collapse_row(ui, "\u{2039}", "TOOLS");
                // Scrolls, because the tool list plus its mode buttons is taller
                // than the panel on a short window and the overflow was simply
                // clipped -- with no indication that anything was missing.
                egui::ScrollArea::vertical().id_salt("tools-scroll").auto_shrink([false; 2]).show(
                    ui,
                    |ui| {
                        tools_panel(ui, &mut v, &mut action);
                    },
                );
                if collapse {
                    *v.tools_open = false;
                }
            });
    } else {
        egui::Panel::left("tools-collapsed")
            .exact_size(RAIL_COLLAPSED_W)
            .frame(theme::panel(4))
            .show(root, |ui| {
                if reopen_strip(ui, "\u{203A}", "Show tools") {
                    *v.tools_open = true;
                }
            });
    }

    // --- inspector ---
    if *v.inspector_open {
        egui::Panel::right("inspector")
            .exact_size(inspector_w)
            .frame(theme::panel(if inspector_w < 220.0 { 10 } else { 14 }))
            .show(root, |ui| {
                let collapse = collapse_row(ui, "\u{203A}", "INSPECTOR");
                egui::ScrollArea::vertical()
                    .id_salt("inspector-scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        inspector_panel(ui, &mut v, &mut action);
                    });
                if collapse {
                    *v.inspector_open = false;
                }
            });
    } else {
        egui::Panel::right("inspector-collapsed")
            .exact_size(RAIL_COLLAPSED_W)
            .frame(theme::panel(4))
            .show(root, |ui| {
                if reopen_strip(ui, "\u{2039}", "Show inspector") {
                    *v.inspector_open = true;
                }
            });
    }

    action
}

/// Width of a collapsed side panel: just enough for the reopen arrow.
pub const RAIL_COLLAPSED_W: f32 = 26.0;

/// A vertical strip that reopens a collapsed panel.
fn reopen_strip(ui: &mut egui::Ui, arrow: &str, tip: &str) -> bool {
    let (rect, resp) =
        ui.allocate_exact_size(vec2(ui.available_width(), ui.available_height()), Sense::click());
    if ui.is_rect_visible(rect) {
        let color = if resp.hovered() { theme::ACCENT } else { theme::MUTED };
        ui.painter().text(
            egui::pos2(rect.center().x, rect.top() + 14.0),
            Align2::CENTER_CENTER,
            arrow,
            FontId::proportional(15.0),
            color,
        );
    }
    resp.on_hover_text(tip).clicked()
}

/// The collapse control at the top of an expanded panel.
fn collapse_row(ui: &mut egui::Ui, arrow: &str, title: &str) -> bool {
    let mut hit = false;
    ui.horizontal(|ui| {
        ui.label(theme::label(title));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let (rect, resp) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::click());
            if ui.is_rect_visible(rect) {
                let color = if resp.hovered() { theme::ACCENT } else { theme::MUTED };
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    arrow,
                    FontId::proportional(14.0),
                    color,
                );
            }
            hit = resp.on_hover_text("Collapse").clicked();
        });
    });
    ui.add_space(4.0);
    hit
}

/// Panel heights, sized to the tallest control each one holds rather than to
/// its text. The toolbar was 46 px, which after 8 px of frame padding on both
/// sides left 30 for a button that wants 35 -- so the row was clipped.
const TOOLBAR_H: f32 = 54.0;
const STATUS_H: f32 = 32.0;

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
             \u{2022}   WASD move, Q/E down/up   \u{2022}   Shift boost"
        }
        (false, Tool::Sculpt, false) => {
            "LMB sculpt   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   [ ] brush size"
        }
        (false, Tool::Select, false) => {
            "LMB pick, drag to move   \u{2022}   Del remove   \u{2022}   MMB pan   \
             \u{2022}   Wheel zoom   \u{2022}   RMB look"
        }
        (false, Tool::Foliage, false) => {
            "LMB plant   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   [ ] brush size"
        }
        (false, Tool::Paint, false) => {
            "LMB paint   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \u{2022}   RMB look   \
             \u{2022}   WASD move, Q/E down/up   \u{2022}   [ ] brush size"
        }
        (false, Tool::Road, false) => {
            "LMB drag to draw the track   \u{2022}   MMB pan   \u{2022}   Wheel zoom   \
             \u{2022}   RMB look   \u{2022}   WASD move, Q/E down/up"
        }
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

    match *v.tool {
        Tool::Sculpt => {
            ui.add_space(20.0);
            ui.label(theme::label("MODE"));
            ui.add_space(6.0);
            for m in SculptMode::ALL {
                let w = ui.available_width();
                if ui
                    .add_sized([w, 32.0], egui::Button::selectable(*v.mode == m, m.label()))
                    .clicked()
                {
                    *v.mode = m;
                }
            }
        }
        Tool::Camera => {
            ui.add_space(20.0);
            ui.label(theme::label("NAVIGATE"));
            ui.add_space(8.0);
            ui.label(theme::muted(
                "LMB drag   orbit\nMMB drag   pan\nWheel      zoom\nRMB drag   look\n\nWASD move, Q/E down/up\nShift      faster",
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
                ui.label(theme::small(
                    "No materials found. Put each texture set in its own folder under \
                     assets/texture and restart.",
                ));
            } else {
                palette_grid(ui, v.palette, v.selected_layer);
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
                ui.label(theme::small("No species available."));
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
}

/// Grass. Shared between the editor inspector and the in-play panel, because
/// it is both an authoring choice and the largest performance dial there is.
pub fn grass_controls(ui: &mut egui::Ui, grass: &mut GrassSettings) {
    ui.checkbox(&mut grass.enabled, "Grass");
    ui.add_enabled_ui(grass.enabled, |ui| {
        ui.horizontal(|ui| {
            for st in GrassStyle::ALL {
                if ui.selectable_label(grass.style == st, st.label()).clicked() && grass.style != st
                {
                    grass.style = st;
                    // A style carries its own height and density; picking one
                    // and keeping meadow numbers gives a mown field of hay.
                    let (h, d, dd) = st.suggested();
                    grass.height_m = h;
                    grass.density = d;
                    grass.draw_distance = dd;
                }
            }
        });
        ui.label(theme::small(match grass.style {
            GrassStyle::Lawn => {
                "Short, even, upright, with cut tips. Needs far more blades to close \
                 the ground, and is invisible past about twenty metres."
            }
            GrassStyle::Field => {
                "Grown, not cut. Uneven lengths, leaning enough to catch light along \
                 the blade, still short enough to read as ground."
            }
            GrassStyle::Meadow => {
                "Tall, uneven, leaning, tapering to a point. Fewer blades cover more, \
                 and it reads much further out."
            }
        }));
        ui.add_space(6.0);
        slider_log(ui, "Blades per m2", &mut grass.density, 200.0..=8000.0);
        slider_log(ui, "Blade height m", &mut grass.height_m, 0.02..=2.0);
        slider_log(ui, "Draw distance m", &mut grass.draw_distance, 10.0..=160.0);
        ui.label(theme::small(
            "Density in the near field. Beyond a few metres the field thins as the \
             square of distance, which holds the blades per pixel steady rather than \
             the blades per square metre -- otherwise nearly the whole budget goes to \
             blades landing inside one pixel.",
        ));
        ui.add_space(4.0);
        slider(ui, "Fade start", &mut grass.fade_start, 0.1..=0.9);
        ui.label(theme::small(
            "Where the dissolve begins, as a fraction of the draw distance. A short \
             band concentrates the dither into a visible ring; a wide one hides it.",
        ));
        ui.add_space(4.0);
        slider(ui, "Wind strength", &mut grass.wind_strength, 0.0..=0.8);
        slider(ui, "Wind speed", &mut grass.wind_speed, 0.0..=4.0);
        ui.label(theme::small(
            "Grass grows on whatever the grass material is painted on. Paint a \
             different material to clear a path through it.",
        ));
    });
}

/// Volumetric fog. Shared between the editor and the in-play panel.
pub fn fog_controls(ui: &mut egui::Ui, fog: &mut FogSettings) {
    ui.checkbox(&mut fog.enabled, "Volumetric fog");
    ui.add_enabled_ui(fog.enabled, |ui| {
        // Extinction per metre. The useful band is narrow and near zero: past
        // about 0.002 a 450 m view is more fog than scene.
        slider_log(ui, "Haze density", &mut fog.density, 0.00005..=0.02);
        slider(ui, "Valley mist", &mut fog.mist_strength, 0.0..=0.01);
        slider(ui, "Mist base height m", &mut fog.mist_base, -100.0..=1200.0);
        slider_log(ui, "Mist falloff", &mut fog.mist_falloff, 0.002..=0.2);
        ui.label(theme::small(
            "Mist thickens below the base height and thins above it, so it pools in \
             valleys and clears off ridges.",
        ));
        ui.add_space(4.0);
        slider(ui, "Forward scattering", &mut fog.anisotropy, 0.0..=0.95);
        ui.label(theme::small(
            "How much the air throws light forward. Near 0 the fog is evenly bright; \
             high values make looking toward the sun glow and away from it stay flat.",
        ));
        ui.add_space(4.0);
        slider_log(ui, "Fog distance m", &mut fog.distance, 100.0..=2000.0);
    });
}

/// Time of day, and the graphics dials that go with it.
///
/// Shared between the editor's inspector and the in-play overlay: the same
/// settings, so a scene lit one way in the editor is lit that way when driven.
pub fn sky_controls(ui: &mut egui::Ui, sky: &mut SkySettings, compact: bool) {
    // `compact` is the in-play panel, where the world's sun is always what you
    // are looking at. In the editor it is a choice.
    if !compact {
        ui.checkbox(&mut sky.editor_preview, "Preview time of day");
        ui.label(theme::small(if sky.editor_preview {
            "The viewport is showing the world's own sun. The cycle runs here too."
        } else {
            "The viewport uses a fixed neutral sun. These settings still apply when you \
             press Play."
        }));
        ui.add_space(8.0);
    }
    let hour = sky.time_of_day;
    ui.label(theme::small(&format!(
        "Time  {:02}:{:02}",
        hour.floor() as u32,
        ((hour.fract()) * 60.0) as u32
    )));
    let w = ui.available_width();
    ui.style_mut().spacing.slider_width = (w - VALUE_BOX_W).max(36.0);
    ui.add(egui::Slider::new(&mut sky.time_of_day, 0.0..=24.0).show_value(false));

    ui.horizontal(|ui| {
        // The presets are what anyone actually reaches for; the slider is for
        // the shot between them.
        for (label, t) in [("Dawn", 6.4), ("Noon", 12.0), ("Dusk", 18.0), ("Night", 1.0)] {
            if ui.small_button(label).clicked() {
                sky.time_of_day = t;
            }
        }
    });
    ui.add_space(4.0);
    ui.checkbox(&mut sky.cycle_running, "Run day/night cycle");
    if sky.cycle_running {
        slider_log(ui, "Hours per second", &mut sky.day_speed, 0.02..=6.0);
    }
    if !compact && sky.cycle_running && !sky.editor_preview {
        // The clock only advances where the sun is being shown, so saying the
        // cycle is "running" here without this reads as a broken setting.
        ui.label(theme::small("The clock advances in Play, or with preview on."));
    }

    if compact {
        return;
    }
    ui.add_space(10.0);
    ui.label(theme::label("SHADOWS"));
    ui.add_space(4.0);
    shadow_controls(ui, sky);

    ui.add_space(10.0);
    ui.label(theme::label("ATMOSPHERE"));
    ui.add_space(4.0);
    atmosphere_controls(ui, sky);
}

/// God rays, haze and exposure. Shared with the in-play panel so a shot set up
/// in the editor is the shot that gets driven.
pub fn atmosphere_controls(ui: &mut egui::Ui, sky: &mut SkySettings) {
    ui.checkbox(&mut sky.temporal_aa, "Temporal AA");
    ui.label(theme::small(
        "Accumulates a jittered sub-pixel offset across frames. It is what \
         resolves grass blades about a pixel wide, and what turns the grass \
         dissolve from a stipple into a fade.",
    ));
    ui.add_space(6.0);
    slider(ui, "Screen-space rays", &mut sky.god_rays, 0.0..=1.5);
    ui.label(theme::small(
        "A screen-space bloom along the sun's direction. The volumetric fog now \
         produces true shafts in 3D; this only adds glare around the sun itself, \
         and only when it is in frame.",
    ));
    slider(ui, "Haze", &mut sky.haze, 0.0..=2.5);
    slider(ui, "Exposure", &mut sky.exposure, 0.4..=2.0);
}

/// The one control most people will touch. Applying a preset writes through to
/// every individual setting, which stay editable afterwards -- picking a preset
/// is a starting point, not a mode.
pub fn quality_presets(
    ui: &mut egui::Ui,
    sky: &mut SkySettings,
    grass: &mut GrassSettings,
    fog: &mut FogSettings,
) {
    ui.horizontal_wrapped(|ui| {
        for q in Quality::ALL {
            if ui.button(q.label()).clicked() {
                let (shadows, distance, rays, taa) = q.sky();
                sky.shadow_quality = shadows;
                sky.shadow_distance = distance;
                sky.god_rays = rays;
                sky.temporal_aa = taa;
                let (on, density, draw) = q.grass();
                grass.enabled = on;
                grass.density = density;
                grass.draw_distance = draw;
                let (fog_on, fog_distance) = q.fog();
                fog.enabled = fog_on;
                fog.distance = fog_distance;
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
fn palette_grid(ui: &mut egui::Ui, palette: &[PaletteEntry<'_>], selected: &mut u32) {
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
                resp.on_hover_text(format!("{}  ({})", entry.name, entry.role));
            }
        });
    }

    if let Some(entry) = palette.get(*selected as usize) {
        ui.add_space(4.0);
        ui.label(theme::small(&format!("{}  \u{2022}  {}", entry.name, entry.role)));
    }
}

fn inspector_panel(ui: &mut egui::Ui, v: &mut EditorView<'_>, action: &mut EditorAction) {
    ui.label(theme::label("QUALITY"));
    ui.add_space(6.0);
    quality_presets(ui, v.sky, v.grass, v.fog);
    ui.add_space(18.0);

    ui.label(theme::label("GRASS"));
    ui.add_space(8.0);
    theme::inset(10).show(ui, |ui| {
        grass_controls(ui, v.grass);
    });
    ui.add_space(18.0);

    ui.label(theme::label("FOG"));
    ui.add_space(8.0);
    theme::inset(10).show(ui, |ui| {
        fog_controls(ui, v.fog);
    });
    ui.add_space(18.0);

    ui.label(theme::label("ENVIRONMENT"));
    ui.add_space(8.0);
    theme::inset(10).show(ui, |ui| {
        sky_controls(ui, v.sky, false);
    });
    ui.add_space(18.0);

    ui.label(theme::label("BRUSH"));
    ui.add_space(8.0);
    theme::inset(10).show(ui, |ui| {
        slider(ui, "Radius m", v.radius, 8.0..=800.0);
        slider(ui, "Strength", v.strength, 0.05..=8.0);
    });

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
                slider_log(ui, "Per hectare", &mut r.density, 1.0..=600.0);
                slider(ui, "Scale min", &mut r.scale_min, 0.1..=3.0);
                slider(ui, "Scale max", &mut r.scale_max, 0.1..=4.0);
                // Keeping these ordered here rather than validating later means
                // the generator never has to deal with an inverted range.
                if r.scale_max < r.scale_min {
                    r.scale_max = r.scale_min;
                }
                slider(ui, "Lean to slope", &mut r.align_to_normal, 0.0..=1.0);
                ui.label(theme::small(
                    "0 stands every instance upright, 1 lays it flat against the hillside. \
                     Trees want a little, rocks want a lot.",
                ));
                ui.add_space(6.0);
                slider(ui, "Max slope", &mut r.slope_max, 0.05..=1.0);
                slider(ui, "Min height m", &mut r.altitude_min, -200.0..=2500.0);
                slider(ui, "Max height m", &mut r.altitude_max, -200.0..=2500.0);
                if r.altitude_max < r.altitude_min {
                    r.altitude_max = r.altitude_min;
                }
                slider(ui, "Bed into ground", &mut r.sink, 0.0..=0.6);
                ui.add_space(6.0);
                slider_log(ui, "Draw distance m", &mut r.cull_distance, 100.0..=3000.0);
                ui.label(theme::small(
                    "Beyond this, instances are not drawn at all. The main cost control: \
                     small props can be cut short, a treeline cannot.",
                ));
            } else {
                ui.label(theme::small("No species selected."));
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
        if i > 0 && (s.len() - i) % 3 == 0 {
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
    ui.label(theme::small(caption));
    let w = ui.available_width();
    ui.style_mut().spacing.slider_width = (w - VALUE_BOX_W).max(36.0);
    ui.add(egui::Slider::new(value, range).logarithmic(log));
    ui.add_space(2.0);
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
    grass: &mut GrassSettings,
    fog: &mut FogSettings,
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
                quality_presets(ui, sky, grass, fog);
                ui.add_space(10.0);
                sky_controls(ui, sky, true);

                ui.add_space(10.0);
                ui.label(theme::label("SHADOWS"));
                ui.add_space(4.0);
                shadow_controls(ui, sky);

                ui.add_space(10.0);
                ui.label(theme::label("GRASS"));
                ui.add_space(4.0);
                grass_controls(ui, grass);

                ui.add_space(10.0);
                ui.label(theme::label("FOG"));
                ui.add_space(4.0);
                fog_controls(ui, fog);

                ui.add_space(10.0);
                ui.label(theme::label("ATMOSPHERE"));
                ui.add_space(4.0);
                atmosphere_controls(ui, sky);
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

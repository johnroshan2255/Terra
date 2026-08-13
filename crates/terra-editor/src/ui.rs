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
use egui::{Align, Align2, CornerRadius, FontId, Layout, Margin, Rect, RichText, Sense, vec2};
use std::path::PathBuf;
use terra_core::WorldSize;
use terra_project::roads::{Road, Surface};
use terra_project::{Library, TerrainParams};
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
    theme::paint_rail_scrim(root.ctx());
    egui::Panel::left("rail")
        .exact_size(theme::RAIL_WIDTH)
        .resizable(false)
        .frame(egui::Frame::NONE.inner_margin(Margin { left: 46, right: 38, top: 42, bottom: 30 }))
        .show(root, |ui| add(ui));
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

            egui::ScrollArea::vertical().show(ui, |ui| {
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
            });
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
        ui.add_sized([w, 34.0], egui::TextEdit::singleline(&mut form.name).hint_text("World name"));

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
            if ui.text_edit_singleline(&mut form.seed_text).changed() {
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
        ui.add(egui::Slider::new(v.camera_speed, 10.0..=600.0).text("Speed m/s"));
        ui.add(egui::Slider::new(v.fov_deg, 40.0..=100.0).text("Field of view"));
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

    egui::Area::new("loading".into())
        .fixed_pos(egui::pos2(screen.left() + 92.0, screen.bottom() - 190.0))
        .show(root.ctx(), |ui| {
            ui.set_width((screen.width() * 0.42).clamp(320.0, 560.0));
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
    Sculpt,
    Road,
}

impl Tool {
    pub const ALL: [Tool; 2] = [Tool::Sculpt, Tool::Road];
    pub fn label(self) -> &'static str {
        match self {
            Tool::Sculpt => "Sculpt",
            Tool::Road => "Road",
        }
    }
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
    /// Re-stamp all roads onto the base terrain.
    RebuildRoads,
    /// Build the physics world and drop a car into it.
    Play,
    /// Leave drive mode and return to editing.
    Stop,
}

pub fn editor(root: &mut egui::Ui, mut v: EditorView<'_>) -> EditorAction {
    let mut action = EditorAction::None;

    // --- toolbar ---
    egui::Panel::top("toolbar").exact_size(46.0).frame(theme::panel(8)).show(root, |ui| {
        ui.horizontal_centered(|ui| {
            ui.add_space(6.0);
            ui.label(RichText::new(v.world_name).size(15.0).strong());
            ui.label(theme::small(v.size.label()));
            if v.unsaved {
                ui.label(RichText::new("\u{25CF} unsaved").size(11.5).color(theme::ACCENT));
            }
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
            });
        });
    });

    // --- status bar ---
    egui::Panel::bottom("status").exact_size(30.0).frame(theme::panel(6)).show(root, |ui| {
        ui.horizontal_centered(|ui| {
            ui.add_space(6.0);
            ui.label(theme::small(if v.playing {
                "W/S throttle   \u{2022}   A/D steer   \u{2022}   Q brake   \u{2022}   \
                 Shift handbrake   \u{2022}   Esc stop"
            } else {
                match *v.tool {
                    Tool::Sculpt => {
                        "LMB sculpt   \u{2022}   RMB look   \u{2022}   WASD move, Q/E down/up   \
                     \u{2022}   Shift boost   \u{2022}   [ ] brush size"
                    }
                    Tool::Road => {
                        "LMB drag to draw the track   \u{2022}   RMB look   \u{2022}   \
                     WASD move, Q/E down/up   \u{2022}   Shift boost"
                    }
                }
            }));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                match v.brush_at {
                    Some((x, z)) => ui.label(theme::small(&format!("{x:.0} m,  {z:.0} m"))),
                    None => ui.label(theme::small("--")),
                };
            });
        });
    });

    // --- tools ---
    egui::Panel::left("tools").exact_size(196.0).frame(theme::panel(14)).show(root, |ui| {
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
            if ui.add_sized([w, 32.0], egui::Button::selectable(*v.tool == t, t.label())).clicked()
            {
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
                            action = EditorAction::UndoPoint;
                        }
                        if ui.add_sized([w, 30.0], egui::Button::new("Finish road")).clicked() {
                            action = EditorAction::FinishRoad;
                        }
                    }
                    None => {
                        ui.label(theme::muted(&format!("{} in this world", v.road_count)));
                        ui.add_space(6.0);
                        if ui
                            .add_sized([w, 34.0], egui::Button::new(theme::heading("New road")))
                            .clicked()
                        {
                            action = EditorAction::NewRoad;
                        }
                    }
                }
                if v.road_count > 0 {
                    ui.add_space(6.0);
                    if ui.add_sized([w, 28.0], egui::Button::new("Clear all roads")).clicked() {
                        action = EditorAction::ClearRoads;
                    }
                }
            }
        }
    });

    // --- inspector ---
    egui::Panel::right("inspector").exact_size(268.0).frame(theme::panel(14)).show(root, |ui| {
        ui.label(theme::label("BRUSH"));
        ui.add_space(8.0);
        theme::inset(10).show(ui, |ui| {
            ui.add(egui::Slider::new(v.radius, 8.0..=800.0).text("Radius m"));
            ui.add(egui::Slider::new(v.strength, 0.05..=8.0).text("Strength"));
        });

        if *v.tool == Tool::Road {
            ui.label(theme::label("ROAD"));
            ui.add_space(8.0);
            theme::inset(10).show(ui, |ui| {
                if let Some(r) = v.active_road.as_deref_mut() {
                    ui.add(egui::Slider::new(&mut r.width_m, 2.0..=12.0).text("Width m"));
                    ui.add(egui::Slider::new(&mut r.shoulder_m, 0.0..=4.0).text("Shoulder m"));
                    ui.add(egui::Slider::new(&mut r.max_grade, 0.02..=0.30).text("Max grade"));
                    ui.add(
                        egui::Slider::new(&mut r.cut_fill_limit_m, 0.5..=25.0).text("Cut/fill m"),
                    );
                    ui.add(egui::Slider::new(&mut r.camber, 0.0..=0.10).text("Camber"));
                    ui.add(egui::Slider::new(&mut r.rut_depth_m, 0.0..=0.30).text("Rut m"));
                    ui.add(egui::Slider::new(&mut r.wander_m, 0.0..=6.0).text("Wander m"));
                    ui.horizontal(|ui| {
                        for sfc in Surface::ALL {
                            if ui.selectable_label(r.surface == sfc, sfc.label()).clicked() {
                                r.surface = sfc;
                            }
                        }
                    });
                    ui.add_space(6.0);
                    let w = ui.available_width();
                    if ui.add_sized([w, 30.0], egui::Button::new("Apply")).clicked() {
                        action = EditorAction::RebuildRoads;
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
            ui.add(
                egui::Slider::new(&mut v.params.rmf.amplitude_m, 100.0..=1600.0).text("Height m"),
            );
            ui.add(
                egui::Slider::new(&mut v.params.rmf.feature_scale_m, 500.0..=12000.0)
                    .logarithmic(true)
                    .text("Feature m"),
            );
            ui.add(
                egui::Slider::new(&mut v.params.rmf.warp_strength_m, 0.0..=1200.0).text("Warp m"),
            );

            ui.add_space(8.0);
            ui.label(theme::small("Hydraulic erosion"));
            ui.add(
                egui::Slider::new(&mut v.params.erosion.iterations, 0..=4000).text("Iterations"),
            );
            ui.add(egui::Slider::new(&mut v.params.erosion.capacity, 0.005..=0.3).text("Carve"));

            ui.add_space(10.0);
            let w = ui.available_width();
            if ui.add_sized([w, 34.0], egui::Button::new(theme::heading("Generate"))).clicked() {
                action = EditorAction::Generate;
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
    });

    action
}

fn row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(theme::muted(key));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(value).size(12.5).color(theme::TEXT));
        });
    });
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
    egui::Area::new("perf".into())
        .anchor(Align2::RIGHT_TOP, vec2(-(v.right_inset + 16.0), 16.0))
        .interactable(false)
        .show(root.ctx(), |ui| {
            theme::floating(12).show(ui, |ui| {
                ui.set_width(if v.graph { 260.0 } else { 168.0 });

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

                if v.graph {
                    ui.add_space(10.0);
                    graph(ui, s);
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
fn graph(ui: &mut egui::Ui, s: &FrameStats) {
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

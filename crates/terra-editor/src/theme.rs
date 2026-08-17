//! Visual style.
//!
//! Menus are **not** modals. They live in a left rail drawn directly over the
//! rendered scene, with no card and no backing panel -- the only background is
//! a gradient scrim that fades to nothing before it reaches the middle of the
//! screen, so the landscape stays fully visible.
//!
//! Editor chrome is the exception: docked panels there do carry a fill, because
//! a tool palette you can see terrain through is unusable.

use egui::{Color32, CornerRadius, Margin, Rect, Stroke, Visuals};

pub const ACCENT: Color32 = Color32::from_rgb(126, 196, 255);

/// Selected-row fill. Unreal's selection is a saturated blue that reads as
/// "chosen" against flat grey, where the pale ACCENT used as a fill reads as
/// "disabled and slightly lit".
pub const SELECT: Color32 = Color32::from_rgb(0, 112, 224);
pub const TEXT: Color32 = Color32::from_rgb(240, 244, 250);
pub const MUTED: Color32 = Color32::from_rgb(156, 167, 182);
pub const DANGER: Color32 = Color32::from_rgb(226, 138, 124);
pub const WARN: Color32 = Color32::from_rgb(240, 180, 110);

/// Widest the menu rail is ever drawn, including its margins.
pub const RAIL_WIDTH: f32 = 400.0;
/// Narrowest the rail may shrink to before its contents start to scroll.
pub const RAIL_MIN_WIDTH: f32 = 248.0;

/// Editor panel fill. Opaque enough to read a tool list against terrain.
pub const PANEL: Color32 = Color32::from_rgb(36, 36, 36);
pub const PANEL_SOFT: Color32 = Color32::from_rgb(45, 45, 45);
// White at a low alpha. In *premultiplied* form that is `rgb == a`, not
// `rgb == 255`: with the channels left at 255 the colour is additive, so what
// was meant as a faint film painted as near-solid white and swallowed the label
// sitting on top of it. `from_rgba_unmultiplied` is not const, hence by hand.
/// White, ~15% opacity.
pub const HAIRLINE: Color32 = Color32::from_rgb(24, 24, 24);
/// White, ~13% opacity.
pub const HOVER: Color32 = Color32::from_rgb(58, 58, 58);

pub fn apply(ctx: &egui::Context) {
    let mut v = Visuals::dark();

    v.panel_fill = Color32::TRANSPARENT;
    v.window_fill = PANEL;
    v.window_corner_radius = CornerRadius::ZERO;
    v.window_stroke = Stroke::new(1.0, HAIRLINE);
    v.override_text_color = Some(TEXT);
    v.extreme_bg_color = Color32::from_rgba_premultiplied(8, 10, 15, 200);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, ACCENT);

    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::ZERO;
    }
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(56, 56, 56);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.hovered.weak_bg_fill = HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(90, 90, 90));
    v.widgets.active.weak_bg_fill = SELECT;
    v.widgets.active.bg_stroke = Stroke::new(1.0, SELECT);
    // Docked panels sit flush against each other, so a drop shadow only muddies
    // the seam between two of them. Unreal does not shadow docked panels either.
    v.window_shadow = egui::Shadow::NONE;
    v.popup_shadow =
        egui::Shadow { offset: [0, 2], blur: 6, spread: 0, color: Color32::from_black_alpha(96) };

    ctx.set_visuals(v);

    for theme in [egui::Theme::Dark, egui::Theme::Light] {
        let mut style = (*ctx.style_of(theme)).clone();
        // Unreal's Details panel is dense. 10 px rows and 14 px button padding
        // gave a third fewer controls per screen for no legibility gain.
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        style.spacing.slider_width = 150.0;
        style.spacing.interact_size.y = 20.0;
        ctx.set_style_of(theme, style);
    }
}

/// Rail width for the current window.
///
/// A fixed 400 px rail is a third of a laptop window and nearly all of a small
/// one, so it tracks the viewport instead and only stops shrinking where the
/// two-line nav rows would start to wrap.
pub fn rail_width(ctx: &egui::Context) -> f32 {
    let w = ctx.viewport_rect().width();
    (w * 0.34).clamp(RAIL_MIN_WIDTH.min(w), RAIL_WIDTH)
}

/// Margins inside the rail, tightened as it narrows so the content keeps a
/// usable width rather than being squeezed between generous gutters.
pub fn rail_margin(rail: f32) -> Margin {
    let t = ((rail - RAIL_MIN_WIDTH) / (RAIL_WIDTH - RAIL_MIN_WIDTH)).clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| (a + (b - a) * t).round() as i8;
    Margin {
        left: lerp(22.0, 46.0),
        right: lerp(18.0, 38.0),
        top: lerp(24.0, 42.0),
        bottom: lerp(20.0, 30.0),
    }
}

/// Widths of the editor's two docked side panels, as `(tools, inspector)`.
///
/// Scaled to the window, and jointly capped: two fixed rails on a small window
/// leave no viewport between them to sculpt in.
pub fn editor_panels(ctx: &egui::Context) -> (f32, f32) {
    let w = ctx.viewport_rect().width();
    let tools = (w * 0.15).clamp(128.0, 196.0);
    let inspector = (w * 0.21).clamp(190.0, 288.0);
    let budget = w * 0.62;
    if tools + inspector > budget {
        let k = budget / (tools + inspector);
        (tools * k, inspector * k)
    } else {
        (tools, inspector)
    }
}

/// Horizontal gradient behind the menu rail, painted on the background layer.
///
/// A solid panel fill would hide the landscape; this fades to fully transparent
/// well before mid-screen, so the scene reads as the background of the app
/// rather than as a picture behind a window.
pub fn paint_rail_scrim(ctx: &egui::Context, rail: f32) {
    let screen = ctx.viewport_rect();
    let width = (rail * 1.8).min(screen.width());
    let rect =
        Rect::from_min_max(screen.left_top(), egui::pos2(screen.left() + width, screen.bottom()));

    let opaque = Color32::from_rgba_premultiplied(6, 8, 13, 226);
    let clear = Color32::from_rgba_premultiplied(0, 0, 0, 0);

    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(rect.left_top(), opaque);
    mesh.colored_vertex(rect.left_bottom(), opaque);
    mesh.colored_vertex(rect.right_top(), clear);
    mesh.colored_vertex(rect.right_bottom(), clear);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(2, 1, 3);

    ctx.layer_painter(egui::LayerId::background()).add(egui::Shape::mesh(mesh));
}

/// Full-screen dim, used by the loading screen.
pub fn paint_dim(ctx: &egui::Context, alpha: u8) {
    ctx.layer_painter(egui::LayerId::background()).rect_filled(
        ctx.viewport_rect(),
        CornerRadius::ZERO,
        Color32::from_rgba_premultiplied(4, 5, 9, alpha),
    );
}

/// Floating overlay, e.g. the performance HUD.
pub fn floating(padding: i8) -> egui::Frame {
    egui::Frame::NONE
        .fill(PANEL)
        .stroke(Stroke::new(1.0, HAIRLINE))
        .inner_margin(Margin::same(padding))
}

/// Docked editor panel frame.
pub fn panel(padding: i8) -> egui::Frame {
    egui::Frame::NONE.fill(PANEL).inner_margin(Margin::same(padding))
}

/// A nested block inside an editor panel.
pub fn inset(padding: i8) -> egui::Frame {
    egui::Frame::NONE.fill(PANEL_SOFT).inner_margin(Margin::same(padding))
}

pub fn display(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(52.0).strong().color(TEXT)
}

pub fn title(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(26.0).strong().color(TEXT)
}

pub fn heading(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(15.0).strong()
}

pub fn label(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(12.0).strong().color(ACCENT)
}

pub fn muted(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(13.0).color(MUTED)
}

pub fn small(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(11.5).color(MUTED)
}

/// Dock chrome, matched to the panel palette.
///
/// Derived from the live egui style rather than written from scratch so the
/// fonts and interaction colours stay in step, then overridden where the
/// defaults clash: `egui_dock` ships a white tab bar and a black separator,
/// which on this dark theme read as two bright seams across the window.
pub fn dock_style(ctx: &egui::Context) -> egui_dock::Style {
    // egui 0.36 dropped `Context::style()` for per-theme lookup; the editor is
    // dark-only, so ask for that one directly.
    let mut s = egui_dock::Style::from_egui(&ctx.style_of(egui::Theme::Dark));
    s.dock_area_padding = None;
    s.main_surface_border_stroke = egui::Stroke::NONE;

    s.tab_bar.bg_fill = PANEL;
    s.tab_bar.height = 26.0;
    s.tab_bar.hline_color = HAIRLINE;
    // Titles sit left at their natural width, as in Unreal. Stretching a lone
    // tab across its whole bar centres the title and makes a single-tab pane
    // look like a window caption rather than a tab.
    s.tab_bar.fill_tab_bar = false;

    // The separator is the resize handle, so it has to be findable without
    // being a visible line. Idle matches the panel; it only lights on hover.
    s.separator.width = 2.0;
    s.separator.extra_interact_width = 4.0;
    s.separator.color_idle = HAIRLINE;
    s.separator.color_hovered = ACCENT;
    s.separator.color_dragged = ACCENT;

    s.tab.active.bg_fill = PANEL;
    s.tab.active.text_color = TEXT;
    s.tab.inactive.bg_fill = PANEL_SOFT;
    s.tab.inactive.text_color = MUTED;
    s.tab.focused.bg_fill = PANEL;
    s.tab.focused.text_color = ACCENT;
    s.tab.hovered.text_color = TEXT;
    s.tab.tab_body.bg_fill = PANEL;

    s.buttons.close_tab_color = MUTED;
    s.buttons.close_tab_active_color = DANGER;
    s
}

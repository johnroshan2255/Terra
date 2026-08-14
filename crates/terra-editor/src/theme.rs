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
pub const TEXT: Color32 = Color32::from_rgb(240, 244, 250);
pub const MUTED: Color32 = Color32::from_rgb(156, 167, 182);
pub const DANGER: Color32 = Color32::from_rgb(226, 138, 124);
pub const WARN: Color32 = Color32::from_rgb(240, 180, 110);

/// Widest the menu rail is ever drawn, including its margins.
pub const RAIL_WIDTH: f32 = 400.0;
/// Narrowest the rail may shrink to before its contents start to scroll.
pub const RAIL_MIN_WIDTH: f32 = 248.0;

/// Editor panel fill. Opaque enough to read a tool list against terrain.
pub const PANEL: Color32 = Color32::from_rgba_premultiplied(17, 20, 27, 232);
pub const PANEL_SOFT: Color32 = Color32::from_rgba_premultiplied(30, 35, 46, 140);
// White at a low alpha. In *premultiplied* form that is `rgb == a`, not
// `rgb == 255`: with the channels left at 255 the colour is additive, so what
// was meant as a faint film painted as near-solid white and swallowed the label
// sitting on top of it. `from_rgba_unmultiplied` is not const, hence by hand.
/// White, ~15% opacity.
pub const HAIRLINE: Color32 = Color32::from_rgba_premultiplied(38, 38, 38, 38);
/// White, ~13% opacity.
pub const HOVER: Color32 = Color32::from_rgba_premultiplied(34, 34, 34, 34);

pub fn apply(ctx: &egui::Context) {
    let mut v = Visuals::dark();

    v.panel_fill = Color32::TRANSPARENT;
    v.window_fill = PANEL;
    v.window_corner_radius = CornerRadius::same(14);
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
        w.corner_radius = CornerRadius::same(9);
    }
    v.widgets.inactive.weak_bg_fill = Color32::from_rgba_premultiplied(48, 56, 72, 150);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgba_premultiplied(70, 84, 108, 200);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
    v.widgets.active.weak_bg_fill = ACCENT.gamma_multiply(0.5);
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);

    ctx.set_visuals(v);

    for theme in [egui::Theme::Dark, egui::Theme::Light] {
        let mut style = (*ctx.style_of(theme)).clone();
        style.spacing.item_spacing = egui::vec2(10.0, 10.0);
        style.spacing.button_padding = egui::vec2(14.0, 9.0);
        style.spacing.slider_width = 150.0;
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
        .corner_radius(CornerRadius::same(14))
        .stroke(Stroke::new(1.0, HAIRLINE))
        .inner_margin(Margin::same(padding))
}

/// Docked editor panel frame.
pub fn panel(padding: i8) -> egui::Frame {
    egui::Frame::NONE.fill(PANEL).inner_margin(Margin::same(padding))
}

/// A nested block inside an editor panel.
pub fn inset(padding: i8) -> egui::Frame {
    egui::Frame::NONE
        .fill(PANEL_SOFT)
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(padding))
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

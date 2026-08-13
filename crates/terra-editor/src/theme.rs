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

/// Width of the menu rail, including its margins.
pub const RAIL_WIDTH: f32 = 400.0;
/// How far the scrim extends past the rail before it is fully transparent.
pub const SCRIM_WIDTH: f32 = 720.0;

/// Editor panel fill. Opaque enough to read a tool list against terrain.
pub const PANEL: Color32 = Color32::from_rgba_premultiplied(17, 20, 27, 232);
pub const PANEL_SOFT: Color32 = Color32::from_rgba_premultiplied(30, 35, 46, 140);
pub const HAIRLINE: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 26);
pub const HOVER: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 20);

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

/// Horizontal gradient behind the menu rail, painted on the background layer.
///
/// A solid panel fill would hide the landscape; this fades to fully transparent
/// well before mid-screen, so the scene reads as the background of the app
/// rather than as a picture behind a window.
pub fn paint_rail_scrim(ctx: &egui::Context) {
    let screen = ctx.viewport_rect();
    let rect = Rect::from_min_max(
        screen.left_top(),
        egui::pos2(screen.left() + SCRIM_WIDTH.min(screen.width()), screen.bottom()),
    );

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

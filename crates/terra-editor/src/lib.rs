//! The terrain and world editor, as a library.
//!
//! `main.rs` is a thin shell over [`app::run`]. The editor is a library first so
//! its UI can be driven by headless tests: `egui_kittest` needs to call the
//! panel-building functions directly, and a bin-only crate cannot be a test
//! dependency of itself.

pub mod app;
pub mod dock;
pub mod theme;
pub mod ui;

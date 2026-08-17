//! The terrain and world editor.
//!
//! Everything lives in the library crate; this is only the entry point. See
//! `lib.rs` for why the split exists.

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    terra_editor::app::run()
}

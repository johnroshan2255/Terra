//! The terrain and world editor.

mod app;
mod theme;
mod ui;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    app::run()
}

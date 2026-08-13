//! The game player: opens a baked project and runs it.
//!
//! Shares `terra-render` with the editor, so the designer's view and the
//! player's view cannot drift apart.

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("terra-runtime: scaffold only, nothing to run yet");
    Ok(())
}

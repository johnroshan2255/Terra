//! Instanced scatter for grass, rocks, and trees.
//!
//! Placement is derived from density masks plus a seed, so instance transforms
//! are never stored per-object -- only the rules are saved. Hand-placed props
//! are the exception and live in `world/edits/props.ron`.

// TODO: GPU scatter -> indirect draw.

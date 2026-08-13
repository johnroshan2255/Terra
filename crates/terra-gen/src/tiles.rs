//! Tier-0 -> tier-1 bake.
//!
//! Bicubic upsample of the eroded global map, plus high-frequency **ridged
//! multifractal** detail below the resolution the solver ran at, then sculpt
//! deltas replayed on top. Output lands in `world/cache/tiles/` and is always
//! reproducible from source + parameters.
//!
//! The detail layer uses the same basis as the tier-0 base. A different basis
//! here would make the boundary between eroded and non-eroded scales read as a
//! change in terrain character.

// TODO: per-tile dispatch; write h_{x}_{z}.r16.

# Terra

A terrain builder and game editor in Rust + WebGPU (`wgpu`).

Terrain is generated with **ridged multifractal noise** and carved by
**compute-shader hydraulic erosion**. World size is chosen once at project
creation and fixed thereafter. Users can create and manage multiple games.

> `terra` is a placeholder name. To rename, change the crate directory names,
> the `name` fields in each `crates/*/Cargo.toml`, the internal entries under
> `[workspace.dependencies]`, and the `ProjectDirs::from(...)` triple in
> `crates/terra-project/src/library.rs`.

## Workspace

| Crate | Role |
|---|---|
| `terra-core` | World scale, tile addressing, CPU heightmap. No GPU, no I/O. |
| `terra-project` | Folder layout, manifests, library index, format versioning. |
| `terra-gen` | Compute shaders: RMF base, hydraulic + thermal erosion, tile bake. |
| `terra-render` | wgpu surface, camera, CDLOD terrain, instanced scatter. |
| `terra-assets` | Shared asset library, glTF import, stable asset ids. |
| `terra-editor` | The tool. Project browser + terrain editing. **bin** |
| `terra-runtime` | The player. Opens a baked project and runs it. **bin** |

Dependency direction is strictly downward: `core` <- `project` <- `gen`/`assets`
<- `render` <- `editor`/`runtime`. Nothing depends upward, so `cargo test -p
terra-project` needs no GPU.

## Fixed constants

Frozen in `terra-core/src/units.rs`, asserted at compile time:

- **2 m/texel** baked tile resolution
- **1024 m** tile footprint, so **512²** texels per tile
- **2048 m** vertical range, `u16` heights → ~3.1 cm precision

| World | Tiles | Tier-0 res | Erodes at | Bake |
|---|---|---|---|---|
| Small — 2 km | 2×2 | 1024² | 2 m/texel | ~0.5 s |
| Medium — 4 km | 4×4 | 1024² | 4 m/texel | ~0.5 s |
| Large — 8 km | 8×8 | 2048² | 4 m/texel | ~2 s |
| Huge — 16 km | 16×16 | 4096² | 4 m/texel | ~8 s |

Hydraulic erosion runs on the **tier-0** map only — a single image, so there are
no tile seams to hide. Tier-1 tiles are upsampled tier-0 plus ridged multifractal
detail below the resolution the solver ran at. Sub-2 m detail is added in the
shader and never stored.

## Noise policy

Height is **always** ridged multifractal. Erosion features **always** come from
the compute-shader solver, never from a noise function shaped to imitate them.

```text
1. heightfield   ridged multifractal + domain warp   -> tier-0 raw
2. thermal       pre-pass, relaxes noise artifacts
3. erosion       6-pass grid hydraulic solver        -> tier-0 carved
4. thermal       post-pass, talus at cliff bases
5. tiles         upsample + ridged detail            -> tier-1
```

Step 1 is *supposed* to look under-detailed. Adding octaves there to compensate
only gives the solver noise to fight.

The one summed-octave function in the codebase is `warp_basis`, which produces
the domain-warp **coordinate offset** — not height. It is deliberately smooth
rather than ridged: ridged noise is C0 but not C1, with a gradient discontinuity
along every ridge, and warping coordinates with a creased field creases the
terrain.

These invariants are enforced by tests in `terra-gen/src/shaders.rs`, not just
by comment — swap the basis for fBm and the terrain still renders, it just stops
looking like mountains.

## Project layout on disk

```
MyGame/
├── project.ron          name, engine version, created
├── thumbnail.png        refreshed on save
├── world/
│   ├── world.ron        size, seed, RMF + erosion parameters
│   ├── source/          AUTHORITATIVE — back this up
│   │   ├── global_height.r16    tier 0, eroded. The one irreplaceable file.
│   │   ├── global_flow.r16      free output of erosion
│   │   ├── global_sediment.r16  free output of erosion
│   │   └── masks/               hand-painted R8 PNGs
│   ├── edits/           USER WORK — never regenerated
│   │   ├── sculpt/              sparse per-tile deltas, zstd
│   │   └── props.ron            hand-placed objects
│   └── cache/           DELETE ANYTIME — gitignored
│       └── tiles/               tier 1, h_{x}_{z}.r16
├── assets/              project-local meshes/textures
└── game/                spawns, config
```

The `source` / `edits` / `cache` split is the load-bearing idea. `cache` is
reproducible from `source` + `world.ron`, so it is never backed up. `edits` is
human effort and is never touched by a regenerate — re-running erosion with new
parameters rebuilds tiles, then replays sculpt deltas on top.

`.r16` is little-endian `u16`. World Machine, Gaea and Unity all import it, so
the built-in generator is never a dead end.

Tiles are origin-centered with signed coordinates (`h_-2_1.r16`), which keeps
float precision best near the middle of the map.

## Library

Multiple games, indexed at:

```
~/Library/Application Support/in.synctric.Terra/
├── library.ron    paths + last-opened, NOT contents
├── assets/        shared across every project
└── projects/      default location only — projects may live anywhere
```

Projects are self-contained and movable; nothing inside one stores an absolute
path. A library entry whose path no longer resolves is greyed out, never an
error — an unplugged drive must not lose the entry.

## Build

Requires a Rust toolchain (stable, edition 2024):

```sh
cargo test --workspace                      # includes GPU integration tests
cargo test -p terra-core -p terra-project   # no GPU needed
cargo run  -p terra-editor
```

`RUST_LOG=info` prints a frame heartbeat every 5 s: FPS, frame / CPU / GPU
milliseconds, 1% low, and presented-vs-skipped counts.

## Plans

Both staged behind the features that create the work they serve, rather than
built up front:

- [docs/culling.md](docs/culling.md) — visibility and culling
- [docs/physics.md](docs/physics.md) — Rapier, vehicles, scatter, weather

## Status

**Working:** project create/open/save, staged loading, fly camera, sculpt brushes
(raise / lower / smooth / flatten), ridged multifractal generation, GPU hydraulic
erosion, thermal erosion, erosion-driven material splatting, performance overlay
with real GPU timestamps.

Generation on an M4, 1024² world: RMF 27 ms, thermal 0.4 s, erosion 3.1 s
(2000 iterations). Runtime: ~1.2 ms CPU, ~1.4 ms GPU against a 5 ms budget.

**Stubs:** `terra-render/src/cdlod.rs`, `instancing.rs`, `terra-assets`,
`terra-runtime`, and the tier-0 → tier-1 tile bake.

**Next**, in order of payoff:

1. **CDLOD** — terrain is currently one uniform 512² grid over the whole world.
   This is the blocker for the 8 and 16 km sizes.
2. **Scatter** — instanced grass and rocks, placed from the erosion masks.
3. **Culling** — see [docs/culling.md](docs/culling.md); staged behind 1 and 2,
   because until they exist there is nothing to cull.

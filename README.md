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
| `terra-voxel` | True 3D: SDF volume, cave modifiers, sculpt brushes, Surface Nets, chunk LOD. |
| `terra-render` | wgpu surface, camera, CDLOD terrain, instanced scatter. |
| `terra-assets` | Shared asset library, glTF import, stable asset ids. |
| `terra-editor` | The tool. Project browser + terrain editing. **bin + lib** |
| `terra-runtime` | The player. Opens a baked project and runs it. **bin** |

Dependency direction is strictly downward: `core` <- `project` <-
`gen`/`assets`/`voxel` <- `render` <- `editor`/`runtime`. Nothing depends
upward, so `cargo test -p terra-project` needs no GPU.

## Two representations

The heightfield is one height per column: the right structure for eroded
landscape, and incapable of an overhang, an arch or a cave. `terra-voxel`
layers over it rather than replacing it — the ground is still RMF noise carved
by the erosion solver, and the volumetric layer adds only what a height per
column cannot express.

```text
base     the heightfield, read as a signed distance   -- free, already in RAM
delta    sparse voxel offsets                         -- freeform 3D sculpting
stack    ordered boolean modifiers                    -- caves, non-destructive
```

sampled as `stack(base(p) + delta(p))`, negative inside rock. Sculpting lands
*under* the modifiers, so boring a tunnel through sculpted rock never destroys
the sculpt, and deleting the tunnel restores it exactly.

Surfaces come back via **Surface Nets** rather than Marching Cubes: at most one
vertex per cell, so triangle density is uniform instead of swinging 5x across a
cave wall, and the mesh is manifold by construction with no disambiguation
table. There are two implementations — a CPU reference and a compute pipeline —
and the tests assert they produce the same triangles.

## Sculpt brushes

Eight modes, in `terra-voxel/src/brush.rs`. Each one is a `target_for` match arm
plus tests; the shared machinery decides a target distance per voxel, blends by
the falloff, and stores the difference.

All eight run today, on the heightfield. Nothing in the palette is disabled.

| Mode | What it does |
|---|---|
| Clay | Fills up toward the brush plane, never cutting ground already above it |
| Raise / Lower | Adds or removes height. Ctrl swaps between them |
| Move | Shifts heights along the cursor travel, so detail moves with the surface |
| Flatten | Converges on the brush plane from both sides |
| Smooth | Four-neighbour average |
| Noise | Displaces by a pattern — built-in basis or an uploaded greyscale map |
| Pinch | Pulls the outer profile inward, narrowing a rise into a ridge |

Clay is not Raise: it converges on the plane and never cuts, so repeated strokes
build a shape instead of inflating one. Noise is a *displacement*, not a target,
so a held stroke keeps adding relief rather than settling — which is what makes
the strength control meaningful for it.

`terra-voxel::Brush` carries the same set for the volumetric path, with one
difference: it has **Inflate** (offset along the surface normal) where the
heightfield has **Lower**. On a tilted plane those differ — a 1 m perpendicular
step is `1/cos θ` of vertical rise, 1.41 m on a 45° face — and world-up is what
landscape work wants, so the 2.5D palette exposes Raise/Lower instead.

### Noise patterns

Signed, `-1..=1`, and **mid-grey means no displacement** — the same convention
as every displacement map, so an ordinary noise texture roughens the surface
without also inflating it. Uploaded images are sampled **triplanar**, weighted
by the stroke normal: projecting straight down world XZ streaks the pattern
across every cliff and cave wall, which is most of what this crate is for.

The built-in ridged/billow basis needs no upload, so the tool works out of the
box. Imports land in the project's own `assets/noise/`.

## Environment Light Mixer

One struct and one panel for everything that lights the world
(`terra-render/src/environment.rs`), in the order light physically arrives:

```text
Sun          the directional source, or the moon at night
Atmosphere   Rayleigh + Mie scattering -- why the sky is blue and the horizon pale
Sky light    the hemisphere bounce that fills shadow
Fog          exponential height fog, and the god rays from marching it
Clouds       volumetric layer between sun and ground
Tone map     radiance to pixels
```

This replaced four separate sections — FOG, ENVIRONMENT, SHADOWS, ATMOSPHERE —
with no ordering between them. They interact: raising fog density without
touching exposure darkens the whole frame, and a user adjusting one section at a
time could not see why.

**Quick Create** presets set the *look* and leave feature toggles alone — switch
clouds on, click Daylight to fix the lighting, and the clouds stay. A preset may
turn something on (Overcast brings clouds with it) but never off; **Reset
environment** is the button that clears everything. Daylight spawns: sun at **-45°** pitch, real **Rayleigh** coefficients,
exponential fog at **0.002**, **ACES** tone mapping. Overcast and Night are
presets too, for the same reason quality is: the settings interact and
half-overcast is not a look.

Scattering coefficients are the real per-metre values for air at sea level, and
that is load-bearing — blue scatters 5.7× as strongly as red, which is why the
sky is this blue and a low sun is warm. Everything downstream (cloud coverage,
god-ray intensity, tone-map contrast) is an artistic dial with no physical claim
and is documented as such.

The sky shader reads `EnvironmentUniform` directly and computes a real
**single-scattering integral**: march the view ray through an exponential
atmosphere, accumulate Rayleigh and Mie in-scattering weighted by their phase
functions, attenuate by transmittance to the sun and back. Multiple scattering is
approximated by a cheap isotropic term, not integrated — the real thing belongs
in a precomputed LUT rather than an inline march. Switching Sky Atmosphere off
falls back to the old gradient.

**Volumetric clouds render**, in their own pass (`clouds.rs`). A raymarched layer:
density from noise shaped by a cumulus height profile, lit by a short march toward
the sun under Beer–Powder, two-lobe Henyey–Greenstein phase for the silver lining,
wind advection, composited by transmittance so it dims the sky rather than
painting over it.

Marched at **half resolution** and **temporally accumulated** — blended against
the previous result reprojected through the previous view-projection, which
smooths the march's noise and carries detail forward as the camera moves.
History is looked up by reprojecting the
*slab entry point*, not the ray direction: the layer sits 1.5–4 km out and the
camera moves at up to 120 m/s, so translation parallax would smear it. History is
rejected when the reprojection lands off-screen or disagrees sharply on
transmittance, which is what stops comet trails behind cloud edges.

The buffer is rendered from an **unjittered** camera on purpose. A 2x2 rotated
sub-pixel grid was tried, to recover the detail half resolution gives up, and it
shook: the accumulated buffer still carried a quarter-pixel of the newest offset,
the sky samples it through its own TAA jitter, and the layer slid every frame.
Cloud edges are soft and the TAA resolve already antialiases them.

Measured at 1280×720 (the cloud march alone):

| | full res | half res |
|---|---|---|
| Low | 4.65 ms | **1.35 ms** |
| Medium | 9.24 ms | **2.58 ms** |
| High | 15.54 ms | **4.56 ms** |

3.4–3.6× from resolution alone, near the theoretical 4×. The clear sky is 2.3 ms
on top. Heavy coverage costs *less* than light, because the march reaches its
transmittance cut-off sooner.

**Cloud shadows reach the ground.** A 512² top-down map of sun transmittance over
an 8 km region following the camera, snapped to whole texels so edges do not crawl
as it moves, sampled by the terrain shader by world XZ and multiplied with the
cascade shadows. A 2D map rather than another cascade because the slab is
kilometres above any terrain, so the transmittance at ground level is a function of
XZ alone to well inside what the eye resolves.

**Tone mapping works.** ACES, Reinhard and None are selectable; white balance,
contrast and saturation are applied in linear before the curve, with contrast
pivoting about 0.18 rather than 0.5. All four were UI-only before — the panel
offered them and the shader hardcoded ACES.

`Environment::apply_to` still bridges the mixer into `SkySettings`/`FogSettings`
for the fog pass and the lighting uniform. It deliberately does *not* touch shadow
resolution, shadow distance or temporal AA: those cost frame time rather than
changing the look, and they stay in Quality where the user set them.

`EnvironmentUniform` is 16 `vec4`s, 256 bytes. Every field is a vector on
purpose: WGSL rounds uniform members up to 16-byte alignment, so a lone `f32`
between two vectors inserts padding the Rust side does not and the whole block
reads shifted from there on.

## Materials

**Nothing ships prebuilt.** An empty project has an empty palette and the terrain
renders as neutral mid-grey — explicitly, in the shader. Sampling the texture
array with no layers returns zeros and the terrain came out *black*, which reads
as a rendering bug rather than as "import a texture". There was a noise-generated fallback of six
layers — soil, grass, rock, gravel, snow, mud — and it made a new project look
furnished with materials the user had not chosen and could not edit.

Content comes from the project's own `assets/textures/`, one folder per material
with albedo, normal and roughness maps in it. Import in the Content pane and the
palette rebuilds immediately — no restart. Then paint it on with the Paint tool.

**Double-click a texture** — in the Content browser or the Paint palette — to open
the Material pane on it:

| Setting | What it does |
|---|---|
| Repeat m | Metres per texture repeat |
| Normal strength | Multiplier on the tangent normal; 0 flattens the layer |
| Roughness | Scales sampled roughness |
| Occlusion | Mixes sampled AO toward unoccluded |
| **Parallax m** | Offsets the lookup by the height channel along the view |
| Blend band | Width of the band this layer contends with its neighbour in |
| Tint | Multiplies the albedo |

`LayerParams` carries **explicit padding before `tint`**, and it is load-bearing.
WGSL aligns a `vec3<f32>` to 16 bytes, so the six scalars fill 0..24 and the shader
places `tint` at offset **32** — where `repr(C)` would put it at 24. The shader then
read `tint.rgb` from bytes 32..44: `tint[2]` followed by two padding floats, or
`(1, 0, 0)`. The terrain rendered **pure red**, and only on the menu backdrop,
because that is the one terrain with a non-empty palette — a fresh project takes the
`layer_count == 0` branch and never reads the tint at all.

The size was already correct; only the offset was wrong, which is why a size
assertion did not catch it. Offsets are now asserted at compile time on both
`LayerParams` and `TerrainUniform`, and a test reads the tint out of the raw bytes
the way the shader does rather than through the Rust field — reading the field
passed while the shader saw red.

`EnvironmentUniform` avoids the whole problem by being 16 `vec4`s, which is why it
never had this bug.

All of it is **per layer**, which the single global tiling scale it replaced could
not be: gravel wants a repeat every metre or two and a cliff face wants ten, and
one shared number leaves one blurred and the other visibly tiled.

Parallax is what makes the surface read as relief rather than a photograph of
relief — stones and cracks occlude each other as the camera moves. It is off by
default because it costs samples and a material with a flat height channel gains
nothing from it.

## Camera

A fly camera with orbit, pan, look and wheel zoom. Three things in it are worth
knowing because each was a bug:

**Pan and zoom scale by the distance to what is on screen**, raycast along the view
direction — not the height above the ground below the camera. Standing five metres
up and looking at a ridge two kilometres away, the old measure moved five metres'
worth per pixel and the drag felt frozen. Aimed at the sky there is no hit, and
height above ground is the fallback.

**One pitch limit, shared.** `right()` is `forward().cross(Y)`, whose length is
`cos(pitch)`; at exactly ±90° that is zero, `normalize()` returns NaN, and the NaN
spreads through `up()`, the view matrix and every pass that reads it. `rotate`,
`orbit` and `look_toward` now share `PITCH_LIMIT` rather than each carrying their
own — `orbit` had a looser value, which is the kind of difference that hides a
division by almost-zero.

**`F` frames the world**, as in every DCC tool, and it is load-bearing rather than
a convenience. Wheel-out is *geometric* — each notch multiplies the distance by
12% and the next notch is computed from the new, larger distance — so sixty notches
from the default view reaches **800 km**, where the terrain is a sliver on the
horizon. Two things then made that unrecoverable: the zoom scale was capped at
6000 m so coming back was *linear* at 720 m a notch (over a thousand notches), and
WASD was a fixed 120 m/s (nearly two hours to cross). All three are fixed — the
cap is now `MAX_VIEW_DIST_M` so both directions are geometric, the reachable
distance is bounded, movement speed scales with the view distance, and `F` always
returns to a known-good framing.

**Movement speed scales with the view distance**, with `camera_speed` as the
multiplier and the camera's starting distance as the reference. Clamped at both
ends: proportional all the way down crawls at half a metre a second when four
metres off the ground, and unbounded at the top overshoots the world in one
keypress.

**Shading is filtered to the pixel footprint.** Zoomed out to tens of kilometres,
several grid quads land inside one pixel — a 4 km world at 1024 texels has a 3.9 m
texel, and from 40 km one pixel spans 51 m, so thirteen texels fall inside it. The
per-vertex normal is sampled at one texel, so the temporal jitter put each frame's
sample on a different one and the terrain shimmered, which reads as the view
shaking.

`fwidth` gives the world span a pixel covers, and the geometric normal is
re-derived by central differences at that width — the same value up close, a smooth
average far away. The material bump fades over the same range and roughness widens
to compensate, because dropping sub-pixel detail without gaining the blur it would
have averaged to leaves a mirror-smooth distant surface that sparkles just as
badly.

The real fix is geometric LOD, so the undersampling never happens —
`terra-render/src/cdlod.rs`, still a stub. This makes the shading stable meanwhile
and stays correct afterwards.

**Accumulated input is dropped when not editing.** Mouse motion and wheel notches
arrive from window and device events regardless of state, and only `update_editor`
consumes them. A wheel spin on the menu, or a drag held through the loading screen
or a Play session, banked up and applied in one go on the first editing frame — the
camera visibly jumped. `Input::clear_motion` runs whenever `is_editing()` is false,
and on the early return inside `update_editor`.

## Viewport visualization modes

Unreal's hotkeys, on the terrain pass (`view_mode.rs`):

| Mode | Key | Answers |
|---|---|---|
| Wireframe | `Alt+2` | Is the topology what I think it is? |
| Unlit | `Alt+3` | Is the albedo right, independent of lighting? |
| Lit | `Alt+4` | The real thing |
| Detail Lighting | `Alt+5` | Are the normal maps doing anything? |
| Lighting Only | `Alt+6` | Is the lighting right, independent of materials? |

Detail Lighting and Lighting Only differ in exactly one respect: both replace
albedo with neutral grey, but Detail Lighting keeps the material normal maps and
Lighting Only shades from the geometric normal alone. So a surface that looks flat
under Detail Lighting has a broken normal map; one that looks flat under Lighting
Only is genuinely flat.

Neutral grey is **0.18 linear**, not 0.5 — mid grey as the eye sees it is 18%
reflectance, and 0.5 linear is a much brighter card that clips the very highlights
the mode exists to inspect.

Fog and god rays are off in every mode but Lit. They are view-dependent haze over
the whole frame, and the point of a debug view is one term with nothing on top of
it.

Whichever mode is active is named in the status bar, as a **button** that returns
to Lit — forgetting a debug mode is on is the classic way to spend ten minutes
debugging a material that was never broken.

**Editing only.** `Screen::Editor` is *not* the right gate: driving is a sub-state
of it, so gating on the screen alone let a wireframe follow the car into Play.
`App::is_editing` requires a world open and `play` unset, and it governs all four
places the mode is used — the draw call, the terrain uniform, the `Alt+` hotkeys,
and the UI. The chosen mode is remembered across a Play session rather than reset,
so stopping returns to the view that was set up.

### Wireframe, and why it is not barycentric

Two paths. `POLYGON_MODE_LINE` is requested when the adapter has it, and the
triangle buffer is drawn with filled faces off. When it does not — it is not in
core WebGPU — a **line-list index buffer** over the grid edges is drawn instead.

The usual barycentric trick cannot work here: on an indexed mesh there is no
per-corner attribute to interpolate, WGSL has no barycentric builtin and no
geometry stage, and `@builtin(vertex_index)` under an index buffer is the index
*value* rather than which corner it is. Getting barycentrics would mean expanding
the grid to three unshared vertices per triangle — tripling the vertex work to
draw a debug view. The line list is exact instead of a screen-space
approximation, needs no optional feature, and is built once at load.

Two of each quad's four sides are emitted, plus the far sides at the boundary,
**plus each quad's diagonal**. The diagonal is the point: the renderer draws
triangles, and a wireframe of only the quad grid looks like graph paper and says
nothing about the actual mesh. It has to be the anti-diagonal `build_indices`
actually splits on — that function emits `[a, c, b, b, c, d]`, so the shared edge
is `b`–`c`, and drawing `a`–`d` instead would be a wireframe of a mesh that is not
being rendered.

Wireframe also **skips the sky** and clears to flat dark grey, as Unreal's does: a
scattering sky behind one-pixel lines makes them unreadable, and reading them is
the whole point. Foliage and props are skipped too — only the terrain has a
wireframe pipeline, and solid trees on a wire mesh looks like the wireframe is
broken rather than incomplete.

**On Metal, `POLYGON_MODE_LINE` is unavailable**, so the fallback is the path that
actually runs on a Mac. It is the one to get right, not the optional-feature path.

## Panels

Square corners, flat opaque greys, dense rows — the Unreal Editor idiom rather
than the rounded translucent panels this started with. Sliders put their value
box on the caption row and the track full-width below, so a panel never demands
more width than it has; the previous layout reserved a fixed 74 px for the value
and clipped it off the edge of any narrow panel.

`egui_dock`, not fixed panels:

| Action | How |
|---|---|
| Move / re-dock | Drag the tab |
| Float in its own window | Right-click the tab → Eject, or **View ▸ Float a panel** |
| Minimize (collapse to title bar) | The ▼ on the tab bar — works docked and floating |
| Resize | Drag a separator, or a floating window's corner |
| Close | The × on the tab |
| **Reopen** | **View ▸ Panels** checkbox |
| Undo a mess | **View ▸ Reset layout** |

The View menu is load-bearing rather than a convenience: without it the × on a
tab is a one-way trip and the pane is gone for the session. Floating windows are
clamped to the viewport (`window_bounds`), so one cannot be dragged off-screen
and lost either.

The toolbar and status bar stay fixed deliberately — Unreal does not undock its
menu bar either, and a floating Save button is a worse editor, not a more
flexible one. The viewport cannot be closed or floated: it is the hole the 3D
scene renders through.

Tool settings live under the tool that owns them, not in one always-on column:
selecting a tool selects its settings. Brush size and strength are `[` and `]`,
with Shift for strength; the panel shows a readout, not a second set of sliders.

`Layout::new` in `dock.rs` documents the one real trap — `split_left` applies
its fraction to the *new* pane, `split_right` to the old one, so reading it the
wrong way round hands the Tools pane 82% of the window.

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

The rule is about *generation*. The Noise sculpt brush is summed-octave and that
is fine: it is a displacement the user aimed by hand, not a generator pretending
to be geology. `terra-voxel/src/noise.rs` says so at the top, so the exemption
is deliberate rather than an oversight.

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
├── assets/              project-local, populated by the content browser
│   ├── textures/                one folder per material set
│   ├── noise/                   greyscale maps for the Noise brush
│   └── models/                  glTF meshes
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

Requires a Rust toolchain (stable, edition 2024, **1.95+** — the floor for
`egui_dock` and `egui_kittest`, both of which track egui 0.36 exactly):

```sh
cargo test --workspace                      # includes GPU integration tests
cargo test -p terra-core -p terra-project   # no GPU needed
cargo run  -p terra-editor
```

The UI is testable without a window. `egui_kittest` drives it through AccessKit,
so panel layout and the contextual tool settings are asserted in `cargo test`:

```sh
cargo test -p terra-editor --test ui_headless
cargo test -p terra-editor --test ui_headless -- --ignored   # renders target/ui-layout.png
```

Note that AccessKit only publishes *interactive* widgets — plain labels and
slider captions are absent, so a section is asserted by the controls it holds.
The `--ignored` snapshot exists for the rest: it renders the whole editor to a
PNG, which is how the split fractions in `dock.rs` were caught.

`RUST_LOG=info` prints a frame heartbeat every 5 s: FPS, frame / CPU / GPU
milliseconds, 1% low, and presented-vs-skipped counts.

## Plans

Both staged behind the features that create the work they serve, rather than
built up front:

- [docs/culling.md](docs/culling.md) — visibility and culling
- [docs/physics.md](docs/physics.md) — Rapier, vehicles, scatter, weather

## Status

**Working:** project create/open/save, staged loading, fly camera, ridged
multifractal generation, GPU hydraulic erosion, thermal erosion, erosion-driven
material splatting, performance overlay with real GPU timestamps, dockable
panels, contextual tool settings, content browser with asset import.

Generation on an M4, 1024² world: RMF 27 ms, thermal 0.4 s, erosion 3.1 s
(2000 iterations). Runtime: ~1.2 ms CPU, ~1.4 ms GPU against a 5 ms budget.

Surface Nets extraction on the same machine, per 32³ chunk: **0.073 ms** on the
GPU against 0.94 ms for the CPU reference — so a sculpt stroke that dirties a
handful of chunks re-extracts well inside one frame.

**Stubs:** `terra-render/src/cdlod.rs`, `instancing.rs`, `terra-assets`,
`terra-runtime`, and the tier-0 → tier-1 tile bake.

**Half-wired:** all eight sculpt modes, the noise patterns and the cave modifier
stack have working UI, and the sculpt modes deform the heightfield for real. What
is missing is the *volumetric* execution path: nothing draws voxel chunks yet, so
a cave modifier is representable and editable but not visible, and neither the
delta field nor the modifier stack has an on-disk format — both belong under
`edits/`, beside the sculpt deltas and road splines already there.

**Next**, in order of payoff:

1. **Wire the volumetric path** — a render pass over extracted chunks, and the
   sculpt tools writing to the SDF instead of the heightfield. This is what
   turns the four disabled brushes on and makes caves visible rather than
   merely representable.
2. **Persist `edits/`** — the delta field and modifier stack. Until these save,
   a cave does not survive closing the project.
3. **CDLOD** — terrain is one uniform 512² grid over the whole world. The
   blocker for the 8 and 16 km sizes.
4. **Culling** — see [docs/culling.md](docs/culling.md); staged behind the
   above, because until chunks exist there is nothing to cull.

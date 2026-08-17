//! Editor shell: window, surface, egui, and the screen state machine.
//!
//! The window, GPU context, sky and menu backdrop are built once in `resumed`
//! and live for the whole session. Navigating between menu panes only changes
//! which controls the left rail draws -- nothing is torn down or rebuilt.

use crate::theme;
use crate::ui::{
    self, Action, CreateForm, EditorAction, EditorView, FoliageEntry, PaintMode, PaletteEntry,
    Pane, PerfView, Tool,
};
use anyhow::Result;
use glam::{Mat4, Quat};
use glam::{Vec2, Vec3, Vec4};
use std::sync::Arc;
use std::time::{Duration, Instant};
use terra_core::{BASE_ELEVATION_M, WorldSize};
use terra_physics::{FIXED_DT, Obstacle, ObstacleShape, PhysicsWorld, Vehicle, VehicleInput};
use terra_project::roads::{Road, RoadNetwork};
use terra_project::{Library, Project, TerrainParams, WorldData};
use terra_render::camera::Camera;
use terra_render::context::RenderContext;
use terra_render::hiz::HiZ;
use terra_render::lighting::{CASCADES, LightMode, Lighting, SkySettings};
use terra_render::material::Materials;
use terra_render::mesh::{Instance, MeshRenderer};
use terra_render::post::Post;
use terra_render::scatter::Scatter;
use terra_render::sky::Sky;
use terra_render::stats::{FrameStats, GpuTimer};
use terra_render::taa::Taa;
use terra_render::terrain::{BrushOp, SculptMode, Terrain, apply_brush};
use terra_render::volumetrics::{FroxelGrids, Volumetrics};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

pub fn run() -> Result<()> {
    let event_loop = EventLoop::new()?;
    // Frames are paced from `about_to_wait` with an explicit deadline rather
    // than by ControlFlow::Poll. Poll never sleeps, and any frame that does not
    // present (occluded window, surface timeout) therefore never blocks on
    // vsync either -- the loop then spins as fast as the CPU allows.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Menu(Pane),
    Loading,
    Editor,
}

struct Settings {
    vsync: bool,
    perf_overlay: bool,
    perf_graph: bool,
    camera_speed: f32,
    fov_deg: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            vsync: true,
            perf_overlay: true,
            perf_graph: true,
            camera_speed: 120.0,
            fov_deg: 60.0,
        }
    }
}

#[derive(Default)]
struct Input {
    fwd: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    boost: bool,
    /// Ctrl held: invert the active brush (raise becomes lower, and so on).
    invert: bool,
    /// Alt held. Only the view-mode hotkeys read it, but it has to be tracked
    /// like any other modifier: a key-up for Alt can arrive while the window is
    /// unfocused, and a latched Alt turns every digit into a mode switch.
    alt: bool,
    looking: bool,
    /// Middle mouse held: dragging the view rather than aiming it.
    panning: bool,
    /// Left mouse held while the Camera tool is active.
    orbiting: bool,
    sculpting: bool,
    /// One-shot: a left press that has not been consumed yet. Sculpting wants a
    /// held state, but placing a road point wants a single event.
    clicked: bool,
    cursor: (f32, f32),
    /// Raw mouse movement this frame, consumed by whichever of look or pan is
    /// active.
    look_delta: (f32, f32),
    /// Wheel notches accumulated since the last frame.
    scroll: f32,
}

impl Input {
    /// Drop accumulated mouse motion and wheel notches.
    ///
    /// Both accumulate from window and device events regardless of what the
    /// editor is doing, and both are only consumed by `update_editor`. So a wheel
    /// spin on the menu, or a drag held through the loading screen or a Play
    /// session, piled up and was applied in one go on the first editing frame --
    /// the camera visibly jumped. Clearing whenever we are not editing keeps
    /// stale input from crossing a state change.
    fn clear_motion(&mut self) {
        self.look_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }

    fn axis(&self) -> Vec3 {
        Vec3::new(
            (self.right as i32 - self.left as i32) as f32,
            (self.up as i32 - self.down as i32) as f32,
            (self.fwd as i32 - self.back as i32) as f32,
        )
    }
}

struct Gfx {
    ctx: RenderContext,
    sky: Sky,
    /// Generated once and shared by the backdrop and every world opened this
    /// session -- the textures are read-only, so there is nothing to duplicate.
    materials: Materials,
    /// Sun, sky and the shadow cascades. Shared by every pass.
    lighting: Lighting,
    /// Volumetric fog. Built before the scene, sampled by every shading pass.
    fog: Volumetrics,
    /// Temporal resolve, between the scene and the post chain.
    taa: Taa,
    /// God rays and the final resolve.
    post: Post,
    /// Depth pyramid built from the previous frame, for occlusion culling.
    hiz: HiZ,
    /// The environment uniform the sky and cloud passes read.
    env_gpu: terra_render::EnvironmentGpu,
    /// Half-res, temporally accumulated cloud layer.
    clouds: terra_render::Clouds,
    /// Foliage species and their instance buffers. Shared like materials: the
    /// meshes are read-only, the per-world painting is not.
    scatter: Scatter,
    meshes: MeshRenderer,
    /// Dimensions of the player's vehicle, measured from the mesh `meshes` draws.
    ///
    /// Kept beside the renderer that owns the mesh, so the collider and the drawn body
    /// cannot come from different sources. `Vehicle::spawn` reads it when play starts.
    vehicle_dims: terra_core::VehicleDims,
    egui_renderer: egui_wgpu::Renderer,
    gpu_timer: Option<GpuTimer>,
}

struct OpenWorld {
    project: Project,
    terrain: Terrain,
    camera: Camera,
    /// Terrain *without* roads. Roads are stamped onto a copy of this every
    /// time one changes, which is what makes them editable after placement --
    /// and what stops a regenerate from destroying them.
    base: Vec<f32>,
    roads: RoadNetwork,
    /// Erosion by-products, kept so they can be saved alongside the heightfield
    /// and reloaded without re-running the solver.
    flow: Vec<f32>,
    deposition: Vec<f32>,
}

/// What happened to a frame. Drives how long the loop waits before the next
/// one -- a skipped frame presents nothing, so nothing else throttles us.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    /// Presented. Vsync (or the frame interval) paces the next one.
    Presented,
    /// Transient: the swapchain needs rebuilding, typically mid-resize. Retry
    /// immediately, or resizing visibly stutters.
    Retry,
    /// The surface is not presentable at all -- covered, minimized, or the
    /// compositor is not asking for frames. Back well off.
    Idle,
}

/// A pose the renderer interpolates toward.
#[derive(Clone, Copy)]
struct Pose {
    translation: Vec3,
    rotation: Quat,
}

/// Everything that only exists while driving.
struct Play {
    world: PhysicsWorld,
    car: Vehicle,
    /// Leftover time that has not yet been consumed by a fixed step.
    accumulator: f32,
    /// Chassis pose before and after the last physics step. Rendering
    /// interpolates between them, because physics runs at 60 Hz and the display
    /// does not -- without this the car visibly stutters even at 75 fps.
    prev: Pose,
    curr: Pose,
    prev_wheels: Vec<Pose>,
    curr_wheels: Vec<Pose>,
    camera: Camera,
}

/// Staged world load. One stage runs per frame so the loading screen actually
/// paints between steps instead of the whole thing happening inside one frozen
/// frame.
struct Loading {
    project: Project,
    stage: u8,
    elapsed: f32,
    terrain: Option<Terrain>,
    heights: Option<Vec<f32>>,
    flow: Vec<f32>,
    deposition: Vec<f32>,
    roads: RoadNetwork,
    foliage: Option<Vec<u8>>,
    props: terra_project::props::PropSet,
}

impl Loading {
    const STAGES: [&'static str; 4] =
        ["Allocating heightfield", "Reading terrain", "Uploading to GPU", "Entering world"];
    /// Held on screen at least this long. A small world loads in ~40 ms, and a
    /// loader that flashes for two frames looks like a glitch.
    const MIN_SECONDS: f32 = 0.75;

    fn new(project: Project) -> Self {
        Self {
            project,
            stage: 0,
            elapsed: 0.0,
            terrain: None,
            heights: None,
            flow: Vec::new(),
            deposition: Vec::new(),
            roads: RoadNetwork::default(),
            foliage: None,
            props: Default::default(),
        }
    }

    fn label(&self) -> &'static str {
        Self::STAGES[(self.stage as usize).min(Self::STAGES.len() - 1)]
    }

    fn progress(&self) -> f32 {
        let work = self.stage as f32 / 3.0;
        let time = self.elapsed / Self::MIN_SECONDS;
        work.min(time).clamp(0.0, 1.0)
    }
}

pub struct App {
    egui_ctx: egui::Context,
    /// Held outside `Gfx` so the UI phase can borrow `&mut self` while the GPU
    /// phase borrows `self.gfx` -- disjoint fields, no conflict.
    egui_state: Option<egui_winit::State>,
    gfx: Option<Gfx>,
    window: Option<Arc<Window>>,
    screen: Screen,
    library: Library,
    settings: Settings,
    form: CreateForm,
    world: Option<OpenWorld>,
    loading: Option<Loading>,
    unsaved: bool,
    /// Landscape shown behind the menus. Generated once at startup and reused
    /// for the whole session.
    backdrop: Option<Terrain>,
    backdrop_cam: Camera,
    time: f32,
    input: Input,
    brush_mode: SculptMode,
    /// Which visualization the viewport is showing. `Alt+2` through `Alt+6`.
    view_mode: terra_render::ViewMode,
    brush_radius: f32,
    brush_strength: f32,
    brush_hit: Option<Vec2>,
    /// Where the brush was last frame, so Move knows which way the cursor
    /// travelled. Cleared when a stroke ends, or the first dab of the next
    /// stroke would drag by the whole gap since the last one.
    last_brush_hit: Option<Vec2>,
    tool: Tool,
    /// Selected palette slot, and how a stroke is applied.
    paint_layer: u32,
    paint_mode: PaintMode,
    /// Weight added per second of painting, so a stroke builds up over time
    /// rather than snapping to full on the first frame.
    paint_flow: f32,
    /// Selected foliage species.
    species: usize,
    /// Whether the in-play graphics panel is showing.
    graphics_open: bool,
    /// Whether each docked side panel is expanded.
    tools_open: bool,
    inspector_open: bool,
    /// Index into the prop list, when the Select tool has one picked.
    selected_prop: Option<usize>,
    /// True while a picked prop is being dragged.
    dragging_prop: bool,
    /// Deferred prop edits, applied after `update_editor` releases its borrows.
    pending_prop_move: Option<Vec3>,
    pending_prop_refresh: bool,
    /// Whether a sculpt stroke was in progress last frame, so its end can be
    /// detected and the foliage rebuilt once rather than per dab.
    was_sculpting: bool,
    /// This frame's sub-pixel camera offset, held so the resolve can use the
    /// same value the scene was rendered with.
    frame_jitter: glam::Vec2,
    /// Where the streamed obstacle set was last built.
    obstacle_origin: Vec3,
    /// Brush stroke waiting to be applied to the scatter palette. Deferred
    /// because painting needs `gfx` mutably while the editor update already
    /// holds it shared.
    pending_foliage: Option<(Vec2, f32)>,
    /// Palette thumbnails, registered with egui on the first editor frame.
    /// Held here rather than rebuilt per frame: uploading six textures every
    /// frame would be a leak in all but name.
    swatches: Vec<egui::TextureHandle>,
    /// Species previews, registered alongside the material swatches.
    species_swatches: Vec<egui::TextureHandle>,
    /// Road currently being drawn, as an index into the network.
    active_road: Option<usize>,
    /// The Environment Light Mixer: the single source of truth for everything
    /// that lights the world. `Lighting::settings` and `Volumetrics::settings`
    /// are derived from it every frame and are written by nothing else.
    env: terra_render::Environment,
    /// What `env` looked like the last time it was written to disk.
    ///
    /// The mixer panel edits `env` in place, so there is no widget to hang a dirty
    /// flag on; comparing against this once a frame is how an environment edit comes
    /// to mark the world unsaved. A few dozen floats, so the compare is free.
    saved_env: terra_render::Environment,
    /// Panel arrangement. Owned by the app so it survives navigating to the
    /// menu and back -- a layout that reset itself every time a world closed
    /// would be worse than not being movable at all.
    layout: crate::dock::Layout,
    /// Non-destructive cave modifiers for the open world.
    modifiers: terra_voxel::ModifierStack,
    selected_modifier: Option<usize>,
    /// Pattern the Noise sculpt brush samples.
    noise: terra_voxel::NoiseField,
    /// Names of greyscale maps imported into this project.
    noise_library: Vec<String>,
    /// Asset names per kind, refreshed when a world opens and after an import.
    assets: [Vec<String>; 3],
    /// Which shelf the content browser is showing.
    asset_kind: crate::ui::AssetKind,
    /// Palette slot the Material pane is editing.
    selected_material: usize,
    /// Set when the palette params were edited, so the change reaches the GPU
    /// once per frame rather than once per slider drag event.
    material_dirty: bool,
    /// Where the viewport pane sits, as of the last frame. `None` until the
    /// editor has drawn once.
    viewport_rect: Option<egui::Rect>,
    /// Raw cursor trail for the stroke in progress, before smoothing and
    /// simplification. Dense and noisy by nature.
    stroke: Vec<[f32; 2]>,
    pending_stroke_finish: bool,
    /// Live simulation while driving. `None` when editing.
    play: Option<Play>,
    last_frame: Instant,
    /// Deadline for the next frame. The event loop sleeps until this.
    next_frame: Instant,
    /// Window is hidden, minimized, or fully covered. Frames are slowed, not
    /// stopped.
    occluded: bool,
    frame_ms: f32,
    /// Frames actually presented since the last heartbeat.
    presented: u32,
    /// Frames skipped because the surface was unavailable. Counted separately:
    /// folding these into the frame rate makes a deliberately idle loop look
    /// like a performance problem.
    skipped: u32,
    /// When the previous frame was presented, if it was. `None` after a skip,
    /// which breaks the chain so the gap is never recorded as a frame time.
    last_present: Option<Instant>,
    last_report: Instant,
    stats: FrameStats,
    /// CPU spent in `update()` this frame.
    update_ms: f32,
    /// `update_ms` plus encoding, excluding both the surface acquire and the
    /// present. Under vsync those are waits, not work, and counting them made
    /// CPU time read as identical to frame time.
    cpu_ms: f32,
}

impl App {
    pub fn new() -> Result<Self> {
        Ok(Self {
            egui_ctx: egui::Context::default(),
            egui_state: None,
            gfx: None,
            window: None,
            screen: Screen::Menu(Pane::Home),
            library: Library::load().unwrap_or_default(),
            settings: Settings::default(),
            form: CreateForm::default(),
            world: None,
            loading: None,
            unsaved: false,
            backdrop: None,
            backdrop_cam: Camera { fov_y: 52f32.to_radians(), ..Camera::default() },
            time: 0.0,
            input: Input::default(),
            brush_mode: SculptMode::Raise,
            view_mode: terra_render::ViewMode::default(),
            brush_radius: 120.0,
            brush_strength: 1.5,
            brush_hit: None,
            last_brush_hit: None,
            tool: Tool::Camera,
            paint_layer: 0,
            paint_mode: PaintMode::Brush,
            paint_flow: 2.0,
            species: 0,
            graphics_open: false,
            tools_open: true,
            inspector_open: true,
            selected_prop: None,
            dragging_prop: false,
            pending_prop_move: None,
            pending_prop_refresh: false,
            was_sculpting: false,
            frame_jitter: glam::Vec2::ZERO,
            obstacle_origin: Vec3::ZERO,
            pending_foliage: None,
            swatches: Vec::new(),
            species_swatches: Vec::new(),
            active_road: None,
            env: terra_render::Environment::daylight(),
            saved_env: terra_render::Environment::daylight(),
            layout: crate::dock::Layout::new(),
            modifiers: terra_voxel::ModifierStack::default(),
            selected_modifier: None,
            noise: terra_voxel::NoiseField::default(),
            noise_library: Vec::new(),
            assets: [Vec::new(), Vec::new(), Vec::new()],
            asset_kind: crate::ui::AssetKind::Texture,
            selected_material: 0,
            material_dirty: false,
            viewport_rect: None,
            stroke: Vec::new(),
            pending_stroke_finish: false,
            play: None,
            last_frame: Instant::now(),
            next_frame: Instant::now(),
            occluded: false,
            frame_ms: 0.0,
            presented: 0,
            skipped: 0,
            last_present: None,
            last_report: Instant::now(),
            stats: FrameStats::default(),
            update_ms: 0.0,
            cpu_ms: 0.0,
        })
    }

    /// Put the whole world back in view.
    ///
    /// Bound to `F`. Frames the terrain's own extent rather than a selection,
    /// because the terrain is what the editor is for and there is always exactly
    /// one of it.
    fn frame_world(&mut self) {
        let Some(world) = self.world.as_mut() else { return };
        let centre = glam::Vec3::new(0.0, BASE_ELEVATION_M, 0.0);
        // Half the diagonal, so the corners fit and not just the edges.
        let radius = world.terrain.extent_m() * 0.5 * std::f32::consts::SQRT_2;
        world.camera.frame(centre, radius);
        log::info!("framed the world from {}", world.camera.pos);
    }

    /// True only while actually editing: a world is open and it is not being
    /// driven.
    ///
    /// `Screen::Editor` is not the same thing -- driving is a sub-state of it,
    /// with `play` set -- and conflating the two is what let the visualization
    /// modes leak into Play. Debug views belong to authoring; while driving, the
    /// viewport is the game and should show what the player sees.
    fn is_editing(&self) -> bool {
        self.screen == Screen::Editor && self.play.is_none()
    }

    /// The visualization the viewport should draw with right now.
    ///
    /// Lit anywhere but the editing viewport. The chosen mode is *remembered*
    /// rather than reset, so leaving Play returns to the view that was set up --
    /// which is what Unreal does when you stop a Play-In-Editor session.
    fn active_view_mode(&self) -> terra_render::ViewMode {
        if self.is_editing() { self.view_mode } else { terra_render::ViewMode::Lit }
    }

    // --- world lifecycle ---

    fn begin_load(&mut self, project: Project) {
        self.library.touch(&project.manifest.name, project.paths.root());
        if let Err(e) = self.library.save() {
            log::warn!("could not update project index: {e}");
        }
        self.loading = Some(Loading::new(project));
        self.screen = Screen::Loading;
    }

    fn create_world(&mut self, name: &str, size: WorldSize, seed: u64) {
        let root = match Library::default_projects_dir() {
            Ok(dir) => dir.join(unique_folder(&dir, name)),
            Err(e) => {
                log::error!("no data directory: {e}");
                return;
            }
        };
        match Project::create(&root, name, size, seed) {
            Ok(p) => {
                log::info!("created {name} at {}", root.display());
                self.begin_load(p);
            }
            Err(e) => log::error!("could not create world: {e}"),
        }
    }

    fn finish_loading(&mut self, l: Loading) {
        let Some(terrain) = l.terrain else { return };
        // The project's own meshes first: everything below either restores painting
        // onto these species or resolves a placed prop against them by name.
        self.reload_species(&l.project.paths);

        // A world opening replaces whatever the last one painted; species are
        // shared, their painting is not.
        if let Some(gfx) = self.gfx.as_mut() {
            for s in &mut gfx.scatter.species {
                s.clear_density();
            }
            if let Some(bytes) = l.foliage.as_deref() {
                gfx.scatter.restore(bytes);
            }
            // Props resolve by species name; one whose model has since been
            // removed is dropped rather than silently becoming something else.
            gfx.scatter.props.clear();
            for p in &l.props.props {
                match gfx.scatter.species.iter().position(|s| s.name == p.species) {
                    Some(species) => {
                        gfx.scatter.place(species, Vec3::from(p.pos), p.scale, p.yaw);
                    }
                    None => log::warn!("placed object references missing species '{}'", p.species),
                }
            }
            gfx.scatter.touch_props();
        }
        self.selected_prop = None;
        // The saved environment, or daylight for a world that has none. Restored
        // here rather than in `Loading` because it needs no disk work worth staging
        // and because a half-loaded world should never be lit by the last one's sun.
        self.env = terra_render::Environment::load(&l.project.paths).unwrap_or_default();
        self.saved_env = self.env;
        // Start above the flat base so a new world is in frame immediately.
        let camera =
            Camera { pos: Vec3::new(0.0, BASE_ELEVATION_M + 380.0, 900.0), ..Camera::default() };
        self.world = Some(OpenWorld {
            project: l.project,
            terrain,
            camera,
            base: l.heights.unwrap_or_default(),
            roads: l.roads,
            flow: l.flow,
            deposition: l.deposition,
        });
        self.unsaved = false;
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.taa.invalidate();
            gfx.clouds.invalidate();
        }
        // The content browser reads the project folder, so it has to be
        // populated once the project is actually open.
        self.refresh_assets();
        self.reload_materials();
        self.modifiers = terra_voxel::ModifierStack::default();
        self.selected_modifier = None;
        self.screen = Screen::Editor;
    }

    /// Rebuild the heightfield from scratch: ridged multifractal base, thermal
    /// relax, GPU hydraulic erosion, then thermal talus.
    ///
    /// Blocking. Generation is an explicit action costing a few seconds, and
    /// staging it across frames would mean carrying the whole pipeline as
    /// resumable state for no benefit the user would notice.
    fn generate_world(&mut self) {
        let (Some(world), Some(gfx)) = (self.world.as_mut(), self.gfx.as_ref()) else {
            return;
        };
        let res = world.terrain.resolution();
        let extent = world.terrain.extent_m();
        let cell = extent / (res - 1) as f32;
        let p = world.project.world.terrain;

        let t0 = Instant::now();
        // Lift off the floor: storage is unsigned, and erosion needs room to
        // cut valleys below the starting surface.
        let mut base = terra_gen::heightfield::generate(res, extent, &p.rmf);
        for h in &mut base {
            *h += BASE_ELEVATION_M;
        }

        let t1 = Instant::now();
        let relaxed =
            terra_gen::thermal::run(&base, res, cell, &p.thermal, p.thermal.pre_iterations);

        let t2 = Instant::now();
        let sim = terra_gen::erosion::Erosion::new(
            &gfx.ctx.device,
            &gfx.ctx.queue,
            res,
            cell,
            &p.erosion,
        );
        let result =
            sim.run(&gfx.ctx.device, &gfx.ctx.queue, &relaxed, p.erosion.iterations, |_| {});

        let t3 = Instant::now();
        let ok = result.height.len() == relaxed.len();
        if !ok {
            log::error!("erosion returned no data; keeping the un-eroded terrain");
        }
        let carved = if ok { result.height } else { relaxed.clone() };

        // Free by-products of the solve: where water ran, and where material
        // moved. These become the material masks -- no hand painting.
        let flow = terra_gen::erosion::Erosion::normalize_flow(&result.flow);
        let deposition = terra_gen::erosion::Erosion::deposition_map(&relaxed, &carved);

        let finished =
            terra_gen::thermal::run(&carved, res, cell, &p.thermal, p.thermal.post_iterations);

        log::info!(
            "generated {res}x{res}: rmf {:.0} ms, thermal {:.0} ms, erosion {:.0} ms ({} iters), \
             talus {:.0} ms",
            (t1 - t0).as_secs_f32() * 1000.0,
            (t2 - t1).as_secs_f32() * 1000.0,
            (t3 - t2).as_secs_f32() * 1000.0,
            p.erosion.iterations,
            t3.elapsed().as_secs_f32() * 1000.0,
        );

        world.terrain.set_masks(&gfx.ctx.queue, &flow, &deposition);
        world.flow = flow;
        world.deposition = deposition;
        world.base = finished;
        self.unsaved = true;
        // Roads survive a regenerate -- that is the whole point of keeping them
        // as splines over a base layer.
        self.rebuild_roads();
        // So does foliage, but only if it is told the ground moved. Scatter is
        // derived, so it regenerates at the new heights; props hold their
        // position as data and have to be put back on the surface.
        self.invalidate_foliage();

        // The clock skips by however long generation took; do not let that
        // land in the frame-time history as a stutter.
        self.last_present = None;
        self.last_frame = Instant::now();
    }

    /// Re-stamp every road onto a fresh copy of the base terrain.
    ///
    /// Always from `base`, never incrementally on the live field: stamping is
    /// destructive, so re-applying onto an already-roaded surface would leave
    /// the cut of a road that has since moved.
    /// Enter drive mode: build the terrain collider and drop a car onto it.
    fn start_play(&mut self) {
        let Some(world) = self.world.as_ref() else { return };

        let mut physics = PhysicsWorld::new();
        // Straight from the CPU heightfield -- no GPU readback, which is why
        // sculpting was kept CPU-side in the first place.
        physics.set_terrain(
            &world.terrain.heights,
            world.terrain.resolution(),
            world.terrain.extent_m(),
        );

        // Drop it in front of the current camera, above the ground.
        let eye = world.camera.pos;
        let ground = world.terrain.height_at(eye.x, eye.z);
        // Obstacles around the spawn. Streamed, per docs/physics.md -- a
        // collider per scatter instance is not viable, and nothing beyond a few
        // hundred metres can be hit before the set is rebuilt.
        if let Some(gfx) = self.gfx.as_ref() {
            let solids = gfx.scatter.obstacles_near(&world.terrain, eye, OBSTACLE_RADIUS_M);
            physics.set_obstacles(&to_obstacles(&solids));
            log::info!("play: {} obstacle colliders", solids.len());
        }

        // Just clear of the ground, not three metres up. The body's origin is now the
        // centre of its contact patch, so a big drop means a 2.9 t vehicle landing on its
        // springs the moment play starts.
        let dims = self.gfx.as_ref().map(|g| g.vehicle_dims).unwrap_or(PLACEHOLDER_VEHICLE);
        let car = Vehicle::spawn(&mut physics, [eye.x, ground + 0.5, eye.z], &dims);

        let (t, r) = car.chassis_pose(&physics);
        let pose = Pose {
            translation: Vec3::from_array(t),
            rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]),
        };
        let wheels = wheel_poses(&car, pose.rotation);

        self.obstacle_origin = eye;
        self.play = Some(Play {
            world: physics,
            car,
            accumulator: 0.0,
            prev: pose,
            curr: pose,
            prev_wheels: wheels.clone(),
            curr_wheels: wheels,
            camera: Camera { fov_y: 62f32.to_radians(), ..Camera::default() },
        });
        log::info!("play: terrain collider built, car spawned");
    }

    fn stop_play(&mut self) {
        self.play = None;
    }

    /// Rebuild the obstacle set when the car has left the one it was given.
    ///
    /// Half the radius is the trigger, so there is always a margin of built
    /// colliders ahead of the car rather than a boundary it can cross before
    /// the rebuild lands.
    fn stream_obstacles(&mut self) {
        let Some(play) = self.play.as_ref() else { return };
        let pos = play.curr.translation;
        if pos.distance(self.obstacle_origin) < OBSTACLE_RADIUS_M * 0.5 {
            return;
        }
        self.obstacle_origin = pos;
        let (Some(gfx), Some(world)) = (self.gfx.as_ref(), self.world.as_ref()) else {
            return;
        };
        let solids = gfx.scatter.obstacles_near(&world.terrain, pos, OBSTACLE_RADIUS_M);
        let obstacles = to_obstacles(&solids);
        if let Some(play) = self.play.as_mut() {
            play.world.set_obstacles(&obstacles);
        }
    }

    /// Advance the simulation at a fixed rate, whatever the frame rate is.
    fn update_play(&mut self, dt: f32) {
        let Some(play) = self.play.as_mut() else { return };

        let input = VehicleInput {
            throttle: (self.input.fwd as i32 - self.input.back as i32) as f32,
            brake: if self.input.down { 1.0 } else { 0.0 },
            steer: (self.input.left as i32 - self.input.right as i32) as f32,
            handbrake: self.input.boost,
        };

        // Clamp the catch-up. Without a cap, one long stall makes the loop try
        // to simulate the whole gap at once and stall again -- the spiral of
        // death.
        play.accumulator = (play.accumulator + dt).min(0.25);
        while play.accumulator >= FIXED_DT {
            play.prev = play.curr;
            play.prev_wheels.clone_from(&play.curr_wheels);

            play.car.update(&mut play.world, input, FIXED_DT);
            play.world.step();

            let (t, r) = play.car.chassis_pose(&play.world);
            play.curr = Pose {
                translation: Vec3::from_array(t),
                rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]),
            };
            play.curr_wheels = wheel_poses(&play.car, play.curr.rotation);
            play.accumulator -= FIXED_DT;
        }

        self.stream_obstacles();
        let Some(play) = self.play.as_mut() else { return };

        // Chase camera follows the interpolated pose, smoothed so it lags the
        // vehicle slightly rather than being welded to it.
        //
        // Placed *behind* the vehicle, which it previously was not: the offset was
        // `- back`, putting the camera ahead of the bonnet, and that only looked right
        // because the vehicle used to drive backwards. Both are fixed together, since one
        // was compensating for the other.
        //
        // Distances scale with the vehicle rather than being fixed at the 9 m that suited a
        // 3.6 m hatchback -- a 5.2 m Hummer at that range fills the frame.
        let alpha = play.accumulator / FIXED_DT;
        let pose = interpolate(play.prev, play.curr, alpha);
        let dims = self.gfx.as_ref().map(|g| g.vehicle_dims).unwrap_or(PLACEHOLDER_VEHICLE);
        let back = pose.rotation * Vec3::new(0.0, 0.0, -1.0);
        let distance = dims.length() * 2.2;
        let height = dims.chassis_centre_y + dims.chassis_half[1] * 2.2;
        let want = pose.translation + back * distance + Vec3::Y * height;
        let follow = 1.0 - (-dt * 9.0).exp();
        play.camera.pos = play.camera.pos.lerp(want, follow);

        // Aim at the roof line rather than the contact patch, or a tall vehicle sits at the
        // bottom of the frame with the sky above it.
        let to_car = (pose.translation + Vec3::Y * dims.chassis_centre_y) - play.camera.pos;
        play.camera.yaw = to_car.z.atan2(to_car.x);
        play.camera.pitch = (to_car.y / to_car.length().max(0.01)).clamp(-1.0, 1.0).asin();
    }

    /// Tell the foliage the terrain moved.
    ///
    /// `docs/physics.md` makes the same point about colliders: rebuild on
    /// stroke end, not per dab. A held brush would otherwise regenerate every
    /// species seventy-five times a second.
    fn invalidate_foliage(&mut self) {
        let Some(world) = self.world.as_ref() else { return };
        let Some(gfx) = self.gfx.as_mut() else { return };
        gfx.scatter.mark_all_dirty();
        gfx.scatter.reground_props(&world.terrain);
    }

    // --- content browser ---

    /// Re-read the project's asset folder.
    ///
    /// Cheap and called after every import rather than maintained
    /// incrementally: the folder is the single source of truth, and a cached
    /// list that drifts from it is how a browser starts showing assets that are
    /// no longer there.
    fn refresh_assets(&mut self) {
        use crate::ui::AssetKind;
        let Some(w) = self.world.as_ref() else {
            self.assets = [Vec::new(), Vec::new(), Vec::new()];
            self.noise_library.clear();
            return;
        };
        let root = w.project.paths.assets_dir();
        for (slot, kind) in AssetKind::ALL.iter().enumerate() {
            let dir = root.join(kind.folder());
            let mut names: Vec<String> = std::fs::read_dir(&dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    let p = e.path();
                    // Skip dotfiles. `.cache` lives beside the material folders and
                    // was being listed as though the user had imported a texture
                    // called ".cache".
                    if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.')) {
                        return None;
                    }
                    // Textures are folders (a set of maps); the others are files.
                    let wanted = if *kind == AssetKind::Texture {
                        p.is_dir()
                    } else {
                        p.extension()
                            .and_then(|x| x.to_str())
                            .is_some_and(|x| kind.extensions().contains(&x.to_lowercase().as_str()))
                    };
                    wanted.then(|| p.file_name()?.to_str().map(str::to_owned))?
                })
                .collect();
            names.sort();
            self.assets[slot] = names;
        }
        self.noise_library = self.assets[1].clone();
    }

    /// Copy a file the user picks into the project's own asset folder.
    ///
    /// Copied, not referenced. A project that stores a path to the user's
    /// Downloads folder stops working the moment it is moved or shared, and
    /// `README.md` already promises projects are self-contained and movable.
    fn import_asset(&mut self, kind: crate::ui::AssetKind) {
        let Some(w) = self.world.as_ref() else { return };
        let dest_dir = w.project.paths.assets_dir().join(kind.folder());
        if let Err(e) = std::fs::create_dir_all(&dest_dir) {
            log::error!("could not create {}: {e}", dest_dir.display());
            return;
        }
        let picked = rfd::FileDialog::new()
            .add_filter(kind.label(), kind.extensions())
            .set_title(format!("Import {}", kind.label()))
            .pick_files();
        let Some(files) = picked else { return };

        for src in files {
            let Some(name) = src.file_name() else { continue };
            let dest = unique_path(&dest_dir, name);
            match std::fs::copy(&src, &dest) {
                Ok(_) => log::info!("imported {}", dest.display()),
                Err(e) => log::error!("could not import {}: {e}", src.display()),
            }
        }
        self.refresh_assets();
        match kind {
            crate::ui::AssetKind::Texture => self.reload_materials(),
            crate::ui::AssetKind::Model => {
                if let Some(paths) = self.world.as_ref().map(|w| w.project.paths.clone()) {
                    self.reload_species(&paths);
                }
            }
            // Noise maps are chosen explicitly from the browser, so there is
            // nothing to rebuild until one is picked.
            crate::ui::AssetKind::Noise => {}
        }
    }

    /// Rebuild the foliage palette from the open project's model folder.
    ///
    /// The same reasoning as `reload_materials`, and previously missing: the palette
    /// was built once at startup from the *repository's* `assets/models`, so a mesh
    /// imported into a project never appeared at all. Nothing the user uploaded could
    /// be scattered, which made the Foliage tool a viewer for four built-in shapes.
    fn reload_species(&mut self, paths: &terra_project::ProjectPaths) {
        let dir = paths.assets_dir().join("models");
        let Some(gfx) = self.gfx.as_mut() else { return };
        let meshes = &gfx.meshes;
        gfx.scatter.reload(&gfx.ctx.device, &gfx.ctx.queue, meshes, &dir);

        // Thumbnails are registered with egui by index, so they have to be dropped
        // for the new palette to register -- the same trap the material swatches hit.
        self.species_swatches.clear();
        self.species = 0;
        self.selected_prop = None;
        self.pending_prop_refresh = true;
    }

    /// Rebuild the material palette from the open project's texture folder.
    ///
    /// Called after an import rather than only at startup: a content browser
    /// whose imports need a restart to appear is a content browser that does not
    /// work. The bind group *layout* is unchanged, so the pipelines stay valid
    /// and only the group and a few uniform fields move -- which is what
    /// `Terrain::set_materials` swaps.
    fn reload_materials(&mut self) {
        let Some(dir) = self.world.as_ref().map(|w| w.project.paths.assets_dir().join("textures"))
        else {
            return;
        };
        let Some(gfx) = self.gfx.as_mut() else { return };
        gfx.materials = Materials::load(&gfx.ctx.device, &gfx.ctx.queue, &dir);
        gfx.materials.upload_params(&gfx.ctx.queue);
        log::info!("palette rebuilt: {} materials", gfx.materials.count());

        // Thumbnails are registered with egui once and keyed by count, so they
        // have to be dropped for the new palette to register.
        self.swatches.clear();
        self.selected_material = 0;
        self.paint_layer = 0;

        let materials = &gfx.materials;
        let queue = &gfx.ctx.queue;
        if let Some(w) = self.world.as_mut() {
            w.terrain.set_materials(queue, materials);
        }
        if let Some(b) = self.backdrop.as_mut() {
            b.set_materials(queue, materials);
        }
    }

    /// Load an imported greyscale map and make it the Noise brush's pattern.
    fn select_noise(&mut self, name: &str) {
        let Some(w) = self.world.as_ref() else { return };
        let path = w.project.paths.assets_dir().join("noise").join(name);
        let img = match image::open(&path) {
            Ok(i) => i.to_luma8(),
            Err(e) => {
                log::error!("could not read noise map {}: {e}", path.display());
                return;
            }
        };
        let (wpx, hpx) = (img.width(), img.height());
        match terra_voxel::NoiseImage::from_gray8(name, wpx, hpx, img.as_raw()) {
            Ok(n) => {
                // Warn rather than reject: a map that is not centred on
                // mid-grey still works, it just shifts the surface as well as
                // roughening it, and the user may well want that.
                let mean = n.mean();
                if (mean - 0.5).abs() > 0.15 {
                    log::warn!(
                        "{name} averages {mean:.2} rather than mid-grey, so it will bias the \
                         surface as well as roughen it"
                    );
                }
                self.noise.pattern = terra_voxel::NoisePattern::Image(n);
                log::info!("noise pattern set to {name} ({wpx}x{hpx})");
            }
            Err(e) => log::error!("{name} is not usable as a noise map: {e}"),
        }
    }

    // --- cave modifiers ---

    /// Bore a tunnel from the camera, along the direction it is looking.
    fn add_tunnel(&mut self) {
        let Some(w) = self.world.as_ref() else { return };
        let cam = &w.camera;
        let dir = cam.forward();
        let start = cam.pos + dir * 20.0;
        let end = cam.pos + dir * 220.0;
        let n = self.modifiers.len() + 1;
        self.modifiers.push(terra_voxel::Modifier::carve(
            format!("Tunnel {n}"),
            terra_voxel::Shape::Tube(terra_voxel::Tube::straight(start, end, 6.0)),
            1.5,
        ));
        self.selected_modifier = Some(self.modifiers.len() - 1);
        self.unsaved = true;
        log::info!("added tunnel {n} from {start} to {end}");
    }

    fn rebuild_roads(&mut self) {
        let (Some(world), Some(gfx)) = (self.world.as_mut(), self.gfx.as_ref()) else {
            return;
        };
        let mut composed = world.base.clone();
        let surface = terra_gen::road::stamp_network(
            &mut composed,
            world.terrain.resolution(),
            world.terrain.extent_m(),
            &world.roads,
        );
        world.terrain.set_heights(&gfx.ctx.queue, composed);
        world.terrain.set_road_masks(&gfx.ctx.queue, &surface.mask, &surface.rut);
        self.unsaved = true;
    }

    fn save_world(&mut self) {
        let Some(w) = self.world.as_ref() else { return };
        // The base, not what is on screen. Roads are geometry and are saved
        // separately; baking them into the heightmap would make them permanent.
        let data = WorldData {
            heights: w.base.clone(),
            flow: w.flow.clone(),
            deposition: w.deposition.clone(),
        };
        // Painting is authored work and is saved even when untouched-and-empty
        // is the common case, because the file's absence is what means "never
        // painted" on the way back in.
        let splat_path = w.project.paths.splat();
        if w.terrain.is_painted() {
            if let Some(parent) = splat_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&splat_path, w.terrain.splat()) {
                log::error!("could not save material painting: {e}");
                return;
            }
        } else {
            let _ = std::fs::remove_file(&splat_path);
        }

        let foliage_path = w.project.paths.foliage();
        if let Some(gfx) = self.gfx.as_ref() {
            if gfx.scatter.is_painted() {
                if let Some(parent) = foliage_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Err(e) = std::fs::write(&foliage_path, gfx.scatter.save()) {
                    log::error!("could not save foliage: {e}");
                    return;
                }
            } else {
                let _ = std::fs::remove_file(&foliage_path);
            }
        }

        if let Some(gfx) = self.gfx.as_ref() {
            let set = terra_project::props::PropSet {
                props: gfx
                    .scatter
                    .props
                    .iter()
                    .filter_map(|p| {
                        let sp = gfx.scatter.species.get(p.species)?;
                        Some(terra_project::props::Prop {
                            species: sp.name.clone(),
                            pos: p.pos.to_array(),
                            yaw: p.yaw,
                            scale: p.scale,
                        })
                    })
                    .collect(),
            };
            if let Err(e) = set.save(&w.project.paths) {
                log::error!("could not save placed objects: {e}");
                return;
            }
        }

        if let Err(e) = w.roads.save(&w.project.paths) {
            log::error!("could not save roads: {e}");
            return;
        }
        // The mixer, so the sun, fog, clouds and time of day are still there on the
        // way back in. Saved even at its defaults: the file's presence is what
        // separates "authored a daylight scene" from "never touched it".
        if let Err(e) = self.env.save(&w.project.paths) {
            log::error!("could not save environment: {e}");
            return;
        }
        if let Err(e) = data.save(&w.project.paths, w.project.size()) {
            log::error!("could not save world data: {e}");
            return;
        }
        if let Err(e) = w.project.save() {
            log::error!("could not save manifests: {e}");
            return;
        }
        log::info!("saved {}", w.project.manifest.name);
        self.saved_env = self.env;
        self.unsaved = false;
    }

    fn exit_editor(&mut self) {
        // Save on the way out rather than prompting. The alternative is a
        // modal, and losing a sculpt session to a mis-click is worse than an
        // extra write of a few megabytes.
        if self.unsaved {
            self.save_world();
        }
        self.world = None;
        self.screen = Screen::Menu(Pane::Worlds);
    }

    // --- per-frame update ---

    fn update(&mut self, dt: f32) {
        self.time += dt;

        // The mixer is the source of truth, so it is pushed into the derived
        // sky and fog settings exactly here -- once per frame, before anything
        // reads them. Doing it at each edit site instead is how the two used to
        // fall out of step.
        self.env.tick(dt);
        // An environment edit is an edit. Without this, `exit_editor` -- which only
        // saves when the world is dirty -- would drop a session spent finding the
        // right dusk, and the panel would look like it did nothing.
        if self.world.is_some() && self.env.differs_for_saving(&self.saved_env) {
            self.unsaved = true;
        }
        if let Some(g) = self.gfx.as_mut() {
            self.env.apply_to(&mut g.lighting.settings, &mut g.fog.settings);
            // And the light values themselves, so the terrain is lit by the same
            // model the sky is drawn with. Without this the two disagree at dusk.
            g.lighting.set_environment(&self.env);
        }
        // The shader branches on the mode, so it has to reach the uniform. Both
        // terrains get it: the backdrop's is forced to Lit at draw time, but
        // leaving its uniform stale would matter the moment that changes.
        let mode = self.active_view_mode();
        if let Some(g) = self.gfx.as_ref() {
            let queue = &g.ctx.queue;
            if let Some(w) = self.world.as_mut() {
                w.terrain.set_view_mode(queue, mode);
            }
            if let Some(b) = self.backdrop.as_mut() {
                b.set_view_mode(queue, terra_render::ViewMode::Lit);
            }
            // The sky and cloud passes read this directly. `self.time` drives
            // wind, so clouds advect with the same clock everything else uses.
            g.env_gpu.upload(&g.ctx.queue, &self.env, self.time);
        }
        // Mouse motion and wheel notches are only consumed while editing, so any
        // that arrive in the menu, during a load, or while driving have to be
        // dropped rather than banked.
        if !self.is_editing() {
            self.input.clear_motion();
        }

        match self.screen {
            Screen::Editor if self.play.is_some() => self.update_play(dt),
            Screen::Editor => {
                self.update_editor(dt);
                if self.pending_stroke_finish {
                    self.pending_stroke_finish = false;
                    self.finish_stroke();
                }
                // Sculpt stroke ended: the ground under the foliage has moved.
                let sculpting = self.input.sculpting && self.tool == Tool::Sculpt;
                if self.was_sculpting && !sculpting {
                    self.invalidate_foliage();
                }
                self.was_sculpting = sculpting;
                if let Some((centre, flow)) = self.pending_foliage.take() {
                    let (index, erase) = (self.species, self.paint_mode == PaintMode::Erase);
                    if let (Some(gfx), Some(w)) = (self.gfx.as_mut(), self.world.as_ref()) {
                        gfx.scatter.paint(
                            index,
                            &w.terrain,
                            centre,
                            self.brush_radius,
                            flow,
                            erase,
                        );
                    }
                }
                if let (Some(pos), Some(i), Some(gfx)) =
                    (self.pending_prop_move.take(), self.selected_prop, self.gfx.as_mut())
                    && let Some(p) = gfx.scatter.props.get_mut(i)
                {
                    p.pos = pos;
                    gfx.scatter.touch_props();
                    self.unsaved = true;
                }
                if self.pending_prop_refresh {
                    self.pending_prop_refresh = false;
                    if let Some(gfx) = self.gfx.as_mut() {
                        gfx.scatter.touch_props();
                    }
                }
                let selected = self.selected_prop;
                if let Some(gfx) = self.gfx.as_mut() {
                    let device = &gfx.ctx.device;
                    gfx.scatter.refresh_props(device, selected);
                }

                // Regenerating here rather than at the point of edit means a
                // held brush or a dragged slider costs one rebuild a frame at
                // most, not one per event.
                if let (Some(gfx), Some(w)) = (self.gfx.as_mut(), self.world.as_ref()) {
                    let (device, scatter) = (&gfx.ctx.device, &mut gfx.scatter);
                    scatter.refresh(device, &w.terrain);
                }
            }
            Screen::Loading => {
                self.update_backdrop();
                self.update_loading(dt);
            }
            Screen::Menu(_) => self.update_backdrop(),
        }

        // Sub-pixel offset for this frame, applied to every camera the scene
        // is drawn from so the resolve can undo exactly what was added.
        self.frame_jitter = self
            .gfx
            .as_ref()
            .map(|g| g.taa.jitter(g.ctx.config.width, g.ctx.config.height))
            .unwrap_or_default();
        let j = self.frame_jitter;
        self.backdrop_cam.jitter = j;
        if let Some(w) = self.world.as_mut() {
            w.camera.jitter = j;
        }
        if let Some(p) = self.play.as_mut() {
            p.camera.jitter = j;
        }

        // The sky follows whichever camera is active this frame.
        let aspect = self.gfx.as_ref().map(|g| g.ctx.aspect()).unwrap_or(1.0);
        // Driving always shows the world's own sun. The editor shows it only
        // when asked, and the menu never does -- the backdrop is a fixed
        // portrait, not a scene anyone is authoring.
        let mode = match (self.screen, self.play.is_some()) {
            (Screen::Editor, true) => LightMode::Scene,
            (Screen::Editor, false) => {
                let preview = self.gfx.as_ref().is_some_and(|g| g.lighting.settings.editor_preview);
                if preview { LightMode::Scene } else { LightMode::Fixed }
            }
            _ => LightMode::Fixed,
        };
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.lighting.update(dt, mode);
            // Resolution changes reallocate the map, so this has to happen
            // outside the frame that samples it.
            let grid = gfx.fog.scattered_view();
            gfx.lighting.apply_quality(&gfx.ctx.device, grid);
        }
        let cam = match (self.screen, self.play.as_ref(), self.world.as_ref()) {
            (Screen::Editor, Some(p), _) => p.camera.clone(),
            (Screen::Editor, None, Some(w)) => w.camera.clone(),
            _ => self.backdrop_cam.clone(),
        };
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.sky.upload_camera(&gfx.ctx.queue, &cam, aspect);
            gfx.meshes.upload_camera(&gfx.ctx.queue, &cam, aspect);
            // Cascades are fitted to the camera actually being rendered from.
            let f = &gfx.fog.settings;
            gfx.lighting.upload(
                &gfx.ctx.queue,
                &cam,
                aspect,
                [
                    gfx.fog.near(),
                    f.distance,
                    if f.enabled { 1.0 } else { 0.0 },
                    terra_render::volumetrics::FROXELS[2] as f32,
                ],
                [gfx.ctx.config.width as f32, gfx.ctx.config.height as f32],
            );
        }
        // While driving, the terrain must be drawn from the chase camera too.
        if let (Some(p), Some(w), Some(gfx)) =
            (self.play.as_ref(), self.world.as_mut(), self.gfx.as_ref())
        {
            w.terrain.upload_camera(&gfx.ctx.queue, &p.camera, aspect);
            // No brush ring while driving.
            w.terrain.set_brush(&gfx.ctx.queue, None, 0.0);
        }
    }

    fn update_loading(&mut self, dt: f32) {
        let Some(gfx) = self.gfx.as_ref() else { return };
        let mut done = false;

        if let Some(l) = self.loading.as_mut() {
            l.elapsed += dt;
            match l.stage {
                0 => {
                    l.terrain = Some(Terrain::new(
                        &gfx.ctx,
                        l.project.size(),
                        &gfx.materials,
                        &gfx.lighting,
                        &gfx.clouds,
                    ));
                    l.stage = 1;
                }
                1 => {
                    let data = WorldData::load(&l.project.paths, l.project.size());
                    l.heights = Some(data.heights);
                    l.flow = data.flow;
                    l.deposition = data.deposition;
                    l.roads = RoadNetwork::load(&l.project.paths);
                    l.stage = 2;
                }
                2 => {
                    if let (Some(t), Some(base)) = (l.terrain.as_mut(), l.heights.clone()) {
                        t.set_masks(&gfx.ctx.queue, &l.flow, &l.deposition);
                        // Roads are stamped onto a copy; `base` stays clean.
                        let mut composed = base;
                        let surface = terra_gen::road::stamp_network(
                            &mut composed,
                            t.resolution(),
                            t.extent_m(),
                            &l.roads,
                        );
                        t.set_heights(&gfx.ctx.queue, composed);
                        t.set_road_masks(&gfx.ctx.queue, &surface.mask, &surface.rut);
                        if let Ok(splat) = std::fs::read(l.project.paths.splat()) {
                            t.set_splat(&gfx.ctx.queue, splat);
                        }
                        l.foliage = std::fs::read(l.project.paths.foliage()).ok();
                        l.props = terra_project::props::PropSet::load(&l.project.paths);
                    }
                    l.stage = 3;
                }
                _ => done = l.elapsed >= Loading::MIN_SECONDS,
            }
        }

        if done && let Some(l) = self.loading.take() {
            self.finish_loading(l);
        }
    }

    /// Slow cinematic orbit around the menu landscape.
    fn update_backdrop(&mut self) {
        const RADIUS: f32 = 1180.0;
        // TERRA_FREEZE pins the orbit so GPU timings are comparable between
        // runs. Without it the camera sweeps a ~140 s cycle and each sample
        // sees a different amount of terrain, which reads as a 20% swing in
        // frame cost that has nothing to do with the change being measured.
        let t = if std::env::var_os("TERRA_FREEZE").is_some() { 0.7 } else { self.time * 0.045 };
        self.backdrop_cam.pos = Vec3::new(t.cos() * RADIUS, 430.0, t.sin() * RADIUS);
        // Face the middle of the map: the inward direction is the orbit angle
        // turned by half a turn.
        self.backdrop_cam.yaw = t + std::f32::consts::PI;
        self.backdrop_cam.pitch = -0.15;

        let aspect = self.gfx.as_ref().map(|g| g.ctx.aspect()).unwrap_or(1.0);
        // Mutable because uploading the camera also reselects the LOD patches for
        // it -- the backdrop's slow orbit needs that as much as the editor camera.
        if let (Some(b), Some(gfx)) = (self.backdrop.as_mut(), self.gfx.as_ref()) {
            b.upload_camera(&gfx.ctx.queue, &self.backdrop_cam, aspect);
        }
    }

    /// Convert the raw cursor trail into control points and append them to the
    /// road being drawn.
    ///
    /// Smooth, then simplify. Raw input carries hand tremor and pixel
    /// quantisation; feeding it straight to the spline would make every jitter
    /// a control point, reproducing the noise instead of the intended line --
    /// and leaving a road with hundreds of handles nobody can edit.
    fn finish_stroke(&mut self) {
        let trail = std::mem::take(&mut self.stroke);
        if trail.len() < 2 {
            return;
        }
        let simplified =
            terra_gen::road::simplify(&terra_gen::road::smooth_path(&trail), STROKE_TOLERANCE_M);

        if let (Some(i), Some(w)) = (self.active_road, self.world.as_mut())
            && let Some(r) = w.roads.roads.get_mut(i)
        {
            // Drawing again extends the same road rather than starting a
            // new one, so a long track can be laid down in passes.
            if r.points.last() == simplified.first() {
                r.points.extend(simplified.into_iter().skip(1));
            } else {
                r.points.extend(simplified);
            }
        }
        self.rebuild_roads();
    }

    fn update_editor(&mut self, dt: f32) {
        let wants_kb = self.egui_ctx.egui_wants_keyboard_input();
        let wants_ptr = self.egui_ctx.egui_wants_pointer_input();
        let aspect = self.gfx.as_ref().map(|g| g.ctx.aspect()).unwrap_or(1.0);

        let (Some(world), Some(gfx)) = (self.world.as_mut(), self.gfx.as_ref()) else {
            // Still clear the motion. Returning with it accumulated is the same
            // bug as never consuming it: the next frame that does get a world
            // applies the whole backlog at once.
            self.input.clear_motion();
            return;
        };
        world.camera.speed = self.settings.camera_speed;
        world.camera.fov_y = self.settings.fov_deg.to_radians();

        // How far away what you are looking *at* is, which is what both pan and
        // zoom have to be scaled by.
        //
        // Previously this was the height above the ground directly below the
        // camera. That is the wrong quantity: standing five metres up and looking
        // at a ridge two kilometres away, a pan moved five metres' worth per
        // pixel and the drag felt frozen, while `pixel_scale` claims the ground
        // stays under the cursor. Raycasting the view direction gives the
        // distance to what is actually on screen.
        //
        // The fallback matters as much as the ray: aimed at the sky there is
        // nothing to hit, and the height above ground is the only sensible
        // stand-in for how big the world looks from here.
        let forward = world.camera.forward();
        let ground = world.terrain.height_at(world.camera.pos.x, world.camera.pos.z);
        let above_ground = (world.camera.pos.y - ground).abs();
        let focus = world.terrain.raycast(world.camera.pos, forward);
        let dist = focus
            .map(|hit| hit.distance(world.camera.pos))
            .unwrap_or(above_ground)
            .clamp(MIN_VIEW_DIST_M, MAX_VIEW_DIST_M);

        let (dx, dy) = self.input.look_delta;
        if self.input.orbiting && self.tool == Tool::Camera {
            // Orbit about what is in front of the camera. The same hit the pan
            // scale uses, so the two controls agree about what "the thing you are
            // looking at" is.
            let pivot = focus.unwrap_or(world.camera.pos + forward * dist);
            world.camera.orbit(pivot, dx * 0.006, dy * 0.006);
        } else if self.input.looking {
            world.camera.rotate(dx * 0.0025, dy * 0.0025);
        } else if self.input.panning {
            let scale = world.camera.pixel_scale(dist, gfx.ctx.config.height as f32);
            world.camera.pan(dx, dy, scale);
        }
        self.input.look_delta = (0.0, 0.0);

        // Wheel zoom. Each notch covers a fixed fraction of the distance to the
        // ground, so approaching a ridge slows down instead of overshooting
        // through it -- and backing off speeds up.
        let scroll = std::mem::take(&mut self.input.scroll);
        if scroll != 0.0 && !wants_ptr {
            let step = (dist * ZOOM_PER_NOTCH * scroll).clamp(-dist * 4.0, dist * 0.9);
            world.camera.dolly(step);
            // Never let a zoom bury the camera in the terrain; there is no way
            // back out of it except flying blind.
            let floor =
                world.terrain.height_at(world.camera.pos.x, world.camera.pos.z) + MIN_VIEW_DIST_M;
            world.camera.pos.y = world.camera.pos.y.max(floor);

            // And never let it strand itself in empty space. See
            // `MAX_VIEW_DIST_M`: the outward direction is geometric, so a few
            // seconds of scrolling reaches hundreds of kilometres.
            let centre = glam::Vec3::new(0.0, BASE_ELEVATION_M, 0.0);
            let offset = world.camera.pos - centre;
            let limit = world.terrain.extent_m() + MAX_VIEW_DIST_M;
            if offset.length() > limit {
                world.camera.pos = centre + offset.normalize_or_zero() * limit;
            }
        }

        if !wants_kb {
            // Speed scales with how far away what you are looking at is, rather
            // than being an absolute rate.
            //
            // A fixed 120 m/s is right at the default view distance and useless
            // anywhere else: from 30 km out the terrain does not visibly move, so
            // the camera reads as broken. `camera_speed` is the multiplier now,
            // and the reference is the distance the camera starts at.
            //
            // Clamped at both ends: proportional all the way down would crawl at
            // 0.5 m/s when four metres from the ground, and unbounded at the top
            // would overshoot the whole world in one keypress.
            const REFERENCE_DIST_M: f32 = 900.0;
            let scale = (dist / REFERENCE_DIST_M).clamp(0.2, 25.0);
            let restore = world.camera.speed;
            world.camera.speed = restore * scale;
            world.camera.translate(self.input.axis(), dt, self.input.boost);
            world.camera.speed = restore;
        }

        self.brush_hit = None;
        let mut finish_stroke = false;
        if !wants_ptr && !self.input.looking && !self.input.panning && self.tool.edits() {
            let (o, d) = {
                let (w, h) = (gfx.ctx.config.width as f32, gfx.ctx.config.height as f32);
                let ndc_x = (self.input.cursor.0 / w) * 2.0 - 1.0;
                let ndc_y = 1.0 - (self.input.cursor.1 / h) * 2.0;
                let inv = (world.camera.projection(aspect) * world.camera.look_at()).inverse();
                // Depth 1.0 is the near plane under reversed-Z.
                let p = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
                let near = p.truncate() / p.w;
                (world.camera.pos, (near - world.camera.pos).normalize_or_zero())
            };
            if self.tool == Tool::Select {
                if self.input.clicked {
                    // A fresh press picks; a press that misses clears, which is
                    // what makes clicking empty ground feel like a deselect
                    // rather than a dead click.
                    self.selected_prop = gfx.scatter.pick(o, d);
                    self.dragging_prop = self.selected_prop.is_some();
                    self.pending_prop_refresh = true;
                }
                if !self.input.sculpting {
                    self.dragging_prop = false;
                }
            }

            if let Some(hit) = world.terrain.raycast(o, d) {
                let c = Vec2::new(hit.x, hit.z);
                self.brush_hit = Some(c);
                // Dragging keeps the object on the surface: an object that
                // moves in a plane ends up floating over a valley.
                if self.dragging_prop && self.input.sculpting && self.tool == Tool::Select {
                    self.pending_prop_move = Some(Vec3::new(hit.x, hit.y, hit.z));
                }
                // Freehand: capture the cursor trail while the button is held.
                // Sampled by distance rather than per frame, so the density of
                // the stroke does not depend on how fast the mouse moved or
                // what the frame rate happened to be.
                if self.tool == Tool::Road && self.input.sculpting && self.active_road.is_some() {
                    const TRAIL_SPACING_M: f32 = 3.0;
                    let far_enough = self
                        .stroke
                        .last()
                        .is_none_or(|p| (c.x - p[0]).hypot(c.y - p[1]) >= TRAIL_SPACING_M);
                    if far_enough {
                        self.stroke.push([c.x, c.y]);
                    }
                }
                if self.input.sculpting && self.tool == Tool::Foliage {
                    let flow = (self.paint_flow * dt).clamp(0.0, 1.0);
                    self.pending_foliage = Some((c, flow));
                    self.unsaved = true;
                }
                if self.input.sculpting && self.tool == Tool::Paint {
                    // Scaled by dt for the same reason sculpting is: a stroke
                    // should deposit the same coverage whatever the frame rate.
                    let flow = (self.paint_flow * dt).clamp(0.0, 1.0);
                    match self.paint_mode {
                        crate::ui::PaintMode::Brush => world.terrain.paint(
                            &gfx.ctx.queue,
                            c,
                            self.brush_radius,
                            flow,
                            self.paint_layer,
                        ),
                        crate::ui::PaintMode::Erase => {
                            world.terrain.erase(&gfx.ctx.queue, c, self.brush_radius, flow)
                        }
                    }
                    self.unsaved = true;
                }
                if self.input.sculpting && self.tool == Tool::Sculpt {
                    // Scale by dt so a held stroke deposits the same material
                    // regardless of frame rate.
                    let strength = self.brush_strength * dt * 60.0;
                    let op = BrushOp {
                        mode: self.brush_mode,
                        drag: self.last_brush_hit.map(|p| c - p).unwrap_or(Vec2::ZERO),
                        noise: Some(&self.noise),
                    };
                    world.terrain.sculpt(&gfx.ctx.queue, c, self.brush_radius, strength, &op);
                    // Same edit into the base, so a road rebuild does not undo
                    // it and so it is what gets saved.
                    apply_brush(
                        &mut world.base,
                        world.terrain.resolution(),
                        world.terrain.extent_m(),
                        c,
                        self.brush_radius,
                        strength,
                        &op,
                    );
                    self.unsaved = true;
                }
            }
        }
        self.last_brush_hit = if self.input.sculpting { self.brush_hit } else { None };
        world.terrain.set_brush(&gfx.ctx.queue, self.brush_hit, self.brush_radius);
        world.terrain.upload_camera(&gfx.ctx.queue, &world.camera, aspect);

        // Consume the click regardless of whether anything used it, or one
        // press would keep firing every frame.
        self.input.clicked = false;

        // Stroke finished: turn the raw trail into control points.
        if self.tool == Tool::Road && !self.input.sculpting && !self.stroke.is_empty() {
            finish_stroke = true;
        }
        if finish_stroke {
            self.pending_stroke_finish = true;
        }
    }

    // --- ui ---

    fn build_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut quit = false;

        match self.screen {
            Screen::Loading => {
                let (name, stage, progress) = match self.loading.as_ref() {
                    Some(l) => (l.project.manifest.name.clone(), l.label(), l.progress()),
                    None => (String::new(), "", 0.0),
                };
                ui::loading(ui, &name, stage, progress);
            }
            Screen::Editor => {
                let height_res = self.world.as_ref().map(|w| w.terrain.resolution()).unwrap_or(0);
                let name = self
                    .world
                    .as_ref()
                    .map(|w| w.project.manifest.name.clone())
                    .unwrap_or_default();
                let size =
                    self.world.as_ref().map(|w| w.project.size()).unwrap_or(WorldSize::Medium);
                let brush_at = self.brush_hit.map(|c| (c.x, c.y));
                self.ensure_swatches(ui.ctx());
                self.ensure_species_swatches(ui.ctx());
                let painted = self.world.as_ref().is_some_and(|w| w.terrain.is_painted());
                let foliage: Vec<FoliageEntry> = self
                    .gfx
                    .as_ref()
                    .map(|g| {
                        g.scatter
                            .species
                            .iter()
                            .enumerate()
                            .map(|(i, s)| FoliageEntry {
                                name: s.name.clone(),
                                instances: s.instance_count(),
                                painted: s.is_painted(),
                                texture: self.species_swatches.get(i).cloned(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let foliage_instances =
                    self.gfx.as_ref().map(|g| g.scatter.total_instances()).unwrap_or(0);
                self.species = self.species.min(foliage.len().saturating_sub(1));
                let species_index = self.species;
                let names: Vec<(String, &'static str)> = self
                    .gfx
                    .as_ref()
                    .map(|g| {
                        g.materials
                            .layers
                            .iter()
                            .map(|l| (l.name.clone(), terra_render::material::role_label(l.role)))
                            .collect()
                    })
                    .unwrap_or_default();
                let palette: Vec<PaletteEntry<'_>> = names
                    .iter()
                    .enumerate()
                    .map(|(i, (name, role))| PaletteEntry {
                        name,
                        role,
                        texture: self.swatches.get(i),
                    })
                    .collect();

                let prop_count = self.gfx.as_ref().map(|g| g.scatter.props.len()).unwrap_or(0);
                // Edited on a copy and written back, so the panel never holds a
                // borrow of  across the rest of the UI build.
                let mut sky_settings =
                    self.gfx.as_ref().map(|g| g.lighting.settings).unwrap_or_default();
                // The selected object's editable fields, copied out and copied
                // back after the UI runs -- the panel cannot hold a borrow into
                // `gfx` while the species rules already do.
                let mut selection_view = self.selected_prop.and_then(|i| {
                    let g = self.gfx.as_ref()?;
                    let p = g.scatter.props.get(i)?;
                    let sp = g.scatter.species.get(p.species)?;
                    Some(ui::SelectionView {
                        species: sp.name.clone(),
                        scale: p.scale,
                        yaw: p.yaw,
                        // The mesh is normalised to one metre, so a prop's scale
                        // factor is its height in metres.
                        height: p.scale,
                    })
                });

                // Copied out and written back: the pane needs `&mut` on one
                // layer's params while the palette above it holds `gfx` shared.
                let mut material_params = self
                    .gfx
                    .as_ref()
                    .and_then(|g| g.materials.params.get(self.selected_material).copied());
                let material_meta = self.gfx.as_ref().and_then(|g| {
                    let l = g.materials.layers.get(self.selected_material)?;
                    Some((l.name.clone(), terra_render::material::role_label(l.role)))
                });
                let material = match (material_meta.as_ref(), material_params.as_mut()) {
                    (Some((name, role)), Some(params)) => Some(ui::MaterialView {
                        name,
                        role,
                        texture: self.swatches.get(self.selected_material),
                        params,
                    }),
                    _ => None,
                };

                // Taken last: this is the only mutable borrow of `gfx` here, so
                // every shared read of it has to have finished first.
                let species_rules = self
                    .gfx
                    .as_mut()
                    .and_then(|g| g.scatter.species.get_mut(species_index))
                    .map(|s| &mut s.rules);

                // Read out of the world before the mutable borrow below: the
                // content browser only needs the path as a string, and taking
                // it after `world.as_mut()` would overlap the two borrows.
                let content_root = self
                    .world
                    .as_ref()
                    .map(|w| w.project.paths.assets_dir().display().to_string())
                    .unwrap_or_default();
                let content = ui::ContentView {
                    root: &content_root,
                    entries: [&self.assets[0], &self.assets[1], &self.assets[2]],
                    kind: self.asset_kind,
                };

                // Everything borrowed from the open world in one place, so the
                // params and the active road come from a single mutable borrow
                // of disjoint fields rather than two overlapping ones.
                let active_index = self.active_road;
                let mut scratch = TerrainParams::default();
                let (params, active_road, road_count) = match self.world.as_mut() {
                    Some(w) => {
                        let count = w.roads.roads.len();
                        let road = active_index.and_then(|i| w.roads.roads.get_mut(i));
                        (&mut w.project.world.terrain, road, count)
                    }
                    None => (&mut scratch, None, 0),
                };

                let act = ui::editor(
                    ui,
                    &mut self.layout,
                    EditorView {
                        mode: &mut self.brush_mode,
                        radius: &mut self.brush_radius,
                        strength: &mut self.brush_strength,
                        world_name: &name,
                        size,
                        unsaved: self.unsaved,
                        brush_at,
                        height_res,
                        params,
                        tool: &mut self.tool,
                        palette: &palette,
                        selected_layer: &mut self.paint_layer,
                        paint_mode: &mut self.paint_mode,
                        paint_flow: &mut self.paint_flow,
                        painted,
                        foliage: &foliage,
                        selected_species: &mut self.species,
                        species_rules,
                        foliage_instances,
                        selection: selection_view.as_mut(),
                        prop_count,
                        tools_open: &mut self.tools_open,
                        inspector_open: &mut self.inspector_open,
                        sky: &mut sky_settings,
                        env: &mut self.env,
                        active_road,
                        road_count,
                        modifiers: &mut self.modifiers,
                        selected_modifier: &mut self.selected_modifier,
                        content: &content,
                        noise: &mut self.noise,
                        noise_library: &self.noise_library,
                        viewport_rect: &mut self.viewport_rect,
                        material,
                        view_mode: &mut self.view_mode,
                        playing: self.play.is_some(),
                        speed_kph: self
                            .play
                            .as_ref()
                            .map(|p| p.car.speed().abs() * 3.6)
                            .unwrap_or(0.0),
                    },
                );
                if let Some(g) = self.gfx.as_mut() {
                    g.lighting.settings = sky_settings;
                    g.taa.enabled = sky_settings.temporal_aa;
                }

                // Slider edits land on the copy; write them back.
                if let (Some(view), Some(i), Some(g)) =
                    (selection_view.as_ref(), self.selected_prop, self.gfx.as_mut())
                    && let Some(p) = g.scatter.props.get_mut(i)
                    && (p.scale != view.scale || p.yaw != view.yaw)
                {
                    p.scale = view.scale;
                    p.yaw = view.yaw;
                    g.scatter.touch_props();
                    self.unsaved = true;
                }

                if let (Some(p), Some(g)) = (material_params, self.gfx.as_mut())
                    && let Some(slot) = g.materials.params.get_mut(self.selected_material)
                    && *slot != p
                {
                    *slot = p;
                    self.material_dirty = true;
                }
                if self.material_dirty {
                    self.material_dirty = false;
                    if let Some(g) = self.gfx.as_ref() {
                        g.materials.upload_params(&g.ctx.queue);
                    }
                }

                // Ctrl inverts the mode that has an opposite.
                self.brush_mode = match (self.brush_mode, self.input.invert) {
                    (SculptMode::Raise, true) => SculptMode::Lower,
                    (SculptMode::Lower, true) => SculptMode::Raise,
                    (other, _) => other,
                };

                match act {
                    EditorAction::SelectAssetKind(k) => self.asset_kind = k,
                    EditorAction::OpenMaterial(i) => {
                        self.selected_material = i;
                        self.layout.focus(crate::dock::Tab::Material);
                    }
                    EditorAction::PaintWithSelectedMaterial => {
                        self.paint_layer = self.selected_material as u32;
                        self.tool = Tool::Paint;
                    }
                    EditorAction::ImportAsset(k) => self.import_asset(k),
                    EditorAction::SelectNoise(name) => self.select_noise(&name),
                    EditorAction::AddTunnel => self.add_tunnel(),
                    EditorAction::DeleteModifier(i) => {
                        self.modifiers.remove(i);
                        self.selected_modifier = None;
                        self.unsaved = true;
                    }
                    EditorAction::Save => self.save_world(),
                    EditorAction::Exit => self.exit_editor(),
                    EditorAction::Generate => self.generate_world(),
                    EditorAction::NewRoad => {
                        if let Some(w) = self.world.as_mut() {
                            w.roads.roads.push(Road::default());
                            self.active_road = Some(w.roads.roads.len() - 1);
                        }
                    }
                    EditorAction::UndoPoint => {
                        if let (Some(i), Some(w)) = (self.active_road, self.world.as_mut())
                            && let Some(r) = w.roads.roads.get_mut(i)
                        {
                            r.points.pop();
                        }
                        self.rebuild_roads();
                    }
                    EditorAction::FinishRoad => {
                        // Drop a road that never got enough points to exist.
                        if let (Some(i), Some(w)) = (self.active_road, self.world.as_mut())
                            && w.roads.roads.get(i).is_some_and(|r| !r.is_drawable())
                        {
                            w.roads.roads.remove(i);
                        }
                        self.active_road = None;
                        self.rebuild_roads();
                    }
                    EditorAction::ClearRoads => {
                        if let Some(w) = self.world.as_mut() {
                            w.roads.roads.clear();
                        }
                        self.active_road = None;
                        self.rebuild_roads();
                    }
                    EditorAction::RebuildRoads => self.rebuild_roads(),
                    EditorAction::FillMaterial => {
                        let layer = self.paint_layer;
                        if let (Some(w), Some(gfx)) = (self.world.as_mut(), self.gfx.as_ref()) {
                            w.terrain.fill(&gfx.ctx.queue, layer);
                            self.unsaved = true;
                        }
                    }
                    EditorAction::FillFoliage => {
                        let i = self.species;
                        if let Some(g) = self.gfx.as_mut() {
                            g.scatter.fill(i);
                        }
                        self.unsaved = true;
                    }
                    EditorAction::ClearFoliage => {
                        let i = self.species;
                        if let Some(g) = self.gfx.as_mut() {
                            g.scatter.clear(i);
                        }
                        self.unsaved = true;
                    }
                    EditorAction::ReseedFoliage => {
                        let i = self.species;
                        if let Some(g) = self.gfx.as_mut()
                            && let Some(s) = g.scatter.species.get_mut(i)
                        {
                            s.rules.seed = s.rules.seed.wrapping_add(1);
                            s.touch();
                        }
                        self.unsaved = true;
                    }
                    EditorAction::PlaceProp => {
                        // Placed where the brush ray hits, which is where the
                        // cursor already shows a ring -- no separate aim.
                        if let (Some(hit), Some(w)) = (self.brush_hit, self.world.as_ref()) {
                            let y = w.terrain.height_at(hit.x, hit.y);
                            let species = self.species;
                            if let Some(g) = self.gfx.as_mut() {
                                let yaw = (ui::fresh_seed() % 628) as f32 / 100.0;
                                // A hand-placed object arrives at the species' own
                                // height rather than one metre, which is what a scale
                                // of 1 would have meant against a unit mesh.
                                let scale = g
                                    .scatter
                                    .species
                                    .get(species)
                                    .map(|sp| sp.rules.height_m)
                                    .unwrap_or(1.0);
                                let i = g.scatter.place(
                                    species,
                                    Vec3::new(hit.x, y, hit.y),
                                    scale,
                                    yaw,
                                );
                                self.selected_prop = Some(i);
                            }
                            self.unsaved = true;
                        }
                    }
                    EditorAction::DeleteProp => {
                        if let (Some(i), Some(g)) = (self.selected_prop, self.gfx.as_mut()) {
                            g.scatter.remove_prop(i);
                        }
                        self.selected_prop = None;
                        self.unsaved = true;
                    }
                    EditorAction::Deselect => {
                        self.selected_prop = None;
                        if let Some(g) = self.gfx.as_mut() {
                            g.scatter.touch_props();
                        }
                    }
                    EditorAction::ClearPaint => {
                        if let (Some(w), Some(gfx)) = (self.world.as_mut(), self.gfx.as_ref()) {
                            w.terrain.clear_paint(&gfx.ctx.queue);
                            self.unsaved = true;
                        }
                    }
                    EditorAction::Play => self.start_play(),
                    EditorAction::Stop => self.stop_play(),
                    EditorAction::None => {}
                }
            }
            Screen::Menu(pane) => {
                let action = match pane {
                    Pane::Home => ui::home(ui, self.library.projects.len()),
                    Pane::Worlds => ui::worlds(ui, &self.library),
                    Pane::Create => ui::create(ui, &mut self.form),
                    Pane::Settings => ui::settings(
                        ui,
                        ui::SettingsView {
                            vsync: &mut self.settings.vsync,
                            perf_overlay: &mut self.settings.perf_overlay,
                            perf_graph: &mut self.settings.perf_graph,
                            camera_speed: &mut self.settings.camera_speed,
                            fov_deg: &mut self.settings.fov_deg,
                            uncapped_supported: self
                                .gfx
                                .as_ref()
                                .map(|g| g.ctx.supports_uncapped())
                                .unwrap_or(false),
                            gpu_timing: self
                                .gfx
                                .as_ref()
                                .map(|g| g.ctx.supports_timestamps())
                                .unwrap_or(false),
                        },
                    ),
                };
                match action {
                    Action::None => {}
                    Action::Go(p) => {
                        if p == Pane::Create {
                            self.form = CreateForm::default();
                        }
                        self.screen = Screen::Menu(p);
                    }
                    Action::Quit => quit = true,
                    Action::Create { name, size, seed } => self.create_world(&name, size, seed),
                    Action::Open(path) => match Project::open(&path) {
                        Ok(p) => self.begin_load(p),
                        Err(e) => log::error!("could not open {}: {e}", path.display()),
                    },
                    Action::Forget(path) => {
                        self.library.forget(&path);
                        let _ = self.library.save();
                    }
                }
            }
        }

        // Graphics panel while driving.
        if self.screen == Screen::Editor && self.play.is_some() {
            let mut sky = self.gfx.as_ref().map(|g| g.lighting.settings).unwrap_or_default();

            let mut open = self.graphics_open;
            ui::play_overlay(ui, &mut sky, &mut self.env, &mut open);
            self.graphics_open = open;
            if let Some(g) = self.gfx.as_mut() {
                g.lighting.settings = sky;
                g.taa.enabled = sky.temporal_aa;
            }
        }

        if self.settings.perf_overlay {
            ui::perf_overlay(
                ui,
                PerfView {
                    stats: &self.stats,
                    gpu_supported: self
                        .gfx
                        .as_ref()
                        .map(|g| g.ctx.supports_timestamps())
                        .unwrap_or(false),
                    graph: self.settings.perf_graph,
                    // Keep clear of whatever is docked to the right of the
                    // viewport. Measured from where the viewport pane actually
                    // ended up rather than from a panel width: under docking
                    // the user can move and resize the panes freely, so any
                    // constant here parks the HUD in the wrong place.
                    right_inset: match self.screen {
                        Screen::Editor => self
                            .viewport_rect
                            .map(|r| (self.egui_ctx.viewport_rect().right() - r.right()).max(0.0))
                            .unwrap_or(0.0),
                        _ => 0.0,
                    },
                },
            );
        }
        quit
    }

    /// Register palette thumbnails with egui once, on first use.
    fn ensure_swatches(&mut self, ctx: &egui::Context) {
        let Some(gfx) = self.gfx.as_ref() else { return };
        if self.swatches.len() == gfx.materials.layers.len() {
            return;
        }
        let size = terra_render::material::THUMB as usize;
        self.swatches = gfx
            .materials
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let image =
                    egui::ColorImage::from_rgba_unmultiplied([size, size], &layer.thumbnail);
                ctx.load_texture(format!("swatch-{i}"), image, egui::TextureOptions::LINEAR)
            })
            .collect();
    }

    /// Register species previews with egui once.
    fn ensure_species_swatches(&mut self, ctx: &egui::Context) {
        let Some(gfx) = self.gfx.as_ref() else { return };
        if self.species_swatches.len() == gfx.scatter.species.len() {
            return;
        }
        let size = terra_render::scatter::THUMB as usize;
        self.species_swatches = gfx
            .scatter
            .species
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let image = egui::ColorImage::from_rgba_unmultiplied([size, size], &s.thumbnail);
                ctx.load_texture(format!("species-{i}"), image, egui::TextureOptions::LINEAR)
            })
            .collect();
    }

    fn render(&mut self) -> Result<Frame> {
        let Some(window) = self.window.clone() else { return Ok(Frame::Idle) };

        // Acquire before building the UI: egui's TexturesDelta asserts in Drop
        // that every delta was applied, so an early return taken afterwards
        // would panic rather than skip a frame.
        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = {
            let Some(gfx) = self.gfx.as_mut() else { return Ok(Frame::Idle) };
            match gfx.ctx.surface.get_current_texture() {
                Cst::Success(f) | Cst::Suboptimal(f) => f,
                Cst::Outdated | Cst::Lost => {
                    gfx.ctx.reconfigure();
                    return Ok(Frame::Retry);
                }
                Cst::Timeout | Cst::Occluded => return Ok(Frame::Idle),
                Cst::Validation => return Err(anyhow::anyhow!("surface validation error")),
            }
        };

        // Clock starts here: everything before this was the acquire wait.
        let cpu_start = Instant::now();
        let Some(mut egui_state) = self.egui_state.take() else { return Ok(Frame::Idle) };

        // --- UI phase: no GPU borrow live across this closure ---
        let ctx = self.egui_ctx.clone();
        let raw_input = egui_state.take_egui_input(&window);
        let mut quit = false;
        let out = ctx.run_ui(raw_input, |ui| {
            quit = self.build_ui(ui);
        });
        let egui::FullOutput {
            platform_output, mut textures_delta, shapes, pixels_per_point, ..
        } = out;
        egui_state.handle_platform_output(&window, platform_output);
        self.egui_state = Some(egui_state);

        let tris = ctx.tessellate(shapes, pixels_per_point);

        // --- GPU phase ---
        //
        // Taken before `gfx` is borrowed mutably: `active_view_mode` reads
        // `self.screen` and `self.play`, which the borrow checker cannot see are
        // disjoint from `self.gfx` across a method call.
        let view_mode = self.active_view_mode();
        let Some(gfx) = self.gfx.as_mut() else {
            textures_delta.clear();
            return Ok(Frame::Idle);
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gfx
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // The depth buffer still holds the previous frame at this point, which
        // is exactly what the pyramid is built from -- reading this frame's
        // instead would mean a read-back and a stall. One frame of staleness is
        // handled by a margin in the test rather than by trying to be exact.
        if self.screen == Screen::Editor && self.world.is_some() {
            gfx.hiz.build(&mut encoder);
        }

        // Culling runs before the render pass that consumes its output.
        if let (Screen::Editor, Some(w)) = (self.screen, self.world.as_ref()) {
            let cam = match self.play.as_ref() {
                Some(p) => &p.camera,
                None => &w.camera,
            };
            gfx.scatter.cull(&mut encoder, &gfx.ctx.queue, cam, gfx.ctx.aspect());
        }

        // Fog needs the shadow cascades, so it is built after them and before
        // anything that samples the grid.

        // Shadow cascades, before anything samples them.
        if gfx.lighting.settings.shadow_quality.enabled() && self.screen == Screen::Editor {
            for cascade in 0..CASCADES {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gfx.lighting.cascade_views[cascade],
                        // Reversed-Z here too, so the compare matches the
                        // scene's and the sampler's `Greater`.
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if let Some(w) = self.world.as_ref() {
                    w.terrain.draw_shadow(&mut pass, &gfx.lighting, cascade);
                }
                gfx.scatter.draw_shadow(&mut pass, &gfx.meshes, &gfx.lighting, cascade);
                gfx.scatter.draw_props_shadow(&mut pass, &gfx.meshes, &gfx.lighting, cascade);
            }
        }

        // Volumetrics, after the cascades it samples and before the scene that
        // samples it.
        {
            let cam = match (self.screen, self.play.as_ref(), self.world.as_ref()) {
                (Screen::Editor, Some(p), _) => p.camera.clone(),
                (Screen::Editor, None, Some(w)) => w.camera.clone(),
                _ => self.backdrop_cam.clone(),
            };
            let aspect = gfx.ctx.aspect();
            let time = self.time;
            let (queue, lighting) = (&gfx.ctx.queue, &gfx.lighting);
            gfx.fog.build(&mut encoder, queue, lighting, &cam, aspect, time);

            // Clouds before the scene pass, because the sky samples the buffer
            // this writes. Its own pass rather than inline in the sky, so it can
            // run at half resolution and accumulate across frames.
            let queue = &gfx.ctx.queue;
            let env_gpu = &gfx.env_gpu;
            gfx.clouds.render(&mut encoder, queue, env_gpu, &cam, aspect);
            // Ground shadows before the cascades and the scene, both of which
            // sample the map.
            let eye = glam::Vec2::new(cam.pos.x, cam.pos.z);
            gfx.clouds.render_shadow(&mut encoder, queue, env_gpu, eye);
        }

        let scene_ts = gfx.gpu_timer.as_ref().and_then(|t| t.scene_writes());
        let ui_ts = gfx.gpu_timer.as_ref().and_then(|t| t.ui_writes());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gfx.ctx.scene_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Alpha 1 means "sky": anything the sky pass does not
                        // cover is still unoccluded as far as the rays care.
                        //
                        // Wireframe clears to a flat dark grey instead and skips
                        // the sky entirely, as Unreal's wireframe view does. A
                        // scattering sky behind one-pixel lines makes the lines
                        // unreadable, and the mode exists to read them.
                        load: wgpu::LoadOp::Clear(if view_mode.is_wireframe() {
                            wgpu::Color { r: 0.014, g: 0.015, b: 0.018, a: 1.0 }
                        } else {
                            wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gfx.ctx.depth_view,
                    // Reversed-Z clears to 0.0 (infinity), not 1.0.
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: scene_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !view_mode.is_wireframe() {
                gfx.sky.draw(&mut pass, &gfx.lighting, &gfx.env_gpu, &gfx.clouds);
            }
            // The backdrop behind the menus is always Lit: it is a portrait, not
            // a viewport, and a wireframe menu background is nobody's intent --
            // `active_view_mode` is what forces that.
            match (self.screen, self.world.as_ref(), self.backdrop.as_ref()) {
                (Screen::Editor, Some(w), _) => {
                    w.terrain.draw(&mut pass, &gfx.lighting, &gfx.clouds, view_mode)
                }
                (_, _, Some(b)) => b.draw(&mut pass, &gfx.lighting, &gfx.clouds, view_mode),
                _ => {}
            }
            // Foliage, after the terrain so the depth buffer rejects most of it.
            //
            // Skipped in wireframe. Only the terrain has a wireframe pipeline, so
            // leaving these in would draw solid trees standing on a wire mesh --
            // which is a worse answer than not drawing them, because it looks like
            // the wireframe is failing rather than not covering them yet.
            if self.screen == Screen::Editor && self.world.is_some() && !view_mode.is_wireframe() {
                gfx.scatter.draw(&mut pass, &gfx.meshes, &gfx.lighting);
                gfx.scatter.draw_props(&mut pass, &gfx.meshes, &gfx.lighting);
            }

            if let Some(p) = self.play.as_ref() {
                // Chassis first, then the four wheels, in one instance buffer.
                let alpha = p.accumulator / FIXED_DT;
                let body = interpolate(p.prev, p.curr, alpha);
                // White, not a tint. The instance colour multiplies the albedo, and it used
                // to carry the vehicle's paint because the vehicle was an untextured box.
                // Against a textured mesh the old dark green multiplied the Hummer's own
                // maps down to almost black.
                let mut instances = vec![Instance::new(
                    Mat4::from_rotation_translation(body.rotation, body.translation),
                    Vec3::ONE,
                )];
                for (a, b) in p.prev_wheels.iter().zip(&p.curr_wheels) {
                    let w = interpolate(*a, *b, alpha);
                    instances.push(Instance::new(
                        Mat4::from_rotation_translation(w.rotation, w.translation),
                        Vec3::ONE,
                    ));
                }
                let n = gfx.meshes.upload_instances(&gfx.ctx.queue, &instances);
                // Every body part reads instance 0 -- they all move with the chassis, and
                // they are separate draws only so each keeps its own material.
                for part in &gfx.meshes.chassis {
                    gfx.meshes.draw(&mut pass, &gfx.lighting, part, 0, 1.min(n));
                }
                // One draw per corner: the wheels are four separate mirrored meshes, and
                // instance `1 + i` is the pose the physics reported for wheel `i`.
                for i in 0..4 {
                    if n > 1 + i as u32 {
                        gfx.meshes.draw(
                            &mut pass,
                            &gfx.lighting,
                            &gfx.meshes.wheels[i],
                            1 + i as u32,
                            1,
                        );
                    }
                }
            }
        }

        // --- temporal resolve ---
        {
            let cam = match (self.screen, self.play.as_ref(), self.world.as_ref()) {
                (Screen::Editor, Some(p), _) => p.camera.clone(),
                (Screen::Editor, None, Some(w)) => w.camera.clone(),
                _ => self.backdrop_cam.clone(),
            };
            let aspect = gfx.ctx.aspect();
            let vp = cam.projection(aspect) * cam.look_at();
            let jitter = self.frame_jitter;
            let queue = &gfx.ctx.queue;
            gfx.taa.resolve(&mut encoder, queue, vp, jitter);
        }

        // --- post: god rays, exposure, encode ---
        {
            let post_cam = match (self.screen, self.play.as_ref(), self.world.as_ref()) {
                (Screen::Editor, Some(p), _) => p.camera.clone(),
                (Screen::Editor, None, Some(w)) => w.camera.clone(),
                _ => self.backdrop_cam.clone(),
            };
            let sky = gfx.lighting.settings;
            let sun = gfx.lighting.sun;
            let active = gfx.post.prepare(
                &gfx.ctx.queue,
                &post_cam,
                gfx.ctx.aspect(),
                sun.direction,
                sun.daylight,
                // No shafts in a debug mode, for the same reason there is no fog:
                // they are a second term laid over the whole frame.
                if sun.night || !self.view_mode.shows_atmosphere() { 0.0 } else { sky.god_rays },
                // Straight from the mixer: exposure, the tone mapper and the
                // grade are all environment, not derived sky settings.
                &self.env.tone,
                gfx.fog.settings.enabled,
            );
            let source = gfx.taa.output_index();
            if active {
                gfx.post.render_rays(&mut encoder, source);
            }
            gfx.post.resolve(&mut encoder, &view, source);
        }

        // --- egui paint ---
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                gfx.egui_renderer.update_texture(&gfx.ctx.device, &gfx.ctx.queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gfx.ctx.config.width, gfx.ctx.config.height],
            pixels_per_point,
        };
        let cmds = gfx.egui_renderer.update_buffers(
            &gfx.ctx.device,
            &gfx.ctx.queue,
            &mut encoder,
            &tris,
            &screen,
        );
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: ui_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            gfx.egui_renderer.render(&mut pass, &tris, &screen);
        }
        for id in &textures_delta.free {
            gfx.egui_renderer.free_texture(id);
        }
        textures_delta.clear();

        if let Some(t) = gfx.gpu_timer.as_ref() {
            t.resolve(&mut encoder);
        }
        gfx.ctx.queue.submit(cmds.into_iter().chain(std::iter::once(encoder.finish())));
        self.cpu_ms = self.update_ms + cpu_start.elapsed().as_secs_f32() * 1000.0;

        let Some(gfx) = self.gfx.as_mut() else { return Ok(Frame::Presented) };
        gfx.ctx.queue.present(frame);
        if let Some(t) = gfx.gpu_timer.as_mut() {
            t.map();
        }

        if quit {
            self.window = None;
        }
        Ok(Frame::Presented)
    }

    /// Minimum spacing between frames.
    ///
    /// Even "uncapped" is capped. Without a floor, any frame that skips
    /// presentation leaves nothing to block on and the loop runs away.
    fn frame_interval(&self) -> Duration {
        if self.occluded {
            // Hidden or fully covered: an occluded surface never presents, so
            // nothing throttles the loop. Slow right down -- but keep drawing,
            // rather than stopping outright, so a mis-reported occlusion
            // cannot leave the window permanently frozen.
            Duration::from_millis(100)
        } else if self.settings.vsync {
            // Present blocks at the real refresh rate; this only stops the
            // loop from spinning when it does not.
            Duration::from_micros(4_000)
        } else {
            Duration::from_micros(500)
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Terra")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0))
            // Do not take focus on open. A tool that steals the foreground
            // every time it is rebuilt makes it unusable to work alongside --
            // and the window still draws, so nothing is lost by opening
            // behind whatever is in front.
            .with_active(false);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("could not create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let ctx = match pollster::block_on(RenderContext::new(window.clone(), self.settings.vsync))
        {
            Ok(c) => c,
            Err(e) => {
                log::error!("could not initialize GPU: {e:#}");
                event_loop.exit();
                return;
            }
        };

        theme::apply(&self.egui_ctx);
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(2048),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &ctx.device,
            ctx.config.format,
            egui_wgpu::RendererOptions::default(),
        );

        // The player's vehicle, split into a body and four wheels and measured, so the
        // collider and the suspension mounts come from the mesh rather than from constants
        // that could disagree with it.
        //
        // A failure here is logged and not fatal: the renderer falls back to a placeholder
        // box, which is obviously a placeholder, and the editor still opens.
        let t_veh = Instant::now();
        let vehicle_rig =
            match terra_assets::VehicleRig::from_gltf(&vehicle_path(), VEHICLE_MASS_KG) {
                Ok(r) => {
                    let d = r.dims;
                    log::info!(
                        "vehicle rigged in {:.0} ms: {:.2} m long, {:.2} m wheelbase, \
                     {:.2} m track, {:.0} kg, {:.2} m tyres",
                        t_veh.elapsed().as_secs_f32() * 1000.0,
                        d.length(),
                        d.wheelbase(),
                        d.track(),
                        d.mass_kg,
                        d.wheel_radius * 2.0
                    );
                    Some(r)
                }
                Err(e) => {
                    log::error!("vehicle mesh unusable, using a placeholder: {e:#}");
                    None
                }
            };

        // Textures first: the light state names the fog grid, and the fog
        // pipelines name the light state.
        let grids = FroxelGrids::new(&ctx.device);
        let lighting = Lighting::new(&ctx.device, SkySettings::default(), grids.scattered());
        let fog = Volumetrics::new(&ctx.device, &lighting, grids);
        let env_gpu = terra_render::EnvironmentGpu::new(&ctx.device);
        let clouds = terra_render::Clouds::new(&ctx, &env_gpu);
        let sky = Sky::new(&ctx, &lighting, &env_gpu, &clouds);
        // Measured dimensions if the mesh loaded, a placeholder otherwise -- matching the
        // placeholder geometry `MeshRenderer` falls back to, so the two still agree.
        let vehicle_dims = vehicle_rig.as_ref().map(|r| r.dims).unwrap_or(PLACEHOLDER_VEHICLE);
        let meshes = MeshRenderer::new(&ctx, vehicle_rig.as_ref(), &lighting);
        let gpu_timer = GpuTimer::new(&ctx.device, &ctx.queue, ctx.supports_timestamps());
        let t_mat = Instant::now();
        let materials = Materials::load(&ctx.device, &ctx.queue, &shared_texture_dir());
        log::info!("materials baked in {:.0} ms", t_mat.elapsed().as_secs_f32() * 1000.0);
        let t_models = Instant::now();
        // Empty at startup: the palette belongs to a project, and none is open yet.
        // `reload_species` fills it from the project's own `assets/models` when a world
        // opens, which is the only place a user's meshes can come from.
        let scatter = Scatter::load(&ctx.device, &ctx.queue, &meshes, &empty_dir());
        log::info!("models loaded in {:.0} ms", t_models.elapsed().as_secs_f32() * 1000.0);
        let mut post = Post::new(&ctx);
        let taa = Taa::new(&ctx);
        post.rebind(&ctx, taa.outputs());
        // Menu backdrop first: it is built once for the session and reused for
        // every world opened, because the textures are read-only.
        let mut backdrop = Terrain::new(&ctx, BACKDROP_SIZE, &materials, &lighting, &clouds);
        let params = backdrop_params();
        let heights =
            terra_gen::heightfield::generate(backdrop.resolution(), backdrop.extent_m(), &params);
        backdrop.set_heights(&ctx.queue, heights);

        let hiz = HiZ::new(&ctx);
        let gfx = Gfx {
            ctx,
            sky,
            lighting,
            fog,
            taa,
            post,
            hiz,
            env_gpu,
            clouds,
            materials,
            scatter,
            vehicle_dims,
            meshes,
            egui_renderer,
            gpu_timer,
        };
        self.backdrop = Some(backdrop);

        self.egui_state = Some(egui_state);
        self.gfx = Some(gfx);
        self.window = Some(window);
        self.last_frame = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.gfx.is_none() {
            return;
        }
        let Some(window) = self.window.clone() else {
            event_loop.exit();
            return;
        };

        let consumed = self
            .egui_state
            .as_mut()
            .map(|s| s.on_window_event(&window, &event).consumed)
            .unwrap_or(false);

        match event {
            WindowEvent::CloseRequested => {
                if self.unsaved {
                    self.save_world();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.ctx.resize(size.width, size.height);
                    // The scene view was recreated, so every bind group holding
                    // it is stale.
                    if g.taa.size() != (g.ctx.config.width, g.ctx.config.height) {
                        g.taa.resize(&g.ctx);
                        // The scene view was recreated too, so every bind group
                        // holding it -- history and post alike -- is stale.
                        g.post.rebind(&g.ctx, g.taa.outputs());
                        // The depth buffer was recreated with it, so the
                        // pyramid's source and every view of it are stale.
                        g.hiz.resize(&g.ctx);
                        // Half-res cloud targets track the surface too, and its
                        // accumulated history is a different shape now.
                        g.clouds.resize(&g.ctx);
                    }
                }
            }
            WindowEvent::Occluded(hidden) => {
                self.occluded = hidden;
                if !hidden {
                    self.next_frame = Instant::now();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !consumed {
                    // A wheel notch and a trackpad's pixel scroll are wildly
                    // different magnitudes; normalise to notches so zooming
                    // feels the same on both.
                    self.input.scroll += match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                    };
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Right => self.input.looking = down,
                    MouseButton::Middle if !consumed => self.input.panning = down,
                    MouseButton::Middle => self.input.panning = false,
                    MouseButton::Left if !consumed => {
                        // In the Camera tool the left button drives the view,
                        // never the brush.
                        if self.tool.edits() {
                            self.input.sculpting = down;
                            self.input.clicked |= down;
                        } else {
                            self.input.orbiting = down;
                        }
                    }
                    MouseButton::Left => {
                        self.input.sculpting = false;
                        self.input.orbiting = false;
                    }
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if consumed {
                    return;
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.input.fwd = down,
                        KeyCode::KeyS => self.input.back = down,
                        KeyCode::KeyA => self.input.left = down,
                        KeyCode::KeyD => self.input.right = down,
                        KeyCode::KeyE => self.input.up = down,
                        KeyCode::KeyQ => self.input.down = down,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.input.boost = down,
                        KeyCode::ControlLeft | KeyCode::ControlRight => self.input.invert = down,
                        KeyCode::AltLeft | KeyCode::AltRight => self.input.alt = down,
                        // Frame the world, as F does in every DCC tool. The hard
                        // recovery: the wheel is geometric, so a few seconds of
                        // scrolling out puts the camera far enough away that
                        // nothing else gets back in reasonable time.
                        KeyCode::KeyF if down && !self.input.alt => self.frame_world(),
                        // Alt + digit selects a visualization mode, as in Unreal.
                        // Guarded on Alt so the digits stay free, and matched on
                        // the physical key so it works on a layout where the
                        // number row is shifted.
                        KeyCode::Digit1
                        | KeyCode::Digit2
                        | KeyCode::Digit3
                        | KeyCode::Digit4
                        | KeyCode::Digit5
                        | KeyCode::Digit6
                        | KeyCode::Digit7
                            if down && self.input.alt && self.is_editing() =>
                        {
                            let digit = match code {
                                KeyCode::Digit1 => 1,
                                KeyCode::Digit2 => 2,
                                KeyCode::Digit3 => 3,
                                KeyCode::Digit4 => 4,
                                KeyCode::Digit5 => 5,
                                KeyCode::Digit6 => 6,
                                _ => 7,
                            };
                            // Unbound digits are ignored rather than falling back
                            // to Lit: Alt+1 is Brush Wireframe in Unreal and
                            // silently switching to Lit would look like a bug.
                            if let Some(m) = terra_render::ViewMode::from_digit(digit)
                                && self.view_mode != m
                            {
                                {
                                    self.view_mode = m;
                                    log::info!("view mode: {}", m.label());
                                    // A mode change alters every pixel, so the
                                    // accumulated history is worthless and
                                    // blending through it reads as a slow wipe.
                                    if let Some(g) = self.gfx.as_mut() {
                                        g.taa.invalidate();
                                        g.clouds.invalidate();
                                    }
                                }
                            }
                        }
                        // Brackets are the brush controls, as in Unreal: size on
                        // their own, strength with Shift. Geometric steps rather
                        // than linear, so one keypress feels the same at 10 m as
                        // at 400 m.
                        KeyCode::BracketLeft if down && self.input.boost => {
                            self.brush_strength = (self.brush_strength * 0.85).max(0.05);
                        }
                        KeyCode::BracketRight if down && self.input.boost => {
                            self.brush_strength = (self.brush_strength * 1.18).min(8.0);
                        }
                        KeyCode::BracketLeft if down => {
                            self.brush_radius = (self.brush_radius * 0.85).max(8.0);
                        }
                        KeyCode::BracketRight if down => {
                            self.brush_radius = (self.brush_radius * 1.18).min(800.0);
                        }
                        KeyCode::Delete | KeyCode::Backspace
                            if down && self.tool == Tool::Select =>
                        {
                            if let (Some(i), Some(g)) = (self.selected_prop, self.gfx.as_mut()) {
                                g.scatter.remove_prop(i);
                                self.selected_prop = None;
                                self.unsaved = true;
                            }
                        }
                        KeyCode::Escape if down && self.play.is_some() => self.stop_play(),
                        KeyCode::Escape if down && self.screen == Screen::Editor => {
                            self.exit_editor();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                // Exponential average; a raw per-frame number is unreadable.
                self.frame_ms += (dt * 1000.0 - self.frame_ms) * 0.1;

                let vsync = self.settings.vsync;
                if let Some(g) = self.gfx.as_mut() {
                    g.ctx.set_vsync(vsync);
                }
                let t_update = Instant::now();
                self.update(dt);
                self.update_ms = t_update.elapsed().as_secs_f32() * 1000.0;

                let outcome = self.render();

                // Drive the timestamp readback and record this frame.
                if let Some(gfx) = self.gfx.as_mut() {
                    let _ = gfx.ctx.device.poll(wgpu::PollType::Poll);
                    if let Some(ms) = gfx.gpu_timer.as_mut().and_then(|t| t.poll()) {
                        self.stats.gpu.push(ms);
                    }
                }
                if outcome.as_ref().is_ok_and(|f| *f == Frame::Presented) {
                    self.presented += 1;
                    // Only time consecutive presented frames. After a skip the
                    // elapsed time is the backoff interval, not a frame.
                    if let Some(prev) = self.last_present {
                        self.stats.frame.push((now - prev).as_secs_f32() * 1000.0);
                        self.stats.cpu.push(self.cpu_ms);
                    }
                    self.last_present = Some(now);
                } else {
                    self.skipped += 1;
                    self.last_present = None;
                }

                // Heartbeat: distinguishes a healthy idle loop from one that
                // has stopped drawing.
                if now.duration_since(self.last_report) >= Duration::from_secs(5) {
                    log::info!(
                        // FPS is derived from measured frame time, the same way
                        // the HUD does it. Presented-count over wall clock is a
                        // different quantity -- it counts deliberate idle as
                        // slowness -- so it is reported separately below.
                        "{:.0} fps | frame {:.2} | cpu {:.2} | gpu {:.2} | 1% low {:.2} ms \
                         | {} presented, {} skipped{}",
                        self.stats.fps(),
                        self.stats.frame.avg(),
                        self.stats.cpu.avg(),
                        self.stats.gpu.avg(),
                        self.stats.frame.p99(),
                        self.presented,
                        self.skipped,
                        if self.occluded { " (occluded)" } else { "" }
                    );
                    self.presented = 0;
                    self.skipped = 0;
                    self.last_report = now;
                }

                match outcome {
                    Ok(Frame::Presented) => {}
                    Ok(Frame::Retry) => self.next_frame = Instant::now(),
                    Ok(Frame::Idle) => {
                        self.next_frame = Instant::now() + Duration::from_millis(100);
                    }
                    Err(e) => {
                        log::error!("render: {e:#}");
                        self.next_frame = Instant::now() + Duration::from_millis(100);
                    }
                }
                if self.window.is_none() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _e: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && (self.input.looking || self.input.panning || self.input.orbiting)
        {
            self.input.look_delta.0 += delta.0 as f32;
            self.input.look_delta.1 += delta.1 as f32;
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(window) = &self.window else { return };

        let now = Instant::now();
        if now >= self.next_frame {
            self.next_frame = now + self.frame_interval();
            window.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

/// Closest the camera is allowed to sit above the surface, and the floor used
/// when scaling pan and zoom. Without it, both stall to nothing the moment the
/// camera touches the ground.
const MIN_VIEW_DIST_M: f32 = 4.0;

/// Furthest the camera is allowed to get from the world, in metres.
///
/// Not cosmetic -- it is what stops the camera being stranded. Wheel-out is
/// geometric: each notch multiplies the distance by `ZOOM_PER_NOTCH`, and the
/// next notch is computed from the new, larger distance. Sixty notches from the
/// default view reaches 800 km, at which point the world is a sliver on the
/// horizon, one notch back in covered 720 m because the scale was capped at
/// 6000, and returning took over a thousand notches. Beyond a few world widths
/// there is nothing to see anyway.
const MAX_VIEW_DIST_M: f32 = 40_000.0;

/// Fraction of the distance to the ground covered by one wheel notch.
const ZOOM_PER_NOTCH: f32 = 0.12;

/// Radius around the car that gets obstacle colliders.
const OBSTACLE_RADIUS_M: f32 = 220.0;

/// Translate renderer solids into physics shapes.
fn to_obstacles(solids: &[terra_render::scatter::Solid]) -> Vec<Obstacle> {
    solids
        .iter()
        .map(|s| Obstacle {
            pos: s.pos.to_array(),
            shape: if s.boulder {
                ObstacleShape::Boulder { radius: s.radius.max(0.15) }
            } else {
                ObstacleShape::Trunk { radius: s.radius.max(0.08), height: s.height }
            },
        })
        .collect()
}

/// How far a simplified control point may sit from the drawn line. Larger
/// values give fewer, smoother control points; smaller ones track the stroke
/// more literally, jitter included.
const STROKE_TOLERANCE_M: f32 = 6.0;

/// Blend two poses. Rotation uses slerp: lerping quaternions and normalising
/// shortens the arc near 180 degrees, which shows up as the car snapping
/// through a roll rather than turning through it.
fn interpolate(a: Pose, b: Pose, t: f32) -> Pose {
    Pose {
        translation: a.translation.lerp(b.translation, t),
        rotation: a.rotation.slerp(b.rotation, t),
    }
}

/// World poses for the four wheels, including steering and roll.
///
/// The wheel mesh lies along X, so roll is about X and steering about Y, both
/// applied inside the chassis frame.
fn wheel_poses(car: &Vehicle, chassis_rotation: Quat) -> Vec<Pose> {
    car.wheel_poses()
        .into_iter()
        .map(|w| Pose {
            translation: Vec3::from_array(w.position),
            rotation: chassis_rotation
                * Quat::from_rotation_y(w.steer)
                * Quat::from_rotation_x(w.roll),
        })
        .collect()
}

/// Where material folders are looked for.
///
/// Relative to the executable's crate root at build time, so running from a
/// checkout finds the same folder the artist drops textures into. A shipped
/// build would resolve this next to the binary instead.
/// Where the shared, cross-project material library lives.
///
/// Repo-relative rather than under the user's data directory, because it is a
/// development convenience: nothing ships in it. A project's own imports live in
/// its `assets/textures/` and take precedence.
fn shared_texture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/texture")
}

/// Where glTF models are looked for, alongside the texture folder.
/// Stand-in dimensions for when the vehicle mesh cannot be read.
///
/// Matches the box-and-cylinders `MeshRenderer` draws in the same situation, so even the
/// fallback has a collider the size of the thing on screen.
const PLACEHOLDER_VEHICLE: terra_core::VehicleDims = terra_core::VehicleDims {
    chassis_half: [1.3, 0.8, 2.6],
    chassis_centre_y: 1.24,
    wheel_radius: 0.57,
    wheel_width: 0.44,
    axle_half_width: 1.0,
    front_axle_z: 1.74,
    rear_axle_z: -1.65,
    mass_kg: VEHICLE_MASS_KG,
};

/// Kerb mass of the player's vehicle, in kilograms.
///
/// The one figure a mesh cannot supply: geometry has no density. A Hummer H1 is about
/// 2.9 tonnes, and this drives every force in the vehicle model, so it is here beside the
/// mesh it belongs to rather than buried in the physics.
const VEHICLE_MASS_KG: f32 = 2900.0;

/// The player's vehicle mesh.
fn vehicle_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/game/hummer.glb")
}

/// A path that holds no models, for building an empty palette before any project is
/// open.
///
/// A named function rather than a bare `Path::new("")` so the intent survives: this
/// used to point at the repository's own `assets/models`, which is how engine-made
/// meshes ended up in every project's palette.
fn empty_dir() -> std::path::PathBuf {
    std::path::PathBuf::new()
}

/// The menu backdrop is its own small world, independent of any project.
const BACKDROP_SIZE: WorldSize = WorldSize::Small;

/// Tuned for the orbit distance rather than for realism: larger features and a
/// taller amplitude than a playable world, so ridgelines read at a glance.
fn backdrop_params() -> terra_project::RmfParams {
    terra_project::RmfParams {
        octaves: 9,
        amplitude_m: 430.0,
        feature_scale_m: 1500.0,
        warp_strength_m: 190.0,
        warp_scale_m: 2400.0,
        ..Default::default()
    }
}

/// Filesystem-safe folder name derived from a world name.
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let s = s.trim_matches('_').to_string();
    if s.is_empty() { "world".to_string() } else { s }
}

/// Folder name that does not collide with an existing project. Two worlds may
/// share a display name; they must not share a directory.
/// A destination filename that does not collide with something already there.
///
/// Imports never overwrite: two different textures both called `rock.png` is
/// entirely normal, and silently replacing the first with the second loses work
/// with no way to notice.
fn unique_path(dir: &std::path::Path, name: &std::ffi::OsStr) -> std::path::PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let name = std::path::Path::new(name);
    let stem = name.file_stem().and_then(|s| s.to_str()).unwrap_or("asset");
    let ext = name.extension().and_then(|s| s.to_str());
    for n in 2..10_000 {
        let candidate = match ext {
            Some(e) => dir.join(format!("{stem} {n}.{e}")),
            None => dir.join(format!("{stem} {n}")),
        };
        if !candidate.exists() {
            return candidate;
        }
    }
    first
}

fn unique_folder(parent: &std::path::Path, name: &str) -> String {
    let base = sanitize(name);
    if !parent.join(&base).exists() {
        return base;
    }
    (2..).map(|n| format!("{base}_{n}")).find(|c| !parent.join(c).exists()).unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_produces_usable_folder_names() {
        assert_eq!(sanitize("Desert Rally"), "Desert_Rally");
        assert_eq!(sanitize("../../etc"), "etc");
        assert_eq!(sanitize("!!!"), "world");
    }

    #[test]
    fn movement_axis_cancels_opposing_keys() {
        let i = Input { fwd: true, back: true, left: true, ..Default::default() };
        assert_eq!(i.axis(), Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn duplicate_names_get_distinct_folders() {
        let tmp = std::env::temp_dir().join("terra-unique-test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("Rally")).unwrap();
        assert_eq!(unique_folder(&tmp, "Rally"), "Rally_2");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn loading_progress_is_bounded_by_both_work_and_time() {
        let mut l = Loading::new_for_test();
        l.stage = 3;
        l.elapsed = 0.0;
        assert_eq!(l.progress(), 0.0, "must not jump to full before the minimum time");
        l.elapsed = Loading::MIN_SECONDS * 2.0;
        assert_eq!(l.progress(), 1.0);
    }

    impl Loading {
        fn new_for_test() -> Self {
            Self::new(dummy_project())
        }
    }

    fn dummy_project() -> Project {
        let tmp = std::env::temp_dir().join("terra-loading-test");
        let _ = std::fs::remove_dir_all(&tmp);
        Project::create(&tmp, "Test", WorldSize::Small, 1).unwrap()
    }
}

#[cfg(test)]
mod view_mode_gate_tests {
    use terra_render::ViewMode;

    /// The gate, restated as pure logic.
    ///
    /// `App::is_editing` and `active_view_mode` need a live GPU device to
    /// construct, so the rule they encode is checked here against the same two
    /// inputs rather than through a window.
    fn active(editor_screen: bool, playing: bool, chosen: ViewMode) -> ViewMode {
        let editing = editor_screen && !playing;
        if editing { chosen } else { ViewMode::Lit }
    }

    #[test]
    fn debug_modes_apply_only_while_editing() {
        // The bug this closes: `Screen::Editor` is still true while driving, so
        // gating on the screen alone let a wireframe follow the car into Play.
        assert_eq!(active(true, false, ViewMode::Wireframe), ViewMode::Wireframe);
        assert_eq!(
            active(true, true, ViewMode::Wireframe),
            ViewMode::Lit,
            "driving must render lit"
        );
        assert_eq!(
            active(false, false, ViewMode::Wireframe),
            ViewMode::Lit,
            "the menu backdrop must render lit"
        );
    }

    #[test]
    fn every_mode_is_suppressed_the_same_way() {
        for m in ViewMode::ALL {
            assert_eq!(active(true, true, m), ViewMode::Lit, "{} leaked into Play", m.label());
            assert_eq!(
                active(false, false, m),
                ViewMode::Lit,
                "{} leaked into the menu",
                m.label()
            );
            assert_eq!(active(true, false, m), m, "{} did not apply while editing", m.label());
        }
    }

    #[test]
    fn the_choice_survives_a_play_session() {
        // Remembered rather than reset, so stopping returns to the view that was
        // set up -- which is what Unreal does when a Play-In-Editor session ends.
        let chosen = ViewMode::LightingOnly;
        assert_eq!(active(true, true, chosen), ViewMode::Lit);
        assert_eq!(active(true, false, chosen), chosen, "the mode was not remembered");
    }
}

#[cfg(test)]
mod camera_input_tests {
    use super::*;

    #[test]
    fn clearing_motion_drops_both_channels() {
        let mut i = Input { look_delta: (120.0, -40.0), scroll: 7.5, ..Default::default() };
        i.clear_motion();
        assert_eq!(i.look_delta, (0.0, 0.0));
        assert_eq!(i.scroll, 0.0);
    }

    #[test]
    fn clearing_motion_leaves_held_keys_and_buttons_alone() {
        // Only the *accumulated* channels are dropped. Zeroing the held state
        // would make W stop working the moment the editor was not editing for a
        // frame, and release events would then never arrive to correct it.
        let mut i = Input {
            fwd: true,
            boost: true,
            looking: true,
            sculpting: true,
            cursor: (300.0, 200.0),
            look_delta: (5.0, 5.0),
            scroll: 1.0,
            ..Default::default()
        };
        i.clear_motion();
        assert!(i.fwd && i.boost && i.looking && i.sculpting);
        assert_eq!(i.cursor, (300.0, 200.0));
    }

    #[test]
    fn a_wheel_spin_outside_the_editor_does_not_bank_a_zoom() {
        // The bug: `scroll` accumulates on any wheel event and is only consumed
        // while editing, so spinning the wheel on the menu or through a load
        // applied the whole backlog in one dolly on the first editing frame.
        let mut i = Input::default();
        for _ in 0..40 {
            i.scroll += 1.0;
        }
        assert_eq!(i.scroll, 40.0, "the backlog is real");
        i.clear_motion();
        assert_eq!(i.scroll, 0.0, "and must not survive into the editor");
    }

    #[test]
    fn a_drag_held_through_a_load_does_not_bank_a_rotation() {
        // Same shape as the wheel case: `look_delta` accumulates while a button
        // is held, and `update_editor` is not running during Loading or Play.
        let mut i = Input { looking: true, ..Default::default() };
        for _ in 0..120 {
            i.look_delta.0 += 9.0;
            i.look_delta.1 += 3.0;
        }
        assert!(i.look_delta.0 > 1000.0);
        i.clear_motion();
        assert_eq!(i.look_delta, (0.0, 0.0));
        assert!(i.looking, "the button is still held; only the backlog is dropped");
    }

    #[test]
    fn movement_axis_is_bounded_and_cancels() {
        // The vector fed to `Camera::translate`. Opposing keys must cancel rather
        // than one winning, and no component may exceed one -- the camera
        // normalizes, but a component outside -1..1 would mean a key state that
        // is neither pressed nor released.
        let all = Input {
            fwd: true,
            back: true,
            left: true,
            right: true,
            up: true,
            down: true,
            ..Default::default()
        };
        assert_eq!(all.axis(), Vec3::ZERO, "opposing keys must cancel");

        let one = Input { fwd: true, right: true, up: true, ..Default::default() };
        let a = one.axis();
        assert_eq!(a, Vec3::new(1.0, 1.0, 1.0));
        assert!(a.to_array().iter().all(|c| (-1.0..=1.0).contains(c)));
    }
}

#[cfg(test)]
mod camera_scale_tests {
    /// The two rules `update_editor` applies, restated so they can be checked
    /// without a GPU: how far the wheel may take you, and how fast WASD moves at
    /// that distance.
    fn clamped_dist(raw: f32) -> f32 {
        raw.clamp(super::MIN_VIEW_DIST_M, super::MAX_VIEW_DIST_M)
    }

    fn speed_multiplier(dist: f32) -> f32 {
        (dist / 900.0).clamp(0.2, 25.0)
    }

    #[test]
    fn zooming_in_and_out_are_symmetric() {
        // The bug: wheel-out is geometric while the step for wheel-in was computed
        // from a distance capped at 6000, so leaving took 30 notches and returning
        // took over a thousand.
        let mut out = 900.0f32;
        let mut notches_out = 0;
        while out < super::MAX_VIEW_DIST_M {
            out += clamped_dist(out) * super::ZOOM_PER_NOTCH;
            notches_out += 1;
            assert!(notches_out < 500, "zooming out never reached the limit");
        }

        let mut back = out;
        let mut notches_in = 0;
        while back > 1000.0 {
            back -= clamped_dist(back) * super::ZOOM_PER_NOTCH;
            notches_in += 1;
            assert!(notches_in < 500, "zooming back in did not converge");
        }

        // Within a small factor, not exact -- the two directions are geometric
        // with the same ratio but opposite signs.
        assert!(
            notches_in < notches_out * 3,
            "leaving took {notches_out} notches and returning {notches_in}"
        );
    }

    #[test]
    fn the_reachable_distance_is_bounded() {
        // However long the wheel is held, the camera cannot end up hundreds of
        // kilometres out where nothing recovers it.
        let mut d = 900.0f32;
        for _ in 0..10_000 {
            d += clamped_dist(d) * super::ZOOM_PER_NOTCH;
        }
        // The scale saturates, so the *step* stops growing -- which is what the
        // position clamp in `update_editor` then bounds for real.
        assert_eq!(clamped_dist(d), super::MAX_VIEW_DIST_M);
    }

    #[test]
    fn movement_speed_follows_the_view_distance() {
        // A fixed rate is right at one distance and useless everywhere else. From
        // 30 km out, 120 m/s does not visibly move the terrain, which is what
        // "camera not working" looked like.
        let near = speed_multiplier(900.0);
        let far = speed_multiplier(30_000.0);
        assert!((near - 1.0).abs() < 1e-6, "the reference distance must be unity");
        assert!(far > 10.0, "far out should be much faster, got {far}");

        // 120 m/s at 30 km would take four minutes to cross it; scaled, it is
        // under half a minute.
        let crossing = 30_000.0 / (120.0 * far);
        assert!(crossing < 30.0, "crossing still takes {crossing:.0} s");
    }

    #[test]
    fn speed_is_clamped_at_both_ends() {
        // Proportional all the way down crawls when close to the ground;
        // unbounded at the top overshoots the world in one keypress.
        assert_eq!(speed_multiplier(1.0), 0.2, "close in must not crawl to a stop");
        assert_eq!(speed_multiplier(1e9), 25.0, "far out must not be unbounded");
        // Monotonic in between, so there is no distance where moving gets slower.
        let mut prev = 0.0;
        for i in 0..200 {
            let m = speed_multiplier(i as f32 * 300.0);
            assert!(m >= prev - 1e-6, "speed fell as distance grew");
            prev = m;
        }
    }
}

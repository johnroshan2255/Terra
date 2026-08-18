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
    /// The water surface. Per world rather than per session, because it binds the
    /// terrain's own height buffer to know how deep it is -- so it has to be rebuilt
    /// whenever the terrain is.
    water: terra_render::water::Water,
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
    /// Where the camera sits around the car, as a world compass angle rather than
    /// an offset from the car's heading.
    ///
    /// Absolute, because that is what a GTA-style chase camera does: the mouse aims
    /// the camera at a direction and the car turns underneath it, so you can look
    /// down a side street while still driving forward. An offset relative to the
    /// heading would drag the view around with every steering input.
    cam_yaw: f32,
    /// Elevation of the camera above the car, radians. Positive looks down.
    cam_pitch: f32,
    /// Wheel zoom, as a multiplier on the vehicle-derived follow distance.
    cam_zoom: f32,
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
    /// Authored water for the open world, and what was last written, compared the same
    /// way the environment is so an edit marks the world unsaved.
    water: terra_render::water::WaterSettings,
    saved_water: terra_render::water::WaterSettings,
    /// Body the Water tool has selected, and where a drag started on the ground.
    selected_water: Option<usize>,
    water_drag_start: Option<Vec2>,
    /// The rectangle being dragged, for the viewport overlay.
    water_drag_preview: Option<(Vec2, Vec2)>,
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
    /// World the delete confirmation is open for.
    ///
    /// Held here rather than in the UI so the modal cannot be the only thing standing
    /// between a click and a recursive delete: the action arrives, this is set, and the
    /// deletion happens on a *second*, explicit action.
    pending_delete: Option<terra_project::ProjectEntry>,
    /// Result of the last import or rescan, and whether it was a failure. Shown
    /// in the Content pane until the next one replaces it.
    ///
    /// Log lines are not an answer here: an import that quietly does nothing is
    /// the exact failure being fixed, and the user is looking at a window rather
    /// than at stderr.
    notice: Option<(String, bool)>,
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
            water: Default::default(),
            saved_water: Default::default(),
            selected_water: None,
            water_drag_start: None,
            water_drag_preview: None,
            layout: crate::dock::Layout::new(),
            modifiers: terra_voxel::ModifierStack::default(),
            selected_modifier: None,
            noise: terra_voxel::NoiseField::default(),
            noise_library: Vec::new(),
            assets: [Vec::new(), Vec::new(), Vec::new()],
            asset_kind: crate::ui::AssetKind::Texture,
            selected_material: 0,
            pending_delete: None,
            notice: None,
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

        // The water surface reads the terrain's own height buffer for depth, so it is
        // built here from the terrain that was just loaded rather than once at startup.
        let Some(gfx) = self.gfx.as_ref() else { return };
        let water = terra_render::water::Water::new(
            &gfx.ctx.device,
            &gfx.lighting,
            &gfx.env_gpu,
            terrain.height_buffer(),
            terrain.extent_m(),
            terrain.resolution(),
        );
        // Whatever was authored, or none. A world from before this existed has no
        // `water.ron` and simply has no water, which must load rather than fail.
        self.water = terra_render::water::WaterSettings::load(&l.project.paths).unwrap_or_default();
        self.saved_water = self.water.clone();
        self.world = Some(OpenWorld {
            project: l.project,
            terrain,
            camera,
            base: l.heights.unwrap_or_default(),
            roads: l.roads,
            flow: l.flow,
            deposition: l.deposition,
            water,
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
        let heading = car.heading(&physics);

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
            // Behind the car to start: the camera sits opposite the heading, which
            // is the heading plus half a turn.
            cam_yaw: heading + std::f32::consts::PI,
            cam_pitch: 0.22,
            cam_zoom: 1.0,
        });
        self.set_cursor_captured(true);
        log::info!("play: terrain collider built, car spawned");
    }

    /// Lock and hide the pointer, or give it back.
    ///
    /// Held for the whole Play session: the camera is aimed by raw motion, so a
    /// visible pointer would crawl to a screen edge and stay there while the view kept
    /// turning. `Locked` is what a driving camera wants; `Confined` is the fallback
    /// where a platform does not offer it, and a platform offering neither is not an
    /// error worth interrupting play for -- the camera still works, the pointer just
    /// wanders.
    fn set_cursor_captured(&self, on: bool) {
        let Some(gfx) = self.gfx.as_ref() else { return };
        let window = gfx.ctx.window();
        if on {
            let locked = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
            if locked.is_err()
                && let Err(e) = window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
            {
                log::debug!("cursor could not be captured: {e}");
            }
        } else if let Err(e) = window.set_cursor_grab(winit::window::CursorGrabMode::None) {
            log::debug!("cursor could not be released: {e}");
        }
        window.set_cursor_visible(!on);
    }

    fn stop_play(&mut self) {
        self.play = None;
        // Give the pointer back, and drop whatever motion arrived on the way out --
        // the editor camera must not inherit the last flick of the driving camera.
        self.set_cursor_captured(false);
        self.input.clear_motion();
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

        // --- mouse-aimed chase camera ---
        //
        // GTA's arrangement: the mouse orbits the camera around the car freely, and
        // when it is left alone the view drifts back behind the car. The camera angle
        // is held in world space rather than relative to the heading, so a steering
        // input does not drag the view with it -- you can look down a side street
        // while still driving straight.
        let (mdx, mdy) = self.input.look_delta;
        self.input.look_delta = (0.0, 0.0);
        let moved_mouse = mdx.abs() + mdy.abs() > 0.0;
        play.cam_yaw += mdx * DRIVE_LOOK_SENSITIVITY;
        // Screen-down should look down at the car from above, which is a *larger*
        // elevation. Inverting this reads as an inverted mouse rather than as a
        // camera bug, so it is worth being explicit.
        play.cam_pitch =
            (play.cam_pitch + mdy * DRIVE_LOOK_SENSITIVITY).clamp(DRIVE_PITCH_MIN, DRIVE_PITCH_MAX);

        // Wheel zooms, consumed here so it cannot bank up for the editor camera.
        if self.input.scroll != 0.0 {
            play.cam_zoom = (play.cam_zoom * (1.0 - self.input.scroll * 0.08))
                .clamp(DRIVE_ZOOM_MIN, DRIVE_ZOOM_MAX);
            self.input.scroll = 0.0;
        }

        let alpha = play.accumulator / FIXED_DT;
        let pose = interpolate(play.prev, play.curr, alpha);
        let dims = self.gfx.as_ref().map(|g| g.vehicle_dims).unwrap_or(PLACEHOLDER_VEHICLE);

        // Recentre behind the car when the mouse is idle and the car is actually
        // going somewhere. Gated on speed because recentring a parked car would spin
        // the view whenever it was nudged, and gated on the mouse because fighting
        // the player's own input is the classic chase-camera annoyance.
        let forward = pose.rotation * Vec3::Z;
        let heading = forward.z.atan2(forward.x);
        let behind = heading + std::f32::consts::PI;
        let speed = play.car.speed().abs() * 3.6;
        if !moved_mouse && speed > 4.0 {
            // Shortest way round, so recentring never takes the long way through a
            // full turn.
            let delta = wrap_angle(behind - play.cam_yaw);
            let rate = 1.0 - (-dt * (speed / 40.0).clamp(0.4, 2.5)).exp();
            play.cam_yaw += delta * rate;
        }

        // Aim at the roof line rather than the contact patch, or a tall vehicle sits
        // at the bottom of the frame with the sky above it.
        let target = pose.translation + Vec3::Y * dims.chassis_centre_y;
        // Distances scale with the vehicle rather than being fixed at the 9 m that
        // suited a 3.6 m hatchback -- a 5.2 m Hummer at that range fills the frame.
        let distance = dims.length() * 2.2 * play.cam_zoom;
        let (sy, cy) = play.cam_yaw.sin_cos();
        let (sp, cp) = play.cam_pitch.sin_cos();
        // Direction from the car to the camera, so a positive pitch lifts it.
        let offset = Vec3::new(cy * cp, sp, sy * cp);
        let want = target + offset * distance;

        // Smoothed so the camera lags the vehicle slightly rather than being welded
        // to it. Position only: the aim is exact, or the car slides around the frame.
        let follow = 1.0 - (-dt * 9.0).exp();
        play.camera.pos = play.camera.pos.lerp(want, follow);
        play.camera.look_toward(target);
    }

    /// Put the car back on its wheels where it currently is.
    ///
    /// Bound to `R`. A raycast vehicle cannot recover from being upside down on its
    /// own -- the wheel rays point at the sky, so there is no contact and neither
    /// throttle nor steering does anything.
    fn reset_car(&mut self) {
        let Some(play) = self.play.as_mut() else { return };
        let Some(world) = self.world.as_ref() else { return };

        let (t, _) = play.car.chassis_pose(&play.world);
        let heading = play.car.heading(&play.world);
        // Lifted clear of the ground it is standing on, not of wherever it fell to:
        // a car wedged in a gully has to come out of the gully.
        let ground = world.terrain.height_at(t[0], t[2]);
        let position = [t[0], ground + 1.0, t[2]];
        play.car.reset(&mut play.world, position, heading);

        // Re-seed the interpolation from the new pose. Left alone, the renderer would
        // smear the car from where it was to where it now is over the next frame,
        // which is a visible dash across the map.
        let (t, r) = play.car.chassis_pose(&play.world);
        let pose = Pose {
            translation: Vec3::from_array(t),
            rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]),
        };
        play.prev = pose;
        play.curr = pose;
        let wheels = wheel_poses(&play.car, pose.rotation);
        play.prev_wheels = wheels.clone();
        play.curr_wheels = wheels;
        play.accumulator = 0.0;
        // Snap the camera behind the reset heading rather than letting it swing round
        // from wherever it was watching the crash from.
        play.cam_yaw = heading + std::f32::consts::PI;
        play.camera.pos = pose.translation;
        log::info!("play: car reset at {:.0}, {:.0}", position[0], position[2]);
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

    /// Copy what the user picks into the project's own asset folder.
    ///
    /// Copied, not referenced. A project that stores a path to the user's
    /// Downloads folder stops working the moment it is moved or shared, and
    /// `README.md` already promises projects are self-contained and movable.
    ///
    /// A texture is a *folder* of maps rather than one image, so it needs a
    /// folder picker. Offering the file picker for it was worse than useless: a
    /// dialog filtered to images cannot select a directory at all, so clicking
    /// Open on a material folder just navigated into it, and picking the images
    /// inside copied them loose into `assets/textures/` where
    /// `texture_set::discover` -- which only ever looks at subdirectories --
    /// ignored them. The import silently did nothing, twice over.
    fn import_asset(&mut self, kind: crate::ui::AssetKind) {
        let Some(w) = self.world.as_ref() else { return };
        let dest_dir = w.project.paths.assets_dir().join(kind.folder());
        if let Err(e) = std::fs::create_dir_all(&dest_dir) {
            log::error!("could not create {}: {e}", dest_dir.display());
            self.notice = Some((format!("Could not create {}: {e}", dest_dir.display()), true));
            return;
        }

        if kind == crate::ui::AssetKind::Texture {
            let picked = rfd::FileDialog::new()
                .set_title("Import material folders  --  one folder per material")
                .pick_folders();
            let Some(dirs) = picked else { return };
            self.import_texture_folders(&dirs, &dest_dir);
            return;
        }

        let picked = rfd::FileDialog::new()
            .add_filter(kind.label(), kind.extensions())
            .set_title(format!("Import {}", kind.label()))
            .pick_files();
        let Some(files) = picked else { return };

        let mut imported = 0usize;
        let mut incomplete: Vec<String> = Vec::new();
        for src in files {
            let Some(name) = src.file_name() else { continue };
            let is_gltf = src.extension().is_some_and(|x| x.eq_ignore_ascii_case("gltf"));
            // A `.gltf` is JSON pointing at a `.bin` and its textures, so it
            // cannot be copied as one file. It goes into a folder of its own with
            // everything it names -- which `terra_assets::mesh::discover` already
            // reads, because it accepts one-model-per-folder as well as loose
            // files. A `.glb` is self-contained and stays loose.
            let dest = if is_gltf {
                let stem = src.file_stem().map(|s| s.to_string_lossy().to_string());
                let folder = unique_folder(&dest_dir, stem.as_deref().unwrap_or("model"));
                let dir = dest_dir.join(&folder);
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    log::error!("could not create {}: {e}", dir.display());
                    continue;
                }
                dir.join(name)
            } else {
                unique_path(&dest_dir, name)
            };
            match std::fs::copy(&src, &dest) {
                Ok(_) => {
                    log::info!("imported {}", dest.display());
                    imported += 1;
                }
                Err(e) => {
                    log::error!("could not import {}: {e}", src.display());
                    continue;
                }
            }
            if !is_gltf {
                continue;
            }
            // The sidecars. A missing one is reported rather than ignored: the
            // model would import "successfully" and then fail to load, with the
            // reason buried in a log line about a path the user never typed.
            let Some(src_dir) = src.parent() else { continue };
            let Some(dest_parent) = dest.parent() else { continue };
            for rel in terra_assets::mesh::external_files(&src) {
                let from = src_dir.join(&rel);
                let to = dest_parent.join(&rel);
                if let Some(p) = to.parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                if let Err(e) = std::fs::copy(&from, &to) {
                    log::error!(
                        "{}: could not copy {}: {e}",
                        name.to_string_lossy(),
                        rel.display()
                    );
                    incomplete.push(format!(
                        "{} (missing {})",
                        name.to_string_lossy(),
                        rel.display()
                    ));
                }
            }
        }
        self.refresh_assets();
        if kind == crate::ui::AssetKind::Model
            && let Some(paths) = self.world.as_ref().map(|w| w.project.paths.clone())
        {
            self.reload_species(&paths);
        }
        self.notice = Some(match incomplete.is_empty() {
            true => (format!("Imported {imported} {}.", kind.label().to_lowercase()), false),
            false => (
                format!(
                    "Imported {imported}, but {} arrived incomplete: {}. A .gltf needs the \
                     files it references beside it.",
                    incomplete.len(),
                    incomplete.join("; ")
                ),
                true,
            ),
        });
    }

    /// Install picked folders as materials and rebuild the palette.
    ///
    /// Accepts either a single material folder or a pack containing several, and
    /// reports every folder it had to skip -- a folder with no colour map used to
    /// be listed by the content browser (which only checks that it is a
    /// directory) and then never appear in the palette, with nothing said.
    fn import_texture_folders(&mut self, dirs: &[std::path::PathBuf], dest_dir: &std::path::Path) {
        let mut installed: Vec<String> = Vec::new();
        let mut rejected: Vec<(String, String)> = Vec::new();
        let mut incomplete: Vec<(String, Vec<&'static str>)> = Vec::new();
        let mut unreadable: Vec<(String, Vec<String>)> = Vec::new();

        for src in dirs {
            match terra_render::texture_set::install(src, dest_dir) {
                Ok(out) => {
                    installed.extend(out.materials);
                    rejected.extend(out.rejected);
                    incomplete.extend(out.incomplete);
                    unreadable.extend(out.unreadable);
                }
                Err(e) => {
                    log::error!("could not import {}: {e}", src.display());
                    rejected.push((
                        src.file_name().unwrap_or_default().to_string_lossy().to_string(),
                        e.to_string(),
                    ));
                }
            }
        }

        // Only rebuild when something landed. `Materials::load` re-decodes every
        // set in the folder, which is seconds of work on a big palette.
        if !installed.is_empty() {
            self.refresh_assets();
            self.reload_materials();
        }
        self.notice = Some(import_summary(&installed, &rejected, &incomplete, &unreadable));
    }

    /// Re-read the project's asset folders and rebuild both palettes.
    ///
    /// Needed because the palettes are otherwise built only on project open and
    /// on import, so a folder copied in by hand -- which was the *only* way to
    /// add a material while the import was broken -- stayed invisible until the
    /// project was closed and reopened.
    fn rescan_assets(&mut self) {
        let Some(paths) = self.world.as_ref().map(|w| w.project.paths.clone()) else { return };
        self.refresh_assets();
        self.reload_materials();
        self.reload_species(&paths);
        let n = self.gfx.as_ref().map_or(0, |g| g.materials.count());
        self.notice = Some((
            match n {
                0 => "Rescanned. No materials found -- each one needs its own folder with a \
                      colour map in it."
                    .to_string(),
                1 => "Rescanned: 1 material.".to_string(),
                n => format!("Rescanned: {n} materials."),
            },
            n == 0,
        ));
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

    /// Remove a world from the library, and its files too when asked.
    ///
    /// The library entry is dropped either way, but only *after* a requested file
    /// delete succeeds -- forgetting first would leave a folder nobody can find again
    /// if the delete then failed on a permission error.
    fn delete_world(&mut self, path: &std::path::Path, files: bool) {
        self.pending_delete = None;

        if files {
            match terra_project::ProjectPaths::delete_project(path) {
                Ok(()) => log::info!("deleted {}", path.display()),
                Err(e) => {
                    // Left in the library on purpose: the folder is still there, and
                    // hiding it would leave a project the user cannot reach.
                    log::error!("could not delete {}: {e}", path.display());
                    self.notice = Some((format!("Could not delete: {e}"), true));
                    return;
                }
            }
        }
        self.library.forget(path);
        if let Err(e) = self.library.save() {
            log::error!("could not save the library: {e}");
        }
    }

    /// Everything in the world, as the outliner shows it.
    ///
    /// Built fresh each frame rather than cached, because it is a few dozen short strings
    /// and every source it reads is already in memory -- a cache would need invalidating
    /// from six places and would be wrong in one of them.
    ///
    /// The order here *is* the order on screen: the pane emits a heading when the kind
    /// changes and never sorts.
    fn outliner_items(&self) -> Vec<ui::OutlinerItem> {
        use ui::{OutlinerItem, OutlinerKind};
        let mut out = Vec::new();

        for (i, r) in self.water.regions.iter().enumerate() {
            let s = r.size();
            out.push(OutlinerItem {
                kind: OutlinerKind::Water,
                index: i,
                name: format!("Water {}", i + 1),
                detail: format!("{:.0} x {:.0} m at {:.0} m", s[0], s[1], r.level_m),
                removable: true,
            });
        }

        if let Some(gfx) = self.gfx.as_ref() {
            // One row per species, whatever the instance count. This is the rule that
            // keeps the list usable: a thousand painted trees are one decision -- the
            // species' rules -- and a thousand rows would be unreadable.
            for (i, sp) in gfx.scatter.species.iter().enumerate() {
                let n = sp.instance_count();
                if n == 0 && !sp.is_painted() {
                    continue;
                }
                out.push(OutlinerItem {
                    kind: OutlinerKind::Species,
                    index: i,
                    name: sp.name.clone(),
                    detail: match n {
                        1 => "1 instance".to_string(),
                        n => format!("{n} instances"),
                    },
                    // Not removable here. A species exists because a mesh is in the
                    // project's `assets/models/`, so removing it means deleting that file
                    // -- clearing its painting is what the Foliage tool's Clear does.
                    removable: false,
                });
            }

            // Hand-placed objects, one row each: each was placed deliberately and has its
            // own transform, which is exactly what a scattered instance does not.
            for (i, p) in gfx.scatter.props.iter().enumerate() {
                let name = gfx
                    .scatter
                    .species
                    .get(p.species)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| "Object".to_string());
                out.push(OutlinerItem {
                    kind: OutlinerKind::Prop,
                    index: i,
                    name,
                    detail: format!("{:.0}, {:.0}", p.pos.x, p.pos.z),
                    removable: true,
                });
            }
        }

        if let Some(w) = self.world.as_ref() {
            for (i, r) in w.roads.roads.iter().enumerate() {
                out.push(OutlinerItem {
                    kind: OutlinerKind::Road,
                    index: i,
                    name: format!("Road {}", i + 1),
                    detail: match r.points.len() {
                        1 => "1 point".to_string(),
                        n => format!("{n} points"),
                    },
                    removable: true,
                });
            }
        }

        for (i, m) in self.modifiers.items.iter().enumerate() {
            out.push(OutlinerItem {
                kind: OutlinerKind::Modifier,
                index: i,
                name: m.name.clone(),
                detail: m.op.label().to_string(),
                removable: true,
            });
        }
        out
    }

    /// Which row the outliner should show as selected.
    ///
    /// Derived from the tool selections rather than stored separately, so the outliner and
    /// the tool panes cannot disagree about what is selected.
    fn outliner_selection(&self) -> Option<(ui::OutlinerKind, usize)> {
        use ui::{OutlinerKind, Tool};
        match self.tool {
            Tool::Water => self.selected_water.map(|i| (OutlinerKind::Water, i)),
            Tool::Foliage => Some((OutlinerKind::Species, self.species)),
            Tool::Select => self.selected_prop.map(|i| (OutlinerKind::Prop, i)),
            Tool::Road => self.active_road.map(|i| (OutlinerKind::Road, i)),
            _ => self.selected_modifier.map(|i| (OutlinerKind::Modifier, i)),
        }
    }

    /// Select an outliner row: switch to the tool that owns its settings, and point that
    /// tool at it.
    ///
    /// Routing to the existing panes rather than duplicating their controls is what keeps
    /// one Details pane in this editor instead of five.
    fn select_outliner(&mut self, kind: ui::OutlinerKind, index: usize) {
        use ui::{OutlinerKind, Tool};
        match kind {
            OutlinerKind::Water => {
                self.tool = Tool::Water;
                self.selected_water = Some(index);
            }
            OutlinerKind::Species => {
                self.tool = Tool::Foliage;
                self.species = index;
            }
            OutlinerKind::Prop => {
                self.tool = Tool::Select;
                self.selected_prop = Some(index);
            }
            OutlinerKind::Road => {
                self.tool = Tool::Road;
                self.active_road = Some(index);
            }
            OutlinerKind::Modifier => {
                self.selected_modifier = Some(index);
                self.layout.focus(crate::dock::Tab::Modifiers);
            }
        }
    }

    /// Remove an outliner row.
    ///
    /// Every arm has to leave the *selection* valid as well as the list: these are
    /// indices, so anything at or after the hole now refers to something else. Clearing
    /// the selection is the honest answer rather than guessing which neighbour was meant.
    fn remove_outliner(&mut self, kind: ui::OutlinerKind, index: usize) {
        use ui::OutlinerKind;
        match kind {
            OutlinerKind::Water => {
                if index < self.water.regions.len() {
                    self.water.regions.remove(index);
                    self.selected_water = None;
                    self.unsaved = true;
                }
            }
            OutlinerKind::Prop => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.scatter.remove_prop(index);
                    self.selected_prop = None;
                    self.unsaved = true;
                }
            }
            OutlinerKind::Road => {
                if let Some(w) = self.world.as_mut()
                    && index < w.roads.roads.len()
                {
                    w.roads.roads.remove(index);
                    self.active_road = None;
                    self.rebuild_roads();
                    self.unsaved = true;
                }
            }
            OutlinerKind::Modifier => {
                if index < self.modifiers.len() {
                    self.modifiers.remove(index);
                    self.selected_modifier = None;
                    self.unsaved = true;
                }
            }
            // A species is a file in the project. Nothing to remove from here.
            OutlinerKind::Species => {}
        }
    }

    /// Whether a physical-pixel cursor position falls inside a logical-point rect.
    ///
    /// The conversion is the whole reason this is its own function: `winit` reports the
    /// cursor in **physical pixels** and egui lays out in **logical points**, so on a 2x
    /// display a cursor at the middle of the window is at *twice* the point coordinate of
    /// the rect that contains it. Skipping the divide makes the gate correct only near the
    /// top-left corner, which is exactly the kind of bug that reads as "sometimes it
    /// works".
    fn point_in_rect(cursor_px: (f32, f32), rect: egui::Rect, pixels_per_point: f32) -> bool {
        if pixels_per_point <= 0.0 {
            return true;
        }
        rect.contains(egui::pos2(cursor_px.0 / pixels_per_point, cursor_px.1 / pixels_per_point))
    }

    /// Whether the pointer is over the 3D view rather than over a panel.
    ///
    /// The scene is rendered across the whole window and the docked panels are drawn on
    /// top of it, which is cheap and looks right -- but it means every pointer event
    /// reaches the camera regardless of what it is over. Scrolling a settings list zoomed
    /// the terrain, which is the bug this exists to stop.
    ///
    /// Unreal's viewport is a bounded region that owns its own input, and this is that
    /// boundary: `Tab::Viewport` reports where it ended up and everything camera-facing
    /// asks here first.
    ///
    /// Two coordinate spaces meet here. `winit` reports the cursor in **physical pixels**
    /// and egui lays out in **logical points**, so the cursor has to be divided by the
    /// scale factor before it can be compared -- on a 2x display, skipping that puts the
    /// pointer at twice its real position and the gate is wrong everywhere but the top
    /// left.
    ///
    /// An unknown rect returns `true`: before the editor has laid out once there is no
    /// viewport to be inside, and a dead viewport is worse than a leaky one.
    fn cursor_in_viewport(&self) -> bool {
        let Some(rect) = self.viewport_rect else { return true };
        Self::point_in_rect(self.input.cursor, rect, self.egui_ctx.pixels_per_point())
    }

    /// Change which role the open material fills automatically.
    ///
    /// `ROLE_NONE` makes it paint-only, which is what someone wants when they would
    /// rather place a material by hand than have the slope and erosion masks decide.
    /// The roles then have to reach the GPU, which is `set_materials` -- a uniform
    /// write, so this is cheap enough to do on a click.
    fn set_material_role(&mut self, role: u32) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        gfx.materials.set_role(self.selected_material, role);
        let materials = &gfx.materials;
        let queue = &gfx.ctx.queue;
        if let Some(w) = self.world.as_mut() {
            w.terrain.set_materials(queue, materials);
        }
        // The menu backdrop shares the palette, so it has to agree or the two show
        // the same material in different places.
        if let Some(b) = self.backdrop.as_mut() {
            b.set_materials(queue, materials);
        }
        self.unsaved = true;
    }

    /// Load an imported greyscale map and make it the Noise brush's pattern.
    fn select_noise(&mut self, name: &str) {
        let Some(w) = self.world.as_ref() else { return };
        let path = w.project.paths.assets_dir().join("noise").join(name);

        // `.r16` is offered by the import dialog and is not an image format:
        // headerless little-endian u16, the same layout the heightfield uses on
        // disk. `image::open` cannot read it -- there is nothing to sniff -- so
        // picking one used to log "could not read noise map" and do nothing,
        // which made the advertised option a dead end.
        let is_r16 = path.extension().is_some_and(|x| x.eq_ignore_ascii_case("r16"));
        let (wpx, hpx, gray) = if is_r16 {
            match read_r16_square(&path) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("{}: {e}", path.display());
                    self.notice = Some((format!("{name}: {e}"), true));
                    return;
                }
            }
        } else {
            let img = match image::open(&path) {
                Ok(i) => i.to_luma8(),
                Err(e) => {
                    log::error!("could not read noise map {}: {e}", path.display());
                    self.notice = Some((format!("{name} could not be read: {e}"), true));
                    return;
                }
            };
            (img.width(), img.height(), img.into_raw())
        };
        let img = image::GrayImage::from_raw(wpx, hpx, gray)
            .expect("dimensions were derived from the buffer length");
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
        // Water, for the same reason and beside it. Written even when disabled: the
        // file's presence is what separates "decided against water" from "never asked".
        if let Err(e) = self.water.save(&w.project.paths) {
            log::error!("could not save water: {e}");
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
        self.saved_water = self.water.clone();
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
        // Same treatment for water: a level someone set is authored work, and losing it
        // on close would make the panel a toy. `WaterSettings` is `PartialEq`, so this
        // needs no bespoke comparison.
        if self.world.is_some() && self.water != self.saved_water {
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
        // Mouse motion and wheel notches are consumed while editing *and* while
        // driving -- the chase camera is mouse-aimed. Anything arriving in the menu
        // or during a load still has to be dropped rather than banked, or it applies
        // in one jump on the first frame that reads it.
        if !self.is_editing() && self.play.is_none() {
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
        if let Some(gfx) = self.gfx.as_mut() {
            gfx.sky.upload_camera(&gfx.ctx.queue, &cam, aspect);
            gfx.meshes.upload_camera(&gfx.ctx.queue, &cam, aspect);
            // Cascades are fitted to the camera actually being rendered from, and kept so
            // the shadow passes can cull against them.
            let f = gfx.fog.settings;
            let near = gfx.fog.near();
            let (width, height) = (gfx.ctx.config.width as f32, gfx.ctx.config.height as f32);
            gfx.lighting.upload(
                &gfx.ctx.queue,
                &cam,
                aspect,
                [
                    near,
                    f.distance,
                    if f.enabled { 1.0 } else { 0.0 },
                    terra_render::volumetrics::FROXELS[2] as f32,
                ],
                [width, height],
            );
        }
        // While driving, the terrain must be drawn from the chase camera too.
        if let (Some(p), Some(w), Some(gfx)) =
            (self.play.as_ref(), self.world.as_mut(), self.gfx.as_ref())
        {
            let lights = gfx.lighting.cascade_frusta();
            w.terrain.upload_camera_culled(&gfx.ctx.queue, &p.camera, aspect, &lights);
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

        // `wants_ptr` only covers hovering an interactive widget, so over a panel's blank
        // area the ray still fired and the brush ring tracked the cursor across the
        // settings. The viewport bound is what stops that.
        //
        // Held strokes are exempt: a sculpt drag that runs past the viewport edge must
        // keep hitting the ground, or the stroke breaks wherever the panel starts.
        //
        // Taken before the world is borrowed mutably, since it reads across `self`.
        let over_view = self.cursor_in_viewport() || self.input.sculpting;
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
        if over_view
            && !wants_ptr
            && !self.input.looking
            && !self.input.panning
            && self.tool.edits()
        {
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
                // Water bodies: press starts a rectangle, release commits it. Recorded
                // on the ground rather than on screen, so the box is in world metres and
                // does not change shape when the camera moves mid-drag.
                if self.tool == Tool::Water {
                    if self.input.sculpting && self.water_drag_start.is_none() {
                        self.water_drag_start = Some(c);
                    }
                    self.water_drag_preview = self.water_drag_start.map(|a| (a, c));
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
        // Released: commit the rectangle. Done here rather than in the branch above,
        // because that branch only runs while the cursor is over the ground and a drag
        // can legitimately end with the pointer off it.
        if self.tool == Tool::Water
            && !self.input.sculpting
            && let Some(start) = self.water_drag_start.take()
        {
            self.water_drag_preview = None;
            if let Some(end) = self.brush_hit.or(self.last_brush_hit) {
                // The level a new body starts at: the ground under the middle of the
                // drag, so water appears immediately rather than needing the slider found
                // before anything is visible. Plus a little, or a basin whose centre is
                // its lowest point comes out with a surface exactly at the mud.
                let mid = (start + end) * 0.5;
                // `world` is already borrowed in this scope, so the terrain is read
                // through it rather than through `self`.
                let level = world.terrain.height_at(mid.x, mid.y) + 2.0;
                let r = terra_render::water::WaterRegion::from_drag(
                    [start.x, start.y],
                    [end.x, end.y],
                    level,
                );
                if r.is_usable() {
                    if self.water.regions.len() < terra_render::water::MAX_REGIONS {
                        self.water.regions.push(r);
                        self.selected_water = Some(self.water.regions.len() - 1);
                        self.unsaved = true;
                    } else {
                        self.notice = Some((
                            format!(
                                "At the limit of {} water bodies. Delete one first.",
                                terra_render::water::MAX_REGIONS
                            ),
                            true,
                        ));
                    }
                }
            }
        }

        self.last_brush_hit = if self.input.sculpting { self.brush_hit } else { None };
        world.terrain.set_brush(&gfx.ctx.queue, self.brush_hit, self.brush_radius);
        let lights = gfx.lighting.cascade_frusta();
        world.terrain.upload_camera_culled(&gfx.ctx.queue, &world.camera, aspect, &lights);

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

                // Built here because it reads across the whole of `self`, and the
                // borrows below take `gfx` mutably. Same reasoning as the note on
                // `species_rules`, one step earlier.
                let outliner = self.outliner_items();
                let outliner_selection = self.outliner_selection();

                // Copied out and written back: the pane needs `&mut` on one
                // layer's params while the palette above it holds `gfx` shared.
                let mut material_params = self
                    .gfx
                    .as_ref()
                    .and_then(|g| g.materials.params.get(self.selected_material).copied());
                let material_meta = self.gfx.as_ref().and_then(|g| {
                    let l = g.materials.layers.get(self.selected_material)?;
                    Some((
                        l.name.clone(),
                        terra_render::material::role_label(l.role),
                        l.role,
                        l.auto_role,
                    ))
                });
                let material = match (material_meta.as_ref(), material_params.as_mut()) {
                    (Some((name, role, role_id, auto_role)), Some(params)) => {
                        Some(ui::MaterialView {
                            name,
                            role,
                            texture: self.swatches.get(self.selected_material),
                            params,
                            role_id: *role_id,
                            auto_role: *auto_role,
                        })
                    }
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
                        water: &mut self.water,
                        selected_water: &mut self.selected_water,
                        outliner: &outliner,
                        outliner_selection,
                        active_road,
                        road_count,
                        modifiers: &mut self.modifiers,
                        selected_modifier: &mut self.selected_modifier,
                        content: &content,
                        notice: self.notice.as_ref(),
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
                    EditorAction::SetMaterialRole(role) => self.set_material_role(role),
                    EditorAction::SelectOutliner(kind, i) => self.select_outliner(kind, i),
                    EditorAction::RemoveOutliner(kind, i) => self.remove_outliner(kind, i),
                    EditorAction::PaintWithSelectedMaterial => {
                        self.paint_layer = self.selected_material as u32;
                        self.tool = Tool::Paint;
                    }
                    EditorAction::ImportAsset(k) => self.import_asset(k),
                    EditorAction::RefreshAssets => self.rescan_assets(),
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
                    Pane::Worlds => ui::worlds(
                        ui,
                        &self.library,
                        self.pending_delete
                            .as_ref()
                            .map(|e| ui::DeleteTarget { name: &e.name, path: &e.path }),
                    ),
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
                    Action::AskDelete(path) => {
                        // Only opens the confirmation. Nothing is touched here.
                        self.pending_delete =
                            self.library.projects.iter().find(|e| e.path == path).cloned();
                    }
                    Action::CancelDelete => self.pending_delete = None,
                    Action::Delete { path, files } => self.delete_world(&path, files),
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

        // Water patch selection and uniforms, before the pass that draws them. Uses the
        // camera the scene is actually rendered from, so the LOD is centred on the view
        // rather than on the editor camera while driving.
        {
            let cam = match (self.screen, self.play.as_ref(), self.world.as_ref()) {
                (Screen::Editor, Some(p), _) => Some(p.camera.clone()),
                (Screen::Editor, None, Some(w)) => Some(w.camera.clone()),
                _ => None,
            };
            let (water, aspect, time) = (self.water.clone(), gfx.ctx.aspect(), self.time);
            if let (Some(cam), Some(w)) = (cam, self.world.as_mut()) {
                w.water.prepare(&gfx.ctx.queue, &water, &cam, aspect, time);
            }
        }

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
            gfx.scatter.cull(&mut encoder, &gfx.ctx.queue, cam, gfx.ctx.aspect(), &gfx.hiz);
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

            // Water is the last draw in the pass, and has to be: it is the only blended
            // pipeline here, so everything that can be seen through it must already be
            // in the colour buffer. Drawn before the vehicle, a submerged car would be
            // painted over the surface instead of appearing beneath it.
            if let (Screen::Editor, Some(w)) = (self.screen, self.world.as_ref())
                && !view_mode.is_wireframe()
            {
                w.water.draw(&mut pass, &gfx.lighting, &gfx.env_gpu);
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
        // Before the scatter, which needs its bind group layout: the cull pass tests
        // instances against the pyramid.
        let hiz = HiZ::new(&ctx);
        let t_models = Instant::now();
        // Empty at startup: the palette belongs to a project, and none is open yet.
        // `reload_species` fills it from the project's own `assets/models` when a world
        // opens, which is the only place a user's meshes can come from.
        let scatter =
            Scatter::load(&ctx.device, &ctx.queue, &meshes, &empty_dir(), hiz.cull_layout());
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
                // `consumed` alone is not enough: egui reports a wheel over a scroll area
                // as consumed only in some cases, so a notch over a settings list reached
                // the camera and zoomed the terrain. The viewport owns the wheel, and
                // only inside its own bounds.
                if !consumed && self.cursor_in_viewport() {
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
                // Gated on the *press* only. A release always releases, and a drag begun
                // inside the viewport has to survive the pointer wandering onto a panel --
                // which is exactly what happens when you orbit past the edge.
                let start = !consumed && self.cursor_in_viewport();
                match button {
                    MouseButton::Right if start || !down => self.input.looking = down,
                    MouseButton::Right => {}
                    MouseButton::Middle if start => self.input.panning = down,
                    MouseButton::Middle => self.input.panning = false,
                    MouseButton::Left if start => {
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
                        // Recover the car. Only while driving, and only on the press
                        // -- a held R must not reset every frame, which would pin the
                        // car in the air.
                        KeyCode::KeyR if down && self.play.is_some() => self.reset_car(),
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
        // Raw relative motion, so it is independent of where the pointer is and keeps
        // working once the cursor is locked. While driving it is accepted with no
        // button held at all: the chase camera is mouse-aimed, as in GTA, and needing
        // to hold a button to look around while steering is not playable.
        if let DeviceEvent::MouseMotion { delta } = event
            && (self.input.looking
                || self.input.panning
                || self.input.orbiting
                || self.play.is_some())
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
/// `a` folded into `-pi..=pi`.
///
/// Used to recentre the chase camera the short way round. Without it, a camera
/// 179 degrees from centre takes the long route through a full turn.
fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut x = a % TAU;
    if x > PI {
        x -= TAU;
    } else if x < -PI {
        x += TAU;
    }
    x
}

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

/// Radians of camera rotation per pixel of mouse movement while driving.
///
/// The editor camera's own sensitivity is applied inside `Camera::rotate`; this
/// path sets the angles directly, so it needs its own figure. Matched to the
/// editor's by eye so switching between driving and editing does not feel like
/// two different mice.
const DRIVE_LOOK_SENSITIVITY: f32 = 0.0035;

/// Elevation limits for the chase camera, radians. Positive looks down at the car.
///
/// Not symmetric: there is more to see from above than from underneath, and the
/// upper bound stops short of straight down, where the view basis degenerates --
/// `Camera::right` is a cross product with world up, which is zero-length when the
/// forward vector is parallel to it.
const DRIVE_PITCH_MIN: f32 = -0.25;
const DRIVE_PITCH_MAX: f32 = 1.25;

/// How far the wheel may pull the chase camera in and out, as a multiplier.
const DRIVE_ZOOM_MIN: f32 = 0.45;
const DRIVE_ZOOM_MAX: f32 = 3.0;

// Checked here rather than in a test, because both sides are constants and a bad
// pair should not compile. At exactly vertical the view basis degenerates:
// `Camera::right` is a cross product with world up, zero-length when forward is
// parallel to it, and the NaN spreads through the whole view matrix.
const _: () = assert!(DRIVE_PITCH_MAX < terra_render::camera::PITCH_LIMIT);
const _: () = assert!(DRIVE_PITCH_MIN > -terra_render::camera::PITCH_LIMIT);
// Neither zoom end may collapse the follow distance to nothing or push the car to a
// dot: the distance is this multiplied by roughly two vehicle lengths.
const _: () = assert!(DRIVE_ZOOM_MIN > 0.0 && DRIVE_ZOOM_MIN < 1.0);
const _: () = assert!(DRIVE_ZOOM_MAX > 1.0 && DRIVE_ZOOM_MAX <= 4.0);

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

/// Read a headerless little-endian `u16` map, down to 8-bit grey.
///
/// `.r16` carries no dimensions, so the side length is derived from the file
/// length and has to come out square -- which every `.r16` Terra writes is, and
/// which World Machine and Gaea also export. Anything else is rejected with the
/// size it would have needed, because "not square" is not something the user can
/// act on without knowing what was expected.
fn read_r16_square(path: &std::path::Path) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() % 2 != 0 || bytes.is_empty() {
        return Err(format!("{} bytes is not a whole number of 16-bit samples", bytes.len()));
    }
    let samples = bytes.len() / 2;
    let n = (samples as f64).sqrt().round() as usize;
    if n * n != samples {
        return Err(format!(
            "{samples} samples is not square -- an r16 map has to be N by N (nearest is {n}x{n})"
        ));
    }
    // Down to 8 bits because the Noise brush samples an 8-bit pattern. The extra
    // 8 bits describe height steps far below what a displacement brush resolves.
    let gray =
        bytes.chunks_exact(2).map(|c| (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8).collect();
    Ok((n as u32, n as u32, gray))
}

/// One line describing what an import did, and whether to show it as a failure.
///
/// A pure function of the outcome so the wording is testable without a file
/// dialog. Rejections are named individually rather than counted: "1 folder
/// skipped" tells the user nothing they can act on, and the reason is the whole
/// value of the message.
fn import_summary(
    installed: &[String],
    rejected: &[(String, String)],
    incomplete: &[(String, Vec<&'static str>)],
    unreadable: &[(String, Vec<String>)],
) -> (String, bool) {
    let mut ok = match installed.len() {
        0 => String::new(),
        1 => format!("Imported {}.", installed[0]),
        n => format!("Imported {n} materials: {}.", installed.join(", ")),
    };
    // Named individually and with the maps they lack, because "incomplete" on its own
    // is not actionable and the consequence is severe: a set with no normal map has no
    // relief for the light to catch and renders as a photograph laid over the ground,
    // which is easy to mistake for the renderer being broken.
    if !incomplete.is_empty() {
        let each: Vec<String> =
            incomplete.iter().map(|(n, m)| format!("{n} has no {}", m.join(" or "))).collect();
        ok.push_str(&format!(
            " {} -- flat defaults are used, so download the missing maps for real relief.",
            each.join("; ")
        ));
    }
    // Named with the file, because the fix is to convert something they already have
    // rather than to go and download it again.
    if !unreadable.is_empty() {
        let each: Vec<String> = unreadable
            .iter()
            .map(|(n, files)| format!("{n} could not read {}", files.join(", ")))
            .collect();
        ok.push_str(&format!(
            " {} -- convert those to PNG. Readable formats are {}.",
            each.join("; "),
            terra_render::texture_set::MAP_EXTENSIONS.join(", ")
        ));
    }
    if rejected.is_empty() {
        return match installed.is_empty() {
            // Reachable when a picked folder holds only empty subfolders.
            true => ("Nothing imported -- no material folders found.".to_string(), true),
            false => (ok, false),
        };
    }
    let why: Vec<String> =
        rejected.iter().map(|(name, reason)| format!("{name}: {reason}")).collect();
    let skipped = format!("Skipped {} -- {}", rejected.len(), why.join("; "));
    match installed.is_empty() {
        true => (skipped, true),
        // A partial success is still a success: the palette changed, so it must
        // not read as an error, but the skipped folder has to be named or the
        // user is left counting swatches to notice it is missing.
        false => (format!("{ok} {skipped}"), false),
    }
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

    // --- chase camera ---

    /// Where the chase camera sits, given the orbit angles and a distance. Mirrors
    /// the expression in `update_play`.
    fn chase_offset(yaw: f32, pitch: f32) -> Vec3 {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    #[test]
    fn the_camera_starts_behind_the_car() {
        // `heading + pi` has to put the camera on the far side of the car from the way
        // it is pointing. Getting this backwards is what previously had the camera
        // looking at the bonnet, and it is also what makes steering *look* inverted:
        // seen head-on, a left turn goes right.
        for heading in [0.0f32, 0.9, -2.2, 3.0] {
            let forward = Vec3::new(heading.cos(), 0.0, heading.sin());
            let offset = chase_offset(heading + std::f32::consts::PI, 0.0);
            assert!(
                offset.dot(forward) < -0.99,
                "at heading {heading} the camera sat {:.2} along forward, not behind it",
                offset.dot(forward)
            );
        }
    }

    #[test]
    fn a_positive_camera_pitch_lifts_it_above_the_car() {
        assert!(chase_offset(0.0, 0.5).y > 0.0);
        assert!(chase_offset(0.0, -0.2).y < 0.0);
    }

    #[test]
    fn wrap_angle_takes_the_short_way_round() {
        use std::f32::consts::PI;
        // The case it exists for: nearly a full turn one way is a hair the other.
        assert!((wrap_angle(2.0 * PI - 0.1) + 0.1).abs() < 1e-4);
        assert!((wrap_angle(-2.0 * PI + 0.1) - 0.1).abs() < 1e-4);
        for a in [0.0f32, 1.0, -1.0, PI - 0.01, -PI + 0.01] {
            assert!((wrap_angle(a) - a).abs() < 1e-4, "{a} should be unchanged");
        }
        // Everything lands inside the half-open turn.
        for k in -8..8 {
            let a = k as f32 * 1.7;
            assert!(wrap_angle(a).abs() <= PI + 1e-4, "{a} wrapped to {}", wrap_angle(a));
        }
    }

    #[test]
    fn recentring_converges_on_the_heading_from_either_side() {
        // The recentre is a lerp along the wrapped delta, so it must approach from
        // whichever side is closer and not orbit the long way.
        let behind = 1.0f32;
        for start in [behind + 3.0, behind - 3.0, behind + 0.2] {
            let mut yaw = start;
            for _ in 0..200 {
                yaw += wrap_angle(behind - yaw) * 0.1;
            }
            assert!(
                wrap_angle(behind - yaw).abs() < 1e-2,
                "from {start} it settled at {yaw}, wanted {behind}"
            );
        }
    }

    #[test]
    fn sanitize_produces_usable_folder_names() {
        assert_eq!(sanitize("Desert Rally"), "Desert_Rally");
        assert_eq!(sanitize("../../etc"), "etc");
        assert_eq!(sanitize("!!!"), "world");
    }

    // --- import feedback ------------------------------------------------------
    //
    // The import used to say nothing at all, in the one case where saying
    // nothing was indistinguishable from working: files landed on disk and no
    // material appeared. These assert the message exists and is actionable.

    #[test]
    fn a_successful_import_names_what_arrived() {
        let (msg, err) = import_summary(&["Ground024".into()], &[], &[], &[]);
        assert_eq!(msg, "Imported Ground024.");
        assert!(!err);
    }

    #[test]
    fn importing_several_materials_counts_and_lists_them() {
        let (msg, err) = import_summary(&["Grass001".into(), "Rock042".into()], &[], &[], &[]);
        assert!(msg.contains("2 materials"), "{msg}");
        assert!(msg.contains("Grass001") && msg.contains("Rock042"), "{msg}");
        assert!(!err);
    }

    #[test]
    fn a_rejected_import_is_an_error_and_gives_the_reason() {
        let rejected = vec![("Screenshots".to_string(), "no colour map found".to_string())];
        let (msg, err) = import_summary(&[], &rejected, &[], &[]);
        assert!(err, "a wholly failed import must not read as success");
        assert!(msg.contains("Screenshots"), "the folder has to be named: {msg}");
        assert!(msg.contains("no colour map found"), "the reason is the point: {msg}");
    }

    #[test]
    fn a_partial_import_reports_both_halves_and_is_not_an_error() {
        // The palette did change, so flagging it red would be wrong -- but the
        // skipped folder still has to be named, or the only clue is a missing
        // swatch the user has to notice by counting.
        let rejected = vec![("Notes".to_string(), "no colour map found".to_string())];
        let (msg, err) = import_summary(&["Grass001".into()], &rejected, &[], &[]);
        assert!(!err, "a partial success is a success: {msg}");
        assert!(msg.contains("Grass001"), "{msg}");
        assert!(msg.contains("Notes"), "{msg}");
    }

    #[test]
    fn an_empty_import_still_says_something() {
        // Reachable when a picked folder holds only empty subfolders. Silence
        // here is the original bug.
        let (msg, err) = import_summary(&[], &[], &[], &[]);
        assert!(err);
        assert!(!msg.is_empty());
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

#[cfg(test)]
mod r16_tests {
    use super::read_r16_square;

    fn write(bytes: &[u8], tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("terra-r16-{tag}.r16"));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn a_square_r16_map_is_read_and_reduced_to_eight_bits() {
        // 4x4 of u16, ascending. The dialog offers `.r16` and `image::open`
        // cannot sniff a headerless format, so this path is the only thing that
        // makes the advertised option work.
        let mut bytes = Vec::new();
        for i in 0u16..16 {
            bytes.extend_from_slice(&(i * 4096).to_le_bytes());
        }
        let p = write(&bytes, "ok");
        let (w, h, gray) = read_r16_square(&p).expect("square map");
        assert_eq!((w, h), (4, 4));
        assert_eq!(gray.len(), 16);
        // 4096 >> 8 = 16, so the ramp survives as a byte ramp.
        assert_eq!(gray[0], 0);
        assert_eq!(gray[1], 16);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn a_non_square_r16_map_says_what_size_was_expected() {
        // 6 samples. "Not square" is useless on its own -- the message has to
        // name the size it wanted or there is nothing to act on.
        let bytes = vec![0u8; 12];
        let p = write(&bytes, "oblong");
        let err = read_r16_square(&p).expect_err("6 samples is not square");
        assert!(err.contains("not square"), "{err}");
        assert!(err.contains("2x2"), "the nearest square has to be named: {err}");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn an_odd_byte_count_is_rejected_rather_than_truncated() {
        let p = write(&[0u8; 7], "odd");
        let err = read_r16_square(&p).expect_err("7 bytes is not whole samples");
        assert!(err.contains("16-bit"), "{err}");
        let _ = std::fs::remove_file(p);
    }
}

#[cfg(test)]
mod viewport_gate_tests {
    use super::App;
    use egui::{Rect, pos2};

    /// A viewport occupying the middle of a 1600x1000 point layout: tools on the left,
    /// details on the right, content along the bottom. Roughly the default dock.
    fn viewport() -> Rect {
        Rect::from_min_max(pos2(256.0, 60.0), pos2(1200.0, 740.0))
    }

    #[test]
    fn a_cursor_over_the_viewport_is_inside() {
        assert!(App::point_in_rect((700.0, 400.0), viewport(), 1.0));
    }

    #[test]
    fn a_cursor_over_a_side_panel_is_outside() {
        // The bug: a wheel notch here zoomed the terrain while scrolling the settings.
        assert!(!App::point_in_rect((80.0, 400.0), viewport(), 1.0), "left panel");
        assert!(!App::point_in_rect((1400.0, 400.0), viewport(), 1.0), "right panel");
        assert!(!App::point_in_rect((700.0, 900.0), viewport(), 1.0), "content browser");
        assert!(!App::point_in_rect((700.0, 20.0), viewport(), 1.0), "toolbar");
    }

    #[test]
    fn the_scale_factor_is_applied() {
        // On a 2x display a cursor at physical (1400, 800) is at logical (700, 400) --
        // inside the viewport. Ignoring the scale factor would place it at (1400, 800),
        // outside, and the viewport would be dead over most of its own area.
        let r = viewport();
        assert!(App::point_in_rect((1400.0, 800.0), r, 2.0), "a 2x cursor was rejected");
        assert!(!App::point_in_rect((1400.0, 800.0), r, 1.0), "at 1x the same pixel is outside");
    }

    #[test]
    fn a_nonsense_scale_factor_does_not_kill_the_viewport() {
        // A dead viewport is worse than a leaky one, so an impossible scale factor errs
        // towards letting input through.
        assert!(App::point_in_rect((80.0, 400.0), viewport(), 0.0));
        assert!(App::point_in_rect((80.0, 400.0), viewport(), -1.0));
    }

    #[test]
    fn the_boundary_belongs_to_the_viewport() {
        // `Rect::contains` is inclusive, so the edge pixel is inside. Either answer is
        // defensible; what matters is that it is not undefined between the two tests that
        // read it.
        let r = viewport();
        assert!(App::point_in_rect((256.0, 60.0), r, 1.0), "the min corner");
        assert!(App::point_in_rect((1200.0, 740.0), r, 1.0), "the max corner");
        assert!(!App::point_in_rect((255.0, 400.0), r, 1.0), "one point left of it");
        assert!(!App::point_in_rect((1201.0, 400.0), r, 1.0), "one point right of it");
    }
}

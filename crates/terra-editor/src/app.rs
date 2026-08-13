//! Editor shell: window, surface, egui, and the screen state machine.
//!
//! The window, GPU context, sky and menu backdrop are built once in `resumed`
//! and live for the whole session. Navigating between menu panes only changes
//! which controls the left rail draws -- nothing is torn down or rebuilt.

use crate::theme;
use crate::ui::{self, Action, CreateForm, EditorAction, EditorView, Pane, PerfView, Tool};
use anyhow::Result;
use glam::{Mat4, Quat};
use glam::{Vec2, Vec3, Vec4};
use std::sync::Arc;
use std::time::{Duration, Instant};
use terra_core::{BASE_ELEVATION_M, WorldSize};
use terra_physics::{FIXED_DT, PhysicsWorld, Vehicle, VehicleInput, vehicle};
use terra_project::roads::{Road, RoadNetwork};
use terra_project::{Library, Project, TerrainParams, WorldData};
use terra_render::camera::Camera;
use terra_render::context::RenderContext;
use terra_render::mesh::{Instance, MeshRenderer};
use terra_render::sky::Sky;
use terra_render::stats::{FrameStats, GpuTimer};
use terra_render::terrain::{SculptMode, Terrain, apply_brush};
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
    looking: bool,
    sculpting: bool,
    /// One-shot: a left press that has not been consumed yet. Sculpting wants a
    /// held state, but placing a road point wants a single event.
    clicked: bool,
    cursor: (f32, f32),
    look_delta: (f32, f32),
}

impl Input {
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
    meshes: MeshRenderer,
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
    brush_radius: f32,
    brush_strength: f32,
    brush_hit: Option<Vec2>,
    tool: Tool,
    /// Road currently being drawn, as an index into the network.
    active_road: Option<usize>,
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
            brush_radius: 120.0,
            brush_strength: 1.5,
            brush_hit: None,
            tool: Tool::Sculpt,
            active_road: None,
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
        let car = Vehicle::spawn(&mut physics, [eye.x, ground + 3.0, eye.z]);

        let (t, r) = car.chassis_pose(&physics);
        let pose = Pose {
            translation: Vec3::from_array(t),
            rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]),
        };
        let wheels = wheel_poses(&car, pose.rotation);

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

        // Chase camera follows the interpolated pose, smoothed so it lags the
        // car slightly rather than being welded to it.
        let alpha = play.accumulator / FIXED_DT;
        let pose = interpolate(play.prev, play.curr, alpha);
        let back = pose.rotation * Vec3::new(0.0, 0.0, -1.0);
        let want = pose.translation - back * 9.0 + Vec3::Y * 3.6;
        let follow = 1.0 - (-dt * 9.0).exp();
        play.camera.pos = play.camera.pos.lerp(want, follow);

        let to_car = (pose.translation + Vec3::Y * 1.0) - play.camera.pos;
        play.camera.yaw = to_car.z.atan2(to_car.x);
        play.camera.pitch = (to_car.y / to_car.length().max(0.01)).clamp(-1.0, 1.0).asin();
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
        if let Err(e) = w.roads.save(&w.project.paths) {
            log::error!("could not save roads: {e}");
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
        match self.screen {
            Screen::Editor if self.play.is_some() => self.update_play(dt),
            Screen::Editor => {
                self.update_editor(dt);
                if self.pending_stroke_finish {
                    self.pending_stroke_finish = false;
                    self.finish_stroke();
                }
            }
            Screen::Loading => {
                self.update_backdrop();
                self.update_loading(dt);
            }
            Screen::Menu(_) => self.update_backdrop(),
        }

        // The sky follows whichever camera is active this frame.
        let aspect = self.gfx.as_ref().map(|g| g.ctx.aspect()).unwrap_or(1.0);
        let cam = match (self.screen, self.play.as_ref(), self.world.as_ref()) {
            (Screen::Editor, Some(p), _) => p.camera.clone(),
            (Screen::Editor, None, Some(w)) => w.camera.clone(),
            _ => self.backdrop_cam.clone(),
        };
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.sky.upload_camera(&gfx.ctx.queue, &cam, aspect);
            gfx.meshes.upload_camera(&gfx.ctx.queue, &cam, aspect);
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
                    l.terrain = Some(Terrain::new(&gfx.ctx, l.project.size()));
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
                    }
                    l.stage = 3;
                }
                _ => done = l.elapsed >= Loading::MIN_SECONDS,
            }
        }

        if done {
            if let Some(l) = self.loading.take() {
                self.finish_loading(l);
            }
        }
    }

    /// Slow cinematic orbit around the menu landscape.
    fn update_backdrop(&mut self) {
        const RADIUS: f32 = 1180.0;
        let t = self.time * 0.045;
        self.backdrop_cam.pos = Vec3::new(t.cos() * RADIUS, 430.0, t.sin() * RADIUS);
        // Face the middle of the map: the inward direction is the orbit angle
        // turned by half a turn.
        self.backdrop_cam.yaw = t + std::f32::consts::PI;
        self.backdrop_cam.pitch = -0.15;

        let aspect = self.gfx.as_ref().map(|g| g.ctx.aspect()).unwrap_or(1.0);
        if let (Some(b), Some(gfx)) = (self.backdrop.as_ref(), self.gfx.as_ref()) {
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

        if let (Some(i), Some(w)) = (self.active_road, self.world.as_mut()) {
            if let Some(r) = w.roads.roads.get_mut(i) {
                // Drawing again extends the same road rather than starting a
                // new one, so a long track can be laid down in passes.
                if r.points.last() == simplified.first() {
                    r.points.extend(simplified.into_iter().skip(1));
                } else {
                    r.points.extend(simplified);
                }
            }
        }
        self.rebuild_roads();
    }

    fn update_editor(&mut self, dt: f32) {
        let wants_kb = self.egui_ctx.egui_wants_keyboard_input();
        let wants_ptr = self.egui_ctx.egui_wants_pointer_input();
        let aspect = self.gfx.as_ref().map(|g| g.ctx.aspect()).unwrap_or(1.0);

        let (Some(world), Some(gfx)) = (self.world.as_mut(), self.gfx.as_ref()) else {
            return;
        };
        world.camera.speed = self.settings.camera_speed;
        world.camera.fov_y = self.settings.fov_deg.to_radians();

        if self.input.looking {
            let (dx, dy) = self.input.look_delta;
            world.camera.rotate(dx * 0.0025, dy * 0.0025);
        }
        self.input.look_delta = (0.0, 0.0);

        if !wants_kb {
            world.camera.translate(self.input.axis(), dt, self.input.boost);
        }

        self.brush_hit = None;
        let mut finish_stroke = false;
        if !wants_ptr && !self.input.looking {
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
            if let Some(hit) = world.terrain.raycast(o, d) {
                let c = Vec2::new(hit.x, hit.z);
                self.brush_hit = Some(c);
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
                if self.input.sculpting && self.tool == Tool::Sculpt {
                    // Scale by dt so a held stroke deposits the same material
                    // regardless of frame rate.
                    let strength = self.brush_strength * dt * 60.0;
                    world.terrain.sculpt(
                        &gfx.ctx.queue,
                        c,
                        self.brush_radius,
                        strength,
                        self.brush_mode,
                    );
                    // Same edit into the base, so a road rebuild does not undo
                    // it and so it is what gets saved.
                    apply_brush(
                        &mut world.base,
                        world.terrain.resolution(),
                        world.terrain.extent_m(),
                        c,
                        self.brush_radius,
                        strength,
                        self.brush_mode,
                    );
                    self.unsaved = true;
                }
            }
        }
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
                        active_road,
                        road_count,
                        playing: self.play.is_some(),
                        speed_kph: self
                            .play
                            .as_ref()
                            .map(|p| p.car.speed().abs() * 3.6)
                            .unwrap_or(0.0),
                    },
                );
                match act {
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
                        if let (Some(i), Some(w)) = (self.active_road, self.world.as_mut()) {
                            if let Some(r) = w.roads.roads.get_mut(i) {
                                r.points.pop();
                            }
                        }
                        self.rebuild_roads();
                    }
                    EditorAction::FinishRoad => {
                        // Drop a road that never got enough points to exist.
                        if let (Some(i), Some(w)) = (self.active_road, self.world.as_mut()) {
                            if w.roads.roads.get(i).is_some_and(|r| !r.is_drawable()) {
                                w.roads.roads.remove(i);
                            }
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
                    // Clear the editor's inspector rather than floating on it.
                    right_inset: if self.screen == Screen::Editor { 268.0 } else { 0.0 },
                },
            );
        }
        quit
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
        let Some(gfx) = self.gfx.as_mut() else {
            textures_delta.clear();
            return Ok(Frame::Idle);
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gfx
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        let scene_ts = gfx.gpu_timer.as_ref().and_then(|t| t.scene_writes());
        let ui_ts = gfx.gpu_timer.as_ref().and_then(|t| t.ui_writes());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.055,
                            g: 0.062,
                            b: 0.085,
                            a: 1.0,
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
            gfx.sky.draw(&mut pass);
            match (self.screen, self.world.as_ref(), self.backdrop.as_ref()) {
                (Screen::Editor, Some(w), _) => w.terrain.draw(&mut pass),
                (_, _, Some(b)) => b.draw(&mut pass),
                _ => {}
            }
            if let Some(p) = self.play.as_ref() {
                // Chassis first, then the four wheels, in one instance buffer.
                let alpha = p.accumulator / FIXED_DT;
                let body = interpolate(p.prev, p.curr, alpha);
                let mut instances = vec![Instance::new(
                    Mat4::from_rotation_translation(body.rotation, body.translation),
                    Vec3::new(0.16, 0.30, 0.20),
                )];
                for (a, b) in p.prev_wheels.iter().zip(&p.curr_wheels) {
                    let w = interpolate(*a, *b, alpha);
                    instances.push(Instance::new(
                        Mat4::from_rotation_translation(w.rotation, w.translation),
                        Vec3::new(0.035, 0.035, 0.040),
                    ));
                }
                let n = gfx.meshes.upload_instances(&gfx.ctx.queue, &instances);
                gfx.meshes.draw(&mut pass, &gfx.meshes.chassis, 0, 1.min(n));
                gfx.meshes.draw(&mut pass, &gfx.meshes.wheel, 1, n.saturating_sub(1));
            }
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
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0));
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

        let sky = Sky::new(&ctx);
        let meshes = MeshRenderer::new(&ctx, vehicle::CHASSIS_HALF, vehicle::WHEEL_RADIUS);
        let gpu_timer = GpuTimer::new(&ctx.device, &ctx.queue, ctx.supports_timestamps());
        let gfx = Gfx { ctx, sky, meshes, egui_renderer, gpu_timer };

        // Menu backdrop: built once, reused for the whole session.
        let mut backdrop = Terrain::new(&gfx.ctx, BACKDROP_SIZE);
        let params = backdrop_params();
        let heights =
            terra_gen::heightfield::generate(backdrop.resolution(), backdrop.extent_m(), &params);
        backdrop.set_heights(&gfx.ctx.queue, heights);
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
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Right => self.input.looking = down,
                    MouseButton::Left if !consumed => {
                        self.input.sculpting = down;
                        self.input.clicked |= down;
                    }
                    MouseButton::Left => self.input.sculpting = false,
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
                        KeyCode::BracketLeft if down => {
                            self.brush_radius = (self.brush_radius * 0.85).max(8.0);
                        }
                        KeyCode::BracketRight if down => {
                            self.brush_radius = (self.brush_radius * 1.18).min(800.0);
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
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.input.looking {
                self.input.look_delta.0 += delta.0 as f32;
                self.input.look_delta.1 += delta.1 as f32;
            }
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

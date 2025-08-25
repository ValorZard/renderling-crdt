//! An example app showing (and verifying) how frustum culling works in
//! `renderling`.
use std::{any::Any, sync::Arc};
mod camera;
mod utils;
use crate::{camera::CameraController, utils::*};
use anyhow::Context as anyContext;
use automerge::{Automerge, ReadDoc, transaction::Transactable};
use clap::Parser;
use glam::*;
use hex::encode;
use iroh::NodeId;
use renderling::{
    Context,
    bvol::{Aabb, BoundingSphere},
    math::hex_to_vec4,
    prelude::*,
    tonemapping::srgba_to_linear,
};
use renderling_crdt::IrohRepo;
use std::str::FromStr;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent},
    event_loop::ActiveEventLoop,
    keyboard::Key,
};

use samod::{DocumentId, PeerId, Samod, storage::TokioFilesystemStorage};

const MIN_SIZE: f32 = 0.5;
const MAX_SIZE: f32 = 4.0;
const MAX_DIST: f32 = 40.0;
const BOUNDS: Aabb = Aabb {
    min: Vec3::new(-MAX_DIST, -MAX_DIST, -MAX_DIST),
    max: Vec3::new(MAX_DIST, MAX_DIST, MAX_DIST),
};

struct AppCamera(Hybrid<Camera>);
struct FrustumCamera(Camera);

#[allow(dead_code)]
struct CullingExample {
    app_camera: AppCamera,
    controller: camera::TurntableCameraController,
    stage: Stage,
    dlights: [AnalyticalLight; 2],
    material_aabb_overlapping: Hybrid<Material>,
    material_aabb_outside: Hybrid<Material>,
    material_frustum: Hybrid<Material>,
    frustum_camera: FrustumCamera,
    frustum_vertices: HybridArray<Vertex>,
    frustum_renderlet: Hybrid<Renderlet>,
    resources: BagOfResources,
    next_k: u64,
    router: RouterContainer,
    iroh_repo_protocol: ProtocolContainer,
    document_id: String,
}

impl CullingExample {
    fn make_aabb(center: Vec3, half_size: Vec3) -> Aabb {
        let min = center - half_size;
        let max = center + half_size;
        Aabb::new(min, max)
    }

    fn make_render_aabb(
        rotation: Quat,
        center: Vec3,
        half_size: Vec3,
        stage: &Stage,
        frustum_camera: &FrustumCamera,
        material_outside: &Hybrid<Material>,
        material_overlapping: &Hybrid<Material>,
    ) -> (
        Gpu<Renderlet>,
        renderling::prelude::HybridArray<renderling::stage::Vertex>,
        renderling::prelude::Hybrid<renderling::transform::Transform>,
    ) {
        let aabb = Self::make_aabb(Vec3::ZERO, half_size);
        let aabb_transform = Transform {
            translation: center,
            rotation,
            ..Default::default()
        };

        let transform = stage.new_transform(aabb_transform);
        let (aabb_vertices, aabb_renderlet) = {
            let material_id = if BoundingSphere::from(aabb)
                .is_inside_camera_view(&frustum_camera.0, transform.get())
                .0
            {
                material_overlapping.id()
            } else {
                material_outside.id()
            };
            let (renderlet, vertices) = stage
                .builder()
                .with_vertices(
                    aabb.get_mesh()
                        .into_iter()
                        .map(|(position, normal)| Vertex {
                            position,
                            normal,
                            ..Default::default()
                        }),
                )
                .with_transform_id(transform.id())
                .with_material_id(material_id)
                .build();
            (renderlet, vertices.into_gpu_only())
        };
        (aabb_renderlet, aabb_vertices, transform)
    }

    fn make_aabbs(
        seed: u64,
        stage: &Stage,
        frustum_camera: &FrustumCamera,
        material_outside: &Hybrid<Material>,
        material_overlapping: &Hybrid<Material>,
    ) -> Box<dyn Any> {
        log::info!("generating aabbs with seed {seed}");
        fastrand::seed(seed);
        Box::new(
            (0..25u32)
                .map(|i| {
                    log::info!("aabb {i}");
                    let x = fastrand::f32() * MAX_DIST - MAX_DIST / 2.0;
                    let y = fastrand::f32() * MAX_DIST - MAX_DIST / 2.0;
                    let z = fastrand::f32() * MAX_DIST - MAX_DIST / 2.0;
                    let w = fastrand::f32() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;
                    let h = fastrand::f32() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;
                    let l = fastrand::f32() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;

                    let rx = std::f32::consts::PI * fastrand::f32();
                    let ry = std::f32::consts::PI * fastrand::f32();
                    let rz = std::f32::consts::PI * fastrand::f32();

                    let rotation = Quat::from_euler(EulerRot::XYZ, rx, ry, rz);

                    let center = Vec3::new(x, y, z);
                    let half_size = Vec3::new(w, h, l);
                    Self::make_render_aabb(
                        rotation,
                        center,
                        half_size,
                        stage,
                        frustum_camera,
                        material_outside,
                        material_overlapping,
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl ApplicationHandler for CullingExample {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        log::info!("culling-example resumed");
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(c),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if c.as_str() == "r" {
                    self.resources.drain();
                    let _ = self.stage.commit();
                    self.resources.push(Self::make_aabbs(
                        self.next_k,
                        &self.stage,
                        &self.frustum_camera,
                        &self.material_aabb_outside,
                        &self.material_aabb_overlapping,
                    ));
                    self.next_k += 1;
                    let document_id = self.document_id.clone();
                    let iroh_repo_protocol = self.iroh_repo_protocol.clone();
                    n0_future::task::spawn(async move {
                        // Lock inside the async block, not before
                        print_document(&document_id, &iroh_repo_protocol).await;
                    });
                }
            }
            winit::event::WindowEvent::Resized(physical_size) => {
                log::info!("window resized to {physical_size:?}");
                let size = UVec2 {
                    x: physical_size.width,
                    y: physical_size.height,
                };
                self.stage.set_size(size);
                self.controller.update_camera(size, &self.app_camera.0);
            }
            event => self.controller.window_event(event),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        self.controller.device_event(event);
    }
}

impl TestAppHandler for CullingExample {
    fn new(
        _event_loop: &ActiveEventLoop,
        _window: Arc<winit::window::Window>,
        ctx: &Context,
        router: RouterContainer,
        iroh_repo_protocol: ProtocolContainer,
        document_id: String,
    ) -> Self {
        let mut seed = 46;
        let mut resources = BagOfResources::default();
        let stage = ctx.new_stage().with_lighting(true);
        let sunlight_a = stage.new_analytical_light(DirectionalLightDescriptor {
            direction: Vec3::new(-0.8, -1.0, 0.5).normalize(),
            color: Vec4::ONE,
            intensity: 10.0,
        });
        let sunlight_b = stage.new_analytical_light(DirectionalLightDescriptor {
            direction: Vec3::new(1.0, 1.0, -0.1).normalize(),
            color: Vec4::ONE,
            intensity: 1.0,
        });

        let dlights = [sunlight_a, sunlight_b];

        let frustum_camera = FrustumCamera({
            let aspect = 1.0;
            let fovy = core::f32::consts::FRAC_PI_4;
            let znear = 4.0;
            let zfar = 1000.0;
            let projection = Mat4::perspective_rh(fovy, aspect, znear, zfar);
            let eye = Vec3::new(0.0, 0.0, 10.0);
            let target = Vec3::ZERO;
            let up = Vec3::Y;
            let view = Mat4::look_at_rh(eye, target, up);
            // let projection = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, -10.0,
            // 10.0); let view = Mat4::IDENTITY;
            Camera::new(projection, view)
        });

        let frustum = frustum_camera.0.frustum();
        log::info!("frustum.planes: {:#?}", frustum.planes);

        let blue_color = srgba_to_linear(hex_to_vec4(0x7EACB5FF));
        let red_color = srgba_to_linear(hex_to_vec4(0xC96868FF));
        let yellow_color = srgba_to_linear(hex_to_vec4(0xFADFA1FF));

        let material_aabb_overlapping = stage.new_material(Material {
            albedo_factor: blue_color,
            ..Default::default()
        });
        let material_aabb_outside = stage.new_material(Material {
            albedo_factor: red_color,
            ..Default::default()
        });
        let material_frustum = stage.new_material(Material {
            albedo_factor: yellow_color,
            ..Default::default()
        });
        let app_camera = AppCamera(stage.new_camera(Camera::default()));
        resources.push(Self::make_aabbs(
            seed,
            &stage,
            &frustum_camera,
            &material_aabb_outside,
            &material_aabb_overlapping,
        ));
        seed += 1;

        let frustum_vertices =
            stage.new_vertices(frustum_camera.0.frustum().get_mesh().into_iter().map(
                |(position, normal)| Vertex {
                    position,
                    normal,
                    ..Default::default()
                },
            ));
        let frustum_renderlet = stage.new_renderlet(Renderlet {
            vertices_array: frustum_vertices.array(),
            material_id: material_frustum.id(),
            ..Default::default()
        });
        stage.add_renderlet(&frustum_renderlet);

        // create player in document
        let document =
            n0_future::future::block_on({ get_document(&document_id, &iroh_repo_protocol) })
                .unwrap();

        let player_id = router.lock().unwrap().endpoint().node_id().to_string();

        let _ = document
            .with_document(|doc| {
                doc.transact(|tx| tx.put(automerge::ROOT, format!("player_{player_id}"), player_id))
            })
            .map_err(debug_err);

        Self {
            next_k: seed,
            app_camera,
            frustum_camera,
            dlights,
            controller: {
                let mut c = camera::TurntableCameraController::default();
                c.reset(BOUNDS);
                c.phi = 0.5;
                c
            },
            stage,
            material_aabb_overlapping,
            material_aabb_outside,
            material_frustum,
            frustum_vertices,
            frustum_renderlet,
            resources,
            router,
            iroh_repo_protocol,
            document_id,
        }
    }

    fn render(&mut self, ctx: &Context) {
        let size = self.stage.get_size();
        self.controller.update_camera(size, &self.app_camera.0);

        let frame = ctx.get_next_frame().unwrap();
        self.stage.render(&frame.view());
        frame.present();
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// A NodeId to connect to and keep syncing with
    #[clap(long)]
    sync_with: Option<NodeId>,

    /// Path where storage files will be created
    #[clap(long, default_value = "./crdt_data")]
    storage_path: String,

    /// Print the secret key
    #[clap(long)]
    print_secret_key: bool,

    #[clap(subcommand)]
    command: Commands,
}

/// Subcommands for the iroh doctor.
#[derive(clap::Subcommand, Debug, Clone)]
pub enum Commands {
    /// Creates a new document with given key-value pairs
    /// and prints its document ID.
    Create {
        /// An initial key to set
        key: String,
        /// An initial value to set
        value: String,
    },
    /// Updates a document with the given document ID, either
    /// inserting a new key-value pair or updating the value at
    /// an existing key.
    Upsert {
        /// The document ID of the document to modify
        doc: String,
        /// The key to set
        key: String,
        /// The value to set at the given key
        value: String,
    },
    /// Prints document's contents by ID
    Print {
        /// The document's ID
        doc: String,
    },
    /// Subscribe to a document and print changes as they occur
    Subscribe {
        /// The document's ID to subscribe to
        doc: String,
    },
    /// Delete a key from the document
    Delete {
        /// The document's ID of the document to modify
        doc: String,

        /// The key you want to delete
        key: String,
    },
    /// Host the app and serve existing documents from the automerge-repo stored in at config-path
    Host,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Pull in the secret key from the environment variable if it exists
    // This key is the public Node ID used to identify the node in the network
    // If not provided, a random key will be generated, and a new Node ID will
    // be assigned each time the app is started
    let secret_key = match std::env::var("IROH_SECRET") {
        Ok(key_hex) => match iroh::SecretKey::from_str(&key_hex) {
            Ok(key) => Some(key),
            Err(_) => {
                println!("invalid IROH_SECRET provided: not valid hex");
                None
            }
        },
        Err(_) => None,
    };

    let secret_key = match secret_key {
        Some(key) => {
            println!("Using existing key: {}", key.public());
            key
        }
        None => {
            println!("Generating new key");
            let mut rng = rand::rngs::OsRng;
            iroh::SecretKey::generate(&mut rng)
        }
    };

    if args.print_secret_key {
        println!("Secret Key: {}", encode(secret_key.to_bytes()));
        println!(
            "Set env var for persistent Node ID: export IROH_SECRET={}",
            encode(secret_key.to_bytes())
        );
    }

    let endpoint = iroh::Endpoint::builder()
        .discovery_n0()
        .secret_key(secret_key)
        .bind()
        .await?;

    println!("Node ID: {}", endpoint.node_id());

    let samod = Samod::build_tokio()
        .with_peer_id(PeerId::from_string(endpoint.node_id().to_string()))
        .with_storage(TokioFilesystemStorage::new(format!(
            "{}/{}",
            args.storage_path,
            endpoint.node_id()
        )))
        .load()
        .await;
    let proto = IrohRepo::new(endpoint.clone(), samod);
    let router = iroh::protocol::Router::builder(endpoint)
        .accept(IrohRepo::SYNC_ALPN, proto.clone())
        .spawn();

    println!("Running as {}", router.endpoint().node_id());

    if let Some(addr) = args.sync_with {
        tokio::spawn({
            let proto = proto.clone();
            async move { proto.sync_with(addr).await }
        });

        proto
            .repo()
            .when_connected(PeerId::from_string(addr.to_string()))
            .await?;
        println!("Connected to {addr}");
    }

    let mut document_id = String::new();

    match args.command {
        Commands::Create { key, value } => {
            let mut doc = Automerge::new();
            doc.transact(|tx| tx.put(automerge::ROOT, key, value))
                .map_err(debug_err)?;
            let doc = proto.repo().create(doc).await?;
            println!("Created document {}", doc.document_id());
            document_id = doc.document_id().to_string();
        }
        Commands::Upsert { doc, key, value } => {
            document_id = doc.clone();
            let doc_id = DocumentId::from_str(&doc).context("Couldn't parse document ID")?;
            let doc = proto
                .repo()
                .find(doc_id)
                .await?
                .context("Couldn't find document with this ID")?;
            doc.with_document(|doc| doc.transact(|tx| tx.put(automerge::ROOT, key, value)))
                .map_err(debug_err)?;
            println!("Updated document");
        }
        Commands::Delete { doc, key } => {
            document_id = doc.clone();
            let doc_id = DocumentId::from_str(&doc).context("Couldn't parse document ID")?;
            let doc = proto
                .repo()
                .find(doc_id)
                .await?
                .context("Couldn't find document with this ID")?;
            doc.with_document(|doc| doc.transact(|tx| tx.delete(automerge::ROOT, key)))
                .map_err(debug_err)?;
            println!("Key deleted!");
        }
        Commands::Print { doc } => {
            document_id = doc.clone();
            let doc_id = DocumentId::from_str(&doc).context("Couldn't parse document ID")?;
            let doc = proto
                .repo()
                .find(doc_id)
                .await?
                .context("Couldn't find document with this ID")?;
            doc.with_document(|doc| {
                for key in doc.keys(automerge::ROOT) {
                    let (value, _) = doc.get(automerge::ROOT, &key)?.expect("missing value");
                    println!("{key}={value}");
                }
                anyhow::Ok(())
            })?;
        }
        Commands::Subscribe { doc } => {
            document_id = doc.clone();
            let doc_id = DocumentId::from_str(&doc).context("Couldn't parse document ID")?;
            let doc = proto
                .repo()
                .find(doc_id.clone())
                .await?
                .context("Couldn't find document with this ID")?;

            println!("Subscribing to document {} for changes...", doc_id);

            // Print initial state
            println!("Initial document state:");
            doc.with_document(|doc| {
                for key in doc.keys(automerge::ROOT) {
                    let (value, _) = doc.get(automerge::ROOT, &key)?.expect("missing value");
                    println!("  {key}={value}");
                }
                anyhow::Ok(())
            })?;

            // Set up polling for changes (no push available yet)
            tokio::spawn(async move {
                // Track the last known heads to detect changes
                let mut last_heads = doc.with_document(|doc| doc.get_heads());
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

                    let current_heads = doc.with_document(|doc| doc.get_heads());
                    if current_heads == last_heads {
                        continue;
                    }

                    last_heads = current_heads;

                    println!("Document changed! New state:");

                    // When changes are detected, print the updated document contents...
                    if let Err(e) = doc.with_document(|current_doc| {
                        for key in current_doc.keys(automerge::ROOT) {
                            let (value, _) = current_doc
                                .get(automerge::ROOT, &key)?
                                .expect("missing value");
                            println!("  {key}={value}");
                        }
                        anyhow::Ok(())
                    }) {
                        eprintln!("Error reading document content: {e}");
                    }
                }
            });
        }
        Commands::Host => {
            println!("Hosting existing documents...");
            println!("Repository is now hosted and ready for sync operations.");
            println!("Other nodes can connect to sync with this repository.");
        }
    }

    println!("Start TestApp");

    TestApp::<CullingExample>::new(
        winit::dpi::LogicalSize::new(800, 600),
        Arc::new(std::sync::Mutex::new(router)),
        Arc::new(tokio::sync::Mutex::new(proto)),
        document_id,
    )
    .run();

    Ok(())
}

fn debug_err(e: impl std::fmt::Debug) -> anyhow::Error {
    anyhow::anyhow!("{e:?}")
}

use std::collections::{HashMap, HashSet};

use craballoc::prelude::{GpuArray, Hybrid};
use renderling::{
    atlas::AtlasImage,
    camera::Camera,
    light::{AnalyticalLight, DirectionalLightDescriptor},
    math::{Mat4, UVec2, Vec2, Vec3, Vec4},
    skybox::Skybox,
    stage::{Animator, GltfDocument, Renderlet, Stage, Vertex},
    ui::{FontArc, Section, Text, Ui, UiPath, UiText},
};

use camera::{CameraControl, TurntableCameraController, WasdMouseCameraController};

pub mod time;
use time::FPSCounter;

const FONT_BYTES: &[u8] = include_bytes!("fonts/Recursive Mn Lnr St Med Nerd Font Complete.ttf");

const DARK_BLUE_BG_COLOR: Vec4 = Vec4::new(
    0x30 as f32 / 255.0,
    0x35 as f32 / 255.0,
    0x42 as f32 / 255.0,
    1.0,
);

pub enum SupportedFileType {
    Gltf,
    Hdr,
}

pub fn is_file_supported(file: impl AsRef<std::path::Path>) -> Option<SupportedFileType> {
    let ext = file.as_ref().extension()?;
    Some(match ext.to_str()? {
        "hdr" => SupportedFileType::Hdr,
        _ => SupportedFileType::Gltf,
    })
}

#[cfg(not(target_arch = "wasm32"))]
lazy_static::lazy_static! {
    static ref START: std::time::Instant = std::time::Instant::now();
}

fn now() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        let doc = web_sys::window().unwrap();
        let perf = doc.performance().unwrap();
        perf.now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let now = std::time::Instant::now();
        let duration = now.duration_since(*START);
        duration.as_secs_f64()
    }
}

struct AppUi {
    ui: Ui,
    fps_text: UiText,
    fps_counter: FPSCounter,
    fps_background: UiPath,
    last_fps_display: f64,
}

impl AppUi {
    fn make_fps_widget(fps_counter: &FPSCounter, ui: &Ui) -> (UiText, UiPath) {
        let translation = Vec2::new(2.0, 2.0);
        let text = format!("{}fps", fps_counter.current_fps_string());
        let fps_text = ui
            .new_text()
            .with_color(Vec3::ZERO.extend(1.0))
            .with_section(Section::new().add_text(Text::new(&text).with_scale(32.0)))
            .build();
        fps_text.transform.set_translation(translation);
        let background = ui
            .new_path()
            .with_fill_color(Vec4::ONE)
            .with_rectangle(fps_text.bounds.0, fps_text.bounds.1)
            .fill();
        background.transform.set_translation(translation);
        background.transform.set_z(-0.9);
        (fps_text, background)
    }

    fn tick(&mut self) {
        self.fps_counter.next_frame();
        let now = now();
        if now - self.last_fps_display >= 1.0 {
            let (fps_text, background) = Self::make_fps_widget(&self.fps_counter, &self.ui);
            self.fps_text = fps_text;
            self.fps_background = background;
            self.last_fps_display = now;
        }
    }
}

#[allow(dead_code)]
pub struct DefaultModel {
    vertices: GpuArray<Vertex>,
    renderlet: Hybrid<Renderlet>,
}

pub enum Model {
    Gltf(Box<GltfDocument>),
    Default(DefaultModel),
    None,
}

pub struct App {
    last_frame_instant: f64,
    skybox_image_bytes: Option<Vec<u8>>,
    loads: Arc<std::sync::Mutex<HashMap<std::path::PathBuf, Vec<u8>>>>,
    pub stage: Stage,
    camera: Hybrid<Camera>,
    _lighting: AnalyticalLight,
    model: Model,
    animators: Option<Vec<Animator>>,
    animations_conflict: bool,
    pub camera_controller: Box<dyn CameraController + 'static>,
    ui: AppUi,
}

impl App {
    pub fn new(ctx: &Context, camera_control: CameraControl) -> Self {
        let stage = ctx
            .new_stage()
            .with_background_color(DARK_BLUE_BG_COLOR)
            .with_bloom_mix_strength(0.5)
            .with_bloom_filter_radius(4.0)
            .with_msaa_sample_count(4);
        let camera = stage.new_camera(Camera::default());
        let directional_light = DirectionalLightDescriptor {
            direction: Vec3::NEG_Y,
            color: renderling::math::hex_to_vec4(0xFDFBD3FF),
            intensity: 10.0,
        };
        let sunlight_bundle = stage.new_analytical_light(directional_light);

        stage
            .set_atlas_size(wgpu::Extent3d {
                width: 2048,
                height: 2048,
                depth_or_array_layers: 32,
            })
            .unwrap();

        let ui = Ui::new(ctx).with_background_color(Vec4::ZERO);
        let _ = ui.add_font(FontArc::try_from_slice(FONT_BYTES).unwrap());
        let fps_counter = FPSCounter::default();
        let (fps_text, fps_background) = AppUi::make_fps_widget(&fps_counter, &ui);

        Self {
            ui: AppUi {
                ui,
                fps_text,
                fps_counter,
                fps_background,
                last_fps_display: now(),
            },
            stage,
            camera,
            _lighting: sunlight_bundle,
            model: Model::None,
            animators: None,
            animations_conflict: false,

            skybox_image_bytes: None,
            loads: Arc::new(std::sync::Mutex::new(HashMap::default())),
            last_frame_instant: now(),

            camera_controller: match camera_control {
                CameraControl::Turntable => Box::new(TurntableCameraController::default()),
                CameraControl::WasdMouse => Box::new(WasdMouseCameraController::default()),
            },
        }
    }

    pub fn tick(&mut self) {
        self.camera_controller.tick();
        self.tick_loads();
        self.update_view();
        self.animate();
        self.ui.tick();
    }

    pub fn render(&self, ctx: &Context) {
        let frame = ctx.get_next_frame().unwrap();
        self.stage.render(&frame.view());
        self.ui.ui.render(&frame.view());
        frame.present();
    }

    pub fn update_view(&mut self) {
        self.camera_controller
            .update_camera(self.stage.get_size(), &self.camera);
    }

    fn load_hdr_skybox(&mut self, bytes: Vec<u8>) {
        let img = AtlasImage::from_hdr_bytes(&bytes).unwrap();
        let skybox = Skybox::new(self.stage.runtime(), img);
        self.skybox_image_bytes = Some(bytes);
        self.stage.set_skybox(skybox);
    }

    pub fn load_default_model(&mut self) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        self.last_frame_instant = now();
        let (vertices, renderlet) = self
            .stage
            .builder()
            .with_vertices(renderling::math::unit_cube().into_iter().map(|(p, n)| {
                let p = p * 2.0;
                min = min.min(p);
                max = max.max(p);
                Vertex::default()
                    .with_position(p)
                    .with_normal(n)
                    .with_color(Vec4::new(1.0, 0.0, 0.0, 1.0))
            }))
            .with_bounds({
                log::info!("default model bounds: {min} {max}");
                BoundingSphere::from((min, max))
            })
            .build();

        self.model = Model::Default(DefaultModel {
            vertices: vertices.into_gpu_only(),
            renderlet,
        });
        self.camera_controller.reset(Aabb::new(min, max));
        self.camera_controller
            .update_camera(self.stage.get_size(), &self.camera);
    }

    fn load_gltf_model(&mut self, path: impl AsRef<std::path::Path>, bytes: &[u8]) {
        log::info!("loading gltf");
        self.camera_controller
            .reset(Aabb::new(Vec3::NEG_ONE, Vec3::ONE));
        self.stage.clear_images().unwrap();
        self.model = Model::None;
        let doc = match self.stage.load_gltf_document_from_bytes(bytes) {
            Err(e) => {
                log::error!("gltf loading error: {e}");
                if cfg!(not(target_arch = "wasm32")) {
                    log::info!("attempting to load by filesystem");
                    match self.stage.load_gltf_document_from_path(path) {
                        Ok(doc) => doc,
                        Err(e) => {
                            log::error!("gltf loading error: {e}");
                            return;
                        }
                    }
                } else {
                    return;
                }
            }
            Ok(doc) => doc,
        };

        // find the bounding box of the model so we can display it correctly
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        let scene = doc.default_scene.unwrap_or(0);
        log::info!("Displaying scene {scene}");
        fn get_children(doc: &GltfDocument, n: usize) -> Vec<usize> {
            let mut children = vec![];
            if let Some(parent) = doc.nodes.get(n) {
                children.extend(parent.children.iter().copied());
                let descendants = parent
                    .children
                    .iter()
                    .copied()
                    .flat_map(|n| get_children(doc, n));
                children.extend(descendants);
            }
            children
        }

        let nodes = doc.nodes_in_scene(scene).flat_map(|n| {
            let mut all_nodes = vec![n];
            for child_index in get_children(&doc, n.index) {
                if let Some(child_node) = doc.nodes.get(child_index) {
                    all_nodes.push(child_node);
                }
            }
            all_nodes
        });
        log::trace!("  nodes:");
        for node in nodes {
            let tfrm = Mat4::from(node.global_transform());
            if let Some(mesh_index) = node.mesh {
                // UNWRAP: safe because we know the node exists
                for primitive in doc.meshes.get(mesh_index).unwrap().primitives.iter() {
                    let bbmin = tfrm.transform_point3(primitive.bounding_box.0);
                    let bbmax = tfrm.transform_point3(primitive.bounding_box.1);
                    min = min.min(bbmin);
                    max = max.max(bbmax);
                }
            }
        }

        log::trace!("Bounding box: {min} {max}");
        let bounding_box = Aabb::new(min, max);
        self.camera_controller.reset(bounding_box);
        self.camera_controller
            .update_camera(self.stage.get_size(), &self.camera);

        self.last_frame_instant = now();

        if doc.animations.is_empty() {
            log::trace!("  animations: none");
        } else {
            log::trace!("  animations:");
        }
        let mut animated_nodes = HashSet::default();
        let mut has_conflicting_animations = false;
        self.animators = Some(
            doc.animations
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let target_nodes = a.target_node_indices().collect::<HashSet<_>>();
                    has_conflicting_animations =
                        has_conflicting_animations || !animated_nodes.is_disjoint(&target_nodes);
                    animated_nodes.extend(target_nodes);

                    log::trace!("    {i} {:?} {}s", a.name, a.length_in_seconds());
                    Animator::new(doc.nodes.iter(), a.clone())
                })
                .collect(),
        );
        if has_conflicting_animations {
            log::trace!("  and some animations conflict");
        }
        self.animations_conflict = has_conflicting_animations;

        self.model = Model::Gltf(Box::new(doc));
    }

    pub fn tick_loads(&mut self) {
        let loaded = std::mem::take(&mut *self.loads.lock().unwrap());
        for (path, bytes) in loaded.into_iter() {
            log::info!("loaded {}bytes from {}", bytes.len(), path.display());
            match is_file_supported(&path) {
                Some(SupportedFileType::Gltf) => self.load_gltf_model(path, &bytes),
                Some(SupportedFileType::Hdr) => self.load_hdr_skybox(bytes),
                None => {}
            }
        }
    }

    /// Queues a load operation.
    pub fn load(&mut self, path: &str) {
        let path = std::path::PathBuf::from(path);
        let loads = self.loads.clone();

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let path_str = format!("{}", path.display());
                let bytes = loading_bytes::load(&path_str).await.unwrap();
                let mut loads = loads.lock().unwrap();
                loads.insert(path, bytes);
                log::debug!("loaded {path_str}");
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = std::thread::spawn(move || {
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|e| panic!("could not load '{}': {e}", path.display()));
                let mut loads = loads.lock().unwrap();
                loads.insert(path, bytes);
            });
        }
    }

    pub fn set_size(&mut self, size: UVec2) {
        self.stage.set_size(size);
    }

    pub fn animate(&mut self) {
        let now = now();
        let dt_seconds = now - self.last_frame_instant;
        self.last_frame_instant = now;
        self.camera_controller.tick();
        if let Some(animators) = self.animators.as_mut() {
            if self.animations_conflict {
                if let Some(animator) = animators.first_mut() {
                    animator.progress(dt_seconds as f32).unwrap();
                }
            } else {
                for animator in animators.iter_mut() {
                    animator.progress(dt_seconds as f32).unwrap();
                }
            }
        }
    }
}

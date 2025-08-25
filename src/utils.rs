//! Example app utilities.

use std::{
    any::Any,
    str::FromStr,
    sync::{Arc, Mutex},
};

use anyhow::Context as anyContext;
use automerge::ReadDoc;
use n0_future::future::block_on;
use renderling::Context;
use renderling_crdt::IrohRepo;
use winit::{monitor::MonitorHandle, platform::windows::WindowAttributesExtWindows};
#[derive(Default)]
pub struct BagOfResources(Vec<Box<dyn Any>>);

impl BagOfResources {
    pub fn push(&mut self, rez: impl Any) {
        self.0.push(Box::new(rez));
    }

    pub fn drain(&mut self) {
        let _ = self.0.drain(..);
    }
}

pub trait TestAppHandler: winit::application::ApplicationHandler {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window: Arc<winit::window::Window>,
        ctx: &Context,
    ) -> Self;
    fn render(&mut self, ctx: &Context);
}

pub(crate) struct InnerTestApp<T> {
    ctx: Context,
    app: T,
}

pub struct TestApp<T> {
    size: winit::dpi::Size,
    inner: Option<InnerTestApp<T>>,
    router: Arc<Mutex<iroh::protocol::Router>>,
    iroh_repo_protocol: Arc<Mutex<IrohRepo>>,
    document_id: String,
}

impl<T: TestAppHandler> winit::application::ApplicationHandler for TestApp<T> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // I have my editor on the high monitor, and I read on the low one,
        // so set the position of the opened window to the lowest monitor so
        // it doesn't constantly switch away from my editor...
        let maybe_lowest_monitor =
            event_loop
                .available_monitors()
                .fold(None::<MonitorHandle>, |mut acc, monitor| {
                    let position = monitor.position();
                    let name = monitor.name().unwrap_or("unknown".to_owned());
                    log::info!("found monitor: {name} {position:?}");
                    if let Some(current_monitor) = acc.as_mut() {
                        // greater than here because y increases downward
                        if monitor.position().y > current_monitor.position().y {
                            acc = Some(monitor);
                        }
                    } else {
                        acc = Some(monitor);
                    }
                    acc
                });

        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::WindowAttributes::default()
                        .with_inner_size(self.size)
                        .with_title("test app")
                        .with_fullscreen(
                            maybe_lowest_monitor
                                .map(|m| winit::window::Fullscreen::Borderless(Some(m))),
                        )
                        // make this false so tokio can run
                        .with_drag_and_drop(false),
                )
                .unwrap(),
        );
        let ctx = Context::try_from_window(None, window.clone()).unwrap();
        let mut app = T::new(event_loop, window, &ctx);
        app.resumed(event_loop);
        self.inner = Some(InnerTestApp { app, ctx });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(inner) = self.inner.as_mut() {
            match event {
                // winit::event::WindowEvent::RedrawRequested => {
                //     inner.ctx.get_device().poll(wgpu::Maintain::Wait);
                // }
                winit::event::WindowEvent::CloseRequested
                | winit::event::WindowEvent::KeyboardInput {
                    event:
                        winit::event::KeyEvent {
                            logical_key:
                                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                            ..
                        },
                    ..
                } => {
                    block_on(async {
                        let doc_id = samod::DocumentId::from_str(&self.document_id).unwrap();
                        let proto = self.iroh_repo_protocol.lock().unwrap();
                        let doc = proto
                            .repo()
                            .find(doc_id)
                            .await
                            .unwrap()
                            .context("Couldn't find document with this ID")
                            .unwrap();
                        doc.with_document(|doc| {
                            for key in doc.keys(automerge::ROOT) {
                                let (value, _) = doc
                                    .get(automerge::ROOT, &key)
                                    .unwrap()
                                    .expect("missing value");
                                println!("{key}={value}");
                            }
                            anyhow::Ok(())
                        })
                        .unwrap();
                        self.router.try_lock().unwrap().shutdown().await.unwrap();
                    });
                    event_loop.exit();
                }
                _ => {
                    inner.app.window_event(event_loop, window_id, event);
                }
            }
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(inner) = self.inner.as_mut() {
            inner.app.device_event(event_loop, device_id, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(inner) = self.inner.as_mut() {
            inner.app.about_to_wait(event_loop);
            inner.app.render(&inner.ctx);
        }
    }
}

impl<T: TestAppHandler> TestApp<T> {
    pub fn new(
        size: impl Into<winit::dpi::Size>,
        router: Arc<Mutex<iroh::protocol::Router>>,
        iroh_repo_protocol: Arc<Mutex<IrohRepo>>,
        document_id: String,
    ) -> Self {
        TestApp {
            size: size.into(),
            inner: None,
            router,
            iroh_repo_protocol,
            document_id,
        }
    }

    pub fn run(&mut self) {
        let event_loop = winit::event_loop::EventLoop::new().unwrap();
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        event_loop.run_app(self).unwrap();
    }
}

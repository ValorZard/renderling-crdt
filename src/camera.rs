//! Camera control.
use std::str::FromStr;

use glam::{Mat4, Quat, UVec2, Vec2, Vec3};
use renderling::bvol::Aabb;
use renderling::prelude::*;
use winit::{event::KeyEvent, keyboard::Key};

const RADIUS_SCROLL_DAMPENING: f32 = 0.001;
const DX_DY_DRAG_DAMPENING: f32 = 0.01;

#[derive(Clone, Copy, Debug, Default)]
pub enum CameraControl {
    #[default]
    Turntable,
    WasdMouse,
}

impl FromStr for CameraControl {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "turntable" => Ok(CameraControl::Turntable),
            "wasdmouse" => Ok(CameraControl::WasdMouse),
            _ => Err("must be 'turntable' or 'wasdmouse'".to_owned()),
        }
    }
}

pub struct TurntableCameraController {
    /// look at
    pub center: Vec3,
    /// distance from the origin
    pub radius: f32,
    /// Determines the distance between the camera's near and far planes
    depth: f32,
    /// anglular position on a circle `radius` away from the origin on x,z
    pub phi: f32,
    /// angular distance from y axis
    pub theta: f32,
    /// is the left mouse button down
    left_mb_down: bool,
}

impl Default for TurntableCameraController {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 6.0,
            depth: 12.0,
            phi: 0.0,
            theta: std::f32::consts::FRAC_PI_4,
            left_mb_down: false,
        }
    }
}

impl CameraController for TurntableCameraController {
    fn tick(&mut self) {}

    fn reset(&mut self, bounds: Aabb) {
        log::debug!("resetting turntable bounds to {bounds:?}");
        let diagonal_length = bounds.diagonal_length();
        self.radius = diagonal_length * 1.25;
        self.depth = 2.0 * diagonal_length;
        self.center = bounds.center();
        self.left_mb_down = false;
    }

    fn update_camera(&self, UVec2 { x: w, y: h }: UVec2, current_camera: &Hybrid<Camera>) {
        let camera_position = Self::camera_position(self.radius, self.phi, self.theta);
        let znear = self.depth / 1000.0;
        let camera = Camera::new(
            Mat4::perspective_infinite_rh(std::f32::consts::FRAC_PI_4, w as f32 / h as f32, znear),
            Mat4::look_at_rh(camera_position, self.center, Vec3::Y),
        );
        debug_assert!(
            camera.view().is_finite(),
            "camera view is borked w:{w} h:{h} camera_position: {camera_position} center: {} \
             radius: {} phi: {} theta: {}",
            self.center,
            self.radius,
            self.phi,
            self.theta
        );
        if current_camera.get() != camera {
            current_camera.set(camera);
        }
    }

    fn mouse_scroll(&mut self, delta: f32) {
        self.zoom(delta);
    }

    fn mouse_moved(&mut self, _position: Vec2) {}

    fn mouse_motion(&mut self, delta: Vec2) {
        self.pan(delta);
    }

    fn mouse_button(&mut self, is_pressed: bool, is_left_button: bool) {
        self.left_mb_down = is_left_button && is_pressed;
    }

    fn key(&mut self, _event: KeyEvent) {}
}

impl TurntableCameraController {
    fn camera_position(radius: f32, phi: f32, theta: f32) -> Vec3 {
        // convert from spherical to cartesian
        let x = radius * theta.sin() * phi.cos();
        let y = radius * theta.sin() * phi.sin();
        let z = radius * theta.cos();
        // in renderling Y is up so switch the y and z axis
        Vec3::new(x, z, y)
    }

    fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - (delta * RADIUS_SCROLL_DAMPENING)).max(0.0);
    }

    fn pan(&mut self, delta: Vec2) {
        if self.left_mb_down {
            self.phi += delta.x * DX_DY_DRAG_DAMPENING;

            let next_theta = self.theta - delta.y * DX_DY_DRAG_DAMPENING;
            self.theta = next_theta.clamp(0.0001, std::f32::consts::PI);
        }
    }
}

pub trait CameraController {
    fn reset(&mut self, bounds: Aabb);
    fn tick(&mut self);
    fn update_camera(&self, size: UVec2, camera: &Hybrid<Camera>);
    fn mouse_scroll(&mut self, delta: f32);
    fn mouse_moved(&mut self, position: Vec2);
    fn mouse_motion(&mut self, delta: Vec2);
    fn mouse_button(&mut self, is_pressed: bool, is_left_button: bool);
    fn key(&mut self, event: KeyEvent);

    fn window_event(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, up) => up,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };

                self.mouse_scroll(delta);
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.mouse_moved(Vec2::new(position.x as f32, position.y as f32));
            }
            winit::event::WindowEvent::MouseInput { state, button, .. } => {
                let is_pressed = matches!(state, winit::event::ElementState::Pressed);
                let is_left_button = matches!(button, winit::event::MouseButton::Left);
                self.mouse_button(is_pressed, is_left_button);
            }
            winit::event::WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.key(event);
            }

            _ => {}
        }
    }

    fn device_event(&mut self, event: winit::event::DeviceEvent) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.mouse_motion(Vec2::new(delta.0 as f32, delta.1 as f32))
        }
    }
}

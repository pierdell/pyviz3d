import pygame as pg

from viz3d.opengl.gl_algebra import *
from enum import Enum


class FPVCamera:
    def __init__(self, height: int, width: int):
        # Surface parameter
        self.height = height
        self.width = width

        # Absolute camera pose (typically set when tracking an object)
        self.absolute_world_to_cam = np.eye(4, dtype=np.float32)

        # Relative camera pose
        self.orientation = np.eye(3, dtype=np.float32)
        self.position = np.array([0, 0, 10], dtype=np.float32)

        # Clip space
        self.z_far = 1000
        self.z_near = 0.1
        self.fov = 45.0

        # Intrinsics
        self.projection_matrix = perspective_matrix(self.fov, self.z_near, self.z_far, aspect_ratio=height / width)

        # Motion parameters
        self.direction_array = np.zeros((3,), dtype=np.float32)
        self.rotation_array = np.zeros((3,), dtype=np.float32)

        # Flag dictating whether the pose of the camera should be modified
        self.mode = self.States.NO_MODE

        # User Input Sensitivity parameters
        self.translation_speed = 10.0
        self.angular_speed = 180.0
        self.radius_orientation_ratio = 0.25

        # Allow an external agent to update the pose
        self._allow_external_pose = True

    class States(Enum):
        NO_MODE = 0,
        FPV_MODE = 1,
        MOVE_MODEL_MODE = 2,
        ORIENTATION_MODE = 3

    def set_camera_pose(self, absolute_world_to_cam: np.ndarray):
        if self._allow_external_pose:
            assert_debug(absolute_world_to_cam.dtype == gl_float_np_dtype())
            check_sizes(absolute_world_to_cam, [4, 4])
            self.absolute_world_to_cam = absolute_world_to_cam

    def get_projection_matrix(self):
        # Update the position
        return self.projection_matrix

    def front_vector(self):
        return self.orientation[:3, 2]

    def right_vector(self):
        return self.orientation[:3, 0]

    def up_vector(self):
        return self.orientation[:3, 1]

    def update_physics(self, tick_sec: float):
        delta_tr = tick_sec * self.translation_speed
        self.position -= delta_tr * self.orientation @ self.direction_array

        # Update the orientation
        delta_rot = tick_sec * self.angular_speed

        # Update rotation along up axis
        delta_theta = rad(delta_rot * self.rotation_array[0])
        c_theta = np.cos(delta_theta)
        s_theta = np.sin(delta_theta)
        rotation_x = farray([[c_theta, 0.0, -s_theta], [0.0, 1.0, 0.0], [s_theta, 0.0, c_theta]])

        # Apply rotation along right axis
        delta_phi = rad(delta_rot * self.rotation_array[1])
        c_phi = np.cos(delta_phi)
        s_phi = np.sin(delta_phi)
        rotation_y = farray([[1.0, 0.0, 0.0], [0.0, c_phi, s_phi], [0.0, -s_phi, c_phi]])

        # Apply rotation along the front axis
        delta_psi = rad(delta_rot * self.rotation_array[2])
        c_psi = np.cos(delta_psi)
        s_psi = np.sin(delta_psi)
        rotation_z = farray([[c_psi, s_psi, 0.0], [-s_psi, c_psi, 0.0], [0.0, 0.0, 1.0]])

        self.orientation = self.orientation @ rotation_x @ rotation_y @ rotation_z
        # Normalize the orientation ?

    def world_to_camera_mat(self):
        pose = farray(np.eye(4, dtype=np.float32))
        pose[:3, :3] = self.orientation.T
        pose[:3, 3] = - self.orientation.T @ self.position
        pose = pose.dot(np.linalg.inv(self.absolute_world_to_cam))

        return pose

    def switch_mode(self, mode: States):
        if mode == self.mode:
            self.mode = self.States.NO_MODE
        else:
            self.mode = mode
        self.rotation_array = np.zeros_like(self.rotation_array)
        self.direction_array = np.zeros_like(self.direction_array)

    def process_key_event(self, event):
        if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
            self.switch_mode(self.States.FPV_MODE)

        if self.mode == self.States.FPV_MODE:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    # Reset the absolute_pose to the default position
                    self.absolute_world_to_cam = np.eye(4, dtype=np.float32)
                if event.key == pg.K_o:
                    # Reset the relative pose to the default position
                    self.orientation = np.eye(3, dtype=np.float32)
                    self.position = np.array([0, 0, 10], dtype=np.float32)
                if event.key == pg.K_LSHIFT:
                    self.translation_speed = 100
                if event.key == pg.K_z:
                    self.direction_array[2] = 1.0
                if event.key == pg.K_s:
                    self.direction_array[2] = -1.0
                if event.key == pg.K_r:
                    self.direction_array[1] = -1.0
                if event.key == pg.K_f:
                    self.direction_array[1] = 1.0
                if event.key == pg.K_q:
                    self.direction_array[0] = 1.0
                if event.key == pg.K_d:
                    self.direction_array[0] = -1.0
                if event.key == pg.K_e:
                    self.rotation_array[2] = 0.5
                if event.key == pg.K_a:
                    self.rotation_array[2] = -0.5
                if event.key == pg.K_p:
                    self._allow_external_pose = not self._allow_external_pose
            elif event.type == pg.KEYUP:
                if event.key == pg.K_LSHIFT:
                    self.translation_speed = 10.0
                if event.key == pg.K_z and self.direction_array[2] == 1.0:
                    self.direction_array[2] = 0.0
                elif event.key == pg.K_s and self.direction_array[2] == -1.0:
                    self.direction_array[2] = 0.0
                if event.key == pg.K_r and self.direction_array[1] == -1.0:
                    self.direction_array[1] = 0.0
                if event.key == pg.K_f and self.direction_array[1] == 1.0:
                    self.direction_array[1] = 0.0
                if event.key == pg.K_q and self.direction_array[0] == 1.0:
                    self.direction_array[0] = 0.0
                elif event.key == pg.K_d and self.direction_array[0] == -1.0:
                    self.direction_array[0] = 0.0
                if event.key == pg.K_e and self.rotation_array[2] > 0.0:
                    self.rotation_array[2] = 0.0
                elif event.key == pg.K_a and self.rotation_array[2] < 0.0:
                    self.rotation_array[2] = 0.0

    def process_mouse_event(self, event):
        if event.type in [pg.MOUSEBUTTONDOWN, pg.MOUSEBUTTONUP]:
            if event.button == pg.BUTTON_LEFT:
                self.switch_mode(self.States.MOVE_MODEL_MODE)
            if event.button == pg.BUTTON_RIGHT:
                self.switch_mode(self.States.ORIENTATION_MODE)

        if self.mode != self.States.NO_MODE and event.type == pg.MOUSEMOTION:
            x, y = event.pos
            x_range = x - (self.width // 2)
            radius_ratio = self.radius_orientation_ratio
            if self.mode == self.States.ORIENTATION_MODE:
                radius_ratio /= 2.0
            diff_x = int(radius_ratio * self.width)

            y_range = y - (self.height // 2)
            diff_y = int(radius_ratio * self.height)

            if self.mode in [self.States.FPV_MODE, self.States.ORIENTATION_MODE]:
                if abs(x_range) > diff_x:
                    value = np.sign(x_range) * (abs(x_range) - diff_x) / (self.width // 2 - diff_x)
                    self.rotation_array[0] = value
                else:
                    self.rotation_array[0] = 0.0

                if abs(y_range) > diff_y:
                    value = np.sign(y_range) * (abs(y_range) - diff_y) / (self.height // 2 - diff_y)
                    self.rotation_array[1] = value
                else:
                    self.rotation_array[1] = 0.0
            elif self.mode == self.States.MOVE_MODEL_MODE:
                self.direction_array[0] = -x_range / self.width
                self.direction_array[1] = y_range / self.height

    def process_event(self, event):
        if event.type == pg.KEYDOWN or event.type == pg.KEYUP:
            self.process_key_event(event)
        if event.type == pg.MOUSEMOTION or event.type == pg.MOUSEBUTTONDOWN or event.type == pg.MOUSEBUTTONUP:
            self.process_mouse_event(event)

import time
from multiprocessing.context import Process
from threading import Thread

from opengl.camera import *
from opengl.model import *
from opengl.camera_shader import CameraAlbedoShader
from pygame.locals import *


class ExplorationEngine:
    """
    An Exploration Engine is an Rendering engine built with a user controlled camera
    Which allow him to move into a scene
    """

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, height: int = 720, width: int = 1080, point_size: int = 3, with_edl: bool = True):
        super().__init__()

        self.height = height
        self.width = width
        self.camera = FPVCamera(self.height, self.width)

        self.background_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.color_shader = CameraAlbedoShader()
        self.edl_shader = None
        self.is_initialized = False

        self.key_to_callback = {}
        self.pc_models = dict()
        self.new_models = dict()
        self.point_size = point_size

        # Messages (typically modified by another thread, than the main thread)
        self.models_to_update = set()
        self._to_close: bool = False
        self._new_camera_pose: Optional[np.ndarray] = None

    # ------------------------------------------------------------------------------------------------------------------
    # PROPERTIES

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, _background_color):
        self._background_color = _background_color

    # ------------------------------------------------------------------------------------------------------------------

    def process_user_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            # Process events
            self.camera.process_event(event)

            if event.type == pg.KEYDOWN and event.key in self.key_to_callback:
                callback = self.key_to_callback[event.key]
                callback(event=event)

    def initialize(self, **kwargs):
        """
        Initializes the OpenGL context and the objects living on GPU (Models, Buffers and Shaders)
        """
        # Initialize the models
        glEnable(GL_DEPTH_TEST)

        # Initialize Shaders
        self.color_shader.init_shader_program(**kwargs)

        self._initialize_models()
        self.is_initialized = True
        self._to_close = False

    def init_window(self):
        pg.init()
        window_size = (self.width, self.height)
        pg.display.set_mode(window_size, DOUBLEBUF | OPENGL)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        self.initialize()

    def _initialize_models(self):
        keys = list(self.new_models.keys())
        for model_id in keys:
            model = self.new_models.pop(model_id)
            model.init_model()
            assert_debug(model_id not in self.pc_models)
            self.pc_models[model_id] = model

    # ------------------------------------------------------------------------------------------------------------------
    def add_model(self, model_id: int, pc_model: PointCloudModel):
        self.delete_model(model_id)
        self.new_models[model_id] = pc_model

    def delete_model(self, model_id):
        if model_id in self.pc_models:
            model = self.pc_models.pop(model_id)
            model.delete()

    def update_model(self, model_id):
        if model_id in self.pc_models:
            self.models_to_update.add(model_id)

    def update_camera(self, camera_pose: np.ndarray):
        self._new_camera_pose = camera_pose

    # ------------------------------------------------------------------------------------------------------------------

    def draw(self):
        proj = self.camera.get_projection_matrix()
        camera_pose = self.camera.world_to_camera_mat()

        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPointSize(self.point_size)
        for model_id, model in self.pc_models.items():
            self.color_shader.draw_model(model, world_to_cam=camera_pose, projection=proj)

        # Swap Buffers
        pg.display.flip()

    def force_close(self):
        """Sends a signal to close the main thread"""
        self._to_close = True

    # ------------------------------------------------------------------------------------------------------------------
    # MAIN LOOP

    def main_loop(self):
        self.init_window()

        last_time = None
        fps = 60.0
        s_per_frame = 1.0 / fps
        while True:

            # Close if the corresponding signal is activated
            if self._to_close:
                break

            current_time = time.perf_counter()

            # Initialize models that need initializing
            self._initialize_models()

            # Update models that need updating
            for i in range(len(self.models_to_update)):
                model_id = self.models_to_update.pop()
                if model_id in self.pc_models:
                    self.pc_models[model_id].update()

            # Update Camera Orientation if it was changed
            if self._new_camera_pose is not None:
                camera_pose = self._new_camera_pose
                self._new_camera_pose = None
                self.camera.set_camera_pose(camera_pose)

            # Process input
            self.process_user_input()

            # Update the physics of the camera
            lag = None
            if last_time is not None:
                lag = current_time - last_time
                self.camera.update_physics(lag)

            # Draw
            last_time = current_time
            self.draw()

            # Wait to update only at 60 fps maximum
            if lag is not None:
                elapsed = time.perf_counter() - last_time
                if elapsed < s_per_frame:
                    pg.time.wait(int((s_per_frame - elapsed) * 1000))

    # ------------------------------------------------------------------------------------------------------------------

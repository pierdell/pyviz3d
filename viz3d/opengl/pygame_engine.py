import time

from pygame.locals import *
import pygame as pg

from viz3d.opengl.camera import FPVCamera
from viz3d.opengl.model import *
from viz3d.opengl.camera_shader import CameraAlbedoShader
from viz3d.opengl.post_process_shader import ScreenShader


class ExplorationEngine:
    """
    An Exploration Engine is a Rendering engine built with a user controlled camera
    Which allow him to move into a scene

    Rendering is done in two passes :
        - A first pass renders the scene into using a simple CameraAlbedoShader (no lighting) into a texture
        - A second pass renders the texture into the screen (adding optionally some post-processing effects)
    """

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, height: int = 720, width: int = 1080,
                 edl_strength: float = 1000, with_edl: bool = True, edl_distance: float = 1.0,
                 num_fps: int = 40):
        super().__init__()

        self.height = height
        self.width = width
        self.camera = FPVCamera(self.height, self.width)
        self.num_fps = num_fps

        self.background_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.is_initialized = False

        # Shader for the first pass
        self.first_pass_shader = CameraAlbedoShader()

        # Framebuffer and textures used to render the first pass
        self.first_pass_framebuffer = None
        self.color_texture = None
        self.depth_texture = None

        # Shader for the post processing and rendering on the screen
        self.screen_shader = ScreenShader(with_edl=with_edl, edl_strength=edl_strength, edl_distance=edl_distance)
        self.screen_framebuffer = 0
        self.screen_model = ScreenModel()

        self.key_to_callback = {}
        self.models = dict()
        self.new_models = dict()

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
                self._to_close = True
            # Process events
            self.camera.process_event(event)

            if event.type == pg.KEYDOWN and event.key in self.key_to_callback:
                callback = self.key_to_callback[event.key]
                callback(event=event)

    def init_framebuffers(self, **kwargs):
        """Initialize the framebuffer for the first pass rendering"""
        self.first_pass_framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.first_pass_framebuffer)

        # Generate color texture for color attachment
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Attach the color to the framebuffer object
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)

        # Generate Depth texture
        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width, self.height,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        # Attach the texture to as the framebuffer's depth buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_texture, 0)

        # Verify that the Framebuffer is correct
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("[ERROR][OPEN_GL]The Framebuffer is not valid")

    def init_shaders(self, **kwargs):
        """Compile And Initialize the shaders"""
        self.first_pass_shader.init_shader_program(**kwargs)
        self.screen_shader.init_shader_program(z_far=self.camera.z_far, z_near=self.camera.z_near)

    def init_models(self, **kwargs):
        """
        Initializes the Models by creating the corresponding opengl buffer on the GPU and populating the data
        """
        # Initialize the models
        glEnable(GL_DEPTH_TEST)

        # Initialize Shaders
        self._initialize_models()

    def init_window(self):
        # Initialize PyGame's window (lets PyGame setup the main Framebuffer)
        pg.init()
        window_size = (self.width, self.height)
        pg.display.set_mode(window_size, DOUBLEBUF | OPENGL)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)

        # Initialize Different OpenGL component used to render the pointclouds on the pygame owned window
        self.init_framebuffers()
        self.init_shaders()
        self.init_models()

    def _initialize_models(self):
        # Initialize the screen model
        self.screen_model.init_model()

        keys = list(self.new_models.keys())
        for model_id in keys:
            model = self.new_models.pop(model_id)
            model.init_model()
            assert_debug(model_id not in self.models)
            self.models[model_id] = model

    # ------------------------------------------------------------------------------------------------------------------

    def _build_model(self, model_data: ModelData) -> Model:
        """Builds a model from model_data"""
        if isinstance(model_data, PointCloudModelData):
            return PointCloudModel(model_data)
        elif isinstance(model_data, CamerasModelData):
            return CamerasModel(model_data)
        else:
            raise NotImplementedError("")

    def add_model(self, model_id: int, model_data: ModelData):
        """Adds a model to set of models rendered by the engine"""
        self.delete_model(model_id)
        self.new_models[model_id] = self._build_model(model_data)

    def delete_model(self, model_id):
        if model_id in self.models:
            model = self.models.pop(model_id)
            model.delete()

    def update_model(self, model_id):
        if model_id in self.models:
            self.models_to_update.add(model_id)

    def update_camera(self, camera_pose: np.ndarray):
        check_sizes(camera_pose, [4, 4])
        self._new_camera_pose = camera_pose

    # ------------------------------------------------------------------------------------------------------------------
    def _draw_in_texture(self):
        # Bind the texture framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.first_pass_framebuffer)

        glEnable(GL_DEPTH_TEST)
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        proj = self.camera.get_projection_matrix()
        camera_pose = self.camera.world_to_camera_mat()

        # Draw each pointcloud model in the
        for model_id, model in self.models.items():
            self.first_pass_shader.draw_model(model, world_to_cam=camera_pose, projection=proj)

    def _draw_on_screen(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.screen_framebuffer)

        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)

        # Draw the screen
        self.screen_shader.draw_model(self.screen_model,
                                      color_texture=self.color_texture,
                                      depth_texture=self.depth_texture)

        # Swap Buffers
        pg.display.flip()

    def force_close(self):
        """Sends a signal to close the main thread"""
        self._to_close = True

    def draw(self):
        self._draw_in_texture()
        self._draw_on_screen()

    # ------------------------------------------------------------------------------------------------------------------
    # MAIN LOOP

    def main_loop(self):
        self.init_window()

        last_time = None
        fps = self.num_fps
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
                if model_id in self.models:
                    self.models[model_id].update()

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

        # Suppresses the Pygame context
        pg.display.quit()
        pg.quit()

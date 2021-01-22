import time

from opengl.camera import *
from opengl.model import *
from opengl.camera_shader import CameraAlbedoShader
from pygame.locals import *

from opengl.post_process_shader import ScreenShader


#
# import cv2
# # Debug : Read the color texture in a numpy array
#         data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
#         glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE, data)
#
#         cv2.imshow("color_texture", data)
#         cv2.waitKey(10)


class ExplorationEngine:
    """
    An Exploration Engine is an Rendering engine built with a user controlled camera
    Which allow him to move into a scene
    """

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, height: int = 720, width: int = 1080, point_size: int = 3):
        super().__init__()

        self.height = height
        self.width = width
        self.camera = FPVCamera(self.height, self.width)

        self.background_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.is_initialized = False

        # Shader for the first pass
        self.first_pass_shader = CameraAlbedoShader()

        # Framebuffer and textures used to render the first pass
        self.first_pass_framebuffer = None
        self.color_texture = None
        self.depth_texture = None

        # Shader for the post processing and rendering on the screen
        self.screen_shader = ScreenShader()
        self.screen_framebuffer = 0
        self.screen_model = ScreenModel()

        self.key_to_callback = {}
        self.pc_models = dict()
        self.models_to_update = set()
        self.point_size = point_size

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
        # Initialize the screen model
        self.screen_model.init_model()

        # Initialize the point cloud models
        for _, model in self.pc_models.items():
            model.init_model()

        self.is_initialized = True

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

    # ------------------------------------------------------------------------------------------------------------------
    def add_model(self, model_id: int, pc_model: PointCloudModel):
        self.delete_model(model_id)
        self.pc_models[model_id] = pc_model
        if self.is_initialized:
            pc_model.init_model()

    def delete_model(self, model_id):
        if model_id in self.pc_models:
            model = self.pc_models.pop(model_id)
            model.delete()

    def update_model(self, model_id):
        if model_id in self.pc_models:
            self.models_to_update.add(model_id)

    # ------------------------------------------------------------------------------------------------------------------
    def _draw_in_texture(self):
        # Bind the texture framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.first_pass_framebuffer)

        glEnable(GL_DEPTH_TEST)
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPointSize(self.point_size)

        proj = self.camera.get_projection_matrix()
        camera_pose = self.camera.world_to_camera_mat()

        # Draw each pointcloud model in the
        for model_id, model in self.pc_models.items():
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

        pg.display.flip()

    def draw(self):
        self._draw_in_texture()
        self._draw_on_screen()

    # ------------------------------------------------------------------------------------------------------------------
    # MAIN LOOP

    def main_loop(self):
        self.init_window()

        last_time = None
        fps = 60.0
        s_per_frame = 1.0 / fps
        while True:
            current_time = time.perf_counter()

            # Update models that need updating
            for i in range(len(self.models_to_update)):
                model_id = self.models_to_update.pop()
                if model_id in self.pc_models:
                    self.pc_models[model_id].update()

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

            if lag is not None:
                elapsed = time.perf_counter() - last_time
                if elapsed < s_per_frame:
                    pg.time.wait(int((s_per_frame - elapsed) * 1000))

    # ------------------------------------------------------------------------------------------------------------------

from abc import abstractmethod
from OpenGL.GL import *

from utils import assert_debug, check_sizes
from opengl.gl_algebra import *


# ----------------------------------------------------------------------------------------------------------------------
# Model
class Model:
    """
    A Model contains all the information to be drawn by the engine
    The Geometric Data (Mesh, Points, Lines),
    And the textures
    """

    @abstractmethod
    def init_model(self):
        """
        Allocates buffers (VBO, EBO, and VAO) on the GPU,
        And defines the layout of the Data (with VAO's vertex attributes)
        """
        # Draw specific Data
        raise NotImplementedError("")


# ----------------------------------------------------------------------------------------------------------------------
# PointCloudModel
class PointCloudModel(Model):

    def __init__(self,
                 pointcloud: np.ndarray,
                 color: Optional[np.ndarray] = None,
                 default_color: np.ndarray = farray([[0.0, 0.63, 1.0]]),
                 storage_mode: str = "dynamic"):
        super().__init__()

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        self.vao = None
        self.vertex_bo = None
        self.color_bo = None
        self.color = None
        self.pointcloud = None
        self.default_color = default_color
        self.set_pointcloud(pointcloud, color)

        assert_debug(storage_mode in ["dynamic", "static"])
        if storage_mode == "dynamic":
            self._storage_mode = GL_DYNAMIC_DRAW
        else:
            self._storage_mode = GL_STATIC_DRAW

    def num_points(self):
        if self.pointcloud is None:
            return None
        return self.pointcloud.shape[0]

    def init_model(self):
        self._is_initialized = True
        self._initialize_buffers()
        self.update()

    def _initialize_buffers(self):
        self.vao = glGenVertexArrays(1)
        self.vertex_bo = glGenBuffers(1)

    def update(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)

        data = np.concatenate([self.pointcloud, self.color], axis=1).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self._storage_mode)

        glVertexAttribPointer(self.points_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.points_location())

        glVertexAttribPointer(self.color_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(3 * type_size(np.float32)))
        glEnableVertexAttribArray(self.color_location())

    def delete(self):
        glDeleteBuffers(1, self.vertex_bo)
        glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    # OpenGL Shader variables and locations
    @staticmethod
    def points_location():
        return 0

    @staticmethod
    def color_location():
        return 1

    # ------------------------------------------------------------------------------------------------------------------
    # PROPERTIES
    def set_pointcloud(self,
                       pointcloud: np.ndarray,
                       color: Optional[np.ndarray] = None):
        check_sizes(pointcloud, [-1, 3])
        self.pointcloud = pointcloud
        if color is not None:
            check_sizes(color, [pointcloud.shape[0], 3])
        else:
            check_sizes(self.default_color, [1, 3])
            color = self.default_color.repeat(self.num_points(), axis=0)
        self.color = color


# ----------------------------------------------------------------------------------------------------------------------
# ScreenModel
class ScreenModel(Model):
    """A Simple Screen model which allows to display texture a screen"""

    def __init__(self):
        super().__init__()

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        self.vao = None
        self.vertex_bo = None

    def init_model(self):
        self._is_initialized = True
        self._initialize_buffers()

    @staticmethod
    def num_points():
        return 6

    def _initialize_buffers(self):
        self.vao = glGenVertexArrays(1)
        self.vertex_bo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)

        data = np.array([
            # First screen triangle
            -1.0, 1.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,

            # Second screen triangle
            -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0
        ]).astype(np.float32)

        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        glVertexAttribPointer(self.points_location(),
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              4 * type_size(np.float32),
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.points_location())

        glVertexAttribPointer(self.tex_coords_location(),
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              4 * type_size(np.float32),
                              ctypes.c_void_p(2 * type_size(np.float32)))
        glEnableVertexAttribArray(self.tex_coords_location())

    def delete(self):
        if self.vertex_bo is not None:
            glDeleteBuffers(1, self.vertex_bo)
        if self.vao is not None:
            glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    # OpenGL Shader variables and locations
    @staticmethod
    def points_location():
        return 0

    @staticmethod
    def tex_coords_location():
        return 1

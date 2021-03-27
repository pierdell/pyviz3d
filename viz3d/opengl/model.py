from abc import abstractmethod

import dataclasses
from OpenGL.GL import *
from dataclasses import dataclass, field

from viz3d.opengl.gl_algebra import *


# ----------------------------------------------------------------------------------------------------------------------
# Model Data
@dataclass
class ModelData:
    """An abstract class for ModelData which reunite all variables needed to define a specific model"""
    default_color: Optional[np.ndarray] = None  # The default color of the model
    instance_model_to_world: Optional[np.ndarray] = None  # The Optional transformation matrix from instance to world


# ----------------------------------------------------------------------------------------------------------------------
# Model
class Model:
    """
    A Model contains all the information to be drawn by the engine
    The Geometric Data (Mesh, Points, Lines),
    And the textures
    """

    def __init__(self, model_data: Optional[ModelData] = None):
        self.model_data = model_data
        self.vao = None
        self.instance_pose_bo = None

        self._last_num_instances = 0

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def model_data(self):
        """Returns the model data saved for this model"""
        return self._model_data

    @model_data.setter
    def model_data(self, _model_data: Optional[ModelData]):
        valid_model_data = self.verify_model_data(_model_data)
        if valid_model_data is not None:
            if valid_model_data.instance_model_to_world is None:
                valid_model_data.instance_model_to_world = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
            check_sizes(valid_model_data.instance_model_to_world, [-1, 4, 4])
        self._model_data = valid_model_data

    def num_instances(self):
        """
        Returns the number of instances of the current model

        More precisely, returns the number of instances corresponding to the last
        Model Data sent to the device via `update_model`
        """
        return self._last_num_instances

    # ------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def init_buffers(self):
        """
        Initializes
        """
        self.vao = glGenVertexArrays(1)
        self.instance_pose_bo = glGenBuffers(1)

    @abstractmethod
    def verify_model_data(self, model_data: Optional[ModelData]):
        """
        Verifies that `model_data` is valid for this instance,
        or that a valid ModelData can be built from it

        Returns a valid ModelData
        """
        assert_debug(model_data is not None)
        return model_data

    @abstractmethod
    def init_model(self):
        """
        Allocates buffers (VBO, EBO, and VAO) on the GPU,
        And defines the layout of the Data (with VAO's vertex attributes)
        """
        # Draw specific Data
        raise NotImplementedError("")

    @abstractmethod
    def update_model(self):
        """
        Updates the model buffers when its state has changed
        """
        # Bind the VAO for the data of the model
        glBindVertexArray(self.vao)

        # Send instance object poses to the GPU
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_pose_bo)
        glBufferData(GL_ARRAY_BUFFER, self.model_data.instance_model_to_world.nbytes,

                     # Need to transpose the array to build the matrix4d consistent with OpenGL layout
                     self.model_data.instance_model_to_world.transpose((0, 2, 1)),
                     GL_STATIC_DRAW)

        # Set the location of
        float_size = type_size(np.float32)
        vec4_size = 4 * float_size
        for i in range(4):
            location = self.instance_pose_location() + i
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location,
                                  4,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  vec4_size * 4,
                                  ctypes.c_void_p(vec4_size * i))
            glVertexAttribDivisor(location, 1)

        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Sets the following variables corresponding to the last updated pointcloud
        self._last_num_instances = self.model_data.instance_model_to_world.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def points_location():
        return 0

    @staticmethod
    def color_location():
        return 1

    @staticmethod
    def instance_pose_location():
        return 3


# ----------------------------------------------------------------------------------------------------------------------
# PointCloudModel
@dataclass
class PointCloudModelData(ModelData):
    """PointCloudModelData"""
    xyz: np.ndarray = field(default=np.zeros((1, 3), dtype=np.float32))  # The xyz pointcloud coordinates
    color: Optional[np.ndarray] = None  # The color of the pointcloud
    point_size: float = 1.0  # The size of points for rendering


class PointCloudModel(Model):
    """A Model To render simple colored PointClouds"""

    def __init__(self,
                 model_data: PointCloudModelData,
                 storage_mode: str = "dynamic"):
        super().__init__(model_data)

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        check_sizes(self.model_data.instance_model_to_world, [-1, 4, 4])
        self.vertex_bo = None
        self.color_bo = None
        self.instance_pose_bo = None

        assert_debug(storage_mode in ["dynamic", "static"])
        if storage_mode == "dynamic":
            self._storage_mode = GL_DYNAMIC_DRAW
        else:
            self._storage_mode = GL_STATIC_DRAW

        # ---------------------------------------------------------
        # Variables of points and instances saved in OpenGL buffers
        self._last_num_points = 0

    def num_points(self):
        return self._last_num_points

    def init_model(self):
        self._is_initialized = True
        self.init_buffers()
        self.update_model()

    def init_buffers(self):
        super().init_buffers()
        self.vertex_bo = glGenBuffers(1)

    def update_model(self):
        super().update_model()
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)

        data = np.concatenate([self.model_data.xyz, self.model_data.color], axis=1).astype(np.float32)
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
        self._last_num_points = self.model_data.xyz.shape[0]

    def delete(self):
        glDeleteBuffers(1, self.vertex_bo)
        glDeleteBuffers(1, self.instance_pose_bo)
        glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    def verify_model_data(self, _model_data: Optional[ModelData]):
        """Safe setter which verifies that the Point Cloud data is correct"""
        assert_debug(_model_data is not None)
        check_sizes(_model_data.xyz, [-1, 3])
        if _model_data.xyz.dtype != np.float32:
            _model_data = dataclasses.replace(_model_data, xyz=_model_data.xyz.astype(np.float32))

        if _model_data.color is None:
            default_color = _model_data.default_color
            if default_color is None:
                default_color = farray([[0.0, 0.63, 1.0]])

            color = default_color.repeat(_model_data.xyz.shape[0], axis=0)
            _model_data = dataclasses.replace(_model_data, color=color)

        check_sizes(_model_data.color, [_model_data.xyz.shape[0], 3])
        return _model_data


# ----------------------------------------------------------------------------------------------------------------------
# CAMERASMODEL

@dataclass
class CamerasModelData(ModelData):
    """CamerasModelData"""
    camera_size: float = 1.0  # The scale factor of the camera model
    width: float = 2  # Width of the line defining the camera model


# ----------------------------------------------------------------------------------------------------------------------
class CamerasModel(Model):
    """A Model to render a set of cameras displayed as a simple edge pyramid"""

    __vertex_data = farray([
        0.0, 0.0, -2.0,
        1.0, 0.6, 0.0,
        1.0, -0.6, 0.0,
        -1.0, -0.6, 0.0,
        -1.0, 0.6, 0.0,
    ]).reshape(-1, 3)

    __edge_indices = idarray([
        0, 1,
        0, 2,
        0, 3,
        0, 4,
        1, 2,
        2, 3,
        3, 4,
        4, 1
    ])

    def __init__(self,
                 model_data: CamerasModelData,
                 storage_mode: str = "dynamic"):
        super().__init__(model_data)

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        self.vertex_bo = None
        self.vertex_ebo = None
        self.color_bo = None

        assert_debug(storage_mode in ["dynamic", "static"])
        if storage_mode == "dynamic":
            self._storage_mode = GL_DYNAMIC_DRAW
        else:
            self._storage_mode = GL_STATIC_DRAW

    def num_elements(self):
        return self.__edge_indices.shape[0]

    def init_model(self):
        self._is_initialized = True
        self.init_buffers()
        self.update_model()

    def init_buffers(self):
        super().init_buffers()
        self.vertex_bo = glGenBuffers(1)
        self.vertex_ebo = glGenBuffers(1)

    def update_model(self):
        super().update_model()
        glBindVertexArray(self.vao)

        # Populate the Vertex Buffer Object with vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)
        data = np.concatenate([self.model_data.camera_size * self.__vertex_data,
                               self.model_data.default_color.repeat(self.__vertex_data.shape[0], axis=0)],
                              axis=1).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self._storage_mode)

        # Define the point attribute in the defined buffer
        glVertexAttribPointer(self.points_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.points_location())

        # Define the color attribute in the defined buffer
        glVertexAttribPointer(self.color_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(3 * type_size(np.float32)))
        glEnableVertexAttribArray(self.color_location())

        # Populate the Element Buffer Object with the edge indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vertex_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.__edge_indices.nbytes, self.__edge_indices, GL_STATIC_DRAW)

    def delete(self):
        glDeleteBuffers(1, self.vertex_bo)
        glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    def verify_model_data(self, _model_data: Optional[ModelData]):
        """Safe setter which verifies that the Point Cloud data is correct"""
        assert_debug(_model_data is not None and isinstance(_model_data, CamerasModelData))
        if _model_data.default_color is None:
            _model_data = dataclasses.replace(_model_data, default_color=np.zeros((1, 3), dtype=np.float32))
        return _model_data


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

    def update_model(self):
        pass

    def delete(self):
        if self.vertex_bo is not None:
            glDeleteBuffers(1, self.vertex_bo)
        if self.vao is not None:
            glDeleteVertexArrays(1, self.vao)

    def verify_model_data(self, model_data: Optional[ModelData]):
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # OpenGL Shader variables and locations
    @staticmethod
    def points_location():
        return 0

    @staticmethod
    def tex_coords_location():
        return 1


# ----------------------------------------------------------------------------------------------------------------------
# ELLIPSES

@dataclass
class EllipsesModelData(ModelData):
    """EllipsesModelData"""
    ellipses_size: float = 1.0  # The scale factor of the camera model
    covariances: Optional[np.ndarray] = None
    means: Optional[np.ndarray] = None

    # TODO colors


# ----------------------------------------------------------------------------------------------------------------------
class EllipsesModel(Model):
    """A Model to render a set of low poly ellipses (80 triangles, 320 points per ellipse)"""

    def __init__(self,
                 model_data: EllipsesModelData,
                 storage_mode: str = "dynamic"):
        super().__init__(model_data)

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        self.vertex_bo = None
        self.vertex_ebo = None
        self.color_bo = None

        assert_debug(storage_mode in ["dynamic", "static"])
        if storage_mode == "dynamic":
            self._storage_mode = GL_DYNAMIC_DRAW
        else:
            self._storage_mode = GL_STATIC_DRAW

        self._num_elements = 0

    def num_elements(self):
        return self._num_elements

    def init_model(self):
        self._is_initialized = True
        self.init_buffers()
        self.update_model()

    def init_buffers(self):
        super().init_buffers()
        self.vertex_bo = glGenBuffers(1)
        self.vertex_ebo = glGenBuffers(1)

    def update_model(self):
        super().update_model()

        # Build the ellipse model
        from viz3d.opengl.primitives.sphere import sphere_model_data
        vertex_and_normals, indices = sphere_model_data()

        num_ellipses = self.model_data.means.shape[0]
        num_points = vertex_and_normals.shape[0]
        num_triangles = indices.shape[0]

        ellipses_data = vertex_and_normals[:, :3].reshape(1, -1, 3).repeat(num_ellipses, axis=0)
        ellipses_indices = indices.reshape(1, -1, 3).repeat(num_ellipses, axis=0)
        for i in range(num_ellipses):
            ellipses_indices[i] += num_points * i

        # Compute the square root of covariance matrices
        covariances = self.model_data.covariances
        u, s, vt = np.linalg.svd(covariances)
        s = np.sqrt(s)
        diags = np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(num_ellipses, axis=0)
        for i in range(3):
            diags[:, i, i] = s[:, i]
        square_root_covs = u @ diags @ vt
        ellipses_data = np.einsum("nij,nmj->nmi", square_root_covs, ellipses_data)

        # Translate ellipse by their means
        ellipses_data += self.model_data.means.reshape(num_ellipses, 1, 3)

        vertex_data = ellipses_data.reshape(num_ellipses * num_points, 3)
        element_indices = ellipses_indices.reshape(num_ellipses * num_triangles * 3).astype(gl_index_np_dtype())
        self._num_elements = element_indices.shape[0]

        glBindVertexArray(self.vao)

        # Populate the Vertex Buffer Object with vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)
        data = np.concatenate([self.model_data.ellipses_size * vertex_data,
                               self.model_data.default_color.repeat(vertex_data.shape[0], axis=0)],
                              axis=1).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self._storage_mode)

        # Define the point attribute in the defined buffer
        glVertexAttribPointer(self.points_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.points_location())

        # Define the color attribute in the defined buffer
        glVertexAttribPointer(self.color_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(3 * type_size(np.float32)))
        glEnableVertexAttribArray(self.color_location())

        # Populate the Element Buffer Object with the edge indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vertex_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, element_indices.nbytes, element_indices, GL_STATIC_DRAW)

    def delete(self):
        glDeleteBuffers(1, self.vertex_bo)
        glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    def verify_model_data(self, _model_data: Optional[ModelData]):
        """Safe setter which verifies that the Point Cloud data is correct"""
        assert (_model_data is not None and isinstance(_model_data, EllipsesModelData))
        if _model_data.default_color is None:
            _model_data = dataclasses.replace(_model_data, default_color=np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        if _model_data.means is None:
            _model_data.means = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            _model_data.covariances = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        else:
            means = _model_data.means
            check_sizes(means, [-1, 3])
            num_ellipses = means.shape[0]

            if _model_data.covariances is None:
                _model_data.covariances = np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(num_ellipses, axis=0)
            else:
                covariances = _model_data.covariances
                check_sizes(covariances, [-1, 3, 3])
                diff = np.max(abs(covariances.transpose(0, 2, 1) - covariances))
                assert_debug(diff == 0.0, "Covariance matrices must be symmetric")

        # TODO COLOR

        return _model_data


# ----------------------------------------------------------------------------------------------------------------------
# LINES
@dataclass
class LinesModelData(ModelData):
    """EllipsesModelData"""
    line_width: float = 1.0  # The scale factor of the camera model
    line_data: np.ndarray = None
    line_color: np.ndarray = None


# ----------------------------------------------------------------------------------------------------------------------
class LinesModel(Model):
    """A Model to render a set of colored lines"""

    def __init__(self,
                 model_data: LinesModelData,
                 storage_mode: str = "dynamic"):
        super().__init__(model_data)

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        self.vertex_bo = None
        self.vertex_ebo = None
        self.color_bo = None

        assert_debug(storage_mode in ["dynamic", "static"])
        if storage_mode == "dynamic":
            self._storage_mode = GL_DYNAMIC_DRAW
        else:
            self._storage_mode = GL_STATIC_DRAW

        self._num_elements: int = 0

    def num_elements(self):
        return self._num_elements

    def init_model(self):
        self._is_initialized = True
        self.init_buffers()
        self.update_model()

    def init_buffers(self):
        super().init_buffers()
        self.vertex_bo = glGenBuffers(1)
        self.vertex_ebo = glGenBuffers(1)

    def update_model(self):
        super().update_model()
        glBindVertexArray(self.vao)

        # Build the line models
        assert isinstance(self.model_data, LinesModelData)
        vertex_data = self.model_data.line_data.reshape(-1, 3)
        color_data = self.model_data.line_color.reshape(-1, 1, 3).repeat(2, axis=1).reshape(-1, 3)
        indices = np.arange(vertex_data.shape[0]).astype(gl_index_np_dtype())
        self._num_elements = indices.shape[0]

        # Populate the Vertex Buffer Object with vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)
        data = np.concatenate([vertex_data,
                               color_data],
                              axis=1).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self._storage_mode)

        # Define the point attribute in the defined buffer
        glVertexAttribPointer(self.points_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.points_location())

        # Define the color attribute in the defined buffer
        glVertexAttribPointer(self.color_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(3 * type_size(np.float32)))
        glEnableVertexAttribArray(self.color_location())

        # Populate the Element Buffer Object with the edge indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vertex_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    def delete(self):
        glDeleteBuffers(1, self.vertex_bo)
        glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    def verify_model_data(self, _model_data: Optional[ModelData]):
        """Safe setter which verifies that the Point Cloud data is correct"""
        assert (_model_data is not None and isinstance(_model_data, LinesModelData))
        if _model_data.default_color is None:
            _model_data = dataclasses.replace(_model_data, default_color=np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        assert_debug(_model_data.line_data is not None)
        check_sizes(_model_data.line_data, [-1, 2, 3])

        _model_data.line_data = _model_data.line_data.astype(np.float32)
        num_lines = _model_data.line_data.shape[0]
        if _model_data.line_color is not None:
            check_sizes(_model_data.line_color, [-1, 3])
        else:
            _model_data.line_color = _model_data.default_color.reshape(1, 1, 3).repeat(num_lines, axis=0)

        return _model_data


# ----------------------------------------------------------------------------------------------------------------------
# VOXELS
@dataclass
class VoxelsModelData(ModelData):
    """VoxelsModelData"""
    voxel_points: np.ndarray = None
    voxel_size: float = 1.0
    line_width: float = 1.0


# ----------------------------------------------------------------------------------------------------------------------
class VoxelsModel(Model):
    """A Model to render a set of cameras displayed as a simple edge pyramid"""

    __voxel_model = 0.5 * farray([
        1., 1., 1.,
        1., 1., -1.,
        1., -1., 1.,
        1., -1., -1.,
        -1., 1., 1.,
        -1., 1., -1.,
        -1., -1., 1.,
        -1., -1., -1.,
    ])

    __voxel_element_indices = idarray([
        0, 1,
        0, 2,
        0, 4,

        3, 1,
        3, 2,
        3, 7,

        6, 7,
        6, 4,
        6, 2,

        5, 4,
        5, 7,
        5, 1
    ])

    def __init__(self,
                 model_data: VoxelsModelData,
                 storage_mode: str = "dynamic"):
        super().__init__(model_data)

        self._is_initialized = False

        # --------------
        # OpenGL buffers
        self.vertex_bo = None
        self.vertex_ebo = None
        self.color_bo = None

        assert_debug(storage_mode in ["dynamic", "static"])
        if storage_mode == "dynamic":
            self._storage_mode = GL_DYNAMIC_DRAW
        else:
            self._storage_mode = GL_STATIC_DRAW

        self._num_elements: int = 0

    def num_elements(self):
        return self._num_elements

    def init_model(self):
        self._is_initialized = True
        self.init_buffers()
        self.update_model()

    def init_buffers(self):
        super().init_buffers()
        self.vertex_bo = glGenBuffers(1)
        self.vertex_ebo = glGenBuffers(1)

    def update_model(self):
        super().update_model()
        glBindVertexArray(self.vao)

        # Build the line models
        assert isinstance(self.model_data, VoxelsModelData)

        voxel_centers = (np.round(self.model_data.voxel_points / \
                                  self.model_data.voxel_size)) * self.model_data.voxel_size

        voxel_centers = np.unique(voxel_centers, axis=0)
        num_points = voxel_centers.shape[0]

        voxel_vertices = self.model_data.voxel_size * self.__voxel_model.reshape(1, -1, 3).repeat(num_points, axis=0)
        num_model_points = self.__voxel_model.shape[0]

        voxel_edge_indices = self.__voxel_element_indices.reshape(1, -1, 2).repeat(num_points, axis=0)
        for i in range(num_points):
            voxel_edge_indices[i] += num_model_points * i
        voxel_vertices += voxel_centers.reshape(num_points, 1, 3)

        voxel_vertices = voxel_vertices.reshape(-1, 3)
        voxel_edge_indices = voxel_edge_indices.reshape(-1)
        color_data = self.model_data.default_color.reshape(1, 3).repeat(voxel_vertices.shape[0], axis=0)

        self._num_elements = voxel_edge_indices.shape[0]

        # Populate the Vertex Buffer Object with vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_bo)
        data = np.concatenate([voxel_vertices,
                               color_data],
                              axis=1).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self._storage_mode)

        # Define the point attribute in the defined buffer
        glVertexAttribPointer(self.points_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.points_location())

        # Define the color attribute in the defined buffer
        glVertexAttribPointer(self.color_location(),
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              6 * type_size(np.float32),
                              ctypes.c_void_p(3 * type_size(np.float32)))
        glEnableVertexAttribArray(self.color_location())

        # Populate the Element Buffer Object with the edge indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vertex_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, voxel_edge_indices.nbytes, voxel_edge_indices, GL_STATIC_DRAW)

    def delete(self):
        glDeleteBuffers(1, self.vertex_bo)
        glDeleteVertexArrays(1, self.vao)

    # ------------------------------------------------------------------------------------------------------------------
    def verify_model_data(self, _model_data: Optional[ModelData]):
        """Safe setter which verifies that the Point Cloud data is correct"""
        assert _model_data is not None
        assert isinstance(_model_data, VoxelsModelData)
        if _model_data.default_color is None:
            _model_data = dataclasses.replace(_model_data, default_color=np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        assert_debug(_model_data.voxel_points is not None)
        check_sizes(_model_data.voxel_points, [-1, 3])
        _model_data.voxel_points = _model_data.voxel_points.astype(np.float32)

        return _model_data

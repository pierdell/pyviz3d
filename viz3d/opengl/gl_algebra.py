from enum import Enum
from typing import Optional, Union

import numpy as np
from OpenGL.constant import IntConstant

from viz3d.utils import check_sizes, assert_debug


def gl_float_np_dtype():
    """Returns the `numpy.dtype` for the vertex data"""
    return np.float32


def gl_index_np_dtype():
    """Returns the `numpy.dtype` used for indexing by OpenGL"""
    return np.uint32


def type_size(type_: np.dtype) -> int:
    """Returns the size of item for a given `numpy.dtype`"""
    return np.dtype(type_).itemsize


def farray(x: Union[np.ndarray, list, tuple]):
    return np.array(x, dtype=gl_float_np_dtype())


def idarray(x: Union[np.ndarray, list, tuple]):
    return np.array(x, dtype=gl_index_np_dtype())


def gl_transpose():
    return True


def rad(angle_degrees: float):
    """
    Converts an angle from degrees to radians
    """
    return angle_degrees * np.pi / 180.0


def perspective_matrix(fov_degrees: float, z_near: float, z_far: float, aspect_ratio: Optional[float] = None):
    """
    Returns an OpenGL projection matrix from
    """
    if aspect_ratio is None:
        aspect_ratio = 1.0
    f = 1.0 / np.tan(rad(fov_degrees) / 2.0)
    return farray([
        f, 0.0, 0.0, 0.0,
        0.0, f / aspect_ratio, 0.0, 0.0,
        0.0, 0.0, 1.0 * (z_far + z_near) / (z_near - z_far), 2.0 * z_far * z_near / (z_near - z_far),
        0.0, 0.0, -1.0, 0.0])


def infinite_perspective_matrix(fov_degrees: float,
                                z_near: float,
                                aspect_ratio: Optional[float] = None):
    """
    Returns an OpenGL Infinite perspective matrix
    """
    if aspect_ratio is None:
        aspect_ratio = 1.0
    f = 1.0 / np.tan(rad(fov_degrees) / 2.0)
    return farray([
        f, 0.0, 0.0, 0.0,
        0.0, f / aspect_ratio, 0.0, 0.0,
        0.0, 0.0, - 1.0, - 2.0 * z_near,
        0.0, 0.0, -1.0, 0.0])


def normalize(vec: np.ndarray):
    """
    Normalizes the one dimensional array
    """
    return vec / np.linalg.norm(vec)


def look_at(position: np.ndarray, destination: np.ndarray, up: Optional[np.ndarray] = None):
    """
    Returns a [4, 4] matrix corresponding to the camera pose looking at a given destination
    """
    if up is None:
        up = farray([0.0, 1.0, 0.0])
    matrix = np.eye(4, dtype=gl_float_np_dtype())
    matrix[:3, 3] = position
    front = normalize(position - destination)
    up_vector = normalize(up - up.dot(front) * front)
    right = normalize(np.cross(up, front))
    matrix[:3, 0] = right
    matrix[:3, 1] = up_vector
    matrix[:3, 2] = front
    return matrix


def aspect_ratio(height: int, width: int):
    return height / width


def rigid_transform_inverse(transform: np.ndarray):
    check_sizes(transform, [4, 4])
    tr_i = np.eye(4, dtype=gl_float_np_dtype())
    rot = transform[:3, :3].T
    tr_i[:3, :3] = rot
    tr_i[:3, 3] = -rot @ transform[:3, 3]
    return tr_i


class GL_TYPES(Enum):
    """
    Enum used to convert OpenGL types to numpy dtype
    """
    GL_INT = np.int32
    GL_FLOAT = np.float32
    GL_UNSIGNED_BYTE = np.uint8
    GL_UNSIGNED_INT = np.int32

    @staticmethod
    def gl_to_numpy(gl_type: IntConstant):
        """
        Returns the numpy dtype corresponding to the OpenGL's data type
        """
        assert_debug(isinstance(gl_type, IntConstant))
        assert_debug(gl_type.name in GL_TYPES.__members__, "No correspondences found")
        return GL_TYPES.__members__[gl_type.name].value

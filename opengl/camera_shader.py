from opengl.gl_algebra import gl_transpose
from opengl.model import PointCloudModel
from opengl.shader import *

import numpy as np


class CameraAlbedoShader(Shader):
    """
    A CameraAlbedoShader is a shader for simple Camera
    """

    # ------------------------------------------------------------------------------------------------------------------
    # VERTEX SHADER SOURCE
    __vertex_shader = """
#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

uniform mat4 world_to_cam;
uniform mat4 projection;

out vec3 _color;

void main() {
    vec4 homogeneous = vec4(in_position, 1.0);
    gl_Position = projection * world_to_cam * homogeneous; 
    _color = in_color;
}
"""

    # ------------------------------------------------------------------------------------------------------------------
    # FRAGMENT SHADER SOURCE
    __fragment_shader = """
#version 330 core

out vec3 fragment_color;
in vec3 _color;

void main() {
    fragment_color = _color;
}
"""

    # ------------------------------------------------------------------------------------------------------------------
    # OVERRIDDEN METHODS
    def initialize_uniform_variables(self, **kwargs):
        """
        Initializes the uniform variables of the program
        """

        pid = self.shader_program.gl_shader_program_id
        glUseProgram(pid)

    def draw_model(self,
                   model: Model,
                   world_to_cam=Optional[np.ndarray],
                   projection: Optional[np.ndarray] = None,
                   **kwargs):
        assert_debug(world_to_cam is not None)
        assert_debug(projection is not None)

        assert isinstance(model, PointCloudModel)
        pid = self.shader_program.gl_shader_program_id
        glUseProgram(pid)

        # Vertex locations
        glUniformMatrix4fv(self.get_ulocation("world_to_cam"), 1, gl_transpose(), world_to_cam)
        glUniformMatrix4fv(self.get_ulocation("projection"), 1, gl_transpose(), projection)

        glBindVertexArray(model.vao)
        glDrawArrays(GL_POINTS, 0, model.num_points())

        # Release buffers
        glBindVertexArray(0)
        glUseProgram(0)

    def init_shader_program(self, **kwargs):
        assert_debug(not self.initialized)
        super().init_shader_program(**kwargs)

    def vertex_shader(self):
        return self.__vertex_shader

    def fragment_shader(self) -> str:
        return self.__fragment_shader

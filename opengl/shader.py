from typing import Optional
from abc import abstractmethod
from itertools import count

from utils import assert_debug
from OpenGL.GL import *
from OpenGL.GL import shaders

from opengl.model import Model


class ShaderProgram:
    """
    A ShaderProgram compiles the different glsl shaders  (vertex, fragment, etc..)
    """

    def __init__(self,
                 vertex_shader: str,
                 fragment_shader: str):
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader

        # Parameters
        self.gl_shader_program_id = None

    def is_compiled(self):
        """
        Whether the shader is compiled and exists in the OpenGL context
        """
        return self.gl_shader_program_id is not None

    @staticmethod
    def __compile_shader(shader: str, gl_shader_type: int):
        try:
            return shaders.compileShader(shader, gl_shader_type)
        except (shaders.ShaderCompilationError, RuntimeError) as e:
            print(shader)
            raise ...

    def compile(self):
        """
        Compiles the shader program and saves the id to use for rendering calls
        """
        # Compute Shader program
        vertex_shader = self.__compile_shader(self.vertex_shader, GL_VERTEX_SHADER)
        fragment_shader = self.__compile_shader(self.fragment_shader, GL_FRAGMENT_SHADER)

        self.gl_shader_program_id = shaders.compileProgram(
            vertex_shader,
            fragment_shader)
        glUseProgram(self.gl_shader_program_id)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)


class Shader:
    """
    A Shader is the object in charge of rendering a scene into the image plane

    It defines and compile glsl shaders as a ShaderProgram
    Sets the variables to interact with the program and draws model in the allocated framebuffer
    """

    def __init__(self, debug=False):
        self.shader_program: Optional[ShaderProgram] = None
        self.initialized = False
        self.debug = debug
        self.__shader_id = next(self.__shader_ids)

    __shader_ids = count(0)

    def init_shader_program(self, **kwargs):
        """
        Initializes the shader program,

        Associates the buffer and uniform variables to it, using the data passed in kwargs

        kwargs : dict
            The dictionary of named arguments passed to child classes methods to initiate the shader program's state
        """
        assert_debug(not self.initialized)
        if self.debug:
            print(f"////////////////////////////"
                  f"VERTEX SHADER \n"
                  f"{self.vertex_shader()}\n\n\n\n")
            print(f"////////////////////////////"
                  f"FRAGMENT SHADER \n"
                  f"{self.fragment_shader()}\n\n\n\n")

        shader_program = ShaderProgram(self.vertex_shader(), self.fragment_shader())

        self.shader_program = shader_program
        self.shader_program.compile()

        self.initialize_uniform_variables(**kwargs)
        self.initialized = True

    @abstractmethod
    def initialize_uniform_variables(self, **kwargs):
        """
        Initialize the uniform variables of the shader program

        The uniform variables passed on the kwargs are should be constant for every draw call
        (eg. Lights position)
        """
        raise NotImplementedError("")

    @abstractmethod
    def vertex_shader(self) -> str:
        """
        Returns the source glsl code of the vertex shader as a string to be compiled
        """
        raise NotImplementedError("")

    @abstractmethod
    def fragment_shader(self) -> str:
        """
        Returns the source glsl code of the fragment shader as a string to be compiled
        """
        raise NotImplementedError("")

    @abstractmethod
    def draw_model(self, model: Model, **kwargs):
        """
        Draws the model

        Notes
        -----
        Not all models can be drawn by each shader,
        A Shader will typically raise exceptions if it cannot draw a given model
        """
        raise NotImplementedError("")

    def get_ulocation(self, location_name: str):
        """
        Returns the uniform location in the shader program associated with a location name
        """
        loc = glGetUniformLocation(self.shader_program.gl_shader_program_id, location_name)
        # assert_debug(loc != -1, f"Location {location_name} is not found in the program")
        return loc

    @property
    def shader_id(self):
        """
        Returns the id of the instance of the shader
        """
        return self.__shader_id

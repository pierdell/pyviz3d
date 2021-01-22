from opengl.model import ScreenModel
from opengl.shader import *
import numpy as np


class ScreenShader(Shader):
    """
    A CameraAlbedoShader is a shader for simple Camera
    """

    # ------------------------------------------------------------------------------------------------------------------
    # VERTEX SHADER SOURCE
    __vertex_shader = """
#version 330 core

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_tex_coord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(in_position.x, in_position.y, 0.0, 1.0); 
    TexCoord = in_tex_coord;
}
"""

    # ------------------------------------------------------------------------------------------------------------------
    # FRAGMENT SHADER SOURCE
    __fragment_shader = """
#version 330 core

out vec4 fragment_color;
in vec2 TexCoord;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;

uniform float A;
uniform float B;
uniform float EDL_STRENGTH;
uniform float EDL_DISTANCE;
uniform bool WITH_EDL;

vec4 color;
float buffer_depth;
float real_depth;

float buffer_depth_to_real_depth(float depth)
{
    return 0.5*(-A * depth + B) / depth + 0.5;
}

vec2 neighborContribution(float log2Depth, vec2 offset)
{
    // Code strongly inspired by 
    // https://github.com/CesiumGS/cesium/blob/master/Source/Shaders/PostProcessStages/PointCloudEyeDomeLighting.glsl
    
    float dist = EDL_DISTANCE;
    vec2 texCoordOrig = TexCoord + offset * dist;
    vec2 texCoord0 = TexCoord + offset * floor(dist);
    vec2 texCoord1 = TexCoord + offset * ceil(dist);

    float depthOrLogDepth0 = texture2D(depthTexture, texCoord0).r;
    float depthOrLogDepth1 = texture2D(depthTexture, texCoord1).r;

    // ignore depth values that are the clear depth
    if (depthOrLogDepth0 == 0.0 || depthOrLogDepth1 == 0.0) {
        return vec2(0.0);
    }

    // interpolate the two adjacent depth values
    float depthMix = mix(depthOrLogDepth0, depthOrLogDepth1, fract(dist));
    float new_depth = buffer_depth_to_real_depth(depthMix);
    return vec2(max(0.0, log2Depth - log2(new_depth)), 1.0);
}

void main() {
    color = texture(colorTexture, TexCoord);
    
    if (WITH_EDL) {
        // Build the Depth
        buffer_depth = texture(depthTexture, TexCoord).r;
        real_depth = buffer_depth_to_real_depth(buffer_depth);
        
        float log2Depth = log2(real_depth);
        if (log2Depth == 0)
            discard;
            
        vec2 texelSize = 1.0 / vec2(720, 1080);
        vec2 responseAndCount = vec2(0.0);
        responseAndCount += neighborContribution(log2Depth, vec2(-texelSize.x, 0.0));
        responseAndCount += neighborContribution(log2Depth, vec2(+texelSize.x, 0.0));
        responseAndCount += neighborContribution(log2Depth, vec2(0.0, -texelSize.y));
        responseAndCount += neighborContribution(log2Depth, vec2(0.0, +texelSize.y));
        
        // Build the Eye Dome Lighting effect PostProcessing
        float response = responseAndCount.x / responseAndCount.y;
        float shade = exp(-response * 300.0 * EDL_STRENGTH);
        color.rgb *= shade;
    }
    
    fragment_color = vec4(color);
}
"""

    def __init__(self, edl_strength=100.0, edl_distance=1.0, with_edl: bool = True):
        super().__init__()
        self.edl_strength = edl_strength
        self.edl_distance = edl_distance
        self.with_edl = with_edl

    # ------------------------------------------------------------------------------------------------------------------
    # OVERRIDDEN METHODS
    def initialize_uniform_variables(self, z_far: float = 1000.0, z_near: float = 0.1, **kwargs):
        """
        Initializes the uniform variables of the program
        """
        pid = self.shader_program.gl_shader_program_id
        glUseProgram(pid)

        # Initialize the texture locations
        color_texture_loc = glGetUniformLocation(pid, "colorTexture")
        depth_texture_loc = glGetUniformLocation(pid, "depthTexture")
        glUniform1i(color_texture_loc, 0)
        glUniform1i(depth_texture_loc, 1)

        # Register the uniform parameters which allows to recover the real depth from the depth buffer value
        a_loc = glGetUniformLocation(pid, "A")
        b_loc = glGetUniformLocation(pid, "B")
        a = - (z_far + z_near) / (z_far - z_near)
        b = -2 * (z_far * z_near) / (z_far - z_near)
        glUniform1f(a_loc, np.float32(a))
        glUniform1f(b_loc, np.float32(b))

        # Register the uniform parameters which allows to recover the real depth from the depth buffer value
        a_loc = glGetUniformLocation(pid, "A")
        b_loc = glGetUniformLocation(pid, "B")
        a = - (z_far + z_near) / (z_far - z_near)
        b = -2 * (z_far * z_near) / (z_far - z_near)
        glUniform1f(a_loc, np.float32(a))
        glUniform1f(b_loc, np.float32(b))

        # Register EDL variables
        edl_strength_loc = glGetUniformLocation(pid, "EDL_STRENGTH")
        edl_dist_loc = glGetUniformLocation(pid, "EDL_DISTANCE")
        with_edl_loc = glGetUniformLocation(pid, "WITH_EDL")
        glUniform1f(edl_strength_loc, np.float32(self.edl_strength))
        glUniform1f(edl_dist_loc, np.float32(self.edl_distance))
        with_edl = np.int32(self.with_edl)
        glUniform1i(with_edl_loc, with_edl)

    def draw_model(self,
                   model: Model,
                   color_texture: int = -1,
                   depth_texture: int = -1,
                   with_edl: bool = True,
                   **kwargs):
        assert isinstance(model, ScreenModel)
        pid = self.shader_program.gl_shader_program_id
        glUseProgram(pid)

        glActiveTexture(GL_TEXTURE0 + 0)
        glBindTexture(GL_TEXTURE_2D, color_texture)

        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, depth_texture)

        # Vertex locations
        glBindVertexArray(model.vao)
        glDisable(GL_DEPTH_TEST)
        glDrawArrays(GL_TRIANGLES, 0, model.num_points())

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

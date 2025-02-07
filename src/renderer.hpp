#pragma once

#include "fluids/fluid_matrix.hpp"
#include "simulation_settings.hpp"
#include "utils.hpp"
#include <GL/glew.h>
#include <string>
#include <vector>

class Renderer {
public:
    /// Get the shader program for rendering the matrix
    static GLuint getShaderProgram(bool useDensityShader);
    static std::vector<float> getDensityVertices(SimulationSettings *settings, FluidMatrix *matrix);
    static std::vector<float> getVelocityVertices(SimulationSettings *settings, FluidMatrix *matrix);
    static void linkDensityVerticesToBuffer(float *vertices, int len);
    static void linkVelocityVerticesToBuffer(float *vertices, int len);

private:
    /// Compile a shader from source code
    static std::optional<GLuint> compileShader(const std::string &shaderSource, GLenum shaderType);
    static void normalizeDensityVertices(SimulationSettings *settings, std::vector<float> &vertices, int n);
    static void normalizeSpeedVertices(SimulationSettings *settings, std::vector<float> &vertices, int n);
};

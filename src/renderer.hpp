#pragma once

#include "fluids/fluid_matrix.hpp"
#include "simulation_settings.hpp"
#include "utils.hpp"
#include <GL/glew.h>
#include <string>
#include <vector>

namespace Renderer {
/// Get the shader program for rendering the matrix
GLuint getShaderProgram(bool useDensityShader);
std::vector<float> getDensityVertices(SimulationSettings *settings, FluidMatrix *matrix);
std::vector<float> getVelocityVertices(SimulationSettings *settings, FluidMatrix *matrix);
void linkDensityVerticesToBuffer(float *vertices, int len);
void linkVelocityVerticesToBuffer(float *vertices, int len);

/// Compile a shader from source code
std::optional<GLuint> compileShader(const std::string &shaderSource, GLenum shaderType);
void normalizeDensityVertices(SimulationSettings *settings, std::vector<float> &vertices, int n);
void normalizeSpeedVertices(SimulationSettings *settings, std::vector<float> &vertices, int n);
}; // namespace Renderer

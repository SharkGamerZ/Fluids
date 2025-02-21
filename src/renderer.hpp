#pragma once

#include "fluids/fluid_matrix.hpp"
#include "simulation_settings.hpp"
#include "utils.hpp"
#include <GL/glew.h>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

namespace Renderer {
/// Get the shader program for rendering the matrix
GLuint getShaderProgram(SimulationAttribute attribute);
/// Get the number of components for a vertex attribute
int getVertexComponentCount(SimulationAttribute attribute);
std::vector<float> getDensityVertices(const SimulationSettings *settings, const FluidMatrix *matrix);
std::vector<float> getVelocityVertices(const SimulationSettings *settings, const FluidMatrix *matrix);
std::vector<float> getVorticityVertices(const SimulationSettings *settings, const FluidMatrix *matrix);
}; // namespace Renderer

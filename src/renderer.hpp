#pragma once

#include "fluids/fluid_simulation.hpp"
#include "simulation_settings.hpp"
#include "utils.hpp"
#include <GL/glew.h>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

namespace Renderer {
/// Set the shader program for rendering the matrix
bool setShaderProgram(SimulationAttribute attribute);
/// Get the number of components for a vertex attribute
int getVertexComponentCount(SimulationAttribute attribute);
/// Get the vertices to be rendered
std::vector<float> getVertices(const SimulationSettings &settings, const FluidSimulation &simulation);
}; // namespace Renderer

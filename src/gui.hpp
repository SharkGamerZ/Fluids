#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "renderer.hpp"
#include "simulation_settings.hpp"
#include "utils.hpp"
#include <GLFW/glfw3.h>
#include <cmath>

namespace GUI {
/// Initialize GUI
void Init(GLFWwindow *window);
/// Main UI rendering function
void Render(SimulationSettings &settings, GLFWwindow *window, FluidMatrix *matrix);
/// Render fluid matrix
void RenderMatrix(const SimulationSettings &settings, const FluidMatrix *fluidMatrix);
/// Cleanup GUI
void Cleanup();
/// Callback for key events
void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
}; // namespace GUI

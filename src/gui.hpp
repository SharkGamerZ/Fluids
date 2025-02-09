#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "renderer.hpp"
#include "simulation_settings.hpp"
#include "utils.hpp"
#include <GLFW/glfw3.h>
#include <cmath>


class GUI {
public:
    /// Initialize GUI
    static void Init(GLFWwindow *window);
    /// Main UI rendering function
    static void Render(SimulationSettings &settings, GLFWwindow *window, FluidMatrix *matrix);
    /// Render fluid matrix
    static void RenderMatrix(SimulationSettings &settings, FluidMatrix *fluidMatrix);
    /// Cleanup GUI
    static void Cleanup();
    /// Callback for key events
    static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
};

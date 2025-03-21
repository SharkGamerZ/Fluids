#include "gui.hpp"

namespace GUI {
void Init(GLFWwindow *window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // Enable keyboard controls
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
}

void Render(SimulationSettings &settings, GLFWwindow *window, FluidMatrix *matrix) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));

    const std::string title = std::format("Simulation Parameters - {} fps: {:.1f} ###SimulationParameters",
                                          settings.executionMode == SERIAL   ? "Serial"
                                          : settings.executionMode == OPENMP ? "OpenMP"
                                                                             : "CUDA",
                                          ImGui::GetIO().Framerate);


    if (ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoResize)) {
        if (ImGui::BeginTabBar("Tabs")) {
            if (ImGui::BeginTabItem("Parameters")) {
                // Simulation parameters
                ImGui::SliderFloat("Viscosity", &settings.viscosity, 0.0f, 0.0001f, "%.7f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("TimeStep", &settings.deltaTime, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Mouse density", &settings.mouse_density, 0.0f, 20.0f, "%.2f", ImGuiSliderFlags_None);
                ImGui::SliderFloat("Mouse velocity", &settings.mouse_velocity, 0.0f, 20.0f, "%.2f", ImGuiSliderFlags_None);

                // Update matrix parameters
                matrix->visc = settings.viscosity;
                matrix->dt = settings.deltaTime;

                // Visualization mode
                constexpr std::array visualizationModeNames{"Density", "Velocity", "Vorticity"};
                ImGui::Combo("Visualization", reinterpret_cast<int *>(&settings.simulationAttribute), visualizationModeNames.data(), visualizationModeNames.size());

                // Execution mode
                constexpr std::array executionModeNames{"Serial", "OpenMP",
#ifdef CUDA_SUPPORT
                                                        "CUDA"
#endif
                };
                ImGui::Combo("Execution mode", reinterpret_cast<int *>(&settings.executionMode), executionModeNames.data(), executionModeNames.size());

                // Toggle wind machine
                if (ImGui::Button(settings.windMachine ? "Stop wind machine" : "Start wind machine")) {
                    settings.windMachine = !settings.windMachine;
                }
                ImGui::SameLine();
                ImGui::Text("Wind Machine: %s", settings.windMachine ? "ON" : "OFF");


                // Start/Stop simulation
                if (ImGui::Button(settings.isSimulationRunning ? "Stop Simulation" : "Start Simulation")) {
                    settings.isSimulationRunning = !settings.isSimulationRunning;
                }
                ImGui::SameLine();
                ImGui::Text("Simulation status: %s", settings.isSimulationRunning ? "Running" : "Stopped");

                // Performance display
                ImGui::Text("Avg %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Keybinds")) {
                ImGui::Text("[ESC]/[Q] - Exit program");
                ImGui::Text("[SPACE] - Pause/Resume simulation");
                ImGui::Text("[R] - Reset simulation");
                ImGui::Text("[A]/[D] - Cycle simulation attribute backwards/forwards");
                ImGui::Text("[M] - Change execution mode");
                ImGui::Text("[W] - Toggle wind machine");
                ImGui::Text("[->] - Frame simulation (when paused)");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    // Run simulation
    {
        // Velocity related mouse controls
        glfwGetCursorPos(window, &settings.xpos, &settings.ypos);
        settings.xposScaled = round(settings.xpos / settings.scalingFactor);
        settings.yposScaled = round(settings.ypos / settings.scalingFactor);
        settings.mouseTime = glfwGetTime();
        settings.mouseTimeDelta = settings.mouseTime - settings.mouseTimePrev;
        settings.deltax = settings.xpos - settings.xposPrev;
        settings.deltay = settings.ypos - settings.yposPrev;
        settings.mouseTimePrev = settings.mouseTime;

        if (settings.xposScaled >= 0 && settings.xposScaled < settings.matrixSize && settings.yposScaled >= 0 && settings.yposScaled < settings.matrixSize) {
            // Add Density
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
                if (settings.executionMode != CUDA) matrix->addDensity(static_cast<int>(settings.xposScaled), static_cast<int>(settings.yposScaled), 20.0f * settings.mouse_density);
#ifdef CUDA_SUPPORT
                else matrix->CUDA_addDensity(static_cast<int>(settings.xposScaled), static_cast<int>(settings.yposScaled), 20.0f * settings.mouse_density);
#endif
            }


            // Calculate velocity
            settings.deltax /= settings.scalingFactor * 2;
            settings.deltay /= settings.scalingFactor * 2;

            // Add Velocity
            if (settings.executionMode != CUDA)
                matrix->addVelocity(static_cast<int>(settings.xposScaled), static_cast<int>(settings.yposScaled), settings.deltax * settings.mouse_velocity, settings.deltay * settings.mouse_velocity);
#ifdef CUDA_SUPPORT
            else
                matrix->CUDA_addVelocity(static_cast<int>(settings.xposScaled), static_cast<int>(settings.yposScaled), settings.deltax * settings.mouse_velocity,
                                         settings.deltay * settings.mouse_velocity);
#endif
        }

        settings.xposPrev = settings.xpos;
        settings.yposPrev = settings.ypos;

        // Wind machine
        if (settings.windMachine) {
            if (settings.executionMode != CUDA) matrix->addVelocity(2, settings.matrixSize / 2, 10, 0.0f);
#ifdef CUDA_SUPPORT
            else matrix->CUDA_addVelocity(2, settings.matrixSize / 2, 10, 0.0f);
#endif
        }

        // Reset simulation
        if (settings.resetSimulation) {
            matrix->reset();
            settings.resetSimulation = false;
        }

        // Run simulation
        if (settings.isSimulationRunning || settings.frameSimulation) {
            switch (settings.executionMode) {
                case SERIAL:
#ifdef CUDA_SUPPORT
                    if (settings.executionModePrev == CUDA) matrix->copyToHost();
#endif
                    matrix->step();
                    break;

                case OPENMP:
#ifdef CUDA_SUPPORT
                    if (settings.executionModePrev == CUDA) matrix->copyToHost();
#endif
                    matrix->OMP_step();
                    break;
#ifdef CUDA_SUPPORT
                case CUDA:
                    if (settings.executionModePrev != CUDA) matrix->copyToDevice();
                    matrix->CUDA_step();
                    matrix->copyToHost();
                    break;
#endif
                default: log(Utils::LogLevel::ERROR, std::cerr, "Unknown execution mode"); return;
            }

            settings.frameSimulation = false;
            settings.executionModePrev = settings.executionMode;
        }
    }

    // Render matrix
    RenderMatrix(settings, matrix);

    // TODO: Check why we have to do this
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void RenderMatrix(const SimulationSettings &settings, const FluidMatrix *matrix) {
    if (const GLuint shaderProgram = Renderer::getShaderProgram(settings.simulationAttribute); !shaderProgram) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to create shader program");
        return;
    }

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    {
        const int viewportSize = settings.viewportSize;
        const auto componentCount = Renderer::getVertexComponentCount(settings.simulationAttribute);

        std::vector<float> verts;
        switch (settings.simulationAttribute) {
            case DENSITY: verts = Renderer::getDensityVertices(&settings, matrix); break;
            case VELOCITY: verts = Renderer::getVelocityVertices(&settings, matrix); break;
            case VORTICITY: verts = Renderer::getVorticityVertices(&settings, matrix); break;
        }

        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, componentCount, GL_FLOAT, GL_FALSE, componentCount * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);

        glPointSize(3.0f); // Render the points bigger to fill the blank spaces
        glDrawArrays(GL_POINTS, 0, viewportSize * viewportSize);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void Cleanup() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void KeyCallback(GLFWwindow *window, const int key, [[maybe_unused]] const int scancode, const int action, [[maybe_unused]] const int mods) {
    auto *settings = static_cast<SimulationSettings *>(glfwGetWindowUserPointer(window));
    if (!settings) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to get window user pointer");
        return;
    }

    // Exit program
    if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);

    // Pause/resume simulation
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) settings->isSimulationRunning = !settings->isSimulationRunning;

    // Frame simulation
    if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) settings->frameSimulation = true;

    // Reset simulation
    if (key == GLFW_KEY_R && action == GLFW_PRESS) settings->resetSimulation = true;

    // Change simulation attribute
    if (key == GLFW_KEY_A && action == GLFW_PRESS) {
        switch (settings->simulationAttribute) {
            case DENSITY: settings->simulationAttribute = VORTICITY; break;
            case VELOCITY: settings->simulationAttribute = DENSITY; break;
            case VORTICITY: settings->simulationAttribute = VELOCITY; break;
        }
    }
    if (key == GLFW_KEY_D && action == GLFW_PRESS) {
        switch (settings->simulationAttribute) {
            case DENSITY: settings->simulationAttribute = VELOCITY; break;
            case VELOCITY: settings->simulationAttribute = VORTICITY; break;
            case VORTICITY: settings->simulationAttribute = DENSITY; break;
        }
    }

    // Change execution mode
    if (key == GLFW_KEY_M && action == GLFW_PRESS) {
#ifdef CUDA_SUPPORT
        settings->executionMode = static_cast<ExecutionMode>((static_cast<int>(settings->executionMode) + 1) % (CUDA + 1));
#else
        settings->executionMode = settings->executionMode == SERIAL ? OPENMP : SERIAL;
#endif
    }
    // Toggle wind machine
    if (key == GLFW_KEY_W && action == GLFW_PRESS) settings->windMachine = !settings->windMachine;
}
} // namespace GUI

#include "gui.hpp"

void GUI::Init(GLFWwindow *window) {
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

void GUI::Render(SimulationSettings &settings, GLFWwindow *window, FluidMatrix *matrix) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("Simulation parameters", nullptr, ImGuiWindowFlags_NoResize)) {
        if (ImGui::BeginTabBar("Tabs")) {
            if (ImGui::BeginTabItem("Parameters")) {
                // Simulation parameters
                ImGui::SliderFloat("Viscosity", &settings.viscosity, 0.0f, 0.0001f, "%.7f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("TimeStep", &settings.deltaTime, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

                // Update matrix parameters
                matrix->visc = settings.viscosity;
                matrix->dt = settings.deltaTime;

                // Visualization mode
                ImGui::Text("Visualization:");
                ImGui::RadioButton("Density", &settings.simulationAttribute, DENSITY);
                ImGui::SameLine();
                ImGui::RadioButton("Velocity", &settings.simulationAttribute, VELOCITY);

                // Execution mode
                ImGui::Text("Execution mode:");
                ImGui::RadioButton("Serial", &settings.executionMode, SERIAL);
                ImGui::SameLine();
                ImGui::RadioButton("OpenMP", &settings.executionMode, OPENMP);
#ifdef __CUDACC__
                ImGui::SameLine();
                ImGui::RadioButton("CUDA", &settings.executionMode, CUDA);
#endif

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
                ImGui::Text("[ESC] - Exit program");
                ImGui::Text("[SPACE] - Pause/Resume simulation");
                ImGui::Text("[R] - Reset simulation");
                ImGui::Text("[V] - Change simulation attribute");
                ImGui::Text("[M] - Change execution mode");
                ImGui::Text("[W] - Toggle wind machine");
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
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
                matrix->addDensity(static_cast<int>(settings.xposScaled), static_cast<int>(settings.yposScaled), 20.0f);
            }

            settings.deltax /= settings.scalingFactor * 2;
            settings.deltay /= settings.scalingFactor * 2;
            matrix->addVelocity(static_cast<int>(settings.xposScaled), static_cast<int>(settings.yposScaled), settings.deltax, settings.deltay);
        }

        settings.xposPrev = settings.xpos;
        settings.yposPrev = settings.ypos;

        // Keybind related actions
        if (settings.windMachine) {
            for (int i = 0; i < settings.matrixSize; i++) {
                matrix->addVelocity(2, i, 0.0f, 0.5f);
            }
        }
        if (settings.resetSimulation) {
            matrix->reset();
            settings.resetSimulation = false;
        }
        // TODO step simulation (add frame mode)
        if (settings.isSimulationRunning) {
            switch (settings.executionMode) {
                case SERIAL: matrix->step(); break;
                case OPENMP: matrix->OMP_step(); break;
#ifdef __CUDACC__
                case CUDA: matrix->CUDA_step(); break;
#endif
            }
        }
    }

    // Render matrix
    RenderMatrix(settings, matrix);
    // NOTE: this makes it go from 60FPS to 15 :(
    // NOTE 2: if this is done before ImGui::Render(), the gui is not bugged off-screen

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::RenderMatrix(SimulationSettings &settings, FluidMatrix *matrix) {
    const GLuint shaderProgram = Renderer::getShaderProgram(settings.simulationAttribute == DENSITY);
    if (!shaderProgram) {
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
        std::vector<float> verts = settings.simulationAttribute == DENSITY ? Renderer::getDensityVertices(&settings, matrix) : Renderer::getVelocityVertices(&settings, matrix);

        if (settings.simulationAttribute == DENSITY) {
            glBindVertexArray(VAO);
            Renderer::linkDensityVerticesToBuffer(verts.data(), viewportSize * viewportSize * 3);
            glDrawArrays(GL_POINTS, 0, viewportSize * viewportSize * 3);
        } else {
            glBindVertexArray(VAO);
            Renderer::linkVelocityVerticesToBuffer(verts.data(), viewportSize * viewportSize * 4);
            glDrawArrays(GL_POINTS, 0, viewportSize * viewportSize * 4);
        }
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
}

void GUI::Cleanup() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GUI::KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto *settings = static_cast<SimulationSettings *>(glfwGetWindowUserPointer(window));
    if (!settings) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to get window user pointer");
        return;
    }

    // Exit program
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
    // Pause/resume simulation
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) settings->isSimulationRunning = !settings->isSimulationRunning;
    // Reset simulation
    if (key == GLFW_KEY_R && action == GLFW_PRESS) settings->resetSimulation = true;
    // Change simulation attribute
    if (key == GLFW_KEY_V && action == GLFW_PRESS) settings->simulationAttribute = settings->simulationAttribute == DENSITY ? VELOCITY : DENSITY;
    // Change execution mode
    if (key == GLFW_KEY_M && action == GLFW_PRESS) {
#ifdef __CUDACC__
        settings->executionMode = (settings->executionMode + 1) % (CUDA + 1);
#else
        settings->executionMode = settings->executionMode == SERIAL ? OPENMP : SERIAL;
#endif
    }
    // Toggle wind machine    // Toggle wind machine
    if (key == GLFW_KEY_W && action == GLFW_PRESS) settings->windMachine = !settings->windMachine;
}

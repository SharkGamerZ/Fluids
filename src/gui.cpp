#include "gui.hpp"

SimulationSettings *GUI::settingsPtr = nullptr;

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

void GUI::Render(SimulationSettings &settings, FluidMatrix *matrix) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Window position and size
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(350.0f, 200.0f));

    if (ImGui::Begin("Simulation parameters", nullptr, ImGuiWindowFlags_NoResize)) {
        // Simulation parameters
        ImGui::SliderFloat("Viscosity", &settings.viscosity, 0.0f, 0.0001f, "%.7f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("TimeStep", &settings.deltaTime, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

        // Visualization mode
        ImGui::Text("Visualization:");
        ImGui::RadioButton("Density", &settings.simulationAttribute, DENSITY_ATTRIBUTE);
        ImGui::SameLine();
        ImGui::RadioButton("Velocity", &settings.simulationAttribute, VELOCITY_ATTRIBUTE);

        // Execution mode
        ImGui::Text("Execution mode:");
        ImGui::RadioButton("Serial", &settings.executionMode, SERIAL);
        ImGui::SameLine();
        ImGui::RadioButton("OpenMP", &settings.executionMode, OPENMP);
        ImGui::SameLine();
        ImGui::RadioButton("CUDA", &settings.executionMode, CUDA);

        // Start/Stop simulation
        if (ImGui::Button(settings.isSimulationRunning ? "Stop Simulation" : "Start Simulation")) {
            settings.isSimulationRunning = !settings.isSimulationRunning;
        }

        // Status display
        ImGui::SameLine();
        ImGui::Text("Status: %s", settings.isSimulationRunning ? "Running" : "Stopped");

        // Performance display
        ImGui::Text("Avg %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    }
    ImGui::End();

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Run simulation
    // TODO reset, step, wind, addDensity, addVelocity calls here

    // Render matrix
    RenderMatrix(settings, matrix); // NOTE: this makes it go from 60FPS to 15 :(
}

void GUI::RenderMatrix(SimulationSettings &settings, FluidMatrix *matrix) {
    const GLuint shaderProgram = Renderer::getShaderProgram(settings.simulationAttribute == DENSITY_ATTRIBUTE);
    if (!shaderProgram) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to create shader program");
        return;
    }

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // TODO: use matrix dimensions
    {
        std::vector<float> verts =
                settingsPtr->simulationAttribute == DENSITY_ATTRIBUTE ? Renderer::getDensityVertices(&settings, matrix) : Renderer::getVelocityVertices(&settings, matrix);

        if (settingsPtr->simulationAttribute == DENSITY_ATTRIBUTE) {
            glBindVertexArray(VAO);
            Renderer::linkDensityVerticesToBuffer(verts.data(), settingsPtr->viewportSize * settingsPtr->viewportSize * 3);
            glDrawArrays(GL_POINTS, 0, settingsPtr->viewportSize * settingsPtr->viewportSize * 3);
        } else {
            glBindVertexArray(VAO);
            Renderer::linkVelocityVerticesToBuffer(verts.data(), settingsPtr->viewportSize * settingsPtr->viewportSize * 4);
            glDrawArrays(GL_POINTS, 0, settingsPtr->viewportSize * settingsPtr->viewportSize * 4);
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
    // Exit program
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    // Pause/resume simulation
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        if (settingsPtr) settingsPtr->isSimulationRunning = !settingsPtr->isSimulationRunning;
    }
    // Reset simulation
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        if (settingsPtr) settingsPtr->ResetMatrix();
    }
}

void GUI::SetSimulationSettings(SimulationSettings *settings) { settingsPtr = settings; }

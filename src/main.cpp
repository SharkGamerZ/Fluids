#include "gui.hpp"

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(800, 800, "Fluids", nullptr, nullptr);
    if (!window) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to create window");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to initialize GLEW");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glViewport(0, 0, 800, 800);

    GUI::Init(window);

    // Init simulation settings
    SimulationSettings settings;
    GUI::SetSimulationSettings(&settings);

    // Register callback for key events
    glfwSetKeyCallback(window, GUI::KeyCallback);

    // Init fluid matrix
    auto matrix = FluidMatrix(settings.matrixSize, 0.0f, settings.viscosity, settings.deltaTime);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Clear screen
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render GUI
        GUI::Render(settings, &matrix);

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    GUI::Cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}

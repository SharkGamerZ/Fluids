#include "renderer.hpp"


namespace {
    // Shader paths
    const std::string DENSITY_VERTEX_SHADER_PATH = "../src/shaders/density.vert";
    const std::string VELOCITY_VERTEX_SHADER_PATH = "../src/shaders/velocity.vert";
    const std::string DENSITY_FRAGMENT_SHADER_PATH = "../src/shaders/density.frag";
    const std::string VELOCITY_FRAGMENT_SHADER_PATH = "../src/shaders/velocity.frag";

} // namespace

GLuint Renderer::getShaderProgram(const bool useDensityShader) {
    const auto vertexShaderSource = useDensityShader ? Utils::readFile(DENSITY_VERTEX_SHADER_PATH) : Utils::readFile(VELOCITY_VERTEX_SHADER_PATH);
    const auto fragmentShaderSource = useDensityShader ? Utils::readFile(DENSITY_FRAGMENT_SHADER_PATH) : Utils::readFile(VELOCITY_FRAGMENT_SHADER_PATH);
    if (!vertexShaderSource || !fragmentShaderSource) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to read shader source files");
        return 0;
    }

    const auto vertexShader = compileShader(vertexShaderSource.value(), GL_VERTEX_SHADER);
    const auto fragmentShader = compileShader(fragmentShaderSource.value(), GL_FRAGMENT_SHADER);
    if (!vertexShader || !fragmentShader) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to compile shaders");
        return 0;
    }

    const GLuint shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to create shader program");
        return 0;
    }

    glAttachShader(shaderProgram, vertexShader.value());
    glAttachShader(shaderProgram, fragmentShader.value());
    glLinkProgram(shaderProgram);

    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, sizeof(infoLog), nullptr, infoLog);
        log(Utils::LogLevel::ERROR, std::cerr, std::format("ERROR::SHADER::PROGRAM::LINKING_FAILED\n{}", infoLog));
        glDeleteProgram(shaderProgram);
        return 0;
    }

    glDeleteShader(vertexShader.value());
    glDeleteShader(fragmentShader.value());

    glUseProgram(shaderProgram); // TODO: maybe move to caller
    return shaderProgram;
}

std::vector<float> Renderer::getDensityVertices(SimulationSettings *settings, FluidMatrix *matrix) {
    int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 3);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vertices[3 * FluidMatrix::index(i, j, n)] = j;
            vertices[3 * FluidMatrix::index(i, j, n) + 1] = i;

            int idx = (i / settings->scalingFactor * matrix->size) + (j / settings->scalingFactor);
            vertices[3 * FluidMatrix::index(i, j, n) + 2] = matrix->density[idx];
        }
    }

    normalizeDensityVertices(settings, vertices, n);

    return vertices;
}

std::vector<float> Renderer::getVelocityVertices(SimulationSettings *settings, FluidMatrix *matrix) {
    int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 4);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vertices[4 * FluidMatrix::index(i, j, n)] = j;
            vertices[4 * FluidMatrix::index(i, j, n) + 1] = i;

            int idx = (i / settings->scalingFactor * matrix->size) + (j / settings->scalingFactor);
            vertices[4 * FluidMatrix::index(i, j, n) + 2] = matrix->Vx[idx];
            vertices[4 * FluidMatrix::index(i, j, n) + 3] = matrix->Vy[idx];
        }
    }

    normalizeSpeedVertices(settings, vertices, n);

    return vertices;
}

void Renderer::linkDensityVerticesToBuffer(float *vertices, int len) {
    glBufferData(GL_ARRAY_BUFFER, len * sizeof(float), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
}

void Renderer::linkVelocityVerticesToBuffer(float *vertices, int len) {
    glBufferData(GL_ARRAY_BUFFER, len * sizeof(float), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
}

void Renderer::normalizeDensityVertices(SimulationSettings *settings, std::vector<float> &vertices, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vertices[3 * FluidMatrix::index(i, j, n)] = (vertices[3 * FluidMatrix::index(i, j, n)] / (static_cast<float>(settings->viewportSize - 1) / 2.0f)) - 1;
            vertices[3 * FluidMatrix::index(i, j, n)] = 1 - (vertices[3 * FluidMatrix::index(i, j, n) + 1] / (static_cast<float>(settings->viewportSize - 1) / 2.0f));
        }
    }
}

void Renderer::normalizeSpeedVertices(SimulationSettings *settings, std::vector<float> &vertices, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vertices[4 * FluidMatrix::index(i, j, n)] = (vertices[4 * FluidMatrix::index(i, j, n)] / (static_cast<float>(settings->viewportSize - 1) / 2.0f)) - 1;
            vertices[4 * FluidMatrix::index(i, j, n) + 1] = 1 - (vertices[4 * FluidMatrix::index(i, j, n) + 1] / (static_cast<float>(settings->viewportSize - 1) / 2.0f));
        }
    }
}

std::optional<GLuint> Renderer::compileShader(const std::string &shaderSource, GLenum shaderType) {
    const GLuint shader = glCreateShader(shaderType);
    const char *shaderSourceCStr = shaderSource.c_str();
    glShaderSource(shader, 1, &shaderSourceCStr, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        log(Utils::LogLevel::ERROR, std::cerr, std::format("ERROR::SHADER::COMPILATION_FAILED\n{}", infoLog));
        glDeleteShader(shader);
        return std::nullopt;
    }
    return shader;
}

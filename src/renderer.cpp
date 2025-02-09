#include "renderer.hpp"


namespace {
// Shader paths
const std::string DENSITY_VERTEX_SHADER_PATH = "../src/shaders/density.vert";
const std::string VELOCITY_VERTEX_SHADER_PATH = "../src/shaders/velocity.vert";
const std::string DENSITY_FRAGMENT_SHADER_PATH = "../src/shaders/density.frag";
const std::string VELOCITY_FRAGMENT_SHADER_PATH = "../src/shaders/velocity.frag";
} // namespace

namespace Renderer {
GLuint getShaderProgram(const bool useDensityShader) {
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

std::vector<float> getDensityVertices(SimulationSettings *settings, FluidMatrix *matrix) {
    const int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 3);

    const float scalingFactorInv = 1.0f / settings->scalingFactor;
    const int matrixSize = matrix->size;

    for (int i = 0; i < n; i++) {
        const int i_scaled = static_cast<int>(i * scalingFactorInv) * matrixSize;

        for (int j = 0; j < n; j++) {
            const int idx = i_scaled + static_cast<int>(j * scalingFactorInv);
            const int vertexIdx = 3 * (i * n + j);

            vertices[vertexIdx] = j;
            vertices[vertexIdx + 1] = i;
            vertices[vertexIdx + 2] = matrix->density[idx];
        }
    }

    normalizeDensityVertices(settings, vertices, n);

    return vertices;
}

std::vector<float> getVelocityVertices(SimulationSettings *settings, FluidMatrix *matrix) {
    const int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 4);

    const float scalingFactorInv = 1.0f / settings->scalingFactor;
    const int matrixSize = matrix->size;

    for (int i = 0; i < n; i++) {
        const int i_scaled = static_cast<int>(i * scalingFactorInv) * matrixSize;

        for (int j = 0; j < n; j++) {
            const int idx = i_scaled + static_cast<int>(j * scalingFactorInv);
            const int vertexIdx = 4 * (i * n + j);

            vertices[vertexIdx] = j;
            vertices[vertexIdx + 1] = i;
            vertices[vertexIdx + 2] = matrix->Vx[idx];
            vertices[vertexIdx + 3] = matrix->Vy[idx];
        }
    }

    normalizeSpeedVertices(settings, vertices, n);

    return vertices;
}

void linkDensityVerticesToBuffer(float *vertices, int len) {
    glBufferData(GL_ARRAY_BUFFER, len * sizeof(float), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
}

void linkVelocityVerticesToBuffer(float *vertices, int len) {
    glBufferData(GL_ARRAY_BUFFER, len * sizeof(float), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
}

void normalizeDensityVertices(SimulationSettings *settings, std::vector<float> &vertices, int n) {
    const float normFactor = 2.0f / (settings->viewportSize - 1);
    float *v = vertices.data();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const int vertIdx = 3 * FluidMatrix::index(i, j, n);

            v[vertIdx] = (v[vertIdx] * normFactor) - 1.0f;
            v[vertIdx + 1] = 1 - (v[vertIdx + 1] * normFactor);
        }
    }
}

void normalizeSpeedVertices(SimulationSettings *settings, std::vector<float> &vertices, int n) {
    const float normFactor = 2.0f / (settings->viewportSize - 1);
    float *v = vertices.data();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const int vertIdx = 4 * FluidMatrix::index(i, j, n);

            v[vertIdx] = (v[vertIdx] * normFactor) - 1.0f;
            v[vertIdx + 1] = 1 - (v[vertIdx + 1] * normFactor);
        }
    }
}

std::optional<GLuint> compileShader(const std::string &shaderSource, GLenum shaderType) {
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
} // namespace Renderer

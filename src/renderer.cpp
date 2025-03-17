#include "renderer.hpp"

namespace {
/// Shader paths struct to hold vertex and fragment shader paths
struct ShaderPaths {
    std::string vertexPath;
    std::string fragmentPath;

    ShaderPaths(std::string vert, std::string frag) : vertexPath(std::move(vert)), fragmentPath(std::move(frag)) {}
};

const std::unordered_map<SimulationAttribute, ShaderPaths> SHADER_PATHS = {{DENSITY, ShaderPaths("../src/shaders/density.vert", "../src/shaders/density.frag")},
                                                                           {VELOCITY, ShaderPaths("../src/shaders/velocity.vert", "../src/shaders/velocity.frag")},
                                                                           {VORTICITY, ShaderPaths("../src/shaders/vorticity.vert", "../src/shaders/vorticity.frag")}};
/// Cache for compiled shaders and shader programs
struct ShaderCache {
    std::unordered_map<std::string, GLuint> compiledShaders;
    std::unordered_map<SimulationAttribute, GLuint> shaderPrograms;

    ~ShaderCache() {
        for (const auto &program: std::views::values(shaderPrograms)) {
            glDeleteProgram(program);
        }
        shaderPrograms.clear();
        for (const auto &shader: std::views::values(compiledShaders)) {
            glDeleteShader(shader);
        }
        compiledShaders.clear();
    }
};

ShaderCache &getShaderCache() {
    static ShaderCache cache;
    return cache;
}

/// Get or compile shader from file
/// @param path shader file path
/// @param shaderType shader type
/// @return compiled shader or nullopt if compilation failed
std::optional<GLuint> getOrCompileShader(const std::string &path, const GLenum shaderType) {
    auto &[compiledShaders, shaderPrograms] = getShaderCache();

    // check if shader is already compiled
    if (const auto it = compiledShaders.find(path); it != compiledShaders.end()) {
        return it->second;
    }

    // compile new shader
    const auto shaderSource = Utils::readFile(path);
    if (!shaderSource) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to read shader source file");
        return std::nullopt;
    }

    const GLuint shader = glCreateShader(shaderType);
    const char *shaderSourceCStr = shaderSource->c_str();
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

    // cache compiled shader
    compiledShaders[path] = shader;
    return shader;
}
} // namespace

namespace Renderer {
GLuint getShaderProgram(const SimulationAttribute attribute) {
    auto &[compiledShaders, shaderPrograms] = getShaderCache();

    // check if shader program is already cached
    if (const auto it = shaderPrograms.find(attribute); it != shaderPrograms.end()) {
        glUseProgram(it->second);
        return it->second;
    }

    // Get shader paths for this attribute
    const auto pathsIt = SHADER_PATHS.find(attribute);
    if (pathsIt == SHADER_PATHS.end()) {
        log(Utils::LogLevel::ERROR, std::cerr, std::format("No shader paths defined for attribute {}", static_cast<int>(attribute)));
        return 0;
    }

    // Get vertex and fragment shaders
    const auto &paths = pathsIt->second;
    const auto vertexShader = getOrCompileShader(paths.vertexPath, GL_VERTEX_SHADER);
    const auto fragmentShader = getOrCompileShader(paths.fragmentPath, GL_FRAGMENT_SHADER);
    if (!vertexShader || !fragmentShader) {
        log(Utils::LogLevel::ERROR, std::cerr, "Failed to compile shaders");
        return 0;
    }

    // create and link shader program
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

    // cache shader program
    shaderPrograms[attribute] = shaderProgram;
    glUseProgram(shaderProgram); // TODO: maybe move to caller
    return shaderProgram;
}

int getVertexComponentCount(const SimulationAttribute attribute) {
    switch (attribute) {
        case DENSITY: return 3;   // x, y, density
        case VELOCITY: return 4;  // x, y, vx, vy
        case VORTICITY: return 3; // x, y, vorticity
        default: log(Utils::LogLevel::ERROR, std::cerr, std::format("Unknown vertex component count for attribute {}", static_cast<int>(attribute))); return 0;
    }
}

std::vector<float> getDensityVertices(const SimulationSettings *settings, const FluidMatrix *matrix) {
    const int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 3);
    const float scalingFactorInv = 1.0f / settings->scalingFactor;
    const float normFactor = 2.0f / (settings->viewportSize - 1);
    const int matrixSize = matrix->size;

#pragma omp parallel for schedule(guided) collapse(2)
    for (int i = 0; i < n; i++) {
        const int i_scaled = static_cast<int>(i * scalingFactorInv) * matrixSize;

        for (int j = 0; j < n; j++) {
            const int idx = i_scaled + static_cast<int>(j * scalingFactorInv);
            const int vertexIdx = 3 * (i * n + j);

            // Generate vertices x, y, density
            vertices[vertexIdx] = j;
            vertices[vertexIdx + 1] = i;
            vertices[vertexIdx + 2] = matrix->density[idx];

            // Normalize coordinates
            vertices[vertexIdx] = (vertices[vertexIdx] * normFactor) - 1.0f;
            vertices[vertexIdx + 1] = 1 - (vertices[vertexIdx + 1] * normFactor);
        }
    }

    return vertices;
}

std::vector<float> getVelocityVertices(const SimulationSettings *settings, const FluidMatrix *matrix) {
    const int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 4);
    const float scalingFactorInv = 1.0f / settings->scalingFactor;
    const float normFactor = 2.0f / (settings->viewportSize - 1);
    const int matrixSize = matrix->size;

#pragma omp parallel for schedule(guided) collapse(2)
    for (int i = 0; i < n; i++) {
        const int i_scaled = static_cast<int>(i * scalingFactorInv) * matrixSize;

        for (int j = 0; j < n; j++) {
            const int idx = i_scaled + static_cast<int>(j * scalingFactorInv);
            const int vertexIdx = 4 * (i * n + j);

            // Generate vertices x, y, vx, vy
            vertices[vertexIdx] = j;
            vertices[vertexIdx + 1] = i;
            vertices[vertexIdx + 2] = matrix->vX[idx];
            vertices[vertexIdx + 3] = matrix->vY[idx];

            // Normalize coordinates
            vertices[vertexIdx] = (vertices[vertexIdx] * normFactor) - 1.0f;
            vertices[vertexIdx + 1] = 1 - (vertices[vertexIdx + 1] * normFactor);
        }
    }

    return vertices;
}

std::vector<float> getVorticityVertices(const SimulationSettings *settings, const FluidMatrix *matrix) {
    const int n = settings->viewportSize;
    std::vector<float> vertices(n * n * 3);
    const float scalingFactorInv = 1.0f / settings->scalingFactor;
    const float normFactor = 2.0f / (settings->viewportSize - 1);
    const int matrixSize = matrix->size;

#pragma omp parallel for schedule(guided) collapse(2)
    for (int i = 0; i < n; i++) {
        const int i_scaled = static_cast<int>(i * scalingFactorInv) * matrixSize;

        for (int j = 0; j < n; j++) {
            const int idx = i_scaled + static_cast<int>(j * scalingFactorInv);
            const int vertexIdx = 3 * (i * n + j);

            // Generate vertices x, y, vorticity
            vertices[vertexIdx] = j;
            vertices[vertexIdx + 1] = i;
            vertices[vertexIdx + 2] = matrix->vorticity[idx];

            // Normalize coordinates
            vertices[vertexIdx] = (vertices[vertexIdx] * normFactor) - 1.0f;
            vertices[vertexIdx + 1] = 1 - (vertices[vertexIdx + 1] * normFactor);
        }
    }

    return vertices;
}
} // namespace Renderer

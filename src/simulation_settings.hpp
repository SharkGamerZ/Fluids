#pragma once

#include "utils.hpp"

enum ExecutionMode { SERIAL, OPENMP, CUDA };
enum SimulationAttribute { DENSITY_ATTRIBUTE, VELOCITY_ATTRIBUTE };

struct SimulationSettings {
    // Used to set up the simulation
    int matrixSize = 400;
    int scalingFactor = 2;
    int viewportSize = matrixSize * scalingFactor;
    int chunkSize = 9;

    // Shown in GUI
    float viscosity = 0.0000001f;
    float deltaTime = 0.2f;
    int executionMode = SERIAL;
    int simulationAttribute = DENSITY_ATTRIBUTE;
    bool isSimulationRunning = false;

    void ResetMatrix() {
        log(Utils::LogLevel::INFO, std::cout, "Matrix reset!");

        viscosity = 0.0000001f;
        deltaTime = 0.2f;

        // TODO rest matrix
    }
};

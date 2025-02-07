#pragma once

enum ExecutionMode { SERIAL, OPENMP, CUDA };
enum SimulationAttribute { DENSITY, VELOCITY };

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
    int simulationAttribute = DENSITY;

    // Mouse state
    double xpos, ypos, xposPrev, yposPrev, deltax, deltay, xposScaled, yposScaled, mouseTime, mouseTimePrev, mouseTimeDelta;

    // Keybind action flags
    bool isSimulationRunning = false;
    bool resetSimulation = false;
    bool windMachine = false;
};

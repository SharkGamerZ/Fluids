#pragma once

enum ExecutionMode { SERIAL, OPENMP, CUDA };
enum SimulationAttribute { DENSITY, VELOCITY, VORTICITY };

struct SimulationSettings {
    // Used to set up the simulation
    int matrixSize = 400;
    int scalingFactor = 2;
    int viewportSize = matrixSize * scalingFactor;
    int chunkSize = 9;

    // Shown in GUI
    float viscosity = 0.0000001f;
    float deltaTime = 0.2f;
    float mouse_density = 1.0f;
    float mouse_velocity = 1.0f;
    ExecutionMode executionMode = SERIAL;
    ExecutionMode executionModePrev = SERIAL;
    SimulationAttribute simulationAttribute = DENSITY;

    // Mouse state
    double xpos, ypos, xposPrev, yposPrev, deltax, deltay, xposScaled, yposScaled, mouseTime, mouseTimePrev, mouseTimeDelta;

    // Keybind action flags
    bool isSimulationRunning = false;
    bool resetSimulation = false;
    bool frameSimulation = false;
    bool windMachine = false;
};

#ifndef FLUIDS_FLUIDMATRIX_H
#define FLUIDS_FLUIDMATRIX_H

#include "../utils.hpp"

#include <vector>
#include <ostream>
#include <iostream>
#include <omp.h>


#define SERIAL 0
#define OPENMP 1
#define CUDA 2
inline int executionMode = SERIAL;


#define IX(i, j) ((j) + (i) * N) ///< index of the matrix

const int ITERATIONS = 20; ///< number of iterations
enum Axis {
    X, Y, ZERO
}; ///< axis of the matrix

class FluidMatrix {
public:
    int size; ///< size of the matrix
    float dt; ///< delta time
    float diff; ///< diffusion
    float visc; ///< viscosity

    std::vector<float> density; ///< density
    std::vector<float> density0; ///< density at previous step

    std::vector<float> Vx; ///< velocity on x axis
    std::vector<float> Vy; ///< velocity on y axis

    std::vector<float> Vx0; ///< velocity on x axis at previous step
    std::vector<float> Vy0; ///< velocity on y axis at previous step

    FluidMatrix(int size, float diffusion, float viscosity, float dt);

    ~FluidMatrix();

    /**
     * Print the matrix using the output operator
     */
    friend std::ostream &operator<<(std::ostream &os, const FluidMatrix &matrix);


    /**
     * Reset the matrix
     */
    void reset();

    // Instance methods
    /**
     * Simulate a time-step
     */
    void step();

    /**
     * Add density to the matrix at the given position
     * @param x coordinate x
     * @param y coordinate y
     * @param amount amount of density to add
     */
    void addDensity(int x, int y, float amount);

    /**
     * Add velocity to the matrix at the given position
     * @param x coordinate x
     * @param y coordinate y
     * @param amountX amount of velocity to add on x axis
     * @param amountY amount of velocity to add on y axis
     */
    void addVelocity(int x, int y, float amountX, float amountY);

private:
    /**
     * Diffuse the matrix
     * @param mode x or y axis
     * @param value value to diffuse
     * @param oldValue value at previous step
     * @param diffusion diffusion
     * @param dt delta time
     */
    void diffuse(Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt) const;

    /**
     * OpenMP version of diffuse
     * @param mode x or y axis
     * @param value value to diffuse
     * @param oldValue value at previous step
     * @param diffusion diffusion
     * @param dt delta time
     */
    void omp_diffuse(Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt) const;

    /**
     * Advect the matrix
     * @param mode x or y axis
     * @param d value to advect
     * @param d0 value at previous step
     * @param vX velocity on x axis
     * @param vY velocity on y axis
     * @param dt delta time
     */
    void advect(Axis mode, std::vector<float> &d, std::vector<float> &d0, std::vector<float> &vX, std::vector<float> &vY, float dt) const;

    /**
     * Project the matrix
     * @param vX velocity on x axis
     * @param vY velocity on y axis
     * @param p
     * @param div
     */
    void project(std::vector<float> &vX, std::vector<float> &vY, std::vector<float> &p, std::vector<float> &div) const;

    /**
     * Set the boundary of the matrix
     * @param mode x or y axis
     * @param attr attribute to set
     */
    void set_bnd(Axis mode, std::vector<float> &attr) const;

    /**
     * Solve the linear equation
     * @param mode x or y axis
     * @param value value to solve
     * @param oldValue value at previous step
     * @param diffusionRate diffusion rate
     */
    void lin_solve(Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusionRate) const;


    void omp_lin_solve(Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusionRate) const;
};


#endif //FLUIDS_FLUIDMATRIX_H

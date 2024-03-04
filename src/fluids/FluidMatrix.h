#ifndef FLUIDS_FLUIDMATRIX_H
#define FLUIDS_FLUIDMATRIX_H

#include "../utils.hpp"

#include <cmath>
#include <iostream>
#include <omp.h>
#include <ostream>
#include <vector>


#define SERIAL 0
#define OPENMP 1
#define CUDA 2
extern int executionMode;

const int ITERATIONS = 15; ///< number of iterations
enum Axis { X, Y, ZERO };  ///< axis of the matrix

class FluidMatrix {
public:
    uint32_t size; ///< size of the matrix
    double dt;     ///< delta time
    double diff;   ///< diffusion
    double visc;   ///< viscosity

    std::vector<double> density;  ///< density
    std::vector<double> density0; ///< density at previous step

    std::vector<double> Vx; ///< velocity on x axis
    std::vector<double> Vy; ///< velocity on y axis

    std::vector<double> Vx0; ///< velocity on x axis at previous step
    std::vector<double> Vy0; ///< velocity on y axis at previous step

    FluidMatrix(uint32_t size, double diffusion, double viscosity, double dt);

    ~FluidMatrix();


    /**
     * Get the index of the matrix
     * @param i coordinate x
     * @param j coordinate y
     * @param matrix_size size of the matrix
     * @return index
     */
    [[nodiscard]] static inline uint32_t index(uint32_t i, uint32_t j, uint32_t matrix_size);

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
     * Simulate a time-step using OpenMP
     */
    void OMPstep();

    /**
     * Add density to the matrix at the given position
     * @param x coordinate x
     * @param y coordinate y
     * @param amount amount of density to add
     */
    void addDensity(uint32_t x, uint32_t y, double amount);

    /**
     * Add velocity to the matrix at the given position
     * @param x coordinate x
     * @param y coordinate y
     * @param amountX amount of velocity to add on x axis
     * @param amountY amount of velocity to add on y axis
     */
    void addVelocity(uint32_t x, uint32_t y, double amountX, double amountY);

private:
    /**
     * Diffuse the matrix
     * @param mode x or y axis
     * @param value value to diffuse
     * @param oldValue value at previous step
     * @param diffusion diffusion
     * @param dt delta time
     */
    void diffuse(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) const;

    /**
     * OpenMP version of diffuse
     * @param mode x or y axis
     * @param value value to diffuse
     * @param oldValue value at previous step
     * @param diffusion diffusion
     * @param dt delta time
     */
    void omp_diffuse(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) const;

    /**
     * Advect the matrix
     * @param mode x or y axis
     * @param d value to advect
     * @param d0 value at previous step
     * @param vX velocity on x axis
     * @param vY velocity on y axis
     * @param dt delta time
     */
    void advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const;

    /**
     * Advect the matrix
     * @param mode x or y axis
     * @param d value to advect
     * @param d0 value at previous step
     * @param vX velocity on x axis
     * @param vY velocity on y axis
     * @param dt delta time
     */
    void omp_advect(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) const;

    /**
     * Project the matrix
     * @param vX velocity on x axis
     * @param vY velocity on y axis
     * @param p
     * @param div
     */
    void project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;

    /**
     * Set the boundary of the matrix
     * @param mode x or y axis
     * @param attr attribute to set
     */
    void set_bnd(Axis mode, std::vector<double> &attr) const;

    /**
     * Set the boundary of the matrix
     * @param mode x or y axis
     * @param attr attribute to set
     */
    void omp_set_bnd(Axis mode, std::vector<double> &attr) const;

    /**
     * Solve the linear equation
     * @param mode x or y axis
     * @param value value to solve
     * @param oldValue value at previous step
     * @param diffusionRate diffusion rate
     */
    void lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;


    void omp_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;


    void fadeDensity(std::vector<double> &density) const;
};


#endif //FLUIDS_FLUIDMATRIX_H

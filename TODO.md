# TODO

- [ ] fix `-Xcompiler=-flto` being applied to other compilers instead of nvcc
- [x] Add CUDA_init in FluidMatrix to preallocate memory for CUDA matrix
- Parameters to add:
    - [x] slider for mouse density added on click
    - [x] slider for mouse velocity added on move
- [x] Add shader caching
- [ ]  Change the diffuse, adding the cRecip as a parameter, so that it can be used inside the advect
- [ ]  See how many iterations are needed in serial Gauss-Siedel vs. parallel Jacobi to converge
- [ ]  Move the memcpy only at the start/end of the CUDA section, getting and setting the various matrixes in RAM only when needed, and using the device pointers in the CUDA_step
- [ ]  Implement CUDA_CalculateVorticity
- [ ]  Modify Vorticity shader to user two colors to represent negative and positive vorticity

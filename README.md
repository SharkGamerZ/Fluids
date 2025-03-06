![C++ Badge](https://img.shields.io/badge/C%2B%2B-17-blue)
![CUDA Badge](https://img.shields.io/badge/CUDA-Enabled-green)
![Personal Badge](https://wakapi.dev/api/badge/SharkGamerZ/interval:any/project:Fluids)

# Fluids
A high-performance fluid simulation project for the **Multicore 2023/24** course.

## 1. Project Overview
This project simulates incompressible fluid dynamics using **Eulerian grids** and a numerical solver. The implementation supports three execution modes: **Serial (CPU), OpenMP (Multithreaded CPU), and CUDA (GPU-accelerated)**.

### **Graphics**
The visualization is powered by **OpenGL**, using custom **vertex and fragment shaders** to render the simulation efficiently. The rendering pipeline includes:
- **GLFW** for window and input management.
- **GLEW** to handle OpenGL extensions.
- **Dear ImGui** for real-time parameter tuning.

_💡 [Insert shader code snippet or diagram here]_  

### **Physics**
The simulation is based on **Jos Stam’s Stable Fluids method**, ensuring numerical stability in real-time applications. The main steps include:
- **Advection**: Moves the fluid according to its velocity field, solved via a **semi-Lagrangian** method.
- **Diffusion**: Spreads velocity values using **implicit diffusion**, solved with a linear system.
- **Projection**: Ensures the velocity field is divergence-free using a **pressure correction step**.

_📜 Reference: [Real-Time Fluid Dynamics for Games (Jos Stam, 2003)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf)_  

### **Implementation**
This project is implemented in **three different ways**:
- **Serial (CPU-only)**: Uses a naive Gauss-Seidel solver.
- **OpenMP (Multicore CPU)**: Switches to a **Jacobi solver** for parallelization.
- **CUDA (GPU-accelerated)**: Optimized with memory management to reduce RAM-VRAM transfers.

_💡 [Insert flowchart or pseudocode for implementation comparison]_  

---

## 2. Running the Simulator

### **Dependencies**
Ensure you have the following installed:
- GLFW
- OpenGL
- GLEW
- OpenMP (for CPU parallelization)
- CUDA Toolkit (for GPU acceleration)

### **Building the Project**
```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### **Running the Simulator**
```sh
./fluids_sim
```

### **Settings and Visualization Modes**
The simulation offers different visualization options:
- **Diffusion Mode**: Shows how particles spread over time.
- **Velocity Field**: Visualizes the motion of the fluid.
- **Vorticity Map**: Highlights areas of rotational motion.

_💡 [Leave space for screenshots of each mode]_  

### **Keybindings**
- `R` - Reset simulation
- `P` - Pause/Resume simulation
- `+/-` - Increase/Decrease timestep

_💡 [Provide a complete list of controls]_  

---

## 3. Performance & Scaling
This section analyzes performance across different implementations, focusing on:
- **Speedup from OpenMP and CUDA over Serial**
- **VRAM vs RAM bandwidth optimizations**
- **Computational cost of Jacobi vs Gauss-Seidel solvers**

_📊 [Leave space for benchmark graphs comparing Serial, OpenMP, and CUDA]_  

---

🚀 **Stay tuned for updates!** Feel free to contribute or report issues.  


# Fluid rendering simulation

---

Using CUDA + OpenGL to simulate fluids. Render the windows using Dear ImGui.

Keynote: we're not using particles, but a grid-based approach.
Each cell can contain multiple fluids, and emulates their behaviour.

## Features

- [ ] 2D Fluid simulation
- [ ] Allow solids, liquids and gases
- [ ] Define interactions between particles
- [ ] Adding parameters to the simulation

## Class graph of the particles

- Particle
    - Solid
        - Movable (sand, etc.)
        - Immovable (stone, etc.)
    - Liquid (water, oil, etc.)
    - Gas (air, smoke, etc.)

Possibly add a Fluid class to handle both liquids and gases.

## Particle properties

Each particle must hae the following properties:

- Position
- Mass
- Velocity
- Acceleration
- Density
- Pressure
- Temperature

More stuff can be added later.

## Particle interactions

Particles can interact with each other.

- Solids usually fall down, but can be pushed by other particles.
- Liquids can flow and be pushed by other particles.
- Gases can flow and push other particles, usually upwards.


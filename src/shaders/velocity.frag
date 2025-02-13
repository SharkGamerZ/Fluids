#version 330 core

// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
in vec2 vel;
out vec4 FragColor;

float mapRange(float val, float inMin, float inMax, float outMin, float outMax) {
    // Division by zero check
    float range = inMax - inMin;
    if (abs(range) < 0.0001) return outMin;

    // Clamp to [0, 1] and map to output range
    float normalized = clamp((val - inMin) / range, 0.0, 1.0);
    return mix(outMin, outMax, normalized);
}

void main() {
    const float VEL_MIN = -0.05;
    const float VEL_MAX = 0.05;

    vec2 mappedVel = vec2(
        mapRange(vel.x, VEL_MIN, VEL_MAX, 0.0, 1.0),
        mapRange(vel.y, VEL_MIN, VEL_MAX, 0.0, 1.0)
    );

    FragColor = vec4(1.0, mappedVel.y, mappedVel.x, 1.0);
}

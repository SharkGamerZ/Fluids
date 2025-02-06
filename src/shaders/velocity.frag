#version 330 core

// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
in vec2 vel;
out vec4 FragColor;

float MapToRange(float val, float minIn, float maxIn, float minOut, float maxOut) {
    float x = (val - minIn) / (maxIn - minIn);
    float result = minOut + (maxOut - minOut) * x;
    return (result < minOut) ? minOut : (result > maxOut) ? maxOut : result;
}

void main()
{
    vec2 velocites;
    velocites.x = MapToRange(vel.x, -0.05, 0.05, 0.0, 1.0);
    velocites.y = MapToRange(vel.y, -0.05, 0.05, 0.0, 1.0);
    FragColor = vec4(1.0, velocites.y, velocites.x, 1.0);
}

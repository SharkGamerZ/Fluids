#version 330 core

// Vertex Shader
//      Prende in input la posizione dei
//      vertici e li passa al Fragment Shader
layout (location = 0) in vec3 aPos;
out float ourColorAlpha;

void main()
{
    ourColorAlpha = aPos.z;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}

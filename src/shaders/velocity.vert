#version 330 core

// Vertex Shader
//      Prende in input la posizione dei
//      vertici e li passa al Fragment Shader
layout (location = 0) in vec4 aPos;
out vec2 vel;

void main()
{
    vel.x = aPos.z;
    vel.y = aPos.w;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}

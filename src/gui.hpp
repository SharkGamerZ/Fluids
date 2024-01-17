#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

static bool simulazioneIsRunning = false;
static ImVec4 clear_color = ImVec4(0.20f, 0.10f, 0.10f, 1.00f);

int openGUI();
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
unsigned int getShaderProgram();

// OpenGL shaders

// Vertex Shader
//      Prende in input la posizione dei
//      vertici e li passa al Fragment Shader
static const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";


// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
static const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";
#ifndef GUI_HPP_
#define GUI_HPP_

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "fluid2d.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers



#define viewportSize 512
inline int display_w, display_h;

static bool dens = true;
inline bool simulazioneIsRunning = false;
inline ImVec4 clear_color = ImVec4(0.20f, 0.10f, 0.10f, 1.00f);

int openGUI();
GLFWwindow *setupWindow(int width, int height);
ImGuiIO *setupImGui(GLFWwindow *window);

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void renderImGui(ImGuiIO *io);


unsigned int getShaderProgram();
void setupBufferAndArray(uint* VBO, uint* VAO);

void linkVerticestoBuffer(float* vertices, int len);
void drawMatrix(FluidMatrix *matrix, int N);




void printMatrix(FluidMatrix *matrix, int N);
void printVertices(float *vertices, int len);
void normalizeVertices(float *vertices, int N);

// OpenGL shaders

// Vertex Shader
//      Prende in input la posizione dei
//      vertici e li passa al Fragment Shader
inline const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
        "vertexColor = aPos.z;\n"
    "}\0";

// inline const char *vertexShaderSourceNorm = "#version 330 core\n"
//     "layout (location = 0) in vec3 aPos;\n"
//     "uniform vec2 viewPort;\n"
//     "void main()\n"
//     "{\n"
//     "   float x = (aPos.x / (viewPort.x / 2)) - 1;\n"
//     "   float y = (aPos.y / (viewPort.y / 2)) - 1;\n"
//     "   gl_Position = vec4(x, y, 0.0, aPos.z);\n"
//     "}\0";

    


// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
inline const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform vec4 ourColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = ourColor;\n"
    "}\n\0";

#endif
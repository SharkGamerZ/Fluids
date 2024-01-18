#ifndef GUI_HPP_
#define GUI_HPP_

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "compute.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

inline int display_w, display_h;


inline bool simulazioneIsRunning = false;
inline ImVec4 clear_color = ImVec4(0.20f, 0.10f, 0.10f, 1.00f);

int openGUI();
GLFWwindow *setupWindow(int width, int height);
ImGuiIO *setupImGui(GLFWwindow *window);

unsigned int getShaderProgram();
uint linkVerticestoBuffer(float* vertices, int len);

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

void renderImGui(ImGuiIO *io);
void render(int width, int height, cell *matrix, uint shaderProgram);


// OpenGL shaders

// Vertex Shader
//      Prende in input la posizione dei
//      vertici e li passa al Fragment Shader
inline const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";

    


// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
inline const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

#endif
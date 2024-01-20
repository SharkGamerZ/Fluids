#ifndef GUI_HPP_
#define GUI_HPP_

#define FM_OLD 1 ///< Set to 1 to use the C version, 0 to use the C++ version

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>

#if FM_OLD
#include "fluid2d.hpp"
#else
#include "FluidMatrix.h"
#endif

#include "utils.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

int openGUI();

/**
 * Funzione per setuppare la finestra e il context di IMGui
 * @return Il puntatore alla finestra o nullptr se c'è stato un errore
 */
GLFWwindow *setupWindow(int width, int height);

/**
 * Funzione per fare il setup di IMGui
 * @param window La finestra di GLFW
 * @return Il puntatore al context di IMGui
 */
ImGuiIO *setupImGui(GLFWwindow *window);

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);

/**
 * Funzione per il rendering di IMGui
 * @param io il context di ImGui
 * @param matrix La matrice a cui (eventualmente) aggiornare i valori
 */
void renderImGui(ImGuiIO *io, FluidMatrix *matrix);

/**
 * Funzione per la creazione del programma di shader
 * @return L'ID del programma di shader o 0 se c'è stato un errore
 */
unsigned int getShaderProgram();

/**
 * Funzione per andare a creare il Vertex Buffer e il Vertex Array
 * @param VBO Il puntatore al Vertex Buffer
 * @param VAO Il puntatore al Vertex Array
 */
void setupBufferAndArray(uint *VBO, uint *VAO);

/**
 * Funzione per andare a linkare i vertici che gli vengono passati, al Vertex Buffer e successivamente al Vertex Array
 * @param vertices I vertici da linkare
 * @param len La lunghezza del vettore di vertici
 */
void linkVerticestoBuffer(float *vertices, int len);

/**
 *  Funzione per il rendering della matrice
 * @param matrix La matrice da renderizzare
 * @param N La dimensione della matrice
 */
void drawMatrix(FluidMatrix *matrix, int N);


void printMatrix(FluidMatrix *matrix, int N);

void printVertices(float *vertices, int len);

void normalizeVertices(float *vertices, int N);

// OpenGL shaders

// inline const char *vertexShaderSourceNorm = "#version 330 core\n"
//     "layout (location = 0) in vec3 aPos;\n"
//     "uniform vec2 viewPort;\n"
//     "void main()\n"
//     "{\n"
//     "   float x = (aPos.x / (viewPort.x / 2)) - 1;\n"
//     "   float y = (aPos.y / (viewPort.y / 2)) - 1;\n"
//     "   gl_Position = vec4(x, y, 0.0, aPos.z);\n"
//     "}\0";

#endif

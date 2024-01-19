#ifndef GUI_HPP_
#define GUI_HPP_

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>

#include "fluid2d.hpp"

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
 * @param mode Cosa far visualzzare
 */
void drawMatrix(FluidMatrix *matrix, int N, int mode);


void printMatrix(FluidMatrix *matrix, int N);

void printVertices(float *vertices, int len);

void normalizeVertices(float *vertices, int N);

// OpenGL shaders

// Vertex Shader
//      Prende in input la posizione dei
//      vertici e li passa al Fragment Shader
inline const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "out float ourColorAlpha;\n"
    "void main()\n"
    "{\n"
    "   ourColorAlpha = aPos.z;\n"
    "   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
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
    "in float ourColorAlpha;\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(ourColorAlpha, ourColorAlpha, ourColorAlpha, ourColorAlpha);\n"
    "}\n\0";

#endif

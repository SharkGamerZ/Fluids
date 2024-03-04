#ifndef GUI_HPP_
#define GUI_HPP_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>


#include "fluids/FluidMatrix.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "utils.hpp"
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
 *  Funzione per il rendering della matrice
 * @param matrix La matrice da renderizzare
 */
void drawMatrix(FluidMatrix *matrix);


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
void linkDensityVerticestoBuffer(float *vertices, int len);


void linkVelocityVerticestoBuffer(float *vertices, int len);

float *getDensityVertices(FluidMatrix *matrix);

float *getVelocityVertices(FluidMatrix *matrix);


void normalizeVertices(float *vertices, int N);

void normalizeSpeedVertices(float *vertices, int N);

#endif

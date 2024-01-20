#include "gui.hpp"

// Defined Valuse
#define DENSITY_ATTRIBUTE 0
#define VELOCITY_ATTRIBUTE 1

// Golbal variables
const int viewportSize = 128;
const ImVec4 clear_color = ImVec4(0.20f, 0.10f, 0.10f, 1.00f);

bool frameSimulation = false;
bool simulazioneIsRunning = false;
int simulationAttribute = DENSITY_ATTRIBUTE;
int display_w, display_h;

// Variabili per il mouse
double xpos, ypos, xpos0, ypos0, deltaX, deltaY;
double mouseTime, mouseTime0, mouseDeltaTime;

int openGUI()
{
    // Setup della finestra e del context di IMGui
    GLFWwindow *window = setupWindow(viewportSize, viewportSize);
    ImGuiIO *io = setupImGui(window);

    // Chiamata a glewInit per andare a caricare tutte le funzioni di OpenGL
    glewInit();

    // Prende l'id del programma di shader
    uint32_t shaderProgram = getShaderProgram();
    if (shaderProgram == 0) return EXIT_FAILURE;

    int size = viewportSize;

    // Creiamo la matrice di fluidi e gli aggiungiamo densità in una cella
#if FM_OLD
    auto matrix = FluidMatrixCreate(size, 0.0f, 1.0f, 0.2f);
    FluidMatrixAddDensity(matrix, size/2, size/2, 10.0f);
#else
    FluidMatrix matrix = FluidMatrix(size, 1.0f, 1.0f, 0.2f);
    matrix.addDensity(size/2, size/2, 10.0);
    matrix.addVelocity(size/2, size/2, 10.0, 10.0);
#endif
    // Creiamo il Vertex Buffer e il Vertex Array
    uint32_t VBO, VAO;
    setupBufferAndArray(&VBO, &VAO);


    // Ciclo principale
    while (!glfwWindowShouldClose(window)) {
        // Rendering di IMGui
#if FM_OLD
        renderImGui(io, matrix);
#else
        renderImGui(io, &matrix);
#endif

        // Aggiunge densità e velocità con il mouse
        int mouseLeftButtonState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        if (mouseLeftButtonState == GLFW_PRESS)
        {
            glfwGetCursorPos(window, &xpos, &ypos);

            if (xpos >= 0 && xpos <= display_w && ypos >= 0 && ypos <= display_h)
            {
                // Aggiunge densità
#if FM_OLD
                FluidMatrixAddDensity(matrix, xpos, ypos, 100.0f);
#else
                matrix.addDensity(xpos, ypos, 100.0f);
#endif

                // Calcola la velocità
                mouseTime = glfwGetTime();
                mouseDeltaTime = mouseTime - mouseTime0;
                deltaX = xpos - xpos0;
                deltaY = ypos - ypos0;
                mouseTime0 = mouseTime;
                xpos0 = xpos;
                ypos0 = ypos;

                // Aggiunge velocità
#if FM_OLD
                FluidMatrixAddVelocity(matrix, xpos, ypos, deltaX, deltaY);
#else
                matrix.addVelocity(xpos, ypos, deltaX, deltaY);
#endif
            }


        }


        // Simulazione
        if (simulazioneIsRunning || frameSimulation)
        {
#if FM_OLD
            FluidMatrixStep(matrix);
#else
            matrix.step();
#endif
            // std::cout<< "DEBUG: A" << std::endl;
            frameSimulation = false;
        }

#if FM_OLD
        drawMatrix(matrix, size);
#else
        drawMatrix(&matrix, size);
#endif

        // In caso di resize della finestra, aggiorna le dimensioni del viewport
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);


        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        // Rendering della matrice
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, size * size * 3);

        // Rendering di IMGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


        // Swappa i buffer e controlla se ci sono stati eventi (input)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}




// Funzione per la gestione degli input
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Se l'utente preme il tasto ESC, chiude la finestra
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Se l'utente preme il tasto SPACE, inverte lo stato della simulazione
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        simulazioneIsRunning = !simulazioneIsRunning;

    if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
        frameSimulation = true;

}


uint32_t getShaderProgram() {
    int  success;
    char infoLog[512];

    // Get the shader source code from the GLSL files
    std::string vertexShaderSource = readFile("../src/shaders/vertexShader.vert");
    const char* vertexShaderSourceCStr = vertexShaderSource.c_str();

    std::string fragmentShaderSource = readFile("../src/shaders/fragmentShader.frag");
    const char* fragmentShaderSourceCStr = fragmentShaderSource.c_str();

    // Creiamo l'id della vertexShader
    uint32_t vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    // Colleghiamo il codice della vertexShader all'id
    glShaderSource(vertexShader, 1, &vertexShaderSourceCStr, nullptr);
    glCompileShader(vertexShader);

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, sizeof(infoLog), nullptr, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }

    // Creiamo l'id della fragmentShader
    uint32_t fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    // Colleghiamo il codice della fragmentShader all'id
    glShaderSource(fragmentShader, 1, &fragmentShaderSourceCStr, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, sizeof(infoLog), nullptr, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }

    // Creiamo l'id del programma di shader
    uint32_t shaderProgram;
    shaderProgram = glCreateProgram();

    // Colleghiamo le due shader al programma
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Verifichiamo che il programma di shader sia stato creato correttamente
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, sizeof(infoLog), nullptr, infoLog);
        return 0;
    }


    // Usiamo il programma di shader
    glUseProgram(shaderProgram);



    // Eliminiamo le shader
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

GLFWwindow *setupWindow(int width, int height) {
    // Setup della finestra
    if (!glfwInit()) return nullptr;

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(width, height, "Fluids", nullptr, nullptr);
    if (window == nullptr) return nullptr;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Impostiamo le callback per la gestione degli input
    glfwSetKeyCallback(window, key_callback);

    return window;
}

ImGuiIO *setupImGui(GLFWwindow *window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    return &io;
}

void renderImGui(ImGuiIO *io, FluidMatrix *matrix) {
    // Avvia il frame di ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // Finestra per i controlli della simulazione
    {
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(350.0f, 200.0f));
        static float diffusione = 1.0f;
        static float deltaTime = 0.2f;
        static float temperatura = 0.0f;

        ImGui::Begin("Parametri di simulazione", nullptr, ImGuiWindowFlags_NoResize);
        ImGui::SliderFloat("Diffusione", &diffusione, 0.0f, 1.0f);
        matrix->diff = diffusione;


        ImGui::SliderFloat("TimeStep", &deltaTime, 0.0f, 1.0f);
        matrix->dt = deltaTime;


        ImGui::RadioButton("Densità", &simulationAttribute, DENSITY_ATTRIBUTE); ImGui::SameLine();
        ImGui::RadioButton("Velocità", &simulationAttribute, VELOCITY_ATTRIBUTE); ImGui::SameLine();

        // Buttons return true when clicked (most widgets return true when edited/activated)
        if (ImGui::Button("Avvio simulazione")) simulazioneIsRunning = !simulazioneIsRunning;


        ImGui::SameLine();
        ImGui::Text("Stato = %B", simulazioneIsRunning);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io->Framerate, io->Framerate);
        ImGui::End();
    }
}




void drawMatrix(FluidMatrix *matrix, int N) {
    // Creiamo un vettore di vertici per la matrice, grande N*N * 3 visto che ho 2 coordinate e 1 colore per ogni vertice
    float* vertices = (float*) calloc(sizeof(float), N * N * 3);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            vertices[3 * IX(i, j)] = i;
            vertices[3 * IX(i, j) + 1] = j;
            if (simulationAttribute == DENSITY_ATTRIBUTE)     vertices[3 * IX(i, j) + 2] = matrix->density[IX(i, j)];
            if (simulationAttribute == VELOCITY_ATTRIBUTE)    vertices[3 * IX(i, j) + 2] = abs(matrix->Vx[IX(i, j)]) +
                                                                                 abs(matrix->Vy[IX(i, j)]);
        }
    }

    // Normalizziamo i vertici da un sistema di coordinate pixel
    // A quello di coordinate OpenGL, detto NDC, che va da -1 a 1
    // TODO da far fare nella shader
    normalizeVertices(vertices, N);

    // Linka i vertici al Vertex Array
    linkVerticestoBuffer(vertices, N * N * 3);

    free(vertices);
}



void linkVerticestoBuffer(float *vertices, int len) {
    // Copia i dati dei vertici nel Vertex Buffer
    // TODO dovrei sostituirlo con glBurfersubData, per evitare di allocare memoria ogni volta
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * len, vertices, GL_DYNAMIC_DRAW);

    // Attacca il Vertex Buffer all'attuale Vertex Array
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

void setupBufferAndArray(uint32_t* VBO, uint32_t* VAO) {
    // Setup del Vertex Array
    glGenVertexArrays(1, VAO);
    // Rende il Vertex Array attivo, creandolo se necessario
    glBindVertexArray(*VAO);

    // Setup del Vertex Buffer
    glGenBuffers(1, VBO);
    // Rende il Vertex Buffer attivo, creandolo se necessario
    glBindBuffer(GL_ARRAY_BUFFER, *VBO);
}



// --------------------------------------------------------------
// Funzioni DEBUG
// --------------------------------------------------------------

void printMatrix(FluidMatrix *matrix, int N) {
    float MAX_DENSITY = 0.9f;
    const char* arrows = "←↑→↓↖↗↘↙";
    const char* brightnessChars = " .:-=+*#%@";

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float velocityX = matrix->Vx[IX(i, j)];
            float velocityY = matrix->Vy[IX(i, j)];

            float density = matrix->density[IX(i, j)];

            float brightness = density / MAX_DENSITY; // Assuming MAX_DENSITY is defined

            int brightnessIndex = brightness * (strlen(brightnessChars) - 1);
            char brightnessChar = brightnessChars[brightnessIndex];

            std::cout << density << " ";

            /*if (velocityX == 0 && velocityY == 0) {
                std::cout << "∘"<< " "; // Print "∘" for zero velocity
            } else {
                // Stampa il giusto carattere
                if (velocityX < 0) {
                    if (velocityY < 0) {
                        std::cout << "↙";
                    } else if (velocityY > 0) {
                        std::cout << "↖";
                    } else {
                        std::cout << "←";
                    }
                } else if (velocityX > 0) {
                    if (velocityY < 0) {
                        std::cout << "↘";
                    } else if (velocityY > 0) {
                        std::cout << "↗";
                    } else {
                        std::cout << "→";
                    }
                } else {
                    if (velocityY < 0) {
                        std::cout << "↓";
                    } else if (velocityY > 0) {
                        std::cout << "↑";
                    }
                }

                std::cout<<" ";
            }*/

        }
        printf("\n");
    }
    printf("\n");
}


void printVertices(float *vertices, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%.2f ", vertices[3 * IX(i, j)]);
            printf("%.2f ", vertices[3 * IX(i, j) + 1]);
            printf("%.2f ", vertices[3 * IX(i, j) + 2]);
            printf("\n");
        }
    }
    printf("\n");
}

// TODO DA AGGIUSTARE, LA NORMALIZZAZIONE NON FUNZIONA
// SPECCHIA ASSE X
void normalizeVertices(float *vertices, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            vertices[3 * IX(i, j)]        = (vertices[3 * IX(i, j)]        / ((float) (viewportSize - 1) / 2.0f)) - 1;
            vertices[3 * IX(i, j) + 1]    = 1 - (vertices[3 * IX(i, j) + 1] / ((float) (viewportSize - 1) / 2.0f));
        }
    }
}

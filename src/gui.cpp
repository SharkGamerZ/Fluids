#include "gui.hpp"


int openGUI()
{
    GLFWwindow *window = setupWindow(viewportSize, viewportSize);
    ImGuiIO *io = setupImGui(window);


    // Linka i vertici al Vertex Array
    glewInit();

    // Prende l'id del programma di shader
    uint shaderProgram = getShaderProgram();
    if (shaderProgram == 0) return EXIT_FAILURE;

    int size = viewportSize;

    FluidMatrix *matrix = FluidMatrixCreate(size, 10.0f, 1.0f, 0.2f);
    FluidMatrixAddDensity(matrix, 8, 8, 10.0f);

    uint VAO;

    // Ciclo principale
    while (!glfwWindowShouldClose(window)) {

        // renderImGui(io);

        // Aggiungo densità al centro della matrice
        if (dens) 
        {
            // Simulazione
            FluidMatrixStep(matrix);
            FluidMatrixAddDensity(matrix, 1, 1, 10.0f);
            FluidMatrixAddVelocity(matrix, 1, 1, 1.0f, -1.0f);
            dens = false;
            VAO = drawMatrix(matrix, size);
        }
        

        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);


        // Rendering del triangolo
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, size * size);

        // Rendering
        // ImGui::Render();
        // ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);


        // check and call events and swap the buffers
        glfwPollEvents();
        glfwSwapBuffers(window);
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

    if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
        dens = true;

}

// Funzione per la creazione del programma di shader
// @return L'ID del programma di shader o 0 se c'è stato un errore
uint getShaderProgram() {
    int  success;
    char infoLog[512];

    // Creiamo l'id della vertexShader 
    uint vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    // Colleghiamo il codice della vertexShader all'id
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }
    
    // Creiamo l'id della fragmentShader
    uint fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    // Colleghiamo il codice della fragmentShader all'id
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 0;
    }

    // Creiamo l'id del programma di shader
    uint shaderProgram;
    shaderProgram = glCreateProgram();

    // Colleghiamo le due shader al programma
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Verifichiamo che il programma di shader sia stato creato correttamente
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        return 0;
    }

    int uniform_viewport = glGetUniformLocation(shaderProgram, "viewPort");

    // Usiamo il programma di shader
    glUseProgram(shaderProgram);

    glUniform2f(uniform_viewport, viewportSize, viewportSize);
    
    // Eliminiamo le shader
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Funzione per setuppare la finestra e il context di IMGui
// @return Il puntatore alla finestra o NULL se c'è stato un errore
GLFWwindow *setupWindow(int width, int height) {
    // Setup della finestra
    if (!glfwInit()) return NULL;

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(width, height, "Fluids", nullptr, nullptr);
    if (window == nullptr) return NULL;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Impostiamo la callback per la gestione degli input
    glfwSetKeyCallback(window, key_callback);

    return window;
}

// Funzione per fare il setup di IMGui
// @param window La finestra di GLFW
// @return Il puntatore al context di IMGui
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

// Funzione per il rendering di IMGui
// @param io Il context di IMGui
void renderImGui(ImGuiIO *io) {
    // Avvia il frame di ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // Finestra per i controlli della simulazione
    {
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(350.0f, 200.0f));
        static float densita = 0.0f;
        static float gravita = 0.0f;
        static float temperatura = 0.0f;

        ImGui::Begin("Parametri di simulazione", NULL, ImGuiWindowFlags_NoResize);
        ImGui::SliderFloat("Densita", &densita, 0.0f, 1.0f);
        ImGui::SliderFloat("Gravità", &gravita, 0.0f, 20.0f);
        ImGui::SliderFloat("Temperatura", &temperatura, 0.0f, 1.0f);


        // Buttons return true when clicked (most widgets return true when edited/activated)
        if (ImGui::Button("Avvio simulazione")) simulazioneIsRunning = !simulazioneIsRunning;


        ImGui::SameLine();
        ImGui::Text("Stato = %B", simulazioneIsRunning);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io->Framerate, io->Framerate);
        ImGui::End();
    }
}

// Funzione per il rendering della matrice
// @param matrix La matrice da renderizzare
// @param N La dimensione della matrice
uint drawMatrix(FluidMatrix *matrix, int N) {
    // Creiamo un vettore di vertici per la matrice, grande N*N * 3 visto che ho 2 coordinate e 1 colore per ogni vertice
    float* vertices = (float*) calloc(sizeof(float), N * N * 3);
    // Loop to iterate on matrix
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            vertices[3 * IX(i, j)] = i;
            vertices[3 * IX(i, j) + 1] = j;
            vertices[3 * IX(i, j) + 2] = matrix->density[IX(i, j)];
        }
    }

    printVertices(vertices, N);

    // Linka i vertici al Vertex Array
    uint VAO = linkVerticestoBuffer(vertices, N * N * 3);

    return VAO;
}


void printVertices(float *vertices, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%2d ", (int) vertices[3 * IX(i, j)]);
            printf("%2d ", (int) vertices[3 * IX(i, j) + 1]);
            printf("%2d ", (int) vertices[3 * IX(i, j) + 2]);
            printf("\n");
        }
    }
    printf("\n");
}


// Funzione per andare a linkare i vertici che gli vengono passati, al Vertex Buffer e successivamente al Vertex Array
// @param vertices I vertici da linkare
// @return L'ID del Vertex Array
uint linkVerticestoBuffer(float *vertices, int len) {
    uint VBO, VAO;
    setupBufferAndArray(&VBO, &VAO);

    // Copia i dati dei vertici nel Vertex Buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * len, vertices, GL_STATIC_DRAW);

    // Attacca il Vertex Buffer all'attuale Vertex Array
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    
    return VAO;
}

void setupBufferAndArray(uint* VBO, uint* VAO) {
    // Setup del Vertex Array 
    glGenVertexArrays(1, VAO);
    // Rende il Vertex Array attivo, creandolo se necessario
    glBindVertexArray(*VAO);

    // Setup del Vertex Buffer
    glGenBuffers(1, VBO);
    // Rende il Vertex Buffer attivo, creandolo se necessario
    glBindBuffer(GL_ARRAY_BUFFER, *VBO);
}

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

            std::cout << brightnessChar << " ";   

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
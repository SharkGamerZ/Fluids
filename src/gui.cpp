#include "gui.hpp"

// Defined Valuse
#define DENSITY_ATTRIBUTE 0
#define VELOCITY_ATTRIBUTE 1

// Golbal variables
const int matrixSize = 100;
const int scalingFactor = 3;
const int viewportSize = matrixSize * scalingFactor;
const int chunkSize = 9;    // Variabile usata quando si va a mostrare la velocità

const ImVec4 clear_color = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);

bool frameSimulation = false;
bool simulazioneIsRunning = false;
bool resetSimulation = false;

int simulationAttribute = DENSITY_ATTRIBUTE;
int display_w, display_h;

// Variabili per il mouse
double xpos, ypos, xpos0, ypos0, deltaX, deltaY;
double xposScaled, yposScaled;
double mouseTime, mouseTime0, mouseDeltaTime;

int openGUI()
{
    // Setup della finestra e del context di IMGui
    GLFWwindow *window = setupWindow(viewportSize, viewportSize);
    ImGuiIO *io = setupImGui(window);

    // Chiamata a glewInit per andare a caricare tutte le funzioni di OpenGL
    glewInit();


    // Creiamo la matrice di fluidi e gli aggiungiamo densità in una cella
    FluidMatrix matrix = FluidMatrix(matrixSize, 1.0f, 1.0f, 0.2f);

    // for (int i = 0; i < matrixSize; i++)
    // {
    //     matrix.addVelocity(1, i, 1000.0, 0.0);
    // }

    // Ciclo principale
    while (!glfwWindowShouldClose(window)) {
        // Rendering di IMGui
        renderImGui(io, &matrix);


        // --------------------------------------------------------------
        // Simulazione

        // Si salva di quanto si è spostato il mouse per poter aggiungere velocità
        glfwGetCursorPos(window, &xpos, &ypos);
        xposScaled = round(xpos / scalingFactor);
        yposScaled = round(ypos / scalingFactor);
        mouseTime = glfwGetTime();
        mouseDeltaTime = mouseTime - mouseTime0;
        deltaX = xpos - xpos0;
        deltaY = ypos - ypos0;
        mouseTime0 = mouseTime;
        xpos0 = xpos;
        ypos0 = ypos;
        
        // Aggiunge densità e velocità con il mouse
        int mouseLeftButtonState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);

        if (xposScaled >= 0 && xposScaled < matrixSize && yposScaled >= 0 && yposScaled < matrixSize)
        {
            // Aggiunge densità
            if (mouseLeftButtonState == GLFW_PRESS)
                matrix.addDensity(xposScaled, yposScaled, 100.0f);

            
            // TODO Al momento disattivato perché se riattivato crea un buco nero dove clicchiamo
            // probabilmente la simulazione è rotta e non riesce a gestire la velocità
            // Calcola la velocità
            deltaX *= 10;
            deltaY *= 10;
            

            // Aggiunge velocità
            matrix.addVelocity(xposScaled, yposScaled, deltaX, deltaY);
        }
        


        // // Aggiunta effetto macchina del vento
        // for (int i = 0; i < matrixSize; i++)
        // {
        //     matrix.addVelocity(2, i, 1000.0, 0.0);
        // }

        // Controlla se la simulazione vada resettata
        if(resetSimulation)
        {
            matrix.reset();
            resetSimulation = false;
        }

        // Simulazione
        if (simulazioneIsRunning || frameSimulation)
        {
            matrix.step();
            frameSimulation = false;
        }

        // --------------------------------------------------------------



        // In caso di resize della finestra, aggiorna le dimensioni del viewport
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);


        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        // Rendering della matrice
        drawMatrix(&matrix);


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

    // Se l'utente preme il tasto RIGHT, fa avanzare la simulazione di un frame
    if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
        frameSimulation = true;

    // Se l'utente preme il tasto R, resetta la simulazione
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        resetSimulation = true;

    // Se l'utente preme il tasto V, cambia l'attributo della simulazione
    if (key == GLFW_KEY_V && action == GLFW_PRESS)
        simulationAttribute = (simulationAttribute + 1) % 2;

}


uint32_t getShaderProgram() {
    int  success;
    char infoLog[512];

    // Get the shader source code from the GLSL files
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    if (simulationAttribute == DENSITY_ATTRIBUTE) 
    {
        vertexShaderSource = readFile("../src/shaders/density.vert");
        fragmentShaderSource = readFile("../src/shaders/density.frag");
    }
    else
    {
        vertexShaderSource = readFile("../src/shaders/velocity.vert");
        fragmentShaderSource = readFile("../src/shaders/velocity.frag");
    }

    const char* vertexShaderSourceCStr = vertexShaderSource.c_str();
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

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

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
        static float diffusione = 0.01f;
        static float deltaTime = 0.2f;
        static float temperatura = 0.0f;

        ImGui::Begin("Parametri di simulazione", nullptr, ImGuiWindowFlags_NoResize);
        ImGui::SliderFloat("Diffusione", &diffusione, 0.0f, 1.0f, "%.3f",ImGuiSliderFlags_Logarithmic);
        matrix->diff = diffusione;


        ImGui::SliderFloat("TimeStep", &deltaTime, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
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




void drawMatrix(FluidMatrix *matrix) {
    // Scegliamo quali shader usare
    GLuint shaderProgram = getShaderProgram();
    
    GLuint VAO, VBO;

    // Setup del Vertex Array
    glGenVertexArrays(1, &VAO);
    // Rende il Vertex Array attivo, creandolo se necessario
    glBindVertexArray(VAO);

    // Setup del Vertex Buffer
    glGenBuffers(1, &VBO);
    // Rende il Vertex Buffer attivo, creandolo se necessario
    glBindBuffer(GL_ARRAY_BUFFER, VBO);


    int N = viewportSize;
    // Creiamo un vettore di vertici per la matrice, grande N*N * 3 visto che ho 2 coordinate e 1 colore per ogni vertice
    float* vertices;
    if (simulationAttribute == DENSITY_ATTRIBUTE)
        vertices = getDensityVertices(matrix);
    else
        vertices = getVelocityVertices(matrix);

    // Linka i vertici al Vertex Array
    if (simulationAttribute == DENSITY_ATTRIBUTE)
    {
        glBindVertexArray(VAO);
        linkVerticestoBuffer(vertices, N * N * 3);
        glDrawArrays(GL_POINTS, 0, N * N * 3);
    }
    else
    {
        N /= chunkSize;
        glBindVertexArray(VAO);
        linkLinesToBuffer(vertices, N * N * 4);
        glDrawArrays(GL_LINES, 0, N*N*4);
    }
    free(vertices);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

}

float *getDensityVertices(FluidMatrix *matrix) {
    int N = viewportSize;
    // Creiamo un vettore di vertici per la matrice, grande N*N * 3 visto che ho 2 coordinate e 1 colore per ogni vertice
    float* vertices = (float*) calloc(sizeof(float), N * N * 3);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            vertices[3 * (IX(i, j))]       = i;
            vertices[3 * (IX(i, j)) + 1]   = j;
            vertices[3 * (IX(i, j)) + 2]   = matrix->density[(i/scalingFactor)*matrixSize + (j/scalingFactor)];
        }
    }

    // Normalizziamo i vertici da un sistema di coordinate pixel
    // A quello di coordinate OpenGL, detto NDC, che va da -1 a 1
    // TODO da far fare nella shader
    normalizeVertices(vertices, N);

    return vertices;
}

float *getVelocityVertices(FluidMatrix *matrix) {
    int N = viewportSize / chunkSize;
    // Creiamo un vettore di vertici per la matrice, grande N*N come la matrice
    //          *4 visto che ho 2 coordinate per il primo punto e 2 per il secondo
    //          /9 per mettere una linea ogni 9 pixel
    float* vertices = (float*) calloc(sizeof(float), (N*N * 4));

    int viewPortI = (chunkSize - 1)/2;
    int viewPortJ = (chunkSize - 1)/2;

    for(int i = 0; i < N; i++) {
        viewPortJ = (chunkSize - 1)/2;
        for(int j = 0; j < N; j++) {
            vertices[4 * (IX(i, j))]       = viewPortI;
            vertices[4 * (IX(i, j)) + 1]   = viewPortJ;

            // Calcolo media velocità
            float vx = 0;
            float vy = 0;
            int newI, newJ;
            int count = 0;
            for (int k = -(chunkSize - 1)/2; k <= (chunkSize - 1)/2; k++) {
                for (int l = -(chunkSize - 1)/2; l <= (chunkSize - 1)/2; l++) {
                    newI = (viewPortI + k)/scalingFactor;
                    newJ = (viewPortJ + l)/scalingFactor;
                    if (newI >=0 && newI < matrixSize && newJ >=0 && newJ < matrixSize)
                    {
                        vx += matrix->Vx[newI*matrixSize + newJ];
                        vy += matrix->Vy[newI*matrixSize + newJ];
                        count++;
                    }
                }
            }
            vx /= count;
            vy /= count;
            vertices[4 * (IX(i, j)) + 2]   = viewPortI + vx + 1;
            vertices[4 * (IX(i, j)) + 3]   = viewPortJ + vy + 1;


            // vertices[4 * (IX(i, j)) + 2]   = viewPortI + matrix->Vx[(viewPortI/scalingFactor)*matrixSize + (viewPortJ/scalingFactor)] + 1;
            // vertices[4 * (IX(i, j)) + 3]   = viewPortJ + matrix->Vy[(viewPortI/scalingFactor)*matrixSize + (viewPortJ/scalingFactor)] + 1;


            viewPortJ +=chunkSize;
        }
        viewPortI +=chunkSize;
    }

    // Normalizziamo i vertici da un sistema di coordinate pixel
    // A quello di coordinate OpenGL, detto NDC, che va da -1 a 1
    // TODO da far fare nella shader
    normalizeSpeedVertices(vertices, N);


    
    return vertices;
}


void linkVerticestoBuffer(float *vertices, int len) {
    // Copia i dati dei vertici nel Vertex Buffer
    // TODO dovrei sostituirlo con glBurfersubData, per evitare di allocare memoria ogni volta
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * len, vertices, GL_DYNAMIC_DRAW);

    // Attacca il Vertex Buffer all'attuale Vertex Array
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

void linkLinesToBuffer(float *vertices, int len) {
    // Copia i dati dei vertici nel Vertex Buffer
    // TODO dovrei sostituirlo con glBurfersubData, per evitare di allocare memoria ogni volta
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * len, vertices, GL_DYNAMIC_DRAW);

    // Attacca il Vertex Buffer all'attuale Vertex Array
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,2 * sizeof(float), (void*)0);
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

// TODO DA AGGIUSTARE, LA NORMALIZZAZIONE NON FUNZIONA
// SPECCHIA ASSE X
void normalizeVertices(float *vertices, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            vertices[3 * IX(i, j)]        = (vertices[3 * IX(i, j)]         / ((float) (viewportSize - 1) / 2.0f)) - 1;
            vertices[3 * IX(i, j) + 1]    = 1-(vertices[3 * IX(i, j) + 1]     / ((float) (viewportSize - 1) / 2.0f));

        }
    }
}

void normalizeSpeedVertices(float *vertices, int N) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            vertices[4 * IX(i, j)]        = (vertices[4 * IX(i, j)]         / ((float) (viewportSize - 1) / 2.0f)) - 1;
            vertices[4 * IX(i, j) + 1]    = 1-(vertices[4 * IX(i, j) + 1]     / ((float) (viewportSize - 1) / 2.0f));

            vertices[4 * IX(i, j) + 2]    = (vertices[4 * IX(i, j) + 2] / ((float) (viewportSize - 1) / 2.0f)) - 1;
            vertices[4 * IX(i, j) + 3]    = 1-(vertices[4 * IX(i, j) + 3] / ((float) (viewportSize - 1) / 2.0f));
        }
    }

}

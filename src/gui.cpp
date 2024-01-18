#include "gui.hpp"


int openGUI()
{
    GLFWwindow *window = setupWindow(1280, 720);
    ImGuiIO *io = setupImGui(window);


    // Definizione dei vertici del triangolo
    // ------------------------------------------------------------------
    float vertices[] = {
         0.5f,  0.5f, 0.0f,  // alto destra
         0.5f, -0.5f, 0.0f,  // basso destra
        -0.5f, -0.5f, 0.0f,  // basso sinistra

         0.5f,  0.5f, 1.0f,  // alto destra
        -0.5f, -0.5f, 1.0f,  // basso sinistra
        -0.5f,  0.5f, 1.0f   // alto sinistra
    };

    // Linka i vertici al Vertex Array
    glewInit();
    uint VAO = linkVerticestoBuffer(vertices, 18);

    // Prende l'id del programma di shader
    uint shaderProgram = getShaderProgram();
    if (shaderProgram == 0) return EXIT_FAILURE;

    // Ciclo principale
    while (!glfwWindowShouldClose(window)) {

        renderImGui(io);

        // Controllo colore 
        if(simulazioneIsRunning) {
            // clear_color = ImVec4(0.00f, 1.00f, 0.00f, 1.00f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        else {
            // clear_color = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        

        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        

        // Rendering del triangolo
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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

    // Usiamo il programma di shader
    glUseProgram(shaderProgram);
    
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

// Funzione per andare a linkare i vertici che gli vengono passati, al Vertex Buffer e successivamente al Vertex Array
// @param vertices I vertici da linkare
// @return L'ID del Vertex Array
uint linkVerticestoBuffer(float *vertices, int len) {
    uint VBO, VAO;
    glGenBuffers(1, &VBO);      // Vertex Buffer

    // Rende il Vertex Buffer attivo, creandolo se necessario
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Copia i dati dei vertici nel Vertex Buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * len, vertices, GL_STATIC_DRAW);



    glGenVertexArrays(1, &VAO); // Vertex Array
    // Rende il Vertex Array attivo, creandolo se necessario
    // Questa operazione è necessaria perché l'Element Buffer è salvato nel Vertex Array
    glBindVertexArray(VAO);

    // Attacca il Vertex Buffer all'attuale Vertex Array
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    printf("VAO = %d\n", VAO);
    
    return VAO;
}
#include "gui.hpp"

int openGUI()
{
    // Setup della finestra
    if (!glfwInit()) return EXIT_FAILURE;

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(1280, 720, "Fluids", nullptr, nullptr);
    if (window == nullptr) return EXIT_FAILURE;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    

    // Set the key callback
    glfwSetKeyCallback(window, key_callback);



    // Definizione dei vertici del triangolo
    float vertices[] = {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
     0.0f,  0.5f, 0.0f
    };  



    // Inizializzazione di GLEW
    glewInit();
    // Creazione del Vertex Buffer Object e salviamo il suo ID in VBO
    uint VBO;
    glGenBuffers(1, &VBO);
    // Colleghiamo il VBO all'Array Buffer di OpenGL
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Copiamo i vertici nel VBO
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices , GL_STATIC_DRAW);


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  

    // Creazione del Vertex Array Object e salviamo il suo ID in VAO
    uint VAO;
    glGenVertexArrays(1, &VAO);


    
    // 1. bind Vertex Array Object
    glBindVertexArray(VAO);
    // 2. copy our vertices array in a buffer for OpenGL to use
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // 3. then set our vertex attributes pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  
    
    uint shaderProgram = getShaderProgram();
    if (shaderProgram == 0) return EXIT_FAILURE;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        // Finestra della simulazione
        if (false){
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin("Finestra simulazione");
            ImGui::Text("Sono la simulazione!");
            if (simulazioneIsRunning) {
                ImGui::Text("Simulazione in corso...");
            } else {
                ImGui::Text("Simulazione non in corso...");
            }
            ImGui::End();
        }

        // Finestra per i controlli della simulazione
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(350.0f, 200.0f));
        {
            static float densita = 0.0f;
            static float gravita = 0.0f;
            static float temperatura = 0.0f;

            ImGui::Begin("Parametri di simulazione");
            ImGui::SliderFloat("Densita", &densita, 0.0f, 1.0f);
            ImGui::SliderFloat("Gravità", &gravita, 0.0f, 20.0f);
            ImGui::SliderFloat("Temperatura", &temperatura, 0.0f, 1.0f);


            // Buttons return true when clicked (most widgets return true when edited/activated)
            if (ImGui::Button("Avvio simulazione")) simulazioneIsRunning = !simulazioneIsRunning;


            ImGui::SameLine();
            ImGui::Text("Stato = %B", simulazioneIsRunning);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // Controllo colore 
        if(simulazioneIsRunning) {
            // clear_color = ImVec4(0.00f, 1.00f, 0.00f, 1.00f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        else {
            // clear_color = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }

        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Rendering del triangolo
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

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
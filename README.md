# Fluids
La fantastica repo per il progetto di Multicore 2023/24

## Fisica
Per iniziare, questa simulazione si basa su una griglia di Eulero, per rappresentare dei fluidi non comprimibili.


## Necessari
- OpenGL
- GLFW
- glew
- ImGUI
- nvcc

## Come funziona il codice
### GUI
L'interfaccia grafica è stata fatta tramite openGL, GLFW e DearIMGui.
- GLFW è il gestore delle finestre, permette di creare una finestra e attaccargli una view.
- OpenGL è la libreria grafica che permette di visualizzare a schermo la simulazione.
- GLEW è usato per ottenere i puntatori alle funzioni di OpenGL specifiche per la versione che stiamo usando.
- ImGUI è un tool di "debugging" che permette di avere delle finestre utili per gestire il render loop, potendo modificare delle variabili a Runtime.

#### GLFW
GLFW è la prima libreria che usiamo, prima con `glfwInit()` e poi con `glfwCreateWindow(...)`.

Quest'ultima funzione ci restituisce un puntatore ad un oggetto di tipo `GLFWwindow`, che rappresenterà la nostra finestra.

Tramite `glfwSetKeyCallback()` impostiamo la nostra funzione `key_callback(...)` come funzione per gestire gli input che vengono dati da tastiera.

Infine, tramite il loop `while (!glfwWindowShouldClose(window))`, andiamo ad eseguire il nostro programma finché la finestra non riceve il segnale di essere chiusa, che può succedere:
- Cliccando la **X** in alto a destra della finestra.
- Premendo **ESC** da tastiera.

---

#### OpenGL
OpenGL è la nostra libreria usata per andare a renderizzare a schermo i nostri oggetti.

Le funzioni di OpenGL sono dette **Shader**, e nel nostro caso ne usiamo due:
- [Vertex Shader](https://www.khronos.org/opengl/wiki/Vertex_Shader): la Vertex Shader processa i singoli vertici, andando a restituire in output altri vertici.
- [Fragment Shader](https://www.khronos.org/opengl/wiki/Fragment_Shader): la Fragment Shader processa i frammenti che gli vengono passati, che sono un'insieme di pixel, per assegnargli dei colori.




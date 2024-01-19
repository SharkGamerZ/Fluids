# Fluids
La fantastica repo per il progetto di Multicore 2023/24

## Necessari
- OpenGL
- GLFW
- glew
- ImGUI
- nvcc

## Fisica
Per iniziare, questa simulazione si basa su una griglia di Eulero, per rappresentare dei fluidi non comprimibili.
La simulazione è rappresentata da 3 fasi principali:
- **Diffusione**: ad ogni timestep ogni cella del fluido tende a "diffondersi" in quelle vicine, come una goccia di salsa di soia messa in una bacinella d'acqua. Questo comporta che se una certa cella ha una velocità, verrà diffusa nei suoi vicini.
- **Project** (non so tradurlo): in ogni momento, la somma di fluidi che entrano/escono da una cella deve essere 0, altrimenti vorrebbe dire che della materia sta sparendo/apparendo dal nulla. Visto che le altre operazioni potrebbero andare ad infrangere questa regola, tramite questa fase ci assicuriamo che venga rispettata.


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



# TODO
## OpenGL

- [x] Mostrare la matrice del fluido a schermo come una serie di vertici
- [x] Usare la densità della matrice per determinare il colore di ogni pixel
- [ ] Modificare `drawMatrix()` per mostrare la velocità
- [ ] Scrivere le shader in dei file e caricarli da lì (Forse tramite classe `Shader`)
- [ ] Normalizzare i vertici nelle shader


### ImGui
- [ ] Rendere i controlli di ImGui legati alle proprie variabili
- [ ] Aggiungere timestep alle variabili
- [ ] Aggiungere una scelta per visualizzare densità o velocità
- [ ] Usare l'accellerazione del mouse per aggiungere velocità al fluido



## Simulazione
- [ ] Far funzionare la `diffuse()` affinché se `diffuse=0` la matrice resti ferma senza evolversi nel tempo. (Probabilmente da controllare che valore si trova nella densità al timestep precedente)
- [ ] Implementare la "**gravità**" tramite un flow laminare di velocità generato dalla prima riga verso il basso.

# Bibliografia
Gran parte di questo lavoro è basato su vari paper/risorse:
- [Real-Time Fluid Dynamics for Games](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf) by Jos Stam
- [Fluid Simulation for Dummies](https://mikeash.com/pyblog/fluid-simulation-for-dummies.html) by Mike Ash
- [But How DO Fluid Simulations Work?](https://www.youtube.com/watch?v=qsYE1wMEMPA) by Gonkee
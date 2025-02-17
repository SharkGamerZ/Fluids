
## OpenGL

- [x] Mostrare la matrice del fluido a schermo come una serie di vertici
- [x] Usare la densità della matrice per determinare il colore di ogni pixel
- [x] Modificare `drawMatrix()` per mostrare la velocità
- [x] Scrivere le shader in dei file e caricarli da lì 
- [x] Rendere la finestra non resizable
- [x] Implementare uno scaling factor
- [x] Fare la visualizzazione della velocità tramite linee
    - [ ] Modificare la visualizzazione del campo delle velocità rendendo la lunghezza delle linee la metà della distanza tra le origini di due linee adiacenti nella griglia e rappresentare la magnitudine tramite range di colori (es. verde - rosso).
    - [x] Fare la visualizzazione della velocità tramite i colori
- [ ] Cambiare glUseProgram per non eseguirlo ogni volta
- [ ] Usare glBufferSubData per efficienza
- [ ] Normalizzare i vertici nelle shader
- [ ] Fixare pixel morto al centro della finestra


### ImGui
- [x] **Da rivedere**(Non sicuro che venga aggiunta la giusta velocità alla matrice) Usare l'accellerazione del mouse per aggiungere velocità al fluido
- [x] Aggiungere una scelta per visualizzare densità o velocità
- [x] Aggiungere timestep alle variabili
- [x] Rendere i controlli di ImGui legati alle proprie variabili
- [x] Aggiungere possibilità di resettare la matrice premendo il pulsante **R**



## Simulazione
- [x] Cambiare nome ad s in density0
- [x] Debuggare Advect (va in seg Fault)
- [x] **PRIORITA-THOMAS'** Far funzionare la `diffuse()`
- [x] Far funzionare la advect
- [ ] Far funzionare la project (assicurare la continuità nel campo delle velocità) ((appare densità dal nulla e sparisce nel nulla))
- [ ] Vedere altri metodi per fare la linear solve

## Parallelizzazione
- [x] **MATTEO** Parallelizzare la linear solve con openMP
- [x]**Matteo** Fare foglio con le statistiche della diffuse in openMP
- [ ] OMP della Advect
- [ ] OMP della Project
- [ ] Aggiustare la CUDA diffuse
- [ ] Fare CUDA della Advect e la Project



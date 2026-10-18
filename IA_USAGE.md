# IA_USAGE — Proyecto Integrador Corte 2 (PENGWIN)

Registro del uso de IA generativa en el proyecto, como pide el enunciado (sección 7). El documento
tiene las dos partes que pide el enunciado:

- **Parte 1. Registro de uso.** Una entrada por cada tema de trabajo en que se usó IA, con la
  herramienta y el modelo, el prompt (exacto o su fragmento clave), qué salió de la IA y qué
  modificó o decidió el equipo a mano.
- **Parte 2. Análisis crítico.** Aciertos, errores, limitaciones y lecciones, con referencia a las
  entradas de la Parte 1.

Equipo: Gustavo Adolfo Chipantiza (2222022), Federico Gonzales Cárdenas (2225657), Juan Jacobo
Delgado Melo (2230110) y María de los Ángeles Valencia Hurtado (2235112).

## Herramientas usadas

| Integrante | Herramienta | Modelo | Entradas |
|---|---|---|---|
| Jacobo | Claude (claude.ai) | Claude Opus 5 | 1–7, 35 |
| Jacobo | Claude Code (extensión de VS Code) | `claude-sonnet-5-5` y, desde el 12 de octubre, `claude-opus-5-5` | 36 |
| María | Claude (claude.ai) | Claude Opus 5.5 | 8, 9, 14–20, 28, 34 |
| Gustavo | Claude, app de escritorio (modo Cowork) | `claude-sonnet-5-5` y, desde el 6 de octubre, `claude-opus-5-5` | 10–13, 21–27, 30–33 |

Algunas entradas citan documentos que ya no están en el repositorio (`docs/plan_semana10.md`,
`docs/plan_semana11.md`, `docs/resumen_semana10.md` y `notebooks/05_casos_dificiles_ejecutado.ipynb`).
Se retiraron en la limpieza final (entrada 33) porque repetían lo que hoy está en la bitácora, el
model card y los documentos de la Semana 11; siguen en el historial de git.

---

# Parte 1. Registro de uso

Las entradas van en orden cronológico por semana del cronograma. Las entradas 1–7 conservan el
texto de Jacobo y las 10–13 y 21–27 el de Gustavo; en estas últimas, la IA redactó el registro a
pedido de Gustavo con los prompts copiados tal cual.

## Semana 8 — datos, EDA y Visualizador 1

### 1. Verificación de reproducibilidad de los datos

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Prompt (resumen del pedido):** confirmar que los datos descargados de Zenodo y colocados en
  `data/raw/` son idénticos a los que usó María, antes de empezar a modificar código.
- **Prompts exactos:**
  > listo ya cree la rama y esta caopaida en lcoal, que continua? lo primreo es ver si hace parte del mimso hash o que es lo primero?

  > no ya no tengo los .zip, los borro solo deje lo descrimpimido
- **Qué produjo la IA:** el procedimiento: correr `build_manifest.py` sobre los datos locales,
  comparar el CSV contra el original (ignorando las columnas de ruta absoluta, que cambian entre
  sistemas operativos) y regenerar `splits.json` para verificar que `git diff` no mostrara cambios.
- **Qué se hizo a mano:** se aceptó el enfoque sin cambios y se ejecutó. Resultado: manifest
  idéntico y splits reproducibles al 100 %.

### 2. Módulo `pengwin_io.py`

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Prompt (resumen del pedido):** centralizar la carga y el preprocesamiento, antes duplicados en
  `build_manifest.py`, `mip_visualizer.py` y `hu_windowing_check.py`, en un solo módulo, con las
  constantes que ya usaba el repo (clip −1024/3000, ventana C400/W1800).
- **Prompts exactos:**
  > empezemos con lo que veas mas conveniente, la idea es cuando se haga ya el commit de esta ram quede complegtamente hecha y repoducible lo que se pide para semana 08
- **Qué produjo la IA:** el módulo completo, incluidas las funciones de redimensionamiento para la
  imagen (bilineal) y para la máscara (vecino más cercano), que no existían y se agregaron pensando
  en la Semana 9.
- **Qué se hizo a mano:** se probó con `python src/pengwin_io.py 001` antes de integrarlo a los
  demás scripts.

### 3. Visualizador 1 — reconstrucción 3D (varios intentos)

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Prompt (resumen del pedido):** reemplazar los PNG estáticos de `mip_visualizer.py` por una
  isosuperficie 3D interactiva que muestre solo hueso, sin camilla ni cables.
- **Prompts exactos:**
  > asi se mira revisa, si te refieres a eso si esta bien o no?

  > asi salieron los tres casos, analizalos y revisalos vea que unos salieron muy incompletos

  > listo mejoro, dime que te parece ahora si podriamos mejorarlo o con eso es suficiente, por que en el caso 025 hay ciertos huecos en la pelvis, puede que esta imagen sea ruidosa pero no se si para tratar de arreglar esto dañemos lo demas
- **Qué produjo la IA y qué se decidió en cada intento:**
  1. *Rechazado.* Apertura morfológica más filtro de componentes por tamaño relativo. Los rieles
     de la camilla resultaron más grandes que los fragmentos fracturados, así que el filtro conservó
     los rieles y descartó hueso real. Se rechazó tras revisar las capturas.
  2. *Aceptado con ajuste.* Una "silueta corporal" (la componente más grande de `HU > −300`, con
     relleno de huecos) intersectada con el umbral óseo. Excluye rieles y cables porque no tocan al
     paciente.
  3. *Rechazado.* Una apertura morfológica extra para quitar rayas de endurecimiento de haz. Perforó
     el hueso cortical (crestas ilíacas rotas, sacro deshecho): a ~0.7–0.8 mm por vóxel la cortical
     es tan delgada como el ruido.
  4. *Aceptado, versión final.* Sin apertura y con un tamaño mínimo por componente de 3000 vóxeles
     (antes 30). Pelvis completa en los 3 casos de prueba.
- **Caso 025:** quedaron varillas rectas y huecos en el ala ilíaca. La IA lo atribuyó a un fijador
  externo y se documentó como limitación. Esa hipótesis era incorrecta (ver entrada 7).
- **Nota de seguimiento:** en la Semana 11 se recuperó el MIP en `src/raw_view.py` (entrada 26), y
  el Visualizador 1 entrega las dos vistas, MIP e isosuperficie (entrada 28).

### 4. EDA de fragmento principal y tamaño

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Prompt (resumen del pedido):** validar si la etiqueta 1/11/21 es siempre el fragmento de mayor
  volumen de su región, y medir el tamaño de cada fragmento en mm³ y en cortes axiales.
- **Prompts exactos:**
  > bien sigamos, tu mira con cual empezamos, igual hay que hacerlos todos
- **Qué produjo la IA:** el script completo, probado primero con un caso sintético que viola la
  convención a propósito (un "11" más chico que un "12") para confirmar que detecta excepciones.
- **Qué se hizo a mano:** se corrió sobre los 100 casos. La convención se cumple en 300 de 300
  combinaciones caso-región.

### 5. `requirements.txt`

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Prompt (resumen del pedido):** el archivo original era un `pip freeze` de macOS (~200 líneas,
  versiones fijas) que no instalaba en Windows.
- **Prompts exactos:**
  > no mi pc no tiene GPU NVIDIA, para qur hago estos utlimos dos comandos que me hiciste hacer para requirements?

  > pero porque mi requiaremnets.txt esta asi bien simple como tu me lo diste, y no esta con todas la lienas que se saben crear?
- **Qué produjo la IA:** un archivo curado con solo las librerías que importa el código, sin
  versiones fijas.
- **Qué se hizo a mano:** en un equipo sin GPU NVIDIA, `pip install torch` instaló la build con
  CUDA, que falló al importar en Windows (`c10_cuda.dll`). Se instaló `torch` aparte desde el índice
  de CPU y se documentó para el resto del equipo.

### 6. Corrección de `docs/resumen_semana8.md`

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Prompt (resumen del pedido):** revisar el documento de María en busca de imprecisiones.
- **Prompts exactos:**
  > lo del .mha el profe nos dijo que si que trabajemos con esos que nos quita un problema, esa era la duda?

  > listo ya lo hice falta algo mas para la semana 8?
- **Qué produjo la IA:** tres correcciones: una afirmación de que "se notificó al profesor por
  escrito" cuando no se había enviado nada; rutas de ejemplo (`~/Downloads/...`) que no coinciden
  con `data/raw/`; y la explicación del clip de HU, que no neutraliza el metal por sí solo (lo aplana
  la ventana C400/W1800).
- **Qué se hizo a mano:** se confirmó directamente con el profesor antes de marcar la notificación
  como resuelta.

### 7. Caso 025: normalización tras la asesoría con el profesor

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "continuacion_semana08"](https://claude.ai/chat/7eb9a270-3b07-4151-9109-0b997bb9c1c3)
- **Contexto:** en asesoría, el profesor corrigió la hipótesis del fijador externo de la entrada 3:
  si fuera metal, no se vería la parte superior de la cresta ilíaca. Según él, es mala distribución
  de HU y hay que normalizar con gamma o ecualización, evaluando cuál conviene.
- **Prompt (resumen del pedido):** diagnosticar el caso 025 contra los 100 casos y comparar métodos
  de normalización con evidencia.
- **Prompts exactos:**
  > tuve una asesoria con el profesor sobre esto que hicimos, hice un breve resumen para que veas el panorama, pero antes te debo comentar que un amigo ya adelanto la semana 09 en esta rama https://github.com/mariavalencia30/proyecto2-analitica-datos/tree/continuacion_semana09. Entonces antes de que te diga el resumen de la asesoria porque no va a tocar retroceder a algunas cosas de la semana 08 quiero analices como esta la rama ahora ver como se realizaron los cambios o que se modificaria, y que tanto cambiaria

  > no importa de todas manera hay que volverlo a reentrenar, asi que cambio lo que se debe, no dejes cosas que a la final no van a servir o sera codigo viejo

  > listo pero entonces quiero ver como aparece este nuevo caso 025 y los otros dos casso para revisar si funciono
- **Qué produjo la IA:** la comparación de percentiles con `data/eda_intensidades_detalle.csv`, sin
  reprocesar volúmenes. El HU máximo y el volumen de metal del 025 no son atípicos (z 1.47 y 1.13);
  lo atípico es la mediana del hueso (z ≈ −2.1), corrida ~150–250 HU hacia abajo. Además, una
  comparación visual de gamma, ecualización global y CLAHE sobre el 025 y un caso típico (007), con
  el criterio de que el método no puede dañar los casos que ya están bien
  (`outputs/eda/diagnostico_normalizacion_caso025.png`).
- **Qué se hizo a mano:** se descartó la ecualización global (aplana el tejido blando) y se eligió
  CLAHE (`clip_limit=0.02`), integrado por defecto en `pengwin_io.window_hu`. Se verificó en los
  casos 001, 025 y 068 (`outputs/eda/verificacion_clahe_casos_ejemplo.png`) y se avisó que los
  cachés y checkpoints de las semanas 9 y 10 había que regenerarlos.
- **Nota de seguimiento:** esta decisión se revirtió después. La comparación de 166 configuraciones
  por contraste-ruido, la revisión de los 100 casos y una nueva asesoría llevaron a cambiar CLAHE por
  **gamma 1.5** (entrada 9). La justificación con Otsu del notebook 03 lo confirma: CLAHE da un Dice
  de 0.223 contra 0.486 de gamma 1.5.

### 8. Documentación de la Semana 8, EDA de intensidades y flujo de git

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai). Septiembre de 2026.
- **Fuente:** [chat "Revisión de avances y documentación del análisis EDA"](https://claude.ai/chat/2473aba1-2d33-405b-a1f6-c211787c2a1e)
- **Prompt (resumido; el texto exacto está en el chat):** revisar el avance de la Semana 8 contra el
  enunciado y documentarlo; armar los notebooks de seguimiento; interpretar los resultados
  ejecutados.
- **Qué produjo la IA:**
  - la actualización de `docs/resumen_semana8.md` con los resultados del EDA y una guía de ejecución;
  - los notebooks `01_semana8_dataset_y_eda.ipynb` y `02_eda_intensidades.ipynb`, que solo importan
    de `src/`;
  - las conclusiones con números reales: la mediana HU del hueso va de 81 a 431 (CV 26.5 %), hay 25
    casos con metal y el endurecimiento de haz es despreciable a nivel de volumen;
  - las instrucciones de git (ramas `feat/*`, PRs con revisión, `.gitignore` de `data/raw/`), el
    cambio de rutas absolutas a relativas y el registro del kernel de Jupyter.
- **Qué se hizo a mano:** el equipo ejecutó los notebooks con los datos reales y aportó las salidas
  sobre las que se escribieron las conclusiones. La decisión de usar ventana fija (C400/W1800) con
  aumento de intensidad la tomó el equipo con esa evidencia.

## Semanas 9 y 10 — modelo, entrenamiento y evaluación

### 9. Modelo de la Semana 9, preprocesamiento y primera versión de la Semana 10

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai). Septiembre a 3 de octubre de
  2026.
- **Fuente:** [chat "Revisar repositorios de GitHub"](https://claude.ai/chat/e42d6060-3925-427b-875d-cccd5b91eec5)
- **Prompt (resumido; el texto exacto está en el chat):** revisar los entregables de las semanas 8 y
  9 contra el enunciado; implementar y depurar el pipeline de la Semana 10; subir el Dice por
  fragmento; atender los pedidos de la asesoría del profesor.
- **Qué produjo la IA:**
  - **Semana 9:** el código y las pruebas del backbone propio, CBAM, la cabeza de detección por grid
    y el NMS propio (`week9_detection.py`, `tests/test_week9_detection.py`); el overfit intencional
    (pérdida de 1.673 a 0.00115, 1454× con un criterio mínimo de 4×, la corrida registrada en el
    notebook 03; una corrida anterior dio 1.687 → 0.002, 758×); y
    `docs/resumen_semana9.md`.
  - **Preprocesamiento:** la comparación de 166 configuraciones por contraste-ruido (CLAHE 3.2,
    ventana sola 6.6, gamma 8.1) y la revisión visual de los 100 casos.
  - **Semana 10 (primera versión):** el modelo multitarea `week10_multitask.py`, la pérdida
    compuesta y la evaluación (`evaluate_week10.py`). La instancia se reformuló de 31 IDs a
    primario/secundario por región, siguiendo la literatura del challenge. Además: espejo sagital,
    peso por línea de fractura, reponderación de secundarios, coseno, 384 px, TTA, cortes sin hueso,
    segundo pase por región y 2.5D.
  - **Pedidos de la asesoría:** γ aprendible por etapa en CBAM, Grad-CAM con y sin atención,
    análisis de fragmentos desplazados y justificación del preprocesamiento con máscaras de Otsu
    (`gradcam_week9.py`, `deteccion_fragmentos_desplazados.py`, `otsu_preprocesamiento.py`).
  - **Documentos:** `sustentacion_semana8_PENGWIN.md`, `documento_tecnico_avance_PENGWIN.md`, la
    hoja de ruta, guiones de presentación y mensajes al equipo y al profesor.
- **Qué se hizo a mano:** los entrenamientos en Colab y Kaggle; la decisión final de gamma 1.5 tras
  la asesoría; la revisión de los resultados y su integración en `continuacion_semana09` y
  `continuacion_semana10_fixed`. El Dice por fragmento subió de 0.667 a ~0.82; el segundo pase por
  región y el 2.5D se descartaron porque no lo movieron.

### 35. Semana 9 de Jacobo: revisión del detector, gamma 1.5, notebook 03 y pedidos del avance

- **Integrante y herramienta:** Jacobo · Claude Opus 5 (claude.ai). 28 de septiembre a 2 de
  octubre de 2026.
- **Fuente:** [chat "este"](https://claude.ai/chat/a1d72462-fd7a-4181-ae09-90c1c4adb69a)
- **Prompts exactos (los que abrieron cada parte del trabajo):**
  > no pero solo necesito que no enfoquemos en la semana 9, la 10 no me importa, entonces volvamos a empezar, que cambios hacer para la semana 09 dejarla full lo que se pide en el proyecto

  > Ya que con esto que el profe nos recimiendo todo lo demás se descarta y no habría islas flotantes ni nada de esto. Verifica si tiene razón el profe y si es correcto entocnes hagamos de una vez lo que nos dice y descartemos esos filtros demás que hicimos que lo que mas hizo fue granular y suavizar las imagen y luego el modelo no va a poder clasificar.

  > si crea el script y me dices donde ponerlo y como ejcutarlo, dime si en este script ya entraria todo lo que hablas, todos esos cambios.

  > si entonces utilicemos gamma = 1.5 pero no se porque no aaprece en la png o cual es la columna para ver cual es el de gamma 1.5

  > No, no, no. No importa que se sobrescriban los outputs. Simplemente quiero subir este notebook con mi nombre. O sea, aparte. No cambiándole el que María ya había subido. Entonces, lo único que quiero es subir este notebook a GitHub al repositorio. Y ya.

  > ya tuvimos el avance y esto fue lo que nos dijo, te paso la transcripcion, por ahora solo entra en contexto, de que fue lo que pide el profe, porque una amiga supiestamnete ya hizo esto en el notebook y me lo mando actulizado con los cambios que dijo el profe, pero hay que revisar si esta todo correcto

  > no los tengo, si quieres podemos continuar con la ultima version que teniemoa del notebook 03 y aplicar esto que dijo el profe o puedes tomar de base este notebook de mi amiga pero arreglado estos puntos malos, tu mira cual es el mejor escenario

  > seria buereno hacer una comparacion de las predicciones de bbox tanto para ventaneo con gamma=1.5 para asi dar mejor justificaciones de que esta dando mejores predicciones el gamma=1.5 dime si esto crees que sea necesario o con lo que tenemos es una justificacion bien basta

  > pues midamos si da mejores predicciones gamma, edita el notebook y mnadamelo para ejecutarlo para ver esta comparacion
- **Qué produjo la IA:** revisó la Semana 9 contra el enunciado; ajustó el detector, el uso de CBAM,
  NMS y las métricas; apoyó la justificación de gamma 1.5 y organizó el notebook 03 con análisis,
  cajas predichas y resultados.
- **Qué se hizo a mano:** Jacobo ejecutó las pruebas, barridos, entrenamientos y notebook con datos
  reales; revisó las salidas y decidió conservar una rama separada para su versión del notebook.
- **Análisis crítico:** algunas primeras interpretaciones de la IA no coincidieron con las salidas
  reales; Jacobo las corrigió comparándolas con la ejecución. También encontró que varios casos que
  se habían tratado como validación eran de entrenamiento, por lo que esas pruebas exploratorias no
  medían generalización.

### 10. Auditoría del repositorio frente al enunciado

- **Integrante y herramienta:** Gustavo · Claude, app de escritorio (Cowork), `claude-sonnet-5-5` y
  `claude-opus-5-5`. 4 de octubre de 2026, rama `unificado_semana09`.
- **Prompt exacto:**
  > entendamos el repositorio y que tiene y mira lo que tenemos que hacer , y miremos en que punto
  > estamos
- **Qué produjo la IA:** una tabla de objetivos contra los resultados de `w10g_384neg` y la lista de
  entregables faltantes (SAM, visualizadores 2 y 3, dashboard, latencia, model card, distancias sobre
  volúmenes). En una segunda lectura de `evaluate()` encontró que el mAP@0.50 de 0.901 se midió sobre
  cajas refinadas con la máscara semántica, y que la cabeza grid cruda nunca se había medido.
- **Qué se hizo a mano:** se aceptó la lista de faltantes y que el Dice por fragmento (0.817) era el
  único objetivo sin cumplir en ese momento.

### 11. Diagnóstico de la Semana 10 y árbol de decisiones

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos. 4 de octubre de 2026.
- **Prompt exacto:**
  > Listo, pero antes de hacer visualizadores y dashboard y todo eso, primero tenemos que hacer eh,
  > un... Tenemos que hacer un notebook o, lo que, o los archivos.py, eh, preferiblemente un notebook
  > que podamos eh, cumplir con todo lo que nos piden, tanto en métricas como en todo. Entonces, ahí,
  > ¿qué podríamos hacer? ¿Sí? Digamos que la idea es que hagamos las cosas... con el notebook número,
  > número 3, que ya es lo que, ahí ya termina la semana 9. Y volvamos a mirar cómo hacemos para,
  > para, para llegar a las métricas, ¿sí? Si se puede hacer un, un notebook, que nos vayamos paso a
  > paso, o a la podemos hacer un árbol de decisiones que tengamos que tomar para poder entrenar esto
  > de acuerdo a lo que encontremos o lo que hemos encontrado y demás. Eso sería eh, importante.
  > Entonces, creo que esa vaina es importante. Que vayamos mirando si, si ya terminamos semana 9, o si
  > hay algo más que podamos agregar o hacer, y pues avancemos con la semana 10.
- **Qué produjo la IA:** que la Semana 9 está cerrada; seis hallazgos sobre la Semana 10 (mAP medido
  sobre cajas refinadas, entrenamiento con 12 de ~244 cortes por caso, modelo sin las opciones de
  CBAM de la Semana 9, Dice por fragmento bajo el objetivo, test evaluado con tres checkpoints,
  distancias y SAM sin hacer); un árbol de decisiones que empieza midiendo la línea base sin
  entrenar; y la estructura del notebook 04.
- **Qué se hizo a mano:** se dejó el plan escrito en `docs/plan_semana10.md` antes de tocar código.

### 12. Carpeta del curso y reinicio de la Semana 10

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos.
- **Prompt exacto:**
  > yo quiero volver a trabajar la semana 10 desde 0 , que tengamos como referencia lo que se habia
  > adelantado ahi , pero ahi te paso la carpeta de lo que hemos hecho con el profe por si sirve
  >
  > igual guiame y dime las preguntas que consideres para hacerles a los companeros y como iniciamos
  > para que vamos avanzando con la semana 10
- **Qué produjo la IA:** que `FundidoraBackbone` del repo no tiene la estructura de FundidoraPC del
  curso (dos convoluciones con SiLU frente a Conv-BN-ReLU-MaxPool), así que sus pesos no cargan
  directamente; que la carpeta del curso no tiene pesos de FundidoraPC, solo los de la Chimera de la
  semana 5; y una lista de preguntas para el equipo y el profesor.
- **Qué se hizo a mano:** se fijó `unificado_semana09` como rama base y se acordó que el registro de
  IA lo llevan Gustavo y la IA.

### 13. Plan v2 de la Semana 10: explorar todo dentro de las restricciones

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos.
- **Prompt exacto:**
  > 3.unificado_semana09
  >
  > el registro de IA usemoslo nosotros ,
  >
  > en cuanto a estas preguntas
  >
  > * ¿El Dice "por fragmento" se mide emparejando fragmentos o basta el Dice semántico por región?
  > * ¿Se permite refinar las cajas con la máscara semántica, o la detección debe medirse con la
  >   cabeza grid pura?
  > * ¿Aceptaría 384 px si la recomendación es 224–256?
  > * ¿Con qué variante y pesos de SAM debemos hacer el baseline?
  >
  > Avancemos con todo lo que podamos , armemos un plan donde exploremos todo , lo que nos pueda
  > ayudar , solo siguiendo las restricciones del profe , mas la recomendaciones da igual , por que al
  > final son recomendaciones
  >
  > y entonces actualicemos el plan , para empezar
- **Qué produjo la IA:** la versión 2 de `docs/plan_semana10.md`: separa restricciones de
  recomendaciones, resuelve las cuatro dudas reportando las dos lecturas y arma un mapa de
  exploración por frentes (datos, backbone, transfer learning, detección, instancia, entrenamiento)
  con prioridades y fases.
- **Qué se hizo a mano:** el plan quedó para revisión del equipo. Cuatro opciones (backbone con
  ImageNet, preentrenamiento autosupervisado, datos externos de pelvis, red 3D) quedaron como zona
  gris para confirmar con el profesor.

### 14. Diagnóstico del Dice por fragmento

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai). 3 a 5 de octubre de 2026.
- **Prompt:** "tenemos que hacer una evaluación exhaustiva para saber qué es lo que falla y cumplir
  con todas las métricas".
- **Qué produjo la IA:** `diagnostico_fragmentos.py`. Mide la estructura del ground truth (islas
  por fragmento, secundarios por región), la matriz de confusión píxel a píxel de la cabeza de
  instancia, seis variantes de post-proceso con la métrica del proyecto, un oráculo (la instancia
  usa la clase primario/secundario real) y el desglose por caso y por zona del volumen.
- **Qué se hizo a mano:** se corrió en CPU sobre `w10g_384neg` y se revisó caso por caso.
  Resultado: la cabeza de instancia clasifica como primario entre el 25 % y el 55 % de los píxeles
  secundarios; con la clase correcta, el Dice llegaría a 0.926. Los casos 012 y 039 concentran el
  30 % de la pérdida y los extremos del volumen son la zona más débil.

### 15. Revisión del estado del arte (challenge PENGWIN 2024)

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai).
- **Prompt:** "revisa estos repositorios y revisa a ver si aquí encontramos la solución para la
  semana 10" (con el sitio del challenge, Zenodo, Hugging Face y otros enlaces).
- **Qué produjo la IA:** un resumen del artículo del challenge (IEEE TMI, arXiv 2504.02382): el
  esquema primario/secundario fusiona fragmentos en contacto, y el ganador (MIC-DKFZ) usó la
  representación núcleo+borde (ABBC).
- **Qué se hizo a mano:** se verificó en los 100 volúmenes que el ID 1/11/21 siempre es el
  fragmento más grande, lo que descarta un error de etiquetas. Como la IA no podía ver el repositorio
  privado, el equipo exportó la información por consola.

### 16. Cabeza de bordes y post-proceso núcleo+borde

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai).
- **Prompt:** "debemos hallar otra solución" (tras el diagnóstico).
- **Qué produjo la IA:** `postproceso_bordes.py` (post-proceso estilo ABBC con la cabeza de bordes
  existente) y `train_boundary_head.py` (reentrena solo esa cabeza con un borde engrosado, un
  `pos_weight` calculado del desbalance real y un Dice global). Detectó que la cabeza de bordes
  estaba muerta en todos los modelos (probabilidad máxima 0.0005), por `pos_weight=3` y un Dice por
  corte con suavizado +1 que premia predecir cero.
- **Qué se hizo a mano:** se reentrenó la cabeza sobre las features congeladas. Se reactivó, pero el
  Dice por fragmento solo subió 0.001.

### 17. Experimentos de entrenamiento

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai).
- **Prompts:** "q vamos a probar?" y similares.
- **Qué produjo la IA:** notebooks de Kaggle para E1 y E2 (repesar secundarios ×6 y Dice de
  instancia ×3), E3 (entrada 2.5D con 5 cortes), E4 (32 canales) y F1–F2 (ajuste fino con
  sobremuestreo de secundarios, EMA y pérdida focal).
- **Qué se hizo a mano:** las corridas en Kaggle. Ninguna mejoró el Dice por fragmento: repesar
  0.820 / 0.816; 2.5D 0.812 en test; ajuste fino 0.854 en validación contra 0.857. Al correr E3 y E4
  en paralelo se agotó la RAM de Kaggle.

### 18. Evaluación de la receta c2 (32 canales) y post-procesos

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai).
- **Prompt:** "dame los pasos exactos de cada cosa que hago".
- **Qué produjo la IA:** `comparar_postprocesos.py` (compara post-procesos sobre las mismas
  predicciones, sin volver a inferir), el notebook `pengwin_c2_postproceso.ipynb` y la regeneración
  de `build_case_cache.py`, que faltaba en la rama. Encontró dos errores en el código del equipo:
  `evaluate_volumes.py` fallaba cuando el último lote tenía un solo corte (casos 067 y 095), y
  `min_size=12` borraba fragmentos pequeños.
- **Qué se hizo a mano:** se corrieron las tres semillas. `min_size=1` mejoró las tres (+0.003) y
  bajó los secundarios perdidos del 11.7 % al 7.9 %, pero solo ~51 % de los fragmentos predichos
  tenía pareja real; el post-proceso 3D empeoró (0.825). La configuración final se congeló con
  `min_size=12` (ver entrada 20).

### 19. Protocolo de evaluación

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai).
- **Prompt:** "entiende que así sean 0.002 no lo podemos dejar así".
- **Qué produjo la IA:** una regla de decisión fijada antes de ver resultados: todo se elige en
  validación; test se corre una sola vez; una variante se adopta solo si mejora ≥ 0.003, con
  bootstrap por paciente (P ≥ 0.90) y mejora en las dos mitades de validación.
- **Qué se hizo a mano:** el equipo adoptó la regla. Cada consulta adicional a test quedó declarada
  (ver Parte 2).

### 20. Últimos intentos de post-proceso

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai).
- **Prompts:** propuestas traídas por el equipo desde otras fuentes (watershed con EDT en mm,
  histéresis guiada por semillas).
- **Qué produjo la IA:** `watershed_bootstrap.py` y `casr.py`, y la evaluación de cada propuesta
  contra lo ya probado. Descartó usar la cabeza de detección como compuerta, porque detecta regiones
  y no fragmentos.
- **Qué se hizo a mano:** las corridas. El ensamble de 3 semillas dio 0.856; el empujón a
  secundarios, 0.857; el watershed, entre 0.776 y 0.851. La mejor configuración de CASR dio 0.8590
  (+0.0022) con P(mejora) = 0.83 y una mitad negativa, así que no cumplió la regla. Combinado con EMA
  (5 modelos), la validación llegó a 0.859, pero la regla falló por 0.0005. Test se consultó una
  segunda vez, declarándolo, y dio 0.847 ± 0.003: la mejora no se transfirió. Se congeló la
  configuración sin CASR.

### 21. Campaña de experimentos de la Semana 10 (familias B a F)

- **Integrante y herramienta:** Gustavo · Claude, app de escritorio (Cowork), `claude-sonnet-5-5`.
  5 de octubre de 2026, rama `semana10-multiples-pruebas-working`.
- **Prompt (resumen; el texto exacto no se conservó):** rehacer la Semana 10 desde cero siguiendo
  `docs/plan_semana10.md`, lanzar los entrenamientos en Kaggle y Colab, probar un cambio a la vez y
  llevar una bitácora de decisiones.
- **Qué produjo la IA:** `src/build_case_cache.py` (caché por caso a 384 px), `src/train_s10.py`,
  `src/evaluate_volumes.py` (las nueve métricas sobre el volumen completo, con caja cruda y
  refinada), `src/volume_postprocess.py`, `src/infer_volume.py`, los lanzadores de `kaggle/` y unas
  25 corridas: ancho del backbone (8, 16, 32, 48), CBAM (ninguno, 1 etapa, 3 etapas), peso a
  secundarios, 2.5D, focal, Tversky, supervisión profunda, EMA, sobremuestreo, preentrenamiento en
  Intel Image y λ.
- **Qué se hizo a mano:** se adoptó la regla de aceptar un cambio solo si supera el ruido medido con
  tres semillas (±0.004). Entraron el ancho 32 y EMA (receta D5); se descartaron 2.5D, focal,
  Tversky, supervisión profunda, sobremuestreo, ancho 48, λ ×2 y la instancia con 31 clases. El
  registro técnico completo está en `docs/decisiones_semana10.md`.

### 22. Distancia en mm, latencia, SAM y primera evaluación en test

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos. 5 de octubre de 2026.
- **Prompts exactos:**
  > Listo, pues demolea eso. Empecemos.  Recuerda actualizar también la bitácora que tenemos de las decisiones que vamos tomando, ¿vale?

  > corrres : benchmark_latency tu?  y      Siii , dale descargalo , dale

  > manda commit , y   siii dale vamos con lo que falta

  > Dale, dale, sí, con uno o dos piensas que, que nos puede ir bien, bien. Dale. Igual, pues, no sé si estamos dándole test a un solo modelo. Y deberíamos darle como a todos los que pasaron el umbral. Darle test a todos los que pasaron el umbral. No sé cómo lo es. Sí.
- **Qué produjo la IA:** `--distance-nofill` en `infer_volume.py` y `fill_unlabeled` en
  `volume_postprocess.py`; `src/benchmark_latency.py`; `src/sam_baseline.py` (SAM ViT-B zero-shot
  con la caja predicha como prompt); y la configuración congelada por escrito antes de tocar test
  (D5 semilla 42, un modelo con TTA, una sola evaluación).
- **Qué se hizo a mano:** se aceptó congelar antes de evaluar. Gustavo pidió además pasar por test
  todos los checkpoints que superaban el umbral en validación; se hizo con la condición, escrita
  antes, de que sirvieran para medir la variación y no para elegir modelo. Resultado oficial de la
  primera evaluación: 0.845 de Dice por fragmento.

### 23. Diagnóstico tras test, serie H y segunda evaluación

- **Integrante y herramienta:** Gustavo · Claude (Cowork), `claude-opus-5-5`. 6 de octubre de 2026.
- **Prompts exactos:**
  > Hola, hola. Listo. Pues sí, o sea, hagamos lo, lo que sea necesario y corramos a otro modelo o lo que sea para que eh, lo hagamos real. Y analicemos, o sea, analicemos fotos o analicemos métricas, o sea, pensemos, sentémonos a pensar un momentico eh, y miremos qué puede ser para que nos funcione, para que logremos todas las métricas en test.

  > ¿Cómo puedo hacer para ver esos casos yo visualmente, tanto el 12 como el 39?  Y bueno, los 87 y 78 quedan da cero.            podemos ver todo esto en un notebook y revisar qué se puede hacer ahí             por ejemplo, una que tú dices que el paciente está girado en el escáner. Pues se pone bien, ¿no? ¿Se gira como debe estar o qué, o qué se puede hacer en esos casos?        Un preprocesamiento de imagen, ¿no? ¿O qué? ¿Se gira o qué?        ¿Se validan como ese tipo de imágenes antes de que entren al modelo y se giran, o qué no sé? Aquí qué hacemos.

  > y simplemente como preprocesar esa foto no sirve como de, de alguna forma detectarlo con código algorítmicamente que está rota la foto y, y rotarla nosotros mismos no no podríamos hacer eso eh, entrenamiento o en inferencia no sé cómo lo ves

  > Siiii corramos test , o ya corrimos?

  > corramoslo varias en paralelo y corramoslo en  GPU ,
- **Qué produjo la IA:** el análisis de fallos por caso (`src/analisis_instancias.py`,
  `src/diagnostico_casos.py`, notebook `05_casos_dificiles`), que encontró que el test lo hundían
  los casos 012 (paciente girado unos 26°) y 039. De ahí salieron el aumento geométrico
  (`--aug-geo`, `--aug-rot-deg`), la orientación canónica según la cabecera del `.mha`
  (`src/orientacion.py`, `--canonical`), la prueba de estrés sobre validación (`--stress`) y los
  modelos H1 a H6. La idea de Gustavo de detectar el giro con código se implementó como
  `orientacion.estimate_angle`.
- **Qué se hizo a mano:** se eligió H6 como modelo final, solo con validación y estrés, y se hizo una
  segunda evaluación en test con la configuración escrita de antemano (0.875). Se descartaron el
  detector geométrico de giro (al revisar a ojo los 16 casos que marcaba como inclinados, 13 estaban
  bien colocados), los filtros de post-proceso y el ensamble como modelo principal.

### 24. Cierre de la Semana 10: notebook 04, SAM en test y documentos

- **Integrante y herramienta:** Gustavo · Claude (Cowork), `claude-opus-5-5`. 6 de octubre de 2026.
- **Prompts exactos:**
  > ya terminamos semana 10?   que faltaria para semana 11? y que nos falta actualizar

  > Listo, entonces quiero que me respondas primero estas preguntas. Luego quiero que vamos por lo que falta en la semana 10 para dejar todo listo. Listo, listo, y también, si quieres    Dame un texto para ponerle a la otra sesión en la que estoy trabajando. Dame un texto para ponérselo a la sesión en la que estoy terminando, la de la semana 11, ¿vale? También, en donde tienen que documentar lo que hagamos en la semana 11. Acá, actualicemos los archivos y los que dices que están desactualizados. Actualicémoslos, por favor.
- **Qué produjo la IA:** el notebook `notebooks/04_semana10_pipeline_multitarea.ipynb`, que lee los
  resultados guardados; `src/resumen_s10.py` y `src/informe_test.py` (tablas, matriz de confusión y
  los cinco peores casos de test con H6); SAM repetido sobre test con H6 (Dice 0.889 de SAM contra
  0.935 del modelo propio); y la actualización de `README.md`, `docs/resumen_semana10.md`,
  `docs/plan_semana10.md` y `outputs/s10/resumen_experimentos.md`.
- **Qué se hizo a mano:** Gustavo pidió no tocar todavía la demo ni el dashboard, que siguen con D5;
  el paso a H6 quedó para la Semana 11.

## Semana 11 — visualizadores, dashboard y despliegue

### 25. Plan de la Semana 11: visualizadores y dashboard

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos.
- **Prompt exacto:**
  > Necesito hacer el visualizador para este proyecto. Miremos a ver si ya tenemos todo para hacerlo,
  > sí, al menos con algunos modelos que mejor han tenido resultado, y damos el visualizador, ¿sí?
  > Como él lo pone ahí O sea, deberíamos poder terminar la semana 11, sí. Terminar la semana 11.
  > Entonces, armemos un plan y revisemos si ya tenemos lo necesario, al menos, para poder hacer esta
  > semana 11, que es el visualizador. hagamos el plan
- **Qué produjo la IA:** un inventario del repo frente a la sección 3.4 del enunciado y
  `docs/plan_semana11.md`: el lado del modelo alcanza para los visualizadores 2 y 3, pero no existe
  nada del lado de la aplicación. Propuso separar la exportación (lenta, una vez por caso) del
  dashboard (solo lee), usar Streamlit y desarrollar con casos de validación, dejando test para
  después de congelar el modelo. La versión 2 del plan tomó D5 como modelo principal e incorporó como
  primer bloqueo que la distancia predicha salía 0.0 mm en los 41 secundarios de validación.
- **Qué se hizo a mano:** se aceptaron Streamlit, la demo en el Mac con túnel de Cloudflare, el
  Visualizador 1 con MIP e isosuperficie y el avance del miércoles 7 de octubre.

### 26. Construcción del dashboard (visualizadores 1, 2 y 3)

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos.
- **Prompt exacto:**
  > Sí, ahí ya tenemos los modelos locales, los .pth. La corrección de distancia es necesaria para
  > esto. No, la otra sección de Claude está trabajando en eso. Empecemos con lo que podamos ir
  > construyendo, ¿vale?
  >
  > De bueno, ahí sería bueno crear una nueva rama tipo semana 11 para esta vaina, listo para hacer
  > esta vaina. No sé si ahora o después, porque estamos esperando los cambios de que haga la otra
  > sesión de Claude.
- **Qué produjo la IA:** `src/predict_volume.py`, `src/demo_export.py`, `src/demo_render.py`
  (visualizadores 2 y 3), `src/raw_view.py` (MIP e isosuperficie del Visualizador 1),
  `src/measure_latency.py`, `app/dashboard.py` (Streamlit), 15 pruebas nuevas y la guía
  `docs/semana11_dashboard.md`. Encontró que el 0.0 mm del caso 024 venía de una fuga de 7 vóxeles en
  `postprocess_3d` y lo corrigió dejando solo la componente mayor del principal (19.0 mm predichos
  para el fragmento real a 19.0 mm).
- **Qué se hizo a mano:** se construyeron solo archivos nuevos mientras otra sesión corregía la
  distancia, y la rama de la Semana 11 se creó a mano. El dashboard oculta los fragmentos menores a
  1000 mm³ (filtro de presentación; la métrica no cambia).

### 27. Despliegue (Streamlit Cloud + API en AWS) y CI/CD

- **Integrante y herramienta:** Gustavo · Claude (Cowork), mismos modelos.
- **Prompts exactos:**
  > Y si subimos el modelo AWS que yo tengo en AWS y el Streamlit lo deployamos en Streamlit Cloud

  > Pero vení, o sea, ¿crees que en un lambda está bien? Porque creo que hay que poner con GPU y sin
  > GPU. No hay un mejor servicio en, en AWS para esto, no sé, SageMaker, Bedrock, no sé. O, o qué se
  > usa normalmente en estos casos..

  > No, las pruebas deberían correrse. Cuando se hace un PR a main, debería correrse este pipeline.
  > Cuando se haga el match, ya cuando estén haciendo cambios en main.
- **Qué produjo la IA:** la separación entre el dashboard (Streamlit Cloud, sin torch) y la
  inferencia en vivo (`api/server.py`, FastAPI en un contenedor para AWS Lambda);
  `deploy/build_streamlit_cloud.py`; `docs/despliegue_aws_streamlit.md`; y dos flujos de GitHub
  Actions, `ci.yml` (pruebas en cada PR a main) y `deploy-api.yml` (imagen a ECR y actualización de
  la Lambda), con la guía `docs/cicd_github_actions.md`. Descartó Hugging Face Spaces porque en
  cuentas gratis el SDK Docker figura como pago.
- **Qué se hizo a mano:** se eligió Lambda en CPU en lugar de SageMaker (el modelo tiene 2.74 M de
  parámetros y no necesita GPU para la demo). La latencia en GPU se mide aparte con
  `src/measure_latency.py` en Kaggle o Colab.

### 28. Cierre de la Semana 8 y reorganización de este registro

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai). 6 de octubre de 2026.
- **Prompt (fragmento clave):** "verificar el MIP con este comando y seguir con los notebooks
  01–03", con el contexto del proyecto y el estado de los entregables.
- **Qué produjo la IA:**
  - comandos de inspección del repo por consola, que mostraron que el MIP ya existía en
    `src/raw_view.py` pero el notebook 01 y `resumen_semana8.md` solo documentaban la
    reconstrucción 3D;
  - `agregar_celdas_notebooks.py`: celdas de objetivo (con la aclaración del formato `.mha`) y de
    conclusiones en los notebooks 01–03, y la sección 6.1 del notebook 01 con el MIP en tres ejes
    sobre un caso estándar y uno girado 180°;
  - `parche_resumen_semana8.py`, que describe el Visualizador 1 como MIP + 3D;
  - el flujo de git con `git worktree` en una rama propia (`docs/cierre-semana8`) para no tocar las
    ramas del equipo, y el texto del PR #4;
  - la confirmación de que `notebooks/05_casos_dificiles_ejecutado.ipynb` se ejecutó con D5
    (`d5_ema_w32_e60_s42.pth`) y de que el modelo entregado es H6 s42;
  - la unificación de este archivo en las dos partes del enunciado, incluidas las entradas 21 a 24;
  - la revisión del PR #6 (publicada con `gh pr review`) y la combinación del README de los PR #5 y
    #6;
  - la actualización del informe con la calibración de los pesos de la pérdida, la matriz de
    confusión, los cinco peores casos y la comparación con SAM, a partir de los resultados de
    Gustavo.
- **Qué se hizo a mano:** la ejecución local de la celda 6.1 (`outputs/figures/mip_visualizador1.png`),
  descartar los HTML del visualizador 3D que se regeneraron al correr el notebook, la revisión de los
  diffs y el PR, revisado por Federico.

### 30. Despliegue real en AWS (cuenta de Almia) y CI/CD con OIDC

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-sonnet-5-5`.
  6 de octubre de 2026.
- **Prompts exactos:**
  > estamos haciendo acá la semana 11 , quiero desplegarel streamlit y el modelo en lambda o lo que veas mejor  la idea es que sea con un CI/CD cuando se pushee a main

  > Sii ejecuta eso en la cuenta de almia ,       2.  yo ya ttengo admin en ese proyecto , de eso ahora se llama : https://github.com/pengwin-analitica-2026/proyecto2-analitica-datos/tree/semana11-visualizadores

  > 1. dale prueba y mira  si puedes hacer todo , intentalo ,         2. hiciste el commit dandote autoria , eso no se deberría hacer , no debes darte autoría ,  si podemos cambiar el commit buenisimo
- **Qué produjo la IA:** en la cuenta de AWS de Almia (`us-east-2`) creó el repositorio ECR
  `pelvis-api`, la función Lambda `pelvis-api` con su Function URL, el rol de ejecución y un rol de
  despliegue que GitHub asume por OIDC, limitado a la rama `main` de este repositorio y a ese ECR y
  esa Lambda (`deploy/aws/`). Cambió `deploy-api.yml` de claves de acceso a OIDC y a desplegar en
  cada push a `main`. Al probar contra la Lambda real encontró y corrigió cuatro fallos: permisos de
  archivo en la imagen (502), el arranque en frío más largo que el chequeo del adaptador (`/health`
  liviano y `/health?deep=1`), `pytest` faltante en el CI y el formato inmutable del `sub` de OIDC
  en este repositorio. Detalle en `docs/cicd_github_actions.md`.
- **Qué se hizo a mano:** Gustavo eligió Streamlit Community Cloud para el dashboard y la cuenta de
  Almia para la API, autorizó crear cada recurso y la URL pública, integró el PR #6 a `main` y
  habilitó las *deploy keys* de la organización para que Streamlit Cloud pudiera leer el repo
  privado. Pidió quitar la coautoría de la IA de los commits; el commit se reescribió.
- **Análisis crítico:** la IA puso en el commit una línea de coautoría que el equipo no quería, y su
  primera política OIDC usaba el `sub` por nombre, que este repositorio no emite; el error solo
  apareció en el primer despliegue real. Los dos bloqueos de permisos (URL pública y merge sin
  revisión) los detuvo el propio entorno y se resolvieron con autorización explícita.

### 31. Demo con H6: orientación desde la cabecera, distancia definitiva y métricas de test

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  6 de octubre de 2026.
- **Prompt exacto (resumen del mensaje, que incluía las instrucciones de la sesión de la Semana 10
  y la revisión del PR #6 de María):**
  > Listo, ya está desplegando. Ahorita nosotros tenemos pendientes las cosas en la semana 11 y varias cositas que cambiaron, entonces sería bueno cambiarlo.  Acá te pongo lo que me pidió otra sesión que te pusiera, sí, la sesión que está terminando la semana 10. […] también, en el anterior PR que subimos, me hicieron estos comentarios. Sería bueno revisar esto. También sé que se subió algo. Se han subido cambios de parte de otra compañera. Sería bueno revisar y mirar que no haya conflictos y demás.

  > No, no, no, no vamos a montar nada en Cloudflare. Ya montamos todo en la lambda. Cloudflare es opcional.
- **Qué produjo la IA:** `orientacion.header_is_rotated` y `flip_inplane`; el enderezado en
  `demo_export.py` (una sola vez al cargar, para imagen, ground truth y vista cruda, decidido por la
  marca `canonical` del checkpoint y la cabecera del `.mha`); el reemplazo del método provisional
  de distancia por `postprocess_3d(fill_unlabeled=False)`; la reexportación de 024, 044, 099 y 039
  con H6 y TTA y el borrado de los casos de D5; H6 como checkpoint por defecto en la API, el
  Dockerfile y el dashboard; la pestaña de métricas con las dos evaluaciones en test, la distancia,
  SAM y la latencia; el orden de los notebooks (Colab en `notebooks/colab/`, 05 rotulado como D5);
  los pesos de H6 s43 y s44 en el repo; y la actualización de esta documentación.
- **Qué se hizo a mano:** Gustavo descartó montar el túnel de Cloudflare (la IA había empezado un
  script para eso y se borró): la demo es Streamlit Cloud + Lambda. La comprobación visual de los
  casos girados la hace el equipo en el dashboard.
- **Análisis crítico:** la revisión de María pedía `from_canonical` a la salida; la IA propuso no
  volver a la orientación original y mostrar todo en la canónica, porque así las tres vistas y el
  ground truth coinciden y la distancia no cambia. Es una decisión de presentación que el equipo
  puede revertir. La IA creyó muerta la primera exportación (su salida quedaba retenida por un
  `grep` en el pipe), la relanzó y las dos corrieron a la vez sobre la misma carpeta; se detectó al
  llegar el aviso de fin y se verificó que los archivos finales quedaran íntegros.

### 32. Mini-paper y licencia de los datos en el model card

- **Integrante y herramienta:** Gustavo · Claude (Cowork), `claude-opus-5-5`. 6 de octubre de 2026.
- **Prompt exacto:**
  > Creo que ya hicimos todo , igual valida , y falta :  * el mini-paper. * La licencia de PENGWIN en el model card.
- **Qué produjo la IA:** el mini-paper de 5 páginas (`docs/mini_paper/`, en `.docx` y PDF, con su
  `build.js` y el script de las dos figuras nuevas); la licencia de los datos (CC BY 4.0) y las
  referencias en la sección 2 del model card; la tabla de latencia de la sección 4.8 con las
  mediciones que ya estaban en `outputs/`; y una revisión del estado del repositorio contra los
  entregables.
- **Qué se hizo a mano:** el equipo revisa la redacción del mini-paper y decide si el formato y la
  extensión son los que pide el profesor; la IA no consultó el enunciado (estaba adjunto en la
  conversación desde el 4 de octubre) y eligió una estructura de artículo corto. Se corrigió en la
  entrada 33.
- **Análisis crítico:** la licencia no se escribió de memoria: se consultó el registro de Zenodo del
  dataset, y ahí se vio que el DOI que aparecía en un README generado para Hugging Face
  (`10990767`) es el del documento de diseño del reto y no el de los datos. Las cifras del
  mini-paper se contrastaron con un script contra los JSON de `outputs/s10/`. El mini-paper no trae
  resultados nuevos; repite el aviso de que el 0.875 es una segunda evaluación en test. La cifra
  del mejor método del reto (IoU 0.930) se tomó del resumen del artículo del reto y no es
  comparable con las nuestras.

### 33. Limpieza de la documentación de las semanas 10 y 11

- **Integrante y herramienta:** Gustavo · Claude (Cowork), `claude-opus-5-5`. 6 de octubre de 2026.
- **Prompts exactos:**
  > quisiera limpiar las documentaciones , revisar cuales  son las definitivas , de acuerdo con lo que hicimos ,  de la semana 10 para arriba ,  porque veo un poco de docs de semana 10  que no se cual o cuales son los definitivos , también se que algunos companeros estuvieron documentando , perro no habiamos terminado la semana 10 y 11 , entonces faltaba por documentar bien , entonces terminemos de documentar bien porfa y dejar solo lo que es y lo definitivo

  > Lo importante es no duplicar información, sino dejar solo los docs que no dupliquen información, o sea, solo dejar los docs que expliquen cada parte de su proceso y no duplicar información.

  > Pues aquí te paso lo que dicen los entregables finales. Dice nomás el tipo paper, pero me gusta ese informe, la verdad. […] Mira que el IA usage.md dice todavía Cloudflare, entonces creería yo que no está actualizado, ¿no? […] revisemos que IA use model card y README estén actualizados
- **Qué produjo la IA:** el inventario de documentos y notebooks con su autor y fecha; la
  integración de la rama `docs/latencia-distancia` de María; el retiro de los planes de las semanas
  10 y 11, de los cambios de la primera versión de la Semana 10, del resumen de la Semana 10, de
  dos copias del notebook 05 y de los cuadernos de Colab anteriores; el cuaderno de Colab regenerado
  con la receta de H6; el cierre de la bitácora; las correcciones del informe técnico (orientación
  en la demo, conteo de metal, tiempo de entrenamiento y consultas a test); la revisión de los
  documentos contra la lista de entregables del enunciado; y la ampliación del mini-paper con lo
  que el enunciado exige y le faltaba (motivo de cada decisión de arquitectura, justificación de los
  pesos de la pérdida, matriz de confusión, galería de los cinco peores casos y tabla de distancias).
- **Qué se hizo a mano:** Gustavo decidió borrar en vez de archivar, incluir los notebooks y el
  criterio de un documento por tema sin repetir. La IA preguntó antes de borrar y mostró la lista.
  También decidió que el entregable del enunciado es el mini-paper y que el informe largo queda
  como versión extendida. Gustavo probó el dashboard publicado desde el celular y confirmó que la
  API en Lambda ya corre con H6.
- **Análisis crítico:** la IA había escrito el resumen de la Semana 10 un día antes y ese mismo
  documento resultó ser el que más repetía a los demás; con el criterio de Gustavo se retiró. La
  discrepancia del metal (41 casos en el informe y 25 en el model card) no era un error de ninguno
  de los dos: son 41 casos con algún vóxel sobre 3000 HU y 25 con más de 100 mm³, y se comprobó
  contra `data/eda_intensidades_detalle.csv` antes de corregir. Dos errores de la IA en este
  tramo: escribió el mini-paper (entrada 32) sin releer el enunciado, que estaba adjunto, y por eso
  le faltaban la matriz de confusión y la galería de los cinco peores casos; y la tabla 2.4 de este
  archivo seguía dando el túnel de Cloudflare como decisión final cuando el despliegue real es
  Streamlit Cloud y Lambda. Los dos los detectó Gustavo al volver a pasar la lista de entregables.
  Al recalcular la matriz de confusión apareció además un redondeo mal hecho en el informe largo
  (recall del coxal izquierdo 0.996, no 0.997).

## Corrección del profesor — orientación, EMA e imágenes "feas"

### 34. Orientación con metadatos, H7 sin EMA, robustez a ruido y dependencia de CBAM

- **Integrante y herramienta:** María · Claude Opus 5.5 (claude.ai). 9 a 12 de octubre de 2026,
  rama `feat/normalizacion-orientacion`.
- **Prompt (fragmentos clave; el texto exacto está en el chat):** la transcripción de la asesoría
  (*"esas rotaciones toca hacerlas a través de productos punto y productos cruz con unos metadatos…
  y el EMA es mejor que no lo utilicemos"*, *"tengo que poner un pipeline super robusto"*), con el
  pedido de cumplirla; luego "dame el notebook todo corregido… que en el entrenamiento yo vea los
  outputs", "lo del aporte del CBAM para cada cabeza… de este modelo" y la lista de mejoras que
  propuso Jacobo (enderezado en casos sin giro, suavizado contra el ruido).
- **Qué produjo la IA:**
  - `src/normalizacion.py`: paso 1, orientación a RAI con productos punto y cruz sobre la cabecera
    (método del artículo del profesor, doi:10.1016/j.mex.2024.103073); paso 2, enderezado del giro
    del paciente por la simetría izquierda-derecha del hueso, en milímetros; con sus pruebas
    (`tests/test_normalizacion.py`, `tests/test_evaluar_normalizado.py`).
  - El giro de entrenamiento en milímetros (`--aug-mm`) y, en `evaluate_volumes.py`, las opciones
    `--normalizar`, `--sin-cbam` y las pruebas de estrés `rotmm=`, `noise=`, `blur=` y `suav=`.
  - Los notebooks 07 (rotación con H6), 09 (entrenamiento de H7 en Kaggle, validación y tercera
    evaluación en test), 10 (ruido, suavizado y CBAM apagado) y 09b (H7 con EMA, solo para
    comparar), y el resumen `outputs/h7/RESUMEN_COMPLETO.pdf`.
- **Qué se hizo a mano:** el entrenamiento de H7 en Kaggle (Colab gratis no alcanzó en RAM), todas
  las corridas de validación, la decisión de que H7 es el modelo final, y la tercera evaluación en
  test, una sola vez y con la configuración decidida en validación (0.879). El equipo (Jacobo) revisó
  el resumen y pidió las pruebas de enderezado sin giro y de suavizado.
- **Análisis crítico:**
  - La primera prueba de estrés de la IA giraba la imagen en píxeles. En el caso 044 (píxeles de
    0.84 × 0.42 mm) el enderezado estimaba +2.8° para un giro de 30°; al revisarlo se vio que el giro
    en píxeles deformaba la pelvis en 10 de los 15 casos de validación. Se corrigió con `rotmm=`.
  - La IA estimó mal la memoria: el entrenamiento falló dos veces en Colab gratis (sin memoria y
    *Bus error*) antes de pasar a Kaggle.
  - En el resumen, la IA mezcló cifras de dos experimentos de CBAM (mAP50 de la Semana 10 con el F1
    de la ablación final) y exageró la conclusión ("H6 y H7 parejos", "H7 igual o mejor en todo").
    Lo detectó Jacobo al revisar; se corrigió con las cifras del model card.
  - En el parche de suavizado propuesto en la revisión, el filtro se aplicaba antes de recortar la
    imagen a 0–255, lo que cancela el ruido negativo del fondo y da un resultado optimista. La IA lo
    cambió para recortar primero.
  - Apagar CBAM al predecir (F1 de 0.992 a 0.444) mide dependencia, no aporte; la IA lo aclaró antes
    de reportarlo para no confundirlo con la ablación entrenada.


### 36. Modelo final H8: experimentos, robustez, métricas finales y visualizadores

- **Integrante y herramienta:** Jacobo · Claude Code, extensión de VS Code. `claude-sonnet-5-5`
  para la revisión inicial del repositorio y de la rama de María (11 y 12 de octubre de 2026) y
  `claude-opus-5-5` desde los experimentos (12 al 16 de octubre). Ramas `feat/normalizacion-orientacion`
  y `modelo_h8`.
- **Prompts exactos** (copiados tal cual; `[…]` marca texto omitido por largo):
  > ahora lo que quiero es hacer unos experimentos. primero este que corrio maria en la rama tengo entendido que es sin EMA pero con el cambio de rotacion, entonces seriaa el primer experimento ya hecho […] Luego el segundo es hacer esa rotacion pero con EMA […] pero antes de ejcutar y hacerlo verifica que todo lo que te digo es cierto y si concuerda cada cosa para asi hacer una buena comparacion de las metricas de cada uno

  > si realiza todo eso tu solo, que no dependas de mi, continua hasta que tengas las respuestas de todos estros experimentos […] 3, Si realiza el H6 sin EMA tambien quisiera hacer este experimento dime como lo ves: Crea y lanza en Kaggle el experimento H8 = receta de H7 + aumento de ruido gaussiano. Solo validación, no tocar test. […] Por ultimo crea una tabla compartiva con todos estos experimentos.

  > quiero que despues ya se haya ejecutado todo, crees un .md donde expliques y documentes cada experimento que es y que hace, luego hagas conlusone sy analissid e la comparacfcion y una conclusion del mejor modelo elegido y su justificacion

  > bien como el H8 fue el ganador ahora quiero que corras igualmente en kaggle este modelo para ver las metricas en test […] **Punto 8 – Aportación de CBAM por Cabeza** […] **Punto 9 – Métricas HD95 y ASSD** […]

  > bueno ya verifique con mi grupo de equipo y damos como mejor modelo definitivo este H8, entonces quiero que añades este modelo ya en el repositorio con el notebook con las salidas, los outputs, el .pth y demas cosas relacionadas. crea una rama que se llame modelo_h8, pero solo quiero que hagas los cambios en local […] que pueda correr en el demo stremlit y dashboard

  > Estamos en la rama modelo_h8. Quiero mejorar los visualizadores 3D del dashboard solo a nivel visual, sin tocar el modelo, el preprocesamiento de entrada (window_hu), las métricas ni las distancias en mm. […]

  > una pregunta la comparacion de SAM zero shot vs H8 esta? — si realizalo

  > si realiza de 1 a 4, que todo se ejecute bien y quede guardado estas salidas, añade estas interpretaciones nuevas que se realicen despues de los resultados en los docs correspondientes

  > bien ahora quiero que ejecutes el modelo h8 con dos semillas más para probar si varía el resultado con esto […]
- **Qué produjo la IA:** la comparación de H6, H6 sin EMA, H7, H7 con EMA y H8 en validación;
  el aumento de ruido de H8; las evaluaciones finales, ablaciones, SAM zero-shot, HD95/ASSD,
  distancias, notebook y documentación; la integración del modelo H8, sus salidas y los kernels de
  Kaggle; y las pruebas de semillas 43 y 44 con una regla previa que impide usarlas para elegir el
  modelo.
- **Qué se hizo a mano:** Jacobo definió las pruebas, autorizó los entrenamientos de Kaggle y las
  subidas de datos; el equipo eligió H8 con resultados de validación. Jacobo revisó los resultados,
  resolvió conflictos al integrar y realizó los commits, push y PR.
- **Análisis crítico:** la IA exageró inicialmente la robustez de H8; el equipo pidió precisar que
  ante ruido σ=25 y giro de 60° sin enderezar no alcanza el objetivo. También se detectaron y
  corrigieron problemas de ejecución de kernels, pruebas heredadas, redondeos de SAM y una prueba de
  relleno 3D que pasaba incluso sin el relleno. Los resultados desfavorables —por ejemplo el error
  de distancia de H8 frente a H6— se mantuvieron reportados.

### 37. Subir un CT propio al dashboard (.mha, NIfTI o carpeta DICOM)

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  15 de octubre de 2026, rama `feat/subir-ct` sobre `modelo_h8`.
- **Prompt exacto:**
  > Listo. Algo importante es que queremos también poder nosotros subir un punto de MHA o subir una carpeta con todos los cortes. Eso fue lo que el profe nos dijo que quería: él quiere poder subir un punto de MHA y, pum, que se vean las cosas, se genere la predicción, todo eso, y se puede empezar a trabajar con eso. Sí, y también una  carpeta con los cortes
- **Qué produjo la IA:** `src/volumen_entrada.py` (lectura de `.mha`/NIfTI/NRRD y de series DICOM,
  orientación con `normalizacion.a_orientacion_estandar`, sin torch); la separación de
  `demo_export.export_case` en `exportar_volumen` (volumen ya cargado + función que predice);
  `src/inferencia_remota.py` y el endpoint `POST /predict_batch` de la API para inferir el volumen
  en AWS cuando el dashboard corre en Streamlit Cloud; la página «Subir un CT» del dashboard; y las
  pruebas `tests/test_volumen_entrada.py` y las nuevas de `tests/test_api_server.py`.
- **Qué se hizo a mano:** la decisión de soportar `.mha` y carpeta de cortes, pedida por el profesor.
- **Análisis crítico:** se interpretó «carpeta con los cortes» como una serie DICOM; una carpeta de
  PNG/JPG se rechaza con explicación porque pierde las HU y el spacing. Para comprobar que el refactor
  no cambió nada se reexportó el caso 024 con H8 y salió idéntico al versionado (meta y arreglos).
  En Streamlit Cloud un volumen muy grande puede no caber en memoria; no se probó ahí todavía porque
  el endpoint nuevo llega a la Lambda solo al integrar a `main`.

### 38. Los 100 casos desde S3 y el preprocesamiento paso a paso

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  15 de octubre de 2026, rama `feat/subir-ct`.
- **Prompts exactos (fragmentos):**
  > algo que el profe pues, dijo que sería chévere es como poner todos los puntos de MHA, que creo que serían 100 […] que haya una sección o haya una parte donde yo pueda escoger cuál de los 100 […] primero nosotros queremos mostrar la imagen como viene, raw, como viene, ¿sí? Luego, pues la imagen de cómo se le, se le agrega pues el giro de 180 grados […] Me imagino que será mejor poner todo esto en un S3 para que los pueda leer de ahí Streamlit, ¿no?

  > me gustan los dos, la verdad. Sería bueno, pues, lo, tener los dos, tanto los precalculados como el procesar al elegir. […] crear el bucket en la cuenta de Almia, sí, créalo.
- **Qué produjo la IA:** el bucket privado en la cuenta de Almia, `deploy/aws/subir_crudos_s3.py`
  (los 100 casos comprimidos sin pérdida), `src/almacen_s3.py`, `deploy/precalcular_casos.py` (4
  procesos en paralelo, reanudable), `src/pasos_preproceso.py` y la pestaña «Preprocesamiento», el
  origen «Casos PENGWIN (100)» con los modos precalculado y procesar ahora, el usuario IAM de solo
  lectura para Streamlit Cloud, y las pruebas `tests/test_almacen_s3.py`.
- **Qué se hizo a mano:** Gustavo eligió tener los dos modos y autorizó el bucket. La clave del
  usuario de solo lectura la crea él y la pone en los Secrets de Streamlit Cloud.
- **Análisis crítico:** Gustavo prefería precalcular en Colab por la GPU; la IA midió que casi la
  mitad del tiempo por caso es CPU (distancias, mallas, vista cruda) y propuso el Mac con 4 procesos,
  con los datos ya en disco. La IA también creyó caídos los procesos de subida por una consulta mal
  hecha y lanzó duplicados; los detectó enseguida y los detuvo sin pérdida.

### 39. Cuantización de H8 en Kaggle

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  16 y 17 de octubre de 2026, rama `feat/cuantizacion-h8`.
- **Prompts exactos (fragmentos):**
  > También me gustaría cuantizar el modelo final que tenemos, Alcir, si se puede hacer. Quisiera primero entender todos los tipos de cuantizaciones que hay, qué es recomendable en ese tipo de casos y cuál sería la mejor a escoger y demás.

  > No, pero si vamos a cuantizar, quisiera que lo hagamos en Kaggle , no en local, por favor.

  > pero usemos la GPU de Kaggle no ? pa eso la tenemos ,    y si lanza con GPU los test que consideres

  > dale mandemos a test o kaggle todas las que sean necesarios
- **Qué produjo la IA:** la explicación de los tipos de cuantización (fp16, bf16, int8 dinámico y
  estático, QAT, int4); la recuperación de `src/cuantizar.py` de la rama `experimentos/pedidos-profesor`
  (hecho para H6) con soporte de GPU; los datasets privados de Kaggle (validación y test) y los kernels
  (`kaggle/h8/armar_cuantizacion.py`): fp32, int8, int8_todo, bf16 y fp16 en validación, en CPU y GPU, y
  en test solo las variantes elegidas (int8 para CPU, fp16 para GPU) más fp32 de referencia. Resultado:
  ninguna pierde Dice; int8 es 2.3 veces más rápido en CPU y fp16 1.8 veces en GPU.
- **Qué se hizo a mano:** Gustavo pidió correrlo en Kaggle y con GPU, generó el token de Kaggle y
  decidió mandar fp16 a test además de int8.
- **Análisis crítico:** la IA empezó a correrlo en el Mac antes de preguntar dónde; se detuvo al pedir
  Gustavo Kaggle. También afirmó que cuantizar las convoluciones transpuestas «rompía la segmentación»,
  que era cierto con H6 pero no con H8: al medirlo con H8 (Kaggle y Mac) el Dice no cambia, y la razón
  para dejarlas en fp32 es solo la velocidad. Se corrigió en `docs/cuantizacion.md`. Se mantuvo el
  protocolo: test una vez y solo con las variantes elegidas en validación (int8_todo y bf16 no fueron).

### 40. H8 en dos precisiones: int8 en CPU (Lambda) y fp16 en GPU

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  17 de octubre de 2026, rama `feat/precision-h8`.
- **Prompts exactos (fragmentos):**
  > tambien me gustaria  tener el H8 normal como lo tenemos normal y otro con el H8 cuantizado ... el modelo cuantizado no se puede tener listo de cuantizado?

  > ¿Qué nos recomiendas: dejar int8 híbrido para cuando escogamos CPU y FP16 cuando escogamos GPU, o qué hacemos?

  > Listo, sí, empecemos entonces con el PR1. Algo muy importante es seguir actualizando con las diferentes documentaciones que hay, sobre todo el IA usage y las demás. Sí, la Lambda quería en 8
- **Qué produjo la IA:** `src/precision.py` (fp32, int8 híbrido exportable ya cuantizado en TorchScript
  de 4.9 MB, fp16, y `auto` según el hardware), la calibración versionada
  (`outputs/s10/calibracion_int8_h8.npz`, 128 cortes de entrenamiento), `load_pipeline(precision=...)`,
  `PRECISION` en la API (la imagen genera el int8 al construirse), el bloque «Procesamiento» del dashboard
  (dónde corre el modelo y con qué precisión, con opción de GPU por `INFERENCE_URL_GPU`) y
  `tests/test_precision.py`.
- **Qué se hizo a mano:** Gustavo decidió que la Lambda use int8 y que la GPU quede como opción.
- **Análisis crítico:** el int8 exportado depende de la versión de torch con que se hizo; por eso no se
  versiona: la imagen de la API lo genera con su propio torch y, si no coincide, se vuelve a cuantizar
  al cargar. La GPU en SageMaker no quedó encendida por costo (~0.74 USD/hora). A pedido de Gustavo («arreglemos los
  tests, por favor, para que pasen todos») se corrigieron las dos pruebas que fallaban en el Mac: no
  eran código viejo sino una incompatibilidad de `train_s10.py` con torch 2.2 (`torch.amp.GradScaler`
  existe desde 2.3; se usa `torch.cuda.amp.GradScaler` si no está) y un caché de prueba en
  `tests/test_cuantizar.py` sin los campos (`spacing_xyz`, `orig_shape`) que el aumento en milímetros
  necesita. Con eso pasa toda la suite también en el Mac.

### 41. El dashboard publicado se caía al procesar un .mha: memoria

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  17 de octubre de 2026, rama `fix/memoria-subida`.
- **Prompt exacto:**
  > tambi';en despues de eso , mrevisa y mira que en produccion subi un .mha y procesarlo se me rompió producción  , sabes porque?
- **Qué produjo la IA:** el diagnóstico a partir del log de Streamlit Cloud (sin error de Python: el
  sistema mató la app por memoria) y la medición del pico de RAM paso a paso con el `.mha` que se subió
  (050, 3.43 GB frente a ~2.7 GB disponibles). Los arreglos: post-proceso, distancias y mallas en la caja
  del hueso, lectura directa en int16 sin copias, vista cruda con menos resolución en volúmenes grandes y
  un tope con mensaje claro. El pico bajó a 1.73 GB (050) y 2.55 GB (001, el más grande).
- **Qué se hizo a mano:** Gustavo encontró el fallo en producción y bajó el log.
- **Análisis crítico:** la subida de CT se había probado en el Mac, que tiene 68 GB de RAM, y en un
  contenedor sin límite de memoria, así que el problema no apareció; faltó medir la memoria contra el
  límite de Streamlit Cloud antes de publicar. La regresión con el caso 024 confirmó que el recorte no
  cambia ningún resultado.

### 42. GPU bajo demanda en SageMaker

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  17 de octubre de 2026, rama `feat/gpu-sagemaker`.
- **Prompts exactos (fragmentos):**
  > en lo de poder procesar en GPU, creo que con SageMaker, aparte del endpoint. Mira si ese es el mejor, si el XLarge G4 DN es el mejor, o si hay uno mejor para esos momentos.

  > Pues no entiendo por qué no se puede subir desde acá el de locales. ¿Cuál sería el problema? Bueno, sí, hagamos lo necesario para que todo funcione. Porfa, hagamos un PR.

  > sí, agrega el apagado automático a las 4 horas , y hay forma de con un botón ir prendiendo la GPU? y que pasa cuando se pushea a main , siempre se crea uno nuevo?
- **Qué produjo la IA:** la comparación de instancias con el cupo y el precio reales de la cuenta
  (T4 elegida); `api/Dockerfile.gpu`, `/ping` e `/invocations` en la API, el puente desde la Lambda
  (`X-Destino: gpu`) para no poner credenciales en Streamlit, la opción GPU del dashboard,
  `deploy/aws/gpu_sagemaker.sh` (imagen, preparar, encender, apagar, estado, borrar) y el workflow
  `build-gpu.yml`; el encendido desde un botón del dashboard con clave de administración y el apagado
  automático a las 4 horas con EventBridge Scheduler (`src/gpu_bajo_demanda.py`, `/gpu/*` en la API);
  la explicación de qué cambia con cada push a `main` y la regla de ECR que conserva 5 imágenes de la
  Lambda; pruebas del puente y del encendido/apagado con SageMaker y Scheduler simulados.
- **Qué se hizo a mano:** Gustavo eligió SageMaker y autorizó crear los recursos; decidió PR separados
  y pidió el apagado automático y el botón.
- **Análisis crítico:** la IA intentó subir la imagen de ~3 GB desde la conexión de la casa y falló en
  cinco intentos (la capa de 3.1 GB se reiniciaba con cada corte); se pasó la construcción a GitHub
  Actions. Como un workflow manual solo corre desde `main`, la imagen y la medición en la T4 real quedan
  para después de integrar el PR. El endpoint no se dejó encendido por su costo por hora. Un botón de
  encendido en una app pública es un riesgo de costo: por eso pide una segunda clave y el apagado no
  depende de que alguien se acuerde, sino de una tarea en AWS. El permiso de la Lambda para crear
  endpoints se limitó a ese nombre y esa configuración, para que el botón no pueda levantar otra
  instancia más cara. Al revisar, la IA encontró que la sincronización del PR #19 había dejado
  `IA_USAGE.md` sin las entradas 37 a 41; se restauraron sin tocar las de main. Al correr `preparar` por primera vez
  en AWS, `CreateModel` falló con «cannot pull»: el rol y su política se acababan de crear y IAM
  todavía no los había propagado; el script ahora reintenta unas veces antes de rendirse.

### 43. Segunda caída del dashboard publicado con un .mha (caso 044)

- **Integrante y herramienta:** Gustavo · Claude Code (app de escritorio), `claude-opus-5-5`.
  17 de octubre de 2026, rama `fix/memoria-distancias`.
- **Prompt exacto:**
  > mmira que me falló otra vez al subir el .mha 044
- **Qué produjo la IA:** el diagnóstico con el log (otra vez sin error de Python: el sistema mató la app
  por memoria) y la medición del pico por función con el 044 y el modelo real: el post-proceso 3D
  (`ndi.sum` y `np.isin` copiaban el volumen de etiquetas a float64 e int64) y la transformada de
  distancia (pedía un mapa de distancias que no se usaba). Los arreglos bajan el pico de 1.83 a 0.96 GB
  con el resultado idéntico, y el dashboard suelta los casos en caché antes de procesar.
- **Qué se hizo a mano:** Gustavo encontró la caída en producción y bajó el log.
- **Análisis crítico:** el arreglo anterior (entrada 41) se midió con los casos 050 y 001, que tienen
  pocos fragmentos y una caja del hueso más chica; el 044 tiene 27 fragmentos y la caja es casi todo
  el volumen, así que recortar no bastaba. Tampoco se había contado la caché de casos de una sesión
  larga. El primer intento (índices en int32) no movió el pico: medir por función mostró que estaba
  en otra parte. Para que no vuelva a pasar falta medir los casos extremos, no solo los grandes.
---

# Parte 2. Análisis crítico

## 2.1 Dónde aportó la IA

- **Diagnósticos con datos en lugar de intuiciones.** El oráculo y la matriz de confusión (14)
  ubicaron el error en la clasificación primario/secundario y no en la segmentación; la comparación
  de percentiles (7) refutó con números la hipótesis del metal en el caso 025.
- **Errores reales en el código.** La cabeza de bordes muerta por su pérdida (16), el lote de un
  solo corte en `evaluate_volumes.py` (18, 21) y el `min_size` que borraba fragmentos (18), la fuga de 7
  vóxeles que daba 0.0 mm de distancia (26) y el mAP medido sobre cajas refinadas y no sobre la
  cabeza grid (10).
- **Verificación antes de usar.** Probar la lógica con datos sintéticos antes de correrla sobre los
  100 casos (4) y comprobar la reproducibilidad del manifest y los splits (1).
- **Disciplina de evaluación.** La regla de decisión fijada antes de ver resultados (19) evitó
  ajustar el modelo mirando test, y obligó a declarar cada consulta adicional.
- **Trabajo de soporte.** Centralizar el preprocesamiento (2), curar `requirements.txt` (5),
  separar exportación y dashboard (25, 26) y documentar cada semana (8, 9, 24, 28).
- **El hallazgo decisivo.** El análisis caso por caso de la entrada 23 encontró que un tercio de los
  volúmenes estaba guardado girado 180°; corregirlo con la cabecera y entrenar con aumento
  geométrico llevó la validación de 0.851 a 0.869. Después, H8 agregó aumento de ruido y obtuvo
  0.835 de Dice por fragmento con ruido σ=25, frente a 0.135 de H6 y 0.228 de H7 (entrada 36).

## 2.2 Errores de la IA y cómo se detectaron

| Error | Entrada | Cómo se detectó |
|---|---|---|
| Filtro por tamaño que conservaba los rieles de la camilla y borraba hueso | 3 | Revisión visual de las capturas |
| Apertura morfológica que perforaba la cortical | 3 | Revisión visual en 3 casos |
| Hipótesis de fijador externo en el caso 025 | 3, 7 | El profesor la corrigió; los percentiles la refutaron |
| Suponer que el metal era osteosíntesis pélvica | 8 | El EDA mostró implantes femorales y objetos externos |
| Instalación de `torch` con CUDA en un equipo sin GPU | 5 | Falló al importar en Windows |
| Tomar el mAP de 0.901 del JSON como objetivo cumplido | 10 | Leyendo cómo lo calcula `evaluate()` |
| Proponer como nuevo el flip sagital, que ya existía | 11 | Revisando el código antes de guardar el plan |
| Proponer unir las islas del primario | 14 | Empeoró el Dice (0.804 contra 0.817) |
| Estimaciones de memoria y tiempo cortas | 17 | Se agotó la RAM de Kaggle con E3 y E4 en paralelo |
| Notebooks sin progreso visible y comandos con comentarios que fallaban en zsh | 17, 28 | Al ejecutarlos |
| Pruebas de la fuga de vóxeles mal armadas | 26 | La prueba fallaba por la razón equivocada |
| Escribir un script de MIP nuevo antes de revisar el repo | 28 | `raw_view.py` ya lo tenía, con mejor limpieza |
| Celdas de notebook agregadas sin campo `id` | 28 | Aviso de `nbformat` al abrir el notebook |
| Error medio de distancia de 1.153 mm que en realidad era un predictor que siempre decía 0.0 mm | 22 | Abriendo el detalle por fragmento |
| Cifras del ensamble en la columna de validación de la tabla de test | 22 | Revisión contra los JSON |
| Texto base64 corrompido al escribirlo a mano en Colab | 22 | Desde ahí, código por dataset privado y verificación con md5 |
| Detector geométrico de giro con el signo al revés y 13 falsos positivos de 16 | 23 | Revisión visual de los casos marcados |
| F1 de 0.991 reportado en vez de 0.990 | 23 | Contraste con el JSON |
| `sam_baseline.py` cargaba los volúmenes sin enderezar | 24 | Detectado antes de correrlo con H6 |
| Prueba de giro en píxeles que deformaba la pelvis | 34 | El enderezado estimaba +2.8° para 30° en el caso 044 |
| Mezclar cifras de dos experimentos de CBAM en el resumen | 34 | Revisión de Jacobo contra el model card |
| Dejar un `index.lock` en `.git` | 10 | La propia IA lo reportó |
| Presentar la robustez de H8 como absoluta | 36 | El equipo comprobó que σ=25 y el giro de 60° sin enderezar quedan bajo el objetivo |
| Lanzar el kernel de test antes de que existiera el dataset | 36 | Falló a los 7 s en la comprobación inicial, sin tocar test |
| JSON de latencia con codificación de Windows | 36 | La lectura en Streamlit Cloud habría fallado en Linux |
| Rama `modelo_h8` siguiendo otra rama | 36 | Se detectó antes del primer push y se corrigió |
| Prueba del relleno del visualizador 3D que pasaba sin el relleno | 36 | Se corrigió y se verificó el caso negativo |
| Redondeo distinto en cifras de SAM entre documentos y dashboard | 36 | Contraste con los JSON guardados |

Patrón común: casi todos los errores eran hipótesis o heurísticas **plausibles** que fallaron al
contrastarlas con los datos, con el código real o con el entorno. Ninguno se detectó por la
explicación de la IA; todos, al ejecutar o medir.

## 2.3 Limitaciones

- **Sin acceso al entorno real.** La IA no tuvo acceso al repositorio privado, a Kaggle ni al Mac
  del equipo (14, 15, 25, 28). Todo se ejecutó y verificó a mano, y la información del repo se le
  pasó por consola.
- **Mediciones del entorno de la IA.** La latencia que se ve en el entorno de la IA (185 ms por
  corte) no vale para el informe (26); las latencias oficiales se midieron en los equipos del grupo,
  en Colab y en Kaggle.
- **No anticipa qué funciona.** Ninguna técnica del entrenamiento (17) ni del post-proceso (18, 20)
  cerró la brecha del Dice por fragmento, y el segundo pase y el 2.5D se descartaron (9). Las mejoras
  decisivas fueron la capacidad del modelo (32 canales) y la receta H6 con orientación canónica,
  aumento geométrico y EMA (entrada 23). De unos quince cambios de la campaña de la entrada 21,
  solo dos superaron el ruido: ninguno atacaba la causa real, que estaba en los datos.
- **Lecturas interpretativas del enunciado.** Las zonas grises (13) y la elección entre MIP e
  isosuperficie (25) son lecturas de la IA que se confirmaron con el profesor o se resolvieron
  entregando las dos vistas (28).
- **Registros redactados por la IA.** Las entradas 10–13 y 21–27 las redactó la IA a pedido de
  Gustavo; el equipo revisó su análisis crítico.

## 2.4 Decisiones que cambiaron en el camino

Para que el registro no se lea como contradictorio, estas son las decisiones que se revirtieron y
su versión final:

| Tema | Primera decisión | Decisión final | Entradas |
|---|---|---|---|
| Normalización | CLAHE por defecto en `window_hu` | Gamma 1.5 (CLAHE: Dice Otsu 0.223 contra 0.486) | 7 → 9 |
| Caso 025 | Fijador externo, limitación conocida | Histograma óseo corrido; se corrige con normalización | 3 → 7 |
| Visualizador 1 | MIP estático → isosuperficie 3D | MIP en tres ejes + isosuperficie | 3 → 26, 28 |
| Post-proceso | `min_size=1` y CASR mejoraban validación | `min_size=12`, sin CASR (no cumplieron la regla) | 18 → 20 |
| Modelo principal | c2 → D5 → H6 s42 → H7 s42 | H8 s42: H7 más aumento de ruido; el más robusto al ruido de cinco modelos comparados, 0.874 en test y 0.875 ± 0.001 con tres semillas | 18, 21 → 23 → 34 → 36 |
| Pesos | EMA 0.999 (D5, H6) | Sin EMA (H7, H8). El aumento de ruido de H8 mejora la robustez; se conservan los resultados de las tres semillas | 21 → 34 → 36 |
| Post-proceso 3D | Tamaños mínimos en vóxeles (30 y 200) | En mm³ en la demo; la evaluación oficial sigue en vóxeles | 36 |
| Enderezado del giro axial | Solo en la evaluación, con `--normalizar` | También en la demo, para giros de 10° o más | 34 → 36 |
| Orientación | Lista de casos LPI / `TransformMatrix` | Productos punto y cruz sobre la cabecera (`normalizacion.py`), mismo resultado en los 100 casos | 23, 31 → 34 |
| Despliegue | Túnel de Cloudflare desde el Mac | Streamlit Community Cloud + API en AWS Lambda, publicados y probados desde computador y celular. El profesor aprobó este despliegue en lugar del túnel de Cloudflare del enunciado | 25, 27 → 30, 31 |
| Detector de giro | Detectar el giro con geometría (`estimate_angle`) | Descartado en la Semana 10; retomado como enderezado por simetría en milímetros, opcional al predecir | 23 → 34 |

**Consultas a test.** Por la regla de la entrada 19, test debía consultarse una sola vez. Se
consultó seis veces para comparar modelos y se declara: receta c2 (0.848), EMA + CASR (0.847), D5
(0.845, la cifra oficial de la primera evaluación), H6 (0.875), H7 (0.879) y H8 (0.874). En la
evaluación de D5 también se pasaron por test los demás checkpoints que superaban el umbral de
validación, con la regla previa de que solo medían la variación y no servían para elegir modelo
(entrada 22). Después, las semillas 43 y 44 de H8 se midieron con la regla escrita antes de
entrenarlas: estimar variación, no elegir modelo. No se ajustó nada con esos resultados. El 0.875 de
H6 es una evaluación posterior al análisis de los casos 012 y 039, y no es una estimación limpia.

## 2.5 Lecciones del equipo

- **Una explicación plausible no es una explicación verificada.** El fijador externo (3, 7) y la
  unión de islas (14) sonaban razonables y eran incorrectas. Cuando la IA propone una causa, se le
  pide contrastarla numéricamente contra el resto del dataset antes de aceptarla.
- **Leer cómo se calcula una métrica, no solo su valor** (10, 18). Un JSON de resultados no dice si
  el mAP se midió sobre la cabeza grid o sobre cajas refinadas, ni cuántos fragmentos sobrantes deja
  pasar la métrica.
- **Validar visualmente en varios casos reales.** Dos de los cuatro intentos del visualizador 3D
  fallaron pese a estar bien razonados (3); en imagen médica, la heurística se prueba en casos
  "sucios" antes de aceptarla.
- **Fijar la regla de decisión antes de ver resultados** (19), y declarar cada excepción.
- **Mirar los datos antes que la arquitectura** (21, 23). Se gastó un día probando pérdidas y
  arquitecturas; el giro de un tercio de los volúmenes estaba en la cabecera desde la Semana 8.
- **Guardar cada resultado apenas sale** (22, 23). La máquina de Colab se cerró a las 24 horas con
  evaluaciones en curso y hubo resultados que solo quedaron en pantalla.
- **Revisar lo que ya existe antes de pedir código nuevo** (28). Varias veces lo que parecía faltar
  ya estaba en el repo pero sin documentar.
- **Los resultados negativos también son evidencia** (17, 20): quedan registrados para justificar
  por qué la mejora vino de la capacidad del modelo y de H6, y no del post-proceso.

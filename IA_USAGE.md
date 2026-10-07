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

> **Pendientes del equipo antes de entregar (borrar este bloque al cerrar):**
> - Federico: sus entradas (no hay ninguna todavía).
> - Gustavo: revisar el análisis crítico de las entradas 21 a 24 y 31 a 33, que redactó la IA.
> - Jacobo: el modelo de Claude que usó (entradas 1–7), los prompts exactos si los tiene, y una
>   entrada para su versión del notebook 03 (Semana 9).
> - Unificar la cifra del overfit de la Semana 9: la entrada 9 dice 1.687 → 0.002 (758×) y el
>   notebook 03 registra 1.673 → 0.00115 (1454×). Son dos corridas; el informe y este archivo deben
>   citar la misma.

## Herramientas usadas

| Integrante | Herramienta | Modelo | Entradas |
|---|---|---|---|
| Jacobo | Claude (claude.ai) | *(completar)* | 1–7 |
| María | Claude (claude.ai) | Claude Opus 5.5 | 8, 9, 14–20, 28 |
| Gustavo | Claude, app de escritorio (modo Cowork) | `claude-sonnet-5-5` y, desde el 6 de octubre, `claude-opus-5-5` | 10–13, 21–27, 30–33 |
| Federico | *(completar)* | *(completar)* | 29 |

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

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Prompt (resumen del pedido):** confirmar que los datos descargados de Zenodo y colocados en
  `data/raw/` son idénticos a los que usó María, antes de empezar a modificar código.
- **Qué produjo la IA:** el procedimiento: correr `build_manifest.py` sobre los datos locales,
  comparar el CSV contra el original (ignorando las columnas de ruta absoluta, que cambian entre
  sistemas operativos) y regenerar `splits.json` para verificar que `git diff` no mostrara cambios.
- **Qué se hizo a mano:** se aceptó el enfoque sin cambios y se ejecutó. Resultado: manifest
  idéntico y splits reproducibles al 100 %.

### 2. Módulo `pengwin_io.py`

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Prompt (resumen del pedido):** centralizar la carga y el preprocesamiento, antes duplicados en
  `build_manifest.py`, `mip_visualizer.py` y `hu_windowing_check.py`, en un solo módulo, con las
  constantes que ya usaba el repo (clip −1024/3000, ventana C400/W1800).
- **Qué produjo la IA:** el módulo completo, incluidas las funciones de redimensionamiento para la
  imagen (bilineal) y para la máscara (vecino más cercano), que no existían y se agregaron pensando
  en la Semana 9.
- **Qué se hizo a mano:** se probó con `python src/pengwin_io.py 001` antes de integrarlo a los
  demás scripts.

### 3. Visualizador 1 — reconstrucción 3D (varios intentos)

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Prompt (resumen del pedido):** reemplazar los PNG estáticos de `mip_visualizer.py` por una
  isosuperficie 3D interactiva que muestre solo hueso, sin camilla ni cables.
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

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Prompt (resumen del pedido):** validar si la etiqueta 1/11/21 es siempre el fragmento de mayor
  volumen de su región, y medir el tamaño de cada fragmento en mm³ y en cortes axiales.
- **Qué produjo la IA:** el script completo, probado primero con un caso sintético que viola la
  convención a propósito (un "11" más chico que un "12") para confirmar que detecta excepciones.
- **Qué se hizo a mano:** se corrió sobre los 100 casos. La convención se cumple en 300 de 300
  combinaciones caso-región.

### 5. `requirements.txt`

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Prompt (resumen del pedido):** el archivo original era un `pip freeze` de macOS (~200 líneas,
  versiones fijas) que no instalaba en Windows.
- **Qué produjo la IA:** un archivo curado con solo las librerías que importa el código, sin
  versiones fijas.
- **Qué se hizo a mano:** en un equipo sin GPU NVIDIA, `pip install torch` instaló la build con
  CUDA, que falló al importar en Windows (`c10_cuda.dll`). Se instaló `torch` aparte desde el índice
  de CPU y se documentó para el resto del equipo.

### 6. Corrección de `docs/resumen_semana8.md`

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Prompt (resumen del pedido):** revisar el documento de María en busca de imprecisiones.
- **Qué produjo la IA:** tres correcciones: una afirmación de que "se notificó al profesor por
  escrito" cuando no se había enviado nada; rutas de ejemplo (`~/Downloads/...`) que no coinciden
  con `data/raw/`; y la explicación del clip de HU, que no neutraliza el metal por sí solo (lo aplana
  la ventana C400/W1800).
- **Qué se hizo a mano:** se confirmó directamente con el profesor antes de marcar la notificación
  como resuelta.

### 7. Caso 025: normalización tras la asesoría con el profesor

- **Integrante y herramienta:** Jacobo · Claude (claude.ai).
- **Contexto:** en asesoría, el profesor corrigió la hipótesis del fijador externo de la entrada 3:
  si fuera metal, no se vería la parte superior de la cresta ilíaca. Según él, es mala distribución
  de HU y hay que normalizar con gamma o ecualización, evaluando cuál conviene.
- **Prompt (resumen del pedido):** diagnosticar el caso 025 contra los 100 casos y comparar métodos
  de normalización con evidencia.
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
    (pérdida de 1.687 a 0.002, 758× con un criterio mínimo de 4×; ver el bloque de pendientes); y
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

### 29. Entradas de Federico *(pendiente)*

- **Integrante y herramienta:** Federico · *(completar)*.
- **Prompt / qué produjo la IA / qué se hizo a mano:** *(completar o declarar que no usó IA)*.

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
  extensión son los que pide el profesor; la IA no tenía el enunciado y eligió una estructura de
  artículo corto.
- **Análisis crítico:** la licencia no se escribió de memoria: se consultó el registro de Zenodo del
  dataset, y ahí se vio que el DOI que aparecía en un README generado para Hugging Face
  (`10990767`) es el del documento de diseño del reto y no el de los datos. Las cifras del
  mini-paper se contrastaron con un script contra los JSON de `outputs/s10/`. El mini-paper no trae
  resultados nuevos; repite el aviso de que el 0.875 es una segunda evaluación en test. La cifra
  del mejor método del reto (IoU 0.930) se tomó del resumen del artículo del reto y no es
  comparable con las nuestras.

### 33. Limpieza de la documentación de las semanas 10 y 11

- **Integrante y herramienta:** Gustavo · Claude (Cowork), `claude-opus-5-5`. 7 de octubre de 2026.
- **Prompt exacto:**
  > quisiera limpiar las documentaciones , revisar cuales  son las definitivas , de acuerdo con lo que hicimos ,  de la semana 10 para arriba ,  porque veo un poco de docs de semana 10  que no se cual o cuales son los definitivos , también se que algunos companeros estuvieron documentando , perro no habiamos terminado la semana 10 y 11 , entonces faltaba por documentar bien , entonces terminemos de documentar bien porfa y dejar solo lo que es y lo definitivo

  > Lo importante es no duplicar información, sino dejar solo los docs que no dupliquen información, o sea, solo dejar los docs que expliquen cada parte de su proceso y no duplicar información.
- **Qué produjo la IA:** el inventario de documentos y notebooks con su autor y fecha; la
  integración de la rama `docs/latencia-distancia` de María; el retiro de los planes de las semanas
  10 y 11, de los cambios de la primera versión de la Semana 10, del resumen de la Semana 10, de
  dos copias del notebook 05 y de los cuadernos de Colab anteriores; el cuaderno de Colab regenerado
  con la receta de H6; el cierre de la bitácora; y las correcciones del informe técnico (orientación
  en la demo, conteo de metal, tiempo de entrenamiento y consultas a test).
- **Qué se hizo a mano:** Gustavo decidió borrar en vez de archivar, incluir los notebooks y el
  criterio de un documento por tema sin repetir. La IA preguntó antes de borrar y mostró la lista.
- **Análisis crítico:** la IA había escrito el resumen de la Semana 10 un día antes y ese mismo
  documento resultó ser el que más repetía a los demás; con el criterio de Gustavo se retiró. La
  discrepancia del metal (41 casos en el informe y 25 en el model card) no era un error de ninguno
  de los dos: son 41 casos con algún vóxel sobre 3000 HU y 25 con más de 100 mm³, y se comprobó
  contra `data/eda_intensidades_detalle.csv` antes de corregir. El mini-paper y el informe técnico
  cubren el mismo trabajo a distinta profundidad; la IA los dejó a los dos porque no tiene el
  enunciado para saber si se piden ambos.

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
  geométrico llevó la validación de 0.851 a 0.869 (H6).

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
| Dejar un `index.lock` en `.git` | 10 | La propia IA lo reportó |

Patrón común: casi todos los errores eran hipótesis o heurísticas **plausibles** que fallaron al
contrastarlas con los datos, con el código real o con el entorno. Ninguno se detectó por la
explicación de la IA; todos, al ejecutar o medir.

## 2.3 Limitaciones

- **Sin acceso al entorno real.** La IA no tuvo acceso al repositorio privado, a Kaggle ni al Mac
  del equipo (14, 15, 25, 28). Todo se ejecutó y verificó a mano, y la información del repo se le
  pasó por consola.
- **Lo que no se ejecutó.** Nada de AWS ni de GitHub Actions se ha corrido (27), y los costos no se
  verificaron. La latencia que se ve en el entorno de la IA (185 ms por corte) no vale para el
  informe (26). La latencia en el Mac sigue pendiente (24).
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
| Modelo principal | c2 → D5 | H6 s42 | 18, 21 → 23 |
| Despliegue | Túnel de Cloudflare | Túnel de Cloudflare, que exige el enunciado; Streamlit Cloud + AWS Lambda queda como alternativa documentada | 25, 27 |
| Detector de giro | Detectar el giro con geometría (`estimate_angle`) | Descartado; se usa la cabecera del `.mha` | 23 |

**Consultas a test.** Por la regla de la entrada 19, test debía consultarse una sola vez. Se
consultó cuatro veces y se declara: receta c2 (0.848), EMA + CASR (0.847), D5 (0.845, la cifra
oficial de la primera evaluación) y H6 (0.875, el modelo entregado). En la evaluación de D5 también
se pasaron por test los demás checkpoints que superaban el umbral de validación, con la regla
escrita de antemano de que solo medían la variación y no servían para elegir modelo (entrada 22).
No se ajustó nada mirando esos resultados. El 0.875 de H6 es una segunda evaluación, posterior al
análisis de fallos de los casos 012 y 039 de test (entrada 23): no es una estimación limpia. Lo que
la respalda es que H6 se eligió con validación (0.869 / 0.868 / 0.872 en las tres semillas) y con
la configuración escrita antes de correrla.

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

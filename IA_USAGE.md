# Registro de uso de IA generativa

Bitácora crítica del uso de Claude (Anthropic) como apoyo durante el desarrollo de la
rama `continuacion_semana08`. No es un volcado de prompts: cada entrada explica qué se
pidió, qué se obtuvo, qué se aceptó tal cual, qué se corrigió y por qué.

**Integrante:** Jacobo
**Herramienta:** Claude (Anthropic), vía claude.ai

---

## 1. Verificación de reproducibilidad de los datos

**Qué se pidió:** confirmar que los datos descargados de Zenodo y colocados en `data/raw/`
son idénticos a los que usó Maria, antes de empezar a modificar código.

**Qué propuso la IA:** correr `build_manifest.py` sobre los datos locales, comparar el
CSV resultante contra el original (ignorando columnas de ruta absoluta, que difieren
entre sistemas operativos) y regenerar `splits.json` para verificar que `git diff` no
mostrara cambios.

**Qué se aceptó:** el enfoque completo, sin modificaciones. Resultado: manifest idéntico
y splits reproducibles al 100 %.

**Análisis crítico:** enfoque correcto y necesario — sin esta verificación, cualquier
diferencia sutil en los datos (por ejemplo, una descarga incompleta) se habría propagado
silenciosamente a todo el trabajo posterior.

---

## 2. Módulo `pengwin_io.py`

**Qué se pidió:** centralizar la carga y el preprocesamiento (antes duplicado en
`build_manifest.py`, `mip_visualizer.py` y `hu_windowing_check.py`) en un solo módulo
reutilizable, siguiendo las mismas constantes ya usadas en el repo (clip -1024/3000,
ventana C400/W1800).

**Qué se aceptó:** el módulo completo tal como se propuso, incluyendo las funciones de
resize para imagen (bilineal) y máscara de etiqueta (vecino más cercano) — estas últimas
no existían antes en el repo y se agregaron pensando en la semana 9.

**Análisis crítico:** correcto y probado con `python src/pengwin_io.py 001` antes de
integrarlo a los demás scripts.

---

## 3. Visualizador 1 — reconstrucción 3D (varios intentos)

**Qué se pidió:** reemplazar los PNG estáticos de `mip_visualizer.py` por una isosuperficie
3D interactiva, mostrando únicamente hueso (sin camilla ni cables), como exige el pliego.

**Intento 1 (rechazado):** la IA propuso limpiar la máscara ósea con apertura morfológica
+ filtro de componentes conectados por tamaño relativo. Al probarlo, los rieles metálicos
de la camilla —rectos y largos— resultaron *más grandes* que los fragmentos óseos
fracturados, así que el filtro hizo lo contrario de lo esperado: mantuvo los rieles y
descartó hueso real. **Se rechazó este resultado** tras revisar las capturas.

**Intento 2 (aceptado con ajuste):** la IA corrigió el enfoque usando una "silueta
corporal" (componente conectado más grande de la máscara `HU > -300`, con relleno de
huecos internos) intersectada con el umbral óseo. Esto sí excluyó correctamente los
rieles y cables, porque no tocan al paciente.

**Intento 3 (rechazado):** para limpiar rayas residuales de *beam hardening*, la IA
agregó una apertura morfológica sobre la máscara ya intersectada. Al probarlo en los
tres casos de ejemplo, el hueso cortical real quedó perforado y fragmentado (crestas
ilíacas rotas, sacro deshecho) — el hueso cortical a este spacing (~0.7-0.8 mm/vóxel)
es tan delgado como el ruido que se quería eliminar, así que la erosión no distinguía
entre ambos. **Se rechazó y se reportaron las capturas comparativas a la IA.**

**Intento 4 (aceptado, versión final):** se quitó la apertura morfológica y se subió el
umbral de tamaño mínimo por componente conectado (de 30 a 3000 vóxeles). Este filtro
"todo o nada" por bloque no erosiona superficies, así que no dañó hueso real. Resultado
final: pelvis anatómicamente completa en los 3 casos de prueba.

**Caso 025 — limitación no resuelta:** en este caso persisten varillas rectas cruzando
la pelvis y huecos en el ala ilíaca. La IA verificó el HU máximo de ese caso en el
manifest (12 939, muy por encima del resto) y propuso como explicación un fijador
externo ortopédico real (plausible: tratamiento estándar en fracturas pélvicas de alta
energía), no un artefacto de limpieza. **Se aceptó no seguir forzando el filtro**, porque
cualquier intento adicional de quitarlo arriesgaba dañar hueso sano en el resto del
dataset — se decidió documentarlo como limitación conocida en vez de perseguir una
limpieza perfecta.

> **Nota de seguimiento (entrada 7):** esta hipótesis de "fijador externo" resultó
> **incorrecta** — el profesor la corrigió en asesoría. Ver entrada 7 para el
> diagnóstico real y la corrección aplicada.

**Análisis crítico:** el proceso de prueba-error fue necesario y valioso — la primera
propuesta de la IA, aunque razonada, fallaba en la práctica y solo se detectó al ver el
resultado visual. Aprendizaje: no aceptar una heurística de limpieza sin validarla contra
varios casos reales del dataset, sobre todo con datos clínicos intencionalmente "sucios".

---

## 4. EDA de fragmento principal y tamaño

**Qué se pidió:** dos análisis pendientes de la semana 8 — validar si la etiqueta 1/11/21
es siempre el fragmento de mayor volumen de su región (necesario para la distancia de
separación de la semana 10), y el tamaño de cada fragmento en mm³/cortes axiales.

**Qué se aceptó:** el script completo. La IA probó la lógica de validación con un caso
sintético diseñado a propósito para violar la convención (fragmento "11" más chico que
un fragmento "12"), confirmando que el script sí detecta excepciones antes de correrlo
sobre los datos reales.

**Resultado sobre los 100 casos:** la convención se cumple al 100% (300/300 combinaciones
caso-región). Se puede usar la etiqueta base directamente como fragmento principal.

**Análisis crítico:** correcto. La verificación con datos sintéticos antes de correr
contra los 100 casos reales dio confianza en el resultado sin tener que auditar el
código manualmente fragmento por fragmento.

---

## 5. `requirements.txt`

**Qué se pidió:** el archivo original era un `pip freeze` completo de macOS (~200 líneas,
versiones fijas) que no instalaba en Windows (`contourpy==1.4.0` no existe para Python <3.12).

**Qué se aceptó:** un archivo curado con solo las librerías importadas directamente por
el código, sin versión fija.

**Problema encontrado durante la instalación (no anticipado por la IA):** en un equipo
sin GPU NVIDIA, `pip install torch` sin más instaló por defecto una build con soporte
CUDA, que falló al importar en Windows (`OSError ... c10_cuda.dll`) por falta de drivers.
Se corrigió instalando `torch` aparte con `--index-url .../whl/cpu`. **Pendiente:**
recrear el entorno virtual desde cero antes de la semana 9, porque la instalación previa
dejó archivos de la build CUDA mezclados que un simple `pip uninstall` no limpió del todo.

**Análisis crítico:** el archivo curado en sí fue la decisión correcta, pero el primer
intento de instalación de `torch` reveló un problema específico de Windows que no se
había anticipado. Se documenta aquí para que el resto del equipo no repita el mismo error.

---

## 6. Corrección de `docs/resumen_semana8.md`

**Qué se pidió:** revisar el documento heredado de Maria en busca de imprecisiones.

**Qué se corrigió:** una afirmación que decía "se notificó al profesor por escrito"
cuando en realidad no se había enviado nada todavía; rutas de ejemplo (`~/Downloads/...`)
que no coinciden con la estructura real del repo (`data/raw/`); y una explicación técnica
imprecisa sobre el clip de HU (el clip a [-1024, 3000] no "neutraliza" el metal por sí
solo — solo evita valores extremos; es la ventana ósea C400/W1800 la que aplana
visualmente cualquier valor por encima del rango de interés).

**Análisis crítico:** revisar un documento de avance escrito por otra persona del equipo
en busca de afirmaciones no verificadas resultó útil — una de ellas (la notificación al
profesor) se habría entregado como falsa de no corregirse a tiempo. Se confirmó
directamente con el profesor antes de marcarla como resuelta en el documento.

---

## 7. Corrección del caso 025 tras asesoría con el profesor

**Contexto:** en asesoría, el profesor corrigió directamente la hipótesis de "fijador
externo" de la entrada 3 (equivocada, salió de la IA): si fuera metal no se vería la
parte superior de la cresta ilíaca. Según él, es mala distribución de HU y hay que
normalizar con corrección gamma o ecualización de histograma, evaluando cuál conviene.

**Qué se pidió:** diagnosticar el problema real del caso 025 contra los 100 casos, y
comparar métodos de normalización con evidencia antes de elegir uno.

**Diagnóstico:** se le pidió a la IA comparar los percentiles de HU del caso 025 contra
el resto del dataset, usando el CSV que ya existía de la semana 8
(`data/eda_intensidades_detalle.csv`), sin volver a procesar los volúmenes crudos. El
resultado con datos reales: el HU máximo y el volumen de metal del caso 025 **no son
atípicos** (z-score 1.47 y 1.13, por debajo del umbral de 2) — la hipótesis de metal no
se sostenía ni con los números. Lo único atípico es `hueso_media` y `hueso_p50`
(z ≈ −2.1): el histograma óseo está corrido y comprimido ~150–250 HU por debajo de lo
típico. Esto confirmó el diagnóstico del profesor con evidencia cuantitativa concreta,
no solo su palabra.

**Elección del método:** la IA propuso comparar gamma, ecualización global y CLAHE
visualmente sobre el caso 025 y un caso típico (007) en paralelo, con un criterio
explícito para decidir: el método se aplica igual a los 100 casos dentro de
`pengwin_io.py`, así que no basta con que arregle el 025 — no puede dañar los casos que
ya están bien. Con esa comparación (`outputs/eda/diagnostico_normalizacion_caso025.png`),
**la decisión de descartar ecualización global fue evidente a simple vista** (aplana el
tejido blando a un bloque gris uniforme, pierde toda la anatomía) y **CLAHE se eligió**
porque recupera la textura trabecular del caso 025 sin distorsionar el caso 007. Gamma
se descartó porque usa la misma curva fija para todos los casos y distorsionaba por
igual el caso ya normal.

**Qué se implementó:** se integró CLAHE (`skimage.exposure.equalize_adapthist`,
`clip_limit=0.02`) directamente dentro de `pengwin_io.window_hu`, activado por defecto,
en vez de dejarlo como una función aparte sin usar — todos los scripts que ya llaman a
`window_hu` (propios y de otros integrantes) quedan corregidos sin tener que tocarlos
uno por uno. Se verificó con datos reales en los 3 casos de referencia del proyecto
(001, 025, 068) que el contraste mejora sin destruir la anatomía en ninguno
(`outputs/eda/verificacion_clahe_casos_ejemplo.png`).

**Análisis crítico:** este es el ejemplo más claro del proyecto de por qué no hay que
aceptar una hipótesis de la IA sin verificarla — la explicación de "fijador externo" de
la entrada 3 sonaba razonable (dato clínico real, plausible en fracturas pélvicas) pero
era incorrecta, y solo se detectó porque el profesor la revisó directamente. La lección
concreta: cuando la IA da una explicación plausible pero no verificada con los datos
mismos, conviene pedirle que la contraste numéricamente contra el resto del dataset
antes de darla por buena, en vez de aceptarla porque "tiene sentido". Además, este
cambio invalida los caches de entrenamiento y checkpoints de semanas 9-10 de otros
integrantes (usan `window_hu` sin CLAHE) — hay que avisarles que toca regenerar datos y
reentrenar.

## Resumen de aprendizajes

- La IA es útil para generar código de preprocesamiento (carga, limpieza de máscaras,
  EDA) rápido, pero las heurísticas de limpieza de imagen médica **requieren validación
  visual contra varios casos reales** antes de aceptarlas — dos de los cuatro intentos
  del visualizador 3D fallaron en la práctica pese a estar bien razonados en teoría.
- Es más seguro pedirle a la IA que pruebe su propia lógica con datos sintéticos antes
  de correrla sobre el dataset completo (se hizo para la validación del fragmento
  principal y para la limpieza de máscaras óseas).
- Ningún resultado generado por la IA se subió al repo sin correrlo primero en el
  entorno local y confirmar la salida.
- Una explicación de la IA que "suena razonable" (entrada 3, fijador externo) no es lo
  mismo que una explicación verificada contra los datos — la entrada 7 muestra que
  contrastarla numéricamente contra el resto del dataset destapó que estaba mal, antes
  de que quedara como limitación permanente en el model card.

---

# Semana 10 — planeación

**Integrante:** Gustavo
**Herramienta:** Claude (Anthropic), app de escritorio en modo Cowork. Modelos configurados
en la sesión: `claude-sonnet-5-5` y `claude-opus-5-5`.
**Fecha:** 4 de octubre de 2026. **Rama:** `unificado_semana09`.

Las entradas 8 a 11 las redactó la IA a pedido de Gustavo, con los prompts copiados tal
cual. El análisis crítico es un borrador: el equipo debe revisarlo y corregirlo.

---

## 8. Auditoría del repositorio frente al enunciado

**Prompt exacto:**

> entendamos el repositorio y que tiene y mira lo que tenemos que hacer , y miremos en que punto estamos

**Qué propuso la IA:** leyó el enunciado, el README, los resúmenes de las semanas 8 a 10 y
los JSON de métricas, y entregó una tabla de objetivos contra resultados de `w10g_384neg`
más la lista de entregables que faltan (SAM, visualizadores 2 y 3, dashboard, latencia,
model card, distancias sobre volúmenes).

**Qué se aceptó:** la lista de faltantes y la observación de que el Dice por fragmento
(0.817) es el único objetivo sin cumplir.

**Análisis crítico:**

- En esta primera respuesta la IA presentó el mAP@0.50 de 0.901 como objetivo cumplido,
  tomando el JSON al pie de la letra. Solo al leer `evaluate()` en la segunda pasada
  encontró que esas cajas están refinadas con la máscara semántica y que la cabeza grid
  cruda nunca se midió. Un JSON de métricas no basta: hay que leer cómo se calculó.
- La IA no pudo correr las pruebas (su entorno no tenía `pytest` ni PyTorch) y lo dijo;
  todo el diagnóstico sale de leer código y resultados guardados, no de ejecutarlos.
- Al correr `git status` dejó un `index.lock` vacío en `.git`. Lo renombró a
  `.git/claude_stale_index.lock.txt` y lo reportó.

---

## 9. Diagnóstico de la Semana 10 y árbol de decisiones

**Prompt exacto:**

> Listo, pero antes de hacer visualizadores y dashboard y todo eso, primero tenemos que hacer eh, un... Tenemos que hacer un notebook o, lo que, o los archivos.py, eh, preferiblemente un notebook que podamos eh, cumplir con todo lo que nos piden, tanto en métricas como en todo. Entonces, ahí, ¿qué podríamos hacer? ¿Sí? Digamos que la idea es que hagamos las cosas... con el notebook número, número 3, que ya es lo que, ahí ya termina la semana 9. Y volvamos a mirar cómo hacemos para, para, para llegar a las métricas, ¿sí? Si se puede hacer un, un notebook, que nos vayamos paso a paso, o a la podemos hacer un árbol de decisiones que tengamos que tomar para poder entrenar esto de acuerdo a lo que encontremos o lo que hemos encontrado y demás. Eso sería eh, importante. Entonces, creo que esa vaina es importante. Que vayamos mirando si, si ya terminamos semana 9, o si hay algo más que podamos agregar o hacer, y pues avancemos con la semana 10.

**Qué propuso la IA:** que la Semana 9 está cerrada; seis hallazgos sobre la Semana 10
(mAP medido sobre cajas refinadas, entrenamiento con 12 de ~244 cortes por caso, modelo sin
las opciones de CBAM de la Semana 9, Dice por fragmento por debajo del objetivo, test
evaluado con tres checkpoints, distancias y SAM sin hacer); un árbol de decisiones que
empieza midiendo la línea base sin entrenar; y la estructura del notebook 04.

**Qué se aceptó:** dejar el plan escrito en `docs/plan_semana10.md` antes de tocar código.

**Análisis crítico:**

- La IA propuso el flip sagital como remedio nuevo para la confusión de lados, cuando ya
  estaba en la receta de la Semana 10 (`--augment`). Lo corrigió al verificar el código
  antes de guardar el documento.
- El equipo respondió que entrenaría en el Mac. La IA revisó el `.venv` y encontró PyTorch
  2.2.2 para x86_64, lo que indica un Mac Intel sin GPU de Apple. Es una inferencia a
  partir del entorno, pendiente de confirmar con `torch.backends.mps.is_available()`.

---

## 10. Carpeta del curso y reinicio de la Semana 10

**Prompt exacto:**

> yo quiero volver a trabajar la semana 10 desde 0 , que tengamos como referencia lo que se habia adelantado ahi , pero      ahi te paso la carpeta de lo que hemos hecho con el profe por si sirve
>
> igual guiame y dime las preguntas que consideres para hacerles a los companeros   y como iniciamos para que vamos avanzando con la semana 10

**Qué propuso la IA:** que `FundidoraBackbone` del repo no tiene la misma estructura que
FundidoraPC del curso (bloques de dos convoluciones con SiLU frente a Conv-BN-ReLU-MaxPool),
así que sus pesos no cargan directamente; que en la carpeta del curso no hay pesos de
FundidoraPC guardados, solo los de la Chimera de la semana 5; y una lista de preguntas para
el equipo y para el profesor.

**Qué se aceptó:** `unificado_semana09` como rama base y que el registro de IA lo lleven
Gustavo y la IA.

**Análisis crítico:**

- La IA no pudo abrir los `.pth` de la Chimera (sin PyTorch en su entorno). Lo que dice de
  esos pesos sale del informe del grupo y de `chimera_blocks.py`, no de inspeccionarlos.
- Que no haya pesos de FundidoraPC es cierto para esa carpeta. Pueden existir en Colab o
  Drive de algún integrante.

---

## 11. Plan v2: explorar todo dentro de las restricciones

**Prompt exacto:**

> 3.unificado_semana09
>
> el registro de IA usemoslo nosotros ,
>
> en cuanto a estas preguntas
>
> * ¿El Dice "por fragmento" se mide emparejando fragmentos o basta el Dice semántico por región?
> * ¿Se permite refinar las cajas con la máscara semántica, o la detección debe medirse con la cabeza grid pura?
> * ¿Aceptaría 384 px si la recomendación es 224–256?
> * ¿Con qué variante y pesos de SAM debemos hacer el baseline?
>
> Avancemos con todo lo que podamos , armemos un plan donde exploremos todo , lo que nos pueda ayudar , solo siguiendo las restricciones del profe , mas la recomendaciones da igual , por que al final son recomendaciones
>
> y entonces actualicemos el plan , para empezar

**Qué propuso la IA:** la versión 2 de `docs/plan_semana10.md`: separa restricciones de
recomendaciones del enunciado, resuelve las cuatro dudas reportando las dos lecturas, y
arma un mapa de exploración por frentes (datos, backbone, transfer learning, detección,
instancia, entrenamiento) con prioridades y un orden de ejecución por fases.

**Qué se aceptó:** pendiente de revisión por el equipo.

**Análisis crítico:**

- La IA dejó cuatro opciones como zona gris en vez de incluirlas (backbone estándar con
  ImageNet, preentrenamiento autosupervisado sobre PENGWIN, datos externos de pelvis, red
  3D). El enunciado no las prohíbe con esas palabras; es su lectura y conviene confirmarla.
- La estimación de tiempo de entrenamiento en CPU es gruesa: sale de una latencia medida
  en Kaggle, no de cronometrar en el Mac.
- Hasta esta entrada no se ha escrito ni ejecutado código nuevo del modelo.

---

## 12. Plan de la Semana 11: visualizadores y dashboard

**Prompt exacto:**

> Necesito hacer el visualizador para este proyecto. Miremos a ver si ya tenemos todo para hacerlo, sí, al menos con algunos modelos que mejor han tenido resultado, y damos el visualizador, ¿sí? Como él lo pone ahí             O sea, deberíamos poder terminar la semana 11, sí. Terminar la semana 11. Entonces, armemos un plan y revisemos si ya tenemos lo necesario, al menos, para poder hacer esta semana 11, que es el visualizador.    hagamos el plan

**Qué propuso la IA:** un inventario del repo frente a la sección 3.4 del enunciado y
`docs/plan_semana11.md`. Conclusión: el lado del modelo alcanza para los visualizadores 2 y
3 (checkpoint c2, inferencia por volumen, NMS, post-proceso 3D, distancias en mm), pero no
existe nada del lado de la aplicación. Propone separar exportación (lenta, una vez por
caso) de dashboard (solo lee), Streamlit, selector de modelo con c2, c1 y B1, y desarrollar
con casos de validación dejando test para después de congelar el modelo.

**Qué se aceptó:** Streamlit (página web propia como segunda fase opcional), demo en el Mac
con túnel de Cloudflare, visualizador 1 con MIP e isosuperficie, y avance el miércoles 7 de
octubre con entrega la semana siguiente.

**Análisis crítico:**

- La IA no ejecutó el modelo ni el post-proceso 3D en esta sesión: desde su entorno no
  puede usar el `.venv` del Mac. Que el post-proceso 3D funciona sobre casos reales sale de
  leer el código y sus pruebas, no de una corrida; por eso el bloque 0 del plan lo comprueba.
- La primera versión del plan decía que los pesos del candidato con EMA no estaban en disco
  y usaba c2. Era cierto al leer el repo, pero los resultados se sincronizaron minutos
  después (`d5_ema_*`, `decisiones_semana10.md`). La versión 2 usa D5 como modelo principal.
- La versión 1 daba por buena la distancia en mm. `decisiones_semana10.md` muestra que la
  distancia predicha sale 0.0 mm en los 41 fragmentos secundarios de validación por el
  relleno del post-proceso 3D. La IA lo incorporó en la versión 2 como primer bloqueo; la
  corrección (`fill_unlabeled`) todavía no está en `src/` de este repo.
- El visualizador 1 actual es una isosuperficie, no la MIP que nombra el enunciado. La IA
  propone mostrar las dos; es su lectura y conviene confirmarla con el profesor.
- Faltan en este registro las entradas de las sesiones de experimentos del 5 de octubre
  (corridas B, C y D de la Semana 10).

---

## 13. Semana 11: construcción del dashboard (visualizadores 1, 2 y 3)

**Prompt exacto:**

> Sí, ahí ya tenemos los modelos locales, los .pth. La corrección de distancia es necesaria para esto. No, la otra sección de Claude está trabajando en eso. Empecemos con lo que podamos ir construyendo, ¿vale?
>
> De bueno, ahí sería bueno crear una nueva rama tipo semana 11 para esta vaina, listo para hacer esta vaina. No sé si ahora o después, porque estamos esperando los cambios de que haga la otra sesión de Claude.

**Qué propuso la IA y qué se construyó:** `src/predict_volume.py` (inferencia sobre un volumen
crudo sin ground truth), `src/demo_export.py` (exporta por caso volumen de IDs, distancias en mm,
comparación con el ground truth, mallas y vista cruda), `src/demo_render.py` (visualizadores 2 y
3), `src/raw_view.py` (MIP e isosuperficie del visualizador 1), `src/measure_latency.py`,
`app/dashboard.py` (Streamlit) y 15 pruebas nuevas. Guía en `docs/semana11_dashboard.md`.

**Qué se aceptó:** construir solo archivos nuevos mientras la otra sesión corrige la distancia, y
dejar la rama de la Semana 11 para después (la crea el equipo a mano desde el commit donde llegue
esa corrección).

**Análisis crítico:**

- La corrección `fill_unlabeled` no estaba en `src/`. La IA no la escribió: usó una versión
  provisional dentro de `demo_export.py` y la dejó marcada en `meta.json` y en el dashboard.
- La primera exportación del caso 024 daba 0.0 mm para el fragmento real a 19.0 mm. La causa no
  era el modelo (Dice 3D 0.90) sino una fuga de 7 vóxeles: `postprocess_3d` absorbe en el principal
  los pedazos menores a `min_voxels` y quedaban sueltos junto al fragmento. La IA lo encontró
  contando las componentes conexas del principal y lo corrigió dejando solo la mayor. Con eso la
  distancia predicha del fragmento es 19.0 mm, y 3.9 mm para el de 3.2 mm. Son dos fragmentos de un
  solo caso: no es una estimación del error.
- El modelo predice 28 fragmentos en el 024 contra 9 reales, casi todos diminutos. El dashboard
  los oculta por debajo de un volumen mínimo ajustable (1000 mm³ por defecto). Es un filtro de
  presentación y no mejora el modelo; la métrica sigue siendo la de la tabla de comparación.
- La IA no corrió nada en el Mac del equipo: la ejecución y las capturas de pantalla se hicieron en
  su propio entorno, con CPU de 2 núcleos. La latencia que se ve (185 ms por corte sin TTA) no vale
  para el informe; hay que medirla con `src/measure_latency.py` en el Mac y en una GPU.
- En el primer intento las pruebas de fuga de vóxeles estaban mal armadas (el bloque suelto tocaba
  el fragmento y `postprocess_3d` lo fusionaba, que es el comportamiento esperado). Se corrigió la
  prueba, no el código, y se comprobó que sin la limpieza la prueba sí falla.

---

## 14. Semana 11: despliegue (Streamlit Cloud + API en AWS) y CI/CD

**Prompts exactos:**

> Y si subimos el modelo AWS que yo tengo en AWS y el Streamlit lo deployamos en Streamlit Cloud

> Pero vení, o sea, ¿crees que en un lambda está bien? Porque creo que hay que poner con GPU y sin GPU. No hay un mejor servicio en, en AWS para esto, no sé, SageMaker, Bedrock, no sé. O, o qué se usa normalmente en estos casos..

> No, las pruebas deberían correrse. Cuando se hace un PR a main, debería correrse este pipeline. Cuando se haga el match, ya cuando estén haciendo cambios en main.

**Qué propuso la IA y qué se construyó:** Hugging Face Spaces se descartó porque en cuentas gratis el
SDK Docker y CPU Basic están como "Paid". Se separó el dashboard (Streamlit Cloud, sin torch) de la
inferencia en vivo (`api/server.py`, FastAPI en una imagen de contenedor para AWS Lambda). Se agregó
`deploy/build_streamlit_cloud.py`, la guía `docs/despliegue_aws_streamlit.md` y dos flujos de GitHub
Actions: `ci.yml` (pruebas en cada pull request a main y en cada cambio de main) y `deploy-api.yml`
(construye la imagen, la sube a ECR y actualiza la Lambda al cambiar main). Guía en
`docs/cicd_github_actions.md`.

**Qué se aceptó y qué no:** Lambda en CPU para la demo, en lugar de SageMaker. SageMaker Serverless no
tiene GPU y los endpoints en tiempo real con GPU no se apagan solos; el modelo (2.74 M de parámetros)
no necesita GPU para la demo. La latencia en GPU se mide con `src/measure_latency.py` en Kaggle o
Colab, sin desplegar nada.

**Análisis crítico:**

- Nada de AWS ni de GitHub Actions se ha ejecutado: se probaron la API y el dashboard en el entorno de
  la IA (con `import torch` bloqueado en el dashboard) y las 87 pruebas en un Python 3.11 limpio sobre
  el último commit más estos archivos. Los workflows, la imagen y la Lambda quedan por probar.
- El remoto es el repositorio de una compañera; los secretos de AWS solo los puede crear un
  administrador y cualquier colaborador con escritura podría leerlos desde un workflow. Se documentó
  el uso de un entorno con revisores y la alternativa de un repositorio propio.
- Los costos de AWS no se verificaron.

---

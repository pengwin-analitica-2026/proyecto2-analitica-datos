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

---


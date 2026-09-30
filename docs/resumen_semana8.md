# Resumen de avance — Semana 8
**Proyecto 2 Analítica de Datos — PENGWIN (fracturas pélvicas)**

## 1. Dataset descargado

- Fuente: Task 1 (CT) del challenge PENGWIN, vía Zenodo (https://doi.org/10.5281/zenodo.10927452).
- Archivos: `PENGWIN_CT_train_images_part1`, `PENGWIN_CT_train_images_part2`, `PENGWIN_CT_train_labels`.
- **Total: 100 casos** (imagen + máscara de fragmentos, uno por paciente).

## 2. Hallazgo importante: formato de archivo

- El dataset viene en formato **`.mha`**, no en NIfTI como asumía el enunciado del proyecto.
- Se ajustó el pipeline para usar **SimpleITK** en vez de `nibabel`. El spacing físico (mm/vóxel) sigue obteniéndose correctamente del header, así que el requisito de la rúbrica se cumple igual.
- **Confirmado con el profesor:** aprobó trabajar con `.mha` en vez de NIfTI, sin objeción.

## 3. Manifest del dataset (`build_manifest.py`)

Script que recorre los 100 casos y extrae metadatos completos (shape, spacing, rango HU, número de fragmentos, y validación de que cada imagen tenga su máscara correspondiente).

**Resultado:**
- **100/100 casos válidos** (ningún caso huérfano ni corrupto).
- Spacing: entre 0.625 y 1.25 mm/vóxel según el eje (rango normal para CT clínico).
- Fragmentos por caso: entre 3 y 9, promedio 5.75.

```text
============================================================
RESUMEN DEL MANIFEST
============================================================
Total de casos encontrados : 100
Casos válidos               : 100
Casos inválidos/incompletos : 0

Rango de spacing (mm/vóxel) en casos válidos:
  spacing_x: 0.658 - 1.220
  spacing_y: 0.658 - 1.220
  spacing_z: 0.625 - 1.250

Rango HU global: -6211.0 a 47685.0

Fragmentos únicos por caso (excluyendo fondo):
  min: 3, max: 9, promedio: 5.75

Manifest guardado en: data/manifest.csv
```

> Nota: el `manifest.csv` versionado se generó en la máquina original de Maria (macOS); la ruta absoluta que imprime el script en consola depende de cada computador, no es parte del contenido del CSV. Regenerado y verificado de forma idéntica (mismos valores, sin contar rutas) en la rama `continuacion_semana08`.

## 4. Diagnóstico de valores HU (`check_hu_outliers.py`)

- Se detectaron 52 casos con valores HU fuera del rango típico de CT (-1100 a 5000 HU).
- **Explicación identificada, no es un error:**
  - Valores mínimos repetidos exactos (ej. -2048, -1023) = **padding** del escáner fuera del campo de reconstrucción.
  - Valores máximos muy altos (hasta 47,685 HU en un caso) = **artefacto de metal**, pero casi nunca está en el hueso pélvico: el EDA de intensidades (notebook 02) muestra que proviene de implantes femorales y de objetos externos al paciente.
- **Decisión:** antes de normalizar se aplica un `clip` a **[-1024, 3000] HU**, que solo acota los picos extremos (evita que un outlier de 47 685 HU distorsione cálculos numéricos aguas abajo). **La neutralización visual real del padding y el metal la hace el ventaneo óseo (C400/W1800, sección 7)**, no el clip: cualquier valor por encima de la ventana (≥1300 HU tras el clip) queda mapeado al mismo blanco saturado (1.0), y por eso el contraste del hueso no se ve afectado por los picos de metal. El clip solo evita overflow/inestabilidad antes de llegar a ese paso.

```text
Total casos válidos: 100

Top 5 casos con HU máximo más alto:
 case_id  hu_min  hu_max     hu_mean
      80 -1023.0 47685.0 -600.436101
      68 -6211.0 30445.0 -757.690200
     100 -6152.0 24970.0 -936.068190
      55 -1023.0 23354.0 -668.465611
      84 -1023.0 19419.0 -546.270047

Casos con valores fuera del rango esperado (-1100 a 5000 HU): 52
```

(Tabla completa de los 52 casos en la salida de `python src/check_hu_outliers.py`.)

## 5. Splits fijos (`make_splits.py`)

- Split por `case_id` (nunca por corte, para evitar data leakage).
- Estratificado por número de fragmentos (buckets bajo/medio/alto).
- Semilla fija = 42.

```text
Bucket 'alto': 26 casos -> train=18, val=4, test=4
Bucket 'bajo': 1 casos -> train=1, val=0, test=0
Bucket 'medio': 73 casos -> train=51, val=11, test=11

Total: train=70, val=15, test=15
Splits guardados en: data/splits.json
```

**Resultado:** train=70, val=15, test=15.

**Limitación conocida (documentada, no es un bug):** solo existe 1 caso en el bucket "bajo" (≤3 fragmentos), así que ese extremo quedó representado únicamente en train y no en val/test. Se deja anotado para el informe final.

## 6. Módulo unificado de carga y preprocesamiento (`pengwin_io.py`)

Nuevo en `continuacion_semana08`. Centraliza en un solo lugar lo que antes estaba repetido en `build_manifest.py`, `mip_visualizer.py` y `hu_windowing_check.py`, para que entrenamiento, inferencia y los tres visualizadores usen exactamente la misma lógica:

- `load_case(case_id)` — carga imagen + label y devuelve el volumen crudo en HU junto con el spacing del header.
- `clip_hu` / `window_hu` — clip [-1024, 3000] + ventana ósea C400/W1800 → `[0, 1]` (misma fórmula que `hu_windowing_check.py`). **Actualizado tras la asesoría:** `window_hu` aplica CLAHE (`skimage.exposure.equalize_adapthist`, `clip_limit=0.02`) por defecto sobre la imagen ya en `[0,1]`, para corregir casos con histograma óseo corrido/comprimido como el 025 (ver sección 8). No toca el HU crudo: `bone_mask`/`body_mask` siguen operando sobre el volumen sin normalizar. Desactivable con `aplicar_clahe=False` para comparación/debug.
- `resize_slice` / `resize_label_slice` — resize a 224–256 px (bilineal para imagen, vecino más cercano para máscaras de etiqueta, para no promediar valores de fragmento).
- `bone_mask`, `body_mask`, `clean_bone_mask` — máscara ósea para el Visualizador 1. **Actualizado:** `clean_bone_mask` ahora incluye filtro por profundidad bajo la piel y recuperación de hueso de baja densidad, activados al pasarle `spacing_zyx`. Detalle y justificación en la sección 8.

**Dos flujos separados que salen del mismo tronco** (`load_case` + `clip_hu`):

| Flujo | Funciones | Escala | Para qué |
|---|---|---|---|
| Entrada al modelo | `window_hu` (+ CLAHE) → `resize_slice` | Grises `[0, 1]` | Entrenamiento e inferencia |
| Visualizador 1 | `clean_bone_mask` (usa `bone_mask` y `body_mask`) | HU reales | Reconstrucción 3D sin modelo |

Las máscaras se calculan sobre HU reales porque los umbrales (250, 150 y −300 HU) solo tienen sentido en esa escala; el ventaneo y CLAHE la destruyen. **Verificado en el código** (búsqueda de `bone_mask|clean_bone_mask` en `src/`): la máscara ósea solo la usan `pengwin_io.py`, `volumen_3d_visualizer.py` y los scripts de diagnóstico. Ningún script de entrenamiento la llama, así que los cambios de la sección 8 **no afectan al modelo, a las métricas ni a la medición de distancias**.

Constantes nuevas en `pengwin_io.py`:

| Constante | Valor | Uso |
|---|---|---|
| `MIN_SURFACE_DEPTH_MM` | 10.0 | Profundidad mínima bajo la piel de un componente pequeño |
| `BIG_COMPONENT_VOXELS` | 50 000 | Componentes mayores (esqueleto, fémures) nunca se filtran |
| `DEPTH_DOWNSAMPLE` | 2 | Submuestreo del mapa de profundidad (memoria), error ~1–2 mm |
| `BONE_LOW_HU_THRESHOLD` | 150 | Umbral bajo para recuperar hueso tenue |
| `CLOSING_RADIUS_VOX` | 2 | Radio del cierre que sella grietas de la cortical |

## 7. Carga de volúmenes y ventaneo HU (`hu_windowing_check.py`)

> Nota: el script se renombró de `hu windowing check.py` (con espacios) a `hu_windowing_check.py`.

- Carga de `.mha` con SimpleITK → array `(z, y, x)` en HU crudo; el spacing sale de `img.GetSpacing()` (orden `x, y, z`).
- Función `apply_hu_window`: primero `clip` a **[-1024, 3000] HU** y luego ventana ósea **C=400 / W=1800**, normalizada a `[0, 1]`. Misma fórmula ahora centralizada en `pengwin_io.window_hu`.
- Verificado visualmente en 3 casos: `001` (normal), `080` (HU máx. 47 685, metal) y `068` (HU mín. -6 211, padding). Salida: `outputs/figures/hu_window_check_{001,068,080}.png`.

## 8. Visualizador 1 — Reconstrucción 3D del volumen crudo (`volumen_3d_visualizer.py`)

Reemplaza la versión anterior (`mip_visualizer.py`, tres PNG estáticos de proyección de máxima intensidad) por una **isosuperficie 3D interactiva en Plotly**, rotable con el mouse — lo que pide literalmente el pliego para el V1 ("reconstrucción 3D inicial del volumen original").

Uso: `python src/volumen_3d_visualizer.py` corre los casos de ejemplo (`001`, `025`, `068`); `python src/volumen_3d_visualizer.py 007` corre un caso puntual. Salida: `outputs/figures/volumen_3d_<id>.html`. Tarda entre 4 y 17 s por caso solo en la máscara, así que para la demo los HTML se dejan pregenerados.

### 8.1 Limpieza final aplicada

Sin intervención del modelo, preprocesamiento clásico, todo dentro de `clean_bone_mask(image, spacing_zyx=(sz, sy, sx))`. El orden importa (ver 8.3, paso 7):

1. **Silueta corporal:** componente conectado más grande de la máscara `HU > -300`, con huecos internos rellenos. La camilla y los cables quedan fuera porque están separados del paciente por aire.
2. **Umbral óseo:** intersección de la silueta con `HU > 250`.
3. **Filtro de tamaño:** se descartan componentes conectados de menos de 3000 vóxeles (motas y rayas de *beam hardening*).
4. **Filtro de profundidad:** se descartan los componentes pequeños (< 50 000 vóxeles) que tocan la zona a menos de 10 mm de la piel (cinturones, electrodos o cables que rozan la silueta). El esqueleto y los fémures se conservan siempre. El resultado de los pasos 1–4 es el **esqueleto limpio**.
5. **Recuperación de hueso tenue:** desde el esqueleto limpio se crece hacia vóxeles de 150–250 HU conectados a él, solo a ≥ 10 mm de la piel. Después, un cierre morfológico (radio 2) sella grietas de la cortical y se rellenan las cavidades 3D encerradas (el interior esponjoso), también solo en profundidad.

Sin `spacing_zyx` la función se comporta como la versión original (solo pasos 1–3), lo que permite reproducir los diagnósticos anteriores.

### 8.2 Decisión inicial: apertura morfológica descartada

Se probó primero con apertura morfológica (erosión + dilatación) para limpiar cables y rayas delgadas, pero **se descartó**: a este spacing (~0.7–0.8 mm/vóxel) el hueso cortical real solo tiene 2–3 vóxeles de grosor, el mismo grosor que el ruido, así que la erosión perforaba y fragmentaba hueso sano (crestas ilíacas rotas, sacro deshecho). El filtro por tamaño de componente conectado es más seguro porque es una decisión de todo-o-nada por bloque completo, nunca le quita una capa a una estructura que sobrevive. Se mantiene `opening_iterations=0` por defecto.

### 8.3 Cómo se llegó a la limpieza final (paso a paso)

Al volver a ejecutar el V1 se detectaron dos problemas: **islas flotantes** alrededor de la pelvis en el caso 001 y **hueso incompleto** en el caso 025. Todas las reglas que se probaron se evaluaron como reglas únicas para los 100 casos, nunca como ajustes para un caso puntual. Cada decisión se tomó con un script de diagnóstico reproducible.

**Paso 1 — Descartar el downsample** (`volumen_3d_visualizer.py` con `DOWNSAMPLE_FACTOR = 1`). Se sospechó que el submuestreo del visualizador (se salta un vóxel de cada dos) creaba la franja punteada. Con factor 1 la imagen quedó igual, así que se descartó.

**Paso 2 — Identificar las islas** (`diag_islas.py`, `diag_islas2.py`). En el caso 001 la máscara tenía 3 componentes: el esqueleto (1 340 208 vóxeles) y dos islas de 11 962 y 10 174 vóxeles. Las islas superaban el filtro de 3000 vóxeles, por eso sobrevivían. Contra las etiquetas de los expertos, las dos islas tenían **0 %** de hueso etiquetado.

**Paso 3 — Descartar subir el tamaño mínimo y filtrar por forma** (`diag_islas3.py`, 20 casos). Fragmentos reales pequeños pesaban 12–15 mil vóxeles, igual que la basura (hasta 14 481), así que subir `MIN_NOISE_VOXELS` borraría fragmentos verdaderos. Se probó el "llenado" (vóxeles del componente / volumen de su caja): los fragmentos reales salieron con llenado bajo (0.033–0.095) y la basura en un rango amplio (0.002–0.528), así que tampoco separa. Los componentes de más de 70 000 vóxeles con 0 % etiquetado resultaron ser fémures (hueso real no incluido en la taxonomía del proyecto), por eso se fijó el umbral de 50 000 vóxeles para no tocarlos nunca.

**Paso 4 — Descartar la distancia al esqueleto** (`diag_islas4.py`, `diag_islas5.py`). Los fragmentos reales estaban a 1.6–2.6 mm del esqueleto, pero dos islas basura estaban a 2.6 y 2.8 mm, así que no separaba. Además, midiendo sobre las etiquetas, **el 7 % de los 59 fragmentos está a más de 3.5 mm de su hueso principal (máximo 73.9 mm)**. Esos fragmentos desplazados son justo los que el proyecto pide medir, y una regla de distancia los habría borrado. El HU medio tampoco separaba (rangos traslapados).

**Paso 5 — Adoptar la profundidad bajo la piel** (`diag_islas5.py`). Las islas del 001 estaban a **1.6 mm** de la piel; todos los demás componentes pequeños (fragmentos reales y hueso no etiquetado) estaban a **36 mm o más**. Justificación anatómica: un fragmento óseo queda rodeado de tejido blando, mientras que un objeto externo que roza la silueta queda en la superficie. Se eligió 10 mm, dentro del hueco y con margen hacia conservar hueso. **Validación en 20 casos no usados** (`diag_islas6.py`, casos 002–097 de 5 en 5): el filtro removió 7 componentes en 4 casos (007, 012, 022, 097), todos con 0 % de hueso etiquetado, y no hizo nada en los otros 16.

**Paso 6 — Medir el problema del hueso incompleto** (`diag_umbral.py`, 22 casos). Con el umbral de 250 HU la máscara capturaba en mediana solo el **57 %** del hueso etiquetado (rango 0.27–0.86). El problema no era exclusivo del 025 (0.31, HU mediano 140): también el 026 (0.27, HU 114), el 066 (0.34, HU 153) y el 001 (0.39, HU 173). En muchos pacientes el hueso esponjoso está por debajo de 250 HU. Se compararon candidatos (`diag_relleno.py`):

| Candidato | Recall (mediana) | Proxy de precisión (mediana) |
|---|---|---|
| Umbral 250 HU | 0.57 | 0.71 |
| Histéresis 150/250 | 0.80 | 0.71 |
| 250 + cierre + relleno | 0.68 | 0.73 |
| **Histéresis 150/250 + cierre + relleno** | **0.92** | **0.72** |

Bajar el umbral a 150 daba casi el mismo recall que la histéresis, sin aportar más. La combinación de histéresis con relleno fue la única que recuperó el interior esponjoso.

**Paso 7 — Error de validación y corrección del orden.** La primera integración aplicaba la recuperación **antes** de la limpieza. Al correr el visualizador, el caso 001 pasó a 3 616 886 vóxeles y aparecieron arcos grandes alrededor de todo el cuerpo. Causa: `diag_relleno.py` recortaba el volumen alrededor de las etiquetas (15 vóxeles de margen), así que nunca vio lo que había lejos de la pelvis. La recuperación fusionó esos arcos con el esqueleto, y el filtro de profundidad (que solo actúa sobre componentes pequeños) dejó de poder borrarlos. Se reordenó la función: **primero limpiar, después recuperar**, creciendo solo desde el esqueleto limpio y solo a ≥ 10 mm de la piel. Así la máscara final siempre contiene al esqueleto limpio y nada de lo borrado puede volver a entrar.

**Paso 8 — Validación final sobre volumen completo** (`diag_final_v1.py`, 22 casos, sin recorte).

- Recall mediano: **0.57 → 0.92**. 18 de 22 casos quedan en 0.80 o más.
- Fuga (vóxeles agregados a más de 25 mm del esqueleto limpio): **0.0 % en 21 casos, 0.1 % en el 021**. 25 mm supera el radio de cualquier estructura ósea de la pelvis o del fémur (la cabeza femoral tiene ~23 mm de radio). Esta métrica no detecta fugas pequeñas pegadas al hueso.
- Se verificó en todos los casos que la máscara final contiene al esqueleto limpio.
- Revisión visual en 001, 025 y 068: sin islas flotantes y sin tejido extraño visible. Conteos finales: 001 = 3 144 617, 025 = 1 120 961, 068 = 2 935 372 vóxeles.

### 8.4 Caso 025: CLAHE y V1 son cosas distintas

La hipótesis original (fijador externo ortopédico, por el HU máximo de 12 939) **se descartó en asesoría con el profesor**: si fuera metal no se vería la parte superior de la cresta ilíaca, y el diagnóstico contra los 100 casos (`data/eda_intensidades_detalle.csv`) confirma que el HU máximo y el volumen de metal del caso 025 **no son estadísticamente atípicos** (z-score < 2). Lo atípico real es `hueso_media` y `hueso_p50` (z ≈ −2.1), es decir, el histograma óseo está corrido y comprimido ~150–250 HU por debajo de lo típico — probablemente por diferencia de kernel de reconstrucción entre instituciones, no por metal.

- **Para el modelo** se corrigió con CLAHE dentro de `pengwin_io.window_hu` (sección 6). Detalle de por qué se descartaron gamma y ecualización global en `IA_USAGE.md`.
- **Para el V1** CLAHE no tiene efecto, porque el visualizador trabaja sobre HU reales y nunca llama a `window_hu`. Por eso el 025 seguía incompleto después de agregar CLAHE. Se mejoró con la recuperación de hueso tenue (sección 8.3, pasos 6–8): recall de 0.31 a 0.51.

### 8.5 Limitaciones conocidas del V1 (para el model card)

- Casos con hueso de muy baja densidad siguen incompletos: 025 (0.51), 026 (0.48) y 066 (0.54).
- Estructuras densas conectadas directamente al hueso (p. ej. artefactos que nacen de un implante) no se separan con filtros de tamaño ni de profundidad.
- La recuperación no actúa a menos de 10 mm de la piel, así que zonas óseas superficiales de baja densidad pueden quedar con huecos.
- La máscara es **solo visual**: el cierre morfológico puede unir huesos separados por un espacio fino o tapar una línea de fractura. No se usa para medir distancias ni alimenta el modelo.

## 9. EDA de fragmentos (`eda_fragments.py`)

Usa la convención oficial de etiquetas de PENGWIN: `0` fondo, `1–10` sacro, `11–20` coxal izquierdo, `21–30` coxal derecho. No apareció ningún label fuera de esa convención (`n_labels_desconocidos = 0` en los 100 casos). El volumen se calcula con el spacing real de cada caso (mm³, no vóxeles).

| Métrica | Valor |
|---|---|
| Casos analizados | 100 |
| Fragmentos totales por caso | mín 3 · máx 9 · media 5.75 |
| Fragmentos promedio — sacro | 1.53 |
| Fragmentos promedio — coxal izquierdo | 2.15 |
| Fragmentos promedio — coxal derecho | 2.07 |
| Casos con sacro sin fractura (1 fragmento) | 55 |
| Casos con coxal izquierdo sin fractura | 34 |
| Casos con coxal derecho sin fractura | 36 |
| Volumen de hueso por caso | 551 193 – 1 095 226 mm³ · media 781 058 mm³ (≈ 0.55 – 1.10 L) |

Distribución de fragmentos totales por caso:

| Fragmentos | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| Casos | 1 | 13 | 32 | 28 | 18 | 6 | 2 |

**Lectura de los resultados:**

- Las tres regiones están presentes en los 100 casos (mínimo = 3 fragmentos = una pelvis con un fragmento por hueso).
- El sacro es el hueso que menos se fractura (55 % intacto); los coxales se fracturan en ~65 % de los casos y son simétricos entre sí.
- Solo 1 caso no tiene ninguna fractura; es el mismo caso único del bucket "bajo" de los splits.
- El volumen de hueso varía ~2× entre casos (tamaño del paciente + campo de visión del CT).

Salidas: `data/eda_fragments_detalle.csv`, `outputs/eda/resumen_estadistico.csv`, `outputs/eda/hist_fragmentos_totales.png`, `outputs/eda/barras_fragmentos_por_region.png`, `outputs/eda/boxplot_volumen_hueso_mm3.png`.

## 10. EDA de fragmento principal y tamaño (`eda_fragmento_principal_y_tamano.py`)

Nuevo en `continuacion_semana08`. Dos análisis a nivel de fragmento individual (575 fragmentos en total, no de caso):

**1. ¿La etiqueta 1/11/21 es siempre el fragmento de mayor volumen de su región?** — necesario para definir el "fragmento principal" del que se mide la distancia de separación (sección 3.3 del proyecto).

```text
Casos-region evaluados : 300
Cumple la convencion   : 300 (100.0%)
NO cumple              : 0 (0.0%)
```

**Confirmado al 100 % en los 300 casos-región (100 casos × 3 regiones).** Se puede usar la etiqueta 1/11/21 directamente como fragmento principal en la semana 10, sin calcular el máximo por caso.

**2. Tamaño de fragmento en mm³ y cortes axiales ocupados** — para estimar riesgo de perder fragmentos pequeños al bajar a 224–256 px.

```text
Fragmentos pequenos (<5 cortes o <500 mm3): 2 de 575 (0.3%)
```

Riesgo bajo: solo el 0.3 % de los fragmentos son pequeños.

**Hallazgo complementario (sección 8.3, paso 4):** el 7 % de los fragmentos (muestra de 59, 20 casos) está a más de 3.5 mm de su fragmento principal, con un máximo de 73.9 mm. Es un primer indicio del rango de distancias de separación que medirá la semana 10.

Salidas: `data/eda_fragmentos_individuales.csv`, `data/eda_fragmento_principal_resumen.csv`, `outputs/eda/hist_tamano_fragmentos_mm3.png`, `outputs/eda/hist_cortes_por_fragmento.png`.

## 11. `requirements.txt`

Se reemplazó el archivo original (un `pip freeze` completo de macOS con ~200 líneas y versiones fijas, que no instalaba en Windows por incompatibilidades de versión — ej. `contourpy==1.4.0` no existe para Python <3.12) por una lista curada de solo las librerías que el código importa directamente, sin versión fija: `SimpleITK`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `scipy`, `scikit-image`, `plotly`, `torch`. `pip` resuelve versiones compatibles según el sistema de cada integrante.

**Nota para equipos sin GPU NVIDIA (Windows):** instalar `torch` primero con `pip install torch --index-url https://download.pytorch.org/whl/cpu` antes de `pip install -r requirements.txt`, o la build CUDA por defecto de pip falla al importar (`OSError ... c10_cuda.dll`) por falta de drivers NVIDIA.

## 12. Mini-resumen: qué se hizo y con qué comando

Todos los comandos se corren **desde la raíz del repo**.

| # | Paso | Comando | Qué produce |
|---|---|---|---|
| 1 | Manifest del dataset | `python src/build_manifest.py` | `data/manifest.csv` |
| 2 | Diagnóstico de HU | `python src/check_hu_outliers.py` | Consola: 52 casos con padding/metal |
| 3 | Splits fijos | `python src/make_splits.py` | `data/splits.json` (70/15/15, seed 42) |
| 4 | Ventaneo HU | `python src/hu_windowing_check.py` | `outputs/figures/hu_window_check_*.png` |
| 5 | Visualizador 1 (3D) | `python src/volumen_3d_visualizer.py` | `outputs/figures/volumen_3d_<id>.html` |
| 6 | EDA de fragmentos | `python src/eda_fragments.py` | `data/eda_fragments_detalle.csv` + `outputs/eda/*` |
| 7 | EDA fragmento principal | `python src/eda_fragmento_principal_y_tamano.py` | `data/eda_fragmentos_individuales.csv` + `outputs/eda/*` |

Orden obligatorio: el paso 1 va primero (2, 3, 6 y 7 leen `data/manifest.csv`).

**Diagnósticos de la limpieza del V1 (opcionales, evidencia de la sección 8.3).** Imprimen en consola; algunos tardan varios minutos porque abren 20–22 volúmenes.

| Script | Qué verifica |
|---|---|
| `python src/diag_islas.py` | Componentes de la máscara del caso 001 y su posición |
| `python src/diag_islas2.py` | % de cada componente dentro del hueso etiquetado (001) |
| `python src/diag_islas3.py` | Tamaño y "llenado" de componentes secundarios (20 casos) |
| `python src/diag_islas4.py` | Distancia al esqueleto y HU medio (20 casos) |
| `python src/diag_islas5.py` | Profundidad bajo la piel y distancia real de fragmentos (20 casos) |
| `python src/diag_islas6.py` | Validación del filtro de profundidad en 20 casos no usados |
| `python src/diag_umbral.py` | Recall del umbral de 250 HU frente a etiquetas (22 casos) |
| `python src/diag_relleno.py` | Comparación de candidatos de recuperación (22 casos, en caja) |
| `python src/diag_final_v1.py` | Validación final en volumen completo: recall, fuga y tiempo (22 casos) |

> `diag_islas.py` a `diag_islas5.py` llaman a `clean_bone_mask` sin `spacing_zyx`, así que reproducen el estado anterior a los filtros nuevos. `diag_islas6.py` se ejecutó antes de agregar la recuperación de hueso tenue; la validación con la versión final es `diag_final_v1.py`.

## 13. Cómo ejecutarlo desde cero

**1. Clonar el repo**

```bash
git clone https://github.com/mariavalencia30/proyecto2-analitica-datos.git
cd proyecto2-analitica-datos
git checkout continuacion_semana08
```

**2. Entorno virtual e instalación**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt     # ver sección 11 sobre torch en equipos sin GPU
```

**3. Descargar los datos** desde Zenodo (https://doi.org/10.5281/zenodo.10927452), descomprimir y dejarlos exactamente así:

```text
data/raw/
├── PENGWIN_CT_train_images_part1/   001.mha …
├── PENGWIN_CT_train_images_part2/   … 100.mha
└── PENGWIN_CT_train_labels/         001.mha … 100.mha
```

**4. Correr el pipeline completo**

```bash
python src/build_manifest.py
python src/check_hu_outliers.py
python src/make_splits.py
python src/hu_windowing_check.py
python src/volumen_3d_visualizer.py
python src/eda_fragments.py
python src/eda_fragmento_principal_y_tamano.py
```

**5. Verificar:** el manifest debe reportar 100/100 válidos, los splits 70/15/15 (semilla fija = reproducible byte a byte), el EDA de fragmento principal debe reportar 300/300 (100 %) y `python src/diag_final_v1.py` debe dar recall mediano 0.92 con fuga máxima 0.1 %.

## 14. Estado de la Semana 8 y pendientes

- [x] Dataset curado con splits fijos.
- [x] Carga de `.mha` + ventaneo HU con clip, probado visualmente en 3 casos.
- [x] EDA de fragmentos por caso.
- [x] Visualizador 1 — reconstrucción 3D interactiva (no solo MIP), limpia de camilla/cables.
- [x] Módulo de preprocesamiento reutilizable (`pengwin_io.py`).
- [x] EDA de tamaño de fragmento (mm³, cortes) y validación del fragmento principal.
- [x] `requirements.txt` curado y probado en Windows sin GPU.
- [x] `hu windowing check.py` renombrado a `hu_windowing_check.py`.
- [x] V1 sin islas flotantes: filtro de profundidad bajo la piel, validado en 20 casos no usados.
- [x] V1 con hueso completo: recuperación de hueso tenue, recall mediano 0.57 → 0.92 sin fuga detectable (22 casos).
- [x] Verificado que la máscara del V1 no alimenta al modelo.
- [ ] Llenar `README.md` e `IA_USAGE.md` (bitácora crítica, no volcado de prompts; incluir la secuencia de la sección 8.3 con los errores).
- [ ] Pasar las limitaciones de la sección 8.5 al model card.
- [ ] Pull request de `continuacion_semana08` → `main` con revisión por pares.

## Archivos generados hasta ahora

- `src/`: `build_manifest.py`, `check_hu_outliers.py`, `make_splits.py`, `hu_windowing_check.py`, `pengwin_io.py`, `volumen_3d_visualizer.py`, `eda_fragments.py`, `eda_fragmento_principal_y_tamano.py`, `diagnostico_normalizacion_caso025.py`, `verificar_clahe_casos_ejemplo.py`, `diag_islas.py`, `diag_islas2.py`, `diag_islas3.py`, `diag_islas4.py`, `diag_islas5.py`, `diag_islas6.py`, `diag_umbral.py`, `diag_relleno.py`, `diag_final_v1.py`
- `data/`: `manifest.csv`, `splits.json`, `eda_fragments_detalle.csv`, `eda_fragmentos_individuales.csv`, `eda_fragmento_principal_resumen.csv`
- `outputs/figures/`: `hu_window_check_{001,068,080}.png`, `volumen_3d_{001,025,068}.html`
- `outputs/eda/`: `resumen_estadistico.csv`, `hist_fragmentos_totales.png`, `barras_fragmentos_por_region.png`, `boxplot_volumen_hueso_mm3.png`, `hist_tamano_fragmentos_mm3.png`, `hist_cortes_por_fragmento.png`, `diagnostico_normalizacion_caso025.png`, `verificacion_clahe_casos_ejemplo.png`
# Plan de la Semana 10 (v3) — reinicio desde cero

Actualizado: 6 de octubre de 2026 (sección 0c con el cierre; lo demás es del 5 de octubre, v3: estado de experimentos, checklist y tabla de modelos). El resto del documento es la v2 sin cambios. Rama base: `unificado_semana09` (commit `a667c41`).
Reemplaza la versión 1 de este mismo documento.

## 0. Decisiones tomadas

- **La Semana 10 se rehace desde cero.** Lo que ya existía (`week10_*`, modelos `w10g_*`)
  queda como referencia y como línea base a superar; no se borra.
- **Rama base:** `unificado_semana09`.
- **Registro de IA:** lo llevan Gustavo y Claude en `IA_USAGE.md`, una entrada por sesión.
- **No se espera al profesor.** En las dudas de interpretación se hacen y se reportan las
  dos lecturas (sección 2).
- **Se explora todo lo que pueda ayudar.** El único límite son las restricciones del
  enunciado. Las recomendaciones (resolución, batch) no limitan.

## 0c. Estado al 6 de octubre de 2026 (17:00 UTC): cierre de la exploración

La sección 0b quedó atrás. El registro completo y actualizado está en el repositorio:
`docs/decisiones_semana10.md` (bitácora de decisiones). Resumen:

- **Modelo final: H6** = receta C2 + EMA (ancho 32, CBAM en 1 etapa, secundarios x3, 60 épocas)
  más aumento geométrico con rotación de +-35 grados y orientación canónica según la cabecera
  del .mha (`--aug-geo 1.0 --aug-rot-deg 35 --canonical`). Archivo:
  `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth`.
- **Test, primera evaluación (D5 s42 + TTA, configuración congelada antes de ver test):**
  Dice de instancias 0.845; no cumplía el objetivo de 0.85. Es la cifra limpia.
- **Diagnóstico:** dos casos hundían el test (012, paciente girado unos 26 grados; 039,
  orientación RAI con aspecto de LPI). El 34 % de los volúmenes está guardado girado 180
  grados y el modelo no veía ningún aumento geométrico.
- **Test, segunda evaluación (H6 s42 + TTA, posterior al análisis de fallos):** Dice de
  instancias 0.875 (primario 0.915, secundario 0.750), F1 0.990, mAP50 0.961, mAP50:95 0.902,
  Dice semántico 0.971. Cumple todos los objetivos. Las semillas 43 y 44 dan 0.875 y 0.876; el
  ensamble de las tres, 0.879. Caso 012: 0.692 → 0.865; caso 039: 0.710 → 0.784.
- **Validación (H6, tres semillas):** 0.869 / 0.868 / 0.872; con la validación girada 30 grados,
  0.854 (D5: 0.559).
- **Distancias en mm (test):** error medio 2.96 → 1.28 mm y correlación 0.54 → 0.82; un solo
  fragmento (caso 007) concentra el 79 % del error.
- **Hecho además:** ablaciones de CBAM y transfer learning, calibración de lambdas, tres
  semillas, post-proceso 3D, SAM zero-shot (en test con H6: 0.889 frente a 0.934 nuestro en Dice
  por región),
  latencia en GPU y CPU.
- **Descartado:** detector geométrico de giro (falla en pacientes bien colocados), filtros de
  post-proceso, ensamble como modelo principal (gana 0.003, dentro del ruido).
- **Análisis de error en test (H6):** la confusión entre regiones bajó a 0.06 % del hueso; el
  error que queda está en dónde se parte el hueso en fragmentos. Peores casos: 039, 061, 007, 054
  y 047 (`outputs/s10/informe_h6_test/`).
- **Documentado:** notebook `notebooks/04_semana10_pipeline_multitarea.ipynb` (lee los resultados
  guardados), `docs/resumen_semana10.md`, `outputs/s10/resumen_experimentos.md`, README e
  `IA_USAGE.md` (entradas 15 a 18).
- **Pendiente:** llevar H6 y el paso de orientación canónica al dashboard y la API de la
  semana 11 (lo hace la sesión de la Semana 11); latencia en el Mac.

## 0b. Estado al 5 de octubre de 2026 (15:30 UTC)

Se actualiza a medida que terminan las corridas. Resultados de validación sobre el volumen completo (15 casos, TTA espejo). Detalle de métricas: `outputs/s10/resumen_experimentos.md`.

Leyenda: **Hecho** · **En curso** · **Parcial** · **Pendiente** · **Descartado**.

### Checklist del mapa de exploración (sección 5)

| # | Qué | Estado | Resultado o nota |
|---|---|---|---|
| A1 | Extensión completa, todos los cortes | Hecho | Cachés a 384 px, 70/15/15 casos; entrenamiento con muestreo por época |
| A2 | Cortes sin hueso | Hecho | `--neg-fraction 0.15`, margen de 10 cortes; AUC 0.99 |
| A3 | Muestreo dirigido a secundarios y extremos | Pendiente | Solo se probó el peso en la pérdida (E4) |
| A4 | Aumentos | Hecho | `--augment` en todas las corridas desde B1; sin ablación aislada |
| A5 | Recorte al cuerpo | Pendiente | |
| A6 | Contexto 2.5D | Pendiente | `--context` no está en `train_s10.py`; hay que portarlo |
| A7 | Resolución 256 / 384 / 512 | Parcial | Solo 384. Faltan 256 y 512 |
| B1 | 16 / 32 / 48 canales | En curso | 8 → 32 canales es la mayor mejora. 16 y 32 hechos; 48 en curso |
| B2 | CBAM ninguno / 1 etapa / 3 etapas con γ | Hecho | 1 etapa gana en detección; no mejora instancias. Una semilla; se repite en la fase 5 |
| B3 | Bloques estilo FundidoraPC | Pendiente | |
| B4 | Más campo receptivo | Pendiente | |
| C1 | Pesos de FundidoraPC | Pendiente | Preguntar al equipo |
| C2 | Preentrenar en Intel Image | Pendiente | Ablación obligatoria; se hace antes de entrenar el modelo final |
| D1 | Medir la cabeza grid cruda | Hecho | c2: mAP50 0.875, mAP50:95 0.474 sin refinar |
| D2 | Varias celdas positivas por región | Pendiente | |
| D3 | Pérdida de caja por IoU | Pendiente | |
| D4–D6 | Objectness por IoU, grid fino, regresión a bordes | Pendiente | |
| D7 | Caja refinada con la semántica | Hecho | Se reportan ambas |
| E1 | Formulaciones de instancia | Parcial | `primary` frente a `ids`: `ids` pierde 45 % de secundarios (Dice 0.787), descartado. Borde, centros y offsets existen como cabezas auxiliares |
| E2 | Segundo pase por región | Pendiente | `--two-pass` solo está en `evaluate_volumes.py` |
| E3 | Post-proceso 3D | Pendiente | No requiere reentrenar; se aplica sobre cualquier modelo ya entrenado |
| E4 | Pérdidas para fragmentos pequeños | Parcial | Peso a secundarios ×3 (B1) y ×6 (B2) y peso por fractura (v4) no mejoran instancias. Faltan focal y Tversky |
| E5 | Supervisión profunda | Pendiente | |
| F1 | Calibrar λ | En curso | Primer ensayo: ancho 32 con instancia ×2, borde ×2, detección ×0.5 |
| F2 | Tres semillas | En curso | B1 hecho (3 semillas); c2 con semillas 43 y 44 en curso |
| F3 | Umbral por clase | Pendiente | |
| F4 | Promedio móvil de pesos | Pendiente | Coseno y calentamiento ya se usan |
| F5 | TTA multiescala | Pendiente | Espejo en uso |
| F6 | HD95 y ASSD | Pendiente | |

### Fases (sección 6)

| Fase | Estado |
|---|---|
| 0 Preparación | Hecho: cachés, evaluador de volumen completo, línea base |
| 1 Modelo desde cero | Hecho |
| 2 Receta base | Hecho como B1 (8 canales, 384 px, CBAM 1 etapa, sec3, 80 ép., 3 semillas). Difiere de lo planeado (256 px, 32 canales) |
| 3 Exploración | En curso: ancho, instancia, λ, peso por fractura |
| 4 Modelo candidato | Pendiente: c2 es el candidato provisional |
| 5 Ablaciones obligatorias | Pendiente |
| 6 Volumen: 3D y distancias | Pendiente |
| 7 SAM | Pendiente |
| 8 Test, latencia, notebook 04 | Pendiente |

### Modelos probados

Todos: CBAM en 1 etapa salvo indicación, resolución 384, batch 16, coseno, aumentos, semilla 42 salvo indicación, instancia `primary` salvo indicación. "ép. × cortes" son épocas por cortes muestreados por época.

| Modelo | Canales | Parámetros | ép. × cortes | Cambio principal | Dónde | Min | mAP50 ref. | mAP50:95 ref. | Dice seg. | Dice inst. | Dice secundarios |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `w10g_384neg` | — | — | — | Línea base de la Semana 10 anterior | Kaggle | — | 0.874 | 0.781 | 0.956 | 0.834 | 0.702 |
| B0 sin CBAM | 8 | 173 mil | 40 × 2 500 | Ablación | Kaggle | — | 0.900 | 0.796 | — | — | — |
| B0 CBAM 1 | 8 | 173 mil | 40 × 2 500 | Ablación | Kaggle | — | 0.926 | 0.835 | — | — | — |
| B0 CBAM 3 + γ | 8 | — | 40 × 2 500 | Ablación | Kaggle | — | 0.907 | 0.814 | — | — | — |
| B1 (s42 / s43 / s44) | 8 | 173 mil | 80 × 2 500 | Secundarios ×3 | Kaggle T4 | 64 | 0.935 / 0.937 / 0.929 | 0.850 / 0.854 / 0.848 | 0.963 / 0.960 / 0.963 | 0.835 / 0.826 / 0.826 | 0.661 / 0.631 / 0.644 |
| B2 | 8 | 173 mil | 50 × 2 500 | Secundarios ×6 | Kaggle T4 | 41 | 0.918 | 0.834 | 0.963 | 0.832 | 0.681 |
| v4 | 8 | 173 mil | 60 × 5 000 | Peso por fractura 2 | Colab L4 | 81 | 0.945 | 0.867 | 0.965 | 0.832 | 0.642 |
| `ids` | 8 | 174 mil | 80 × 2 500 | Instancia con 31 clases | Colab L4 | 62 | 0.933 | 0.844 | 0.962 | 0.787 | 0.460 |
| c1 | 16 | 688 mil | 80 × 2 500 | Ancho 16 | Kaggle T4 | 70 | 0.955 | 0.886 | 0.969 | 0.839 | 0.649 |
| **c2** | **32** | **2.74 M** | **60 × 2 500** | **Ancho 32** | Kaggle T4 | 62 | **0.962** | **0.889** | **0.969** | **0.849** | **0.687** |

En curso: c2 semillas 43 y 44 (Kaggle), ancho 48 con 6.15 M de parámetros (Colab) y ancho 32 con λ ajustados (Colab).

### Qué es código nuevo y cuándo se ejecuta

- **Antes de entrenar** (cambia lo que aprende el modelo): contexto 2.5D, pérdida focal o Tversky, supervisión profunda, promedio móvil de pesos, pérdida de caja por IoU, varias celdas positivas, y el **preentrenamiento en Intel Image (C2)**, que es un paso previo: se entrena el backbone en clasificación de imágenes naturales, se guardan sus pesos y el entrenamiento en PENGWIN parte de ahí.
- **Después de entrenar** (no se reentrena nada; se aplica a un modelo ya entrenado): el post-proceso 3D, las distancias en mm, HD95 y ASSD, SAM, el umbral por clase, la latencia y la evaluación en test. Estos pasos se pueden desarrollar mientras otros modelos entrenan.
- **Intermedio:** el segundo pase por región funciona al entrenar o solo al inferir; se prueba primero solo al inferir con el modelo c2.

### Por qué 60 u 80 épocas

Aquí una "época" no es una pasada sobre todo el conjunto: cada una toma 2 500 cortes al azar de los ~18 000 de entrenamiento. Ochenta épocas son unos 200 000 cortes vistos, cerca de once pasadas completas; 60 épocas con 5 000 cortes (v4) son unas diecisiete. Las curvas de validación indican que no sobra: en B1 el Dice de instancia sigue subiendo hasta la época 72 (0.805 en la 24, 0.832 en la 72) y la pérdida de entrenamiento aún baja. En c2 la mejor época es la 40 (instancia 0.851) y de ahí oscila (0.847, 0.843), lo que sugiere que con el ancho 32 ya converge sobre esas 60 y lo que falta no es más tiempo.

## 1. Restricciones del enunciado y recomendaciones

Restricciones. No se negocian:

| Tema | Restricción |
|---|---|
| Taxonomía | Tres regiones: sacro, coxal izquierdo, coxal derecho. No subdividir ilion, isquion ni pubis |
| Etiquetas | Se usan las del dataset. No se generan etiquetas nuevas, ni a mano ni con SAM |
| Backbone | Extensión propia de FundidoraPC o arquitectura equivalente diseñada por el equipo, compartido por las tres cabezas |
| Atención | CBAM (canal y espacial) en el backbone, antes de la bifurcación a las cabezas |
| Cabezas | Clasificación, detección (grid propio y NMS propio) y segmentación de instancia, entrenadas desde cero |
| Segmentación | Salida semántica más salida de instancia, con lógica de dos etapas: primero regiones, luego fragmentos dentro de cada región |
| Transfer learning | Solo en el backbone; pesos de FundidoraPC del curso, ImageNet u otro dataset natural |
| Pérdida | Compuesta multitarea, con λ calibrados y justificados; no copiados |
| Prohibido | YOLO, Detectron2, Mask R-CNN preentrenado y cualquier framework de detección o segmentación de alto nivel |
| SAM | Solo como baseline zero-shot, con la caja predicha como prompt, sobre test. Nunca en el pipeline |
| Distancia | Mínima borde a borde, en mm, con `distance_transform_edt` y el spacing del header; también sobre el ground truth |
| Cómputo | Precisión mixta con `torch.cuda.amp`; inferencia ejecutable en GPU y en CPU; latencia por imagen medida en ambas |
| Informe | Ablación con y sin CBAM, y con y sin transfer learning; matriz de confusión; cinco peores casos; distancia predicción contra GT |
| Demo | Corte a corte, aunque el modelo use contexto 2.5D (pedido del avance del 2 de octubre) |

Recomendaciones. Se pueden explorar sin límite:

| Tema | Lo que dice el enunciado |
|---|---|
| Resolución y batch | "Se recomienda" 224–256 px y batch 8–16 |
| Objetivos de métricas | Son "objetivos sugeridos" |
| HD95 y ASSD | Opcionales; pueden dar puntos extra |

## 2. Las cuatro dudas, resueltas sin esperar

| Duda | Qué se hace |
|---|---|
| ¿Dice por fragmento o semántico? | Se reportan tres cifras: semántico por región, por fragmento emparejado en 2D y por fragmento en 3D sobre el volumen. El objetivo de trabajo es el más estricto: ≥ 0.85 por fragmento |
| ¿Cajas crudas o refinadas? | Se reportan las dos. Se trabaja para que la cabeza grid cruda cumpla por sí sola; así la respuesta deja de importar |
| ¿384 px? | Es una recomendación. Se comparan 256, 384 y 512 y se elige por validación y latencia |
| ¿Qué SAM? | ViT-B oficial como base por costo; ViT-H si la GPU alcanza. Prompt: la caja predicha, por región y por fragmento. Dice e IoU sobre los mismos cortes de test que el modelo propio |

## 3. Qué se conserva, qué se rehace

| | Archivos | Motivo |
|---|---|---|
| Se conserva | `pengwin_io.py`, `build_manifest.py`, `make_splits.py`, `data/splits.json`, EDA | Semanas 8 y 9, validado |
| Se conserva | `week9_detection.py` (backbone, CBAM, grid, NMS) y sus pruebas | Probado con overfit y pruebas unitarias |
| Se conserva | `pengwin_metrics.py`, `fragment_distances.py`, `instance_postprocess.py` | Se amplían, no se reescriben |
| Se rehace | Modelo de tres cabezas, entrenamiento, evaluación, inferencia | Archivos nuevos: `multitask_model.py`, `train_multitask.py`, `evaluate_multitask.py`, `infer_volume.py` |
| Referencia | `week10_*.py`, `outputs/gamma/w10g_*` | No se tocan. `w10g_384neg` es la línea base |

Salidas nuevas en `outputs/s10/`. Notebook: `notebooks/04_semana10_pipeline_multitarea.ipynb`.

Línea base a superar (`w10g_384neg`, test, 180 cortes con hueso, TTA): F1 0.988, AUC 0.930,
mAP50 refinado 0.901, Dice semántico 0.942, Dice por fragmento 0.817, IoU por fragmento
0.747. Con la regla de medición nueva estas cifras pueden bajar; se vuelven a medir en la
fase 0.

## 4. Regla de medición

- Checkpoint, umbrales, TTA y post-proceso se eligen en validación. Test se corre una vez,
  al final, con todo congelado.
- Las métricas se calculan sobre el volumen completo de cada caso, no sobre una muestra
  del centro.
- Un cambio se acepta solo si mejora validación por encima del ruido. El ruido se mide con
  tres semillas en la receta base, como se hizo en la Semana 9.
- Cada corrida guarda configuración, semilla, historial por época y métricas en JSON.

## 5. Mapa de exploración

Prioridad 1: ataca un objetivo que hoy no se cumple o es obligatorio. Prioridad 2: mejora
probable. Prioridad 3: se prueba si sobra cómputo.

### A. Datos

| # | Qué probar | Evidencia | P |
|---|---|---|:-:|
| A1 | Extensión completa y todos los cortes | Hoy se usan 12 de ~244 cortes por caso; 3 de 5 fragmentos desplazados están fuera del rango entrenado | 1 |
| A2 | Cortes sin hueso | A igual resolución, el AUC en test sube de 0.75 a 0.93 al agregarlos | 1 |
| A3 | Muestreo dirigido: más peso a cortes con fragmentos secundarios y a los extremos | 275 de 575 fragmentos son secundarios y ocupan pocos píxeles | 2 |
| A4 | Aumentos geométricos leves e intensidad | Brillo del hueso varía de 0.19 a 0.40 entre casos (notebook 03, sección 1.1) | 2 |
| A5 | Recorte al cuerpo antes de redimensionar | Más resolución efectiva sin más píxeles | 2 |
| A6 | Contexto 2.5D (cortes vecinos como canales) | Ayuda a decidir primario y secundario; ya existe `--context` | 2 |
| A7 | Resolución 256, 384, 512 | Pasar a 384 subió el Dice por fragmento de 0.803 a 0.835 en validación | 2 |

### B. Backbone y atención

| # | Qué probar | Evidencia | P |
|---|---|---|:-:|
| B1 | 16, 32 y 48 canales base | La Semana 9 mostró que 32 supera a 8; la Semana 10 usó 16 sin compararlo | 1 |
| B2 | CBAM apagado, en una etapa y en tres etapas con γ | Ablación obligatoria; Grad-CAM favoreció tres etapas (55 % contra 46 %) | 1 |
| B3 | Bloques al estilo FundidoraPC del curso frente a los actuales | Necesario si aparecen pesos de FundidoraPC | 3 |
| B4 | Más campo receptivo: etapa extra o convoluciones dilatadas | El coxal es largo y curvo; su centro cae en tejido blando | 3 |

### C. Transfer learning

| # | Qué probar | Evidencia | P |
|---|---|---|:-:|
| C1 | Ubicar pesos de FundidoraPC del curso | No están en la carpeta del curso; hay que preguntar al equipo | 1 |
| C2 | Preentrenar nuestro backbone en Intel Image Classification y cargarlo | Dataset natural, ya está en disco (67 MB, 6 clases); ablación obligatoria | 1 |

### D. Detección

| # | Qué probar | Evidencia | P |
|---|---|---|:-:|
| D1 | Medir la cabeza grid cruda | Nunca se midió en la Semana 10 | 1 |
| D2 | Varias celdas positivas por región | Hoy hay un solo positivo por región y por corte | 1 |
| D3 | Pérdida de caja basada en IoU, implementación propia | Las cajas quedan cortas o corridas en coxales largos | 1 |
| D4 | Objectness entrenado contra el IoU de la caja | El puntaje no refleja la calidad de la caja (sección 5.1 del notebook 03) | 2 |
| D5 | Grid más fino o dos escalas (/8 y /16) | El sacro delgado del caso 067 no se detectó | 2 |
| D6 | Regresión de distancias a los bordes, sin anclas | Alternativa a centro y tamaño | 3 |
| D7 | Una caja por región y refinamiento con la semántica | Ya existen; se reportan como segunda etapa | — |

### E. Segmentación e instancia

| # | Qué probar | Evidencia | P |
|---|---|---|:-:|
| E1 | Comparar formulaciones de instancia con los mismos datos: primario/secundario, borde, centros y offsets, IDs | El código de las cuatro existe; se compararon antes del cambio a gamma y con 12 cortes por caso | 1 |
| E2 | Segundo pase por región recortada | Es la lógica de dos etapas que pide el enunciado; existe `--two-pass` | 1 |
| E3 | Post-proceso 3D: componentes conectados en el volumen, primario = el mayor, voto entre cortes vecinos | Un corte 2D no sabe cuál fragmento es el principal del hueso | 1 |
| E4 | Pérdidas para fragmentos pequeños: peso de fractura, peso de secundarios, focal o Tversky | El Dice por fragmento es el único objetivo que no se cumple | 2 |
| E5 | Supervisión profunda en el decodificador | Mejora habitual en segmentación con pocos datos | 3 |

### F. Entrenamiento e inferencia

| # | Qué probar | Evidencia | P |
|---|---|---|:-:|
| F1 | Calibrar λ: igualar la magnitud inicial de cada término y comparar con ponderación aprendida por incertidumbre | Obligatorio; hoy hay nueve términos con pesos por defecto | 1 |
| F2 | Tres semillas en las comparaciones | Con una semilla la ablación de la Semana 9 no fue concluyente | 1 |
| F3 | Calibrar el umbral de clasificación por clase | F1 baja a 0.94 al incluir cortes sin hueso | 2 |
| F4 | Promedio móvil de pesos, calentamiento y coseno | Validación oscila entre épocas | 2 |
| F5 | TTA con espejo y multiescala | El espejo ya existe; cuesta latencia | 3 |
| F6 | HD95 y ASSD por fragmento | Opcional con puntos extra | 2 |

### Zonas grises: no se usan sin confirmar con el profesor

- Backbone estándar con pesos de ImageNet (por ejemplo ResNet): el enunciado permite pesos
  de ImageNet, pero exige backbone propio o equivalente diseñado por el equipo.
- Preentrenamiento autosupervisado sobre los mismos CT de PENGWIN.
- Datos externos de pelvis para preentrenar.
- Red con convoluciones 3D: el proyecto está planteado como 2D corte a corte.

## 6. Orden de ejecución

```
FASE 0 · Preparación (sin entrenar)
  Cachés con extensión completa y negativos · evaluador de volumen completo
  · línea base de w10g_384neg · cronometrar una época
  Compuerta: tabla de línea base con las tres lecturas de Dice y detección
             cruda y refinada

FASE 1 · Modelo desde cero, pieza por pieza
  Backbone con CBAM configurable → clasificación → detección →
  semántica → instancia. Prueba y overfit de cada cabeza.
  Compuerta: overfit conjunto con reducción de pérdida ≥ 4× y pruebas en verde

FASE 2 · Receta base B0
  Extensión completa, negativos, 256 px, 32 canales, CBAM en una etapa,
  tres semillas.
  Compuerta: métricas de validación y ruido entre semillas

FASE 3 · Exploración, un cambio a la vez contra B0
  Primero toda la prioridad 1 (A1-A2, B1-B2, C2, D1-D3, E1-E3, F1),
  luego la 2. Se conserva lo que supere el ruido.

FASE 4 · Modelo candidato
  Combinar los cambios ganadores y entrenar con tres semillas.

FASE 5 · Ablaciones obligatorias sobre el candidato
  Con y sin CBAM · con y sin transfer learning · λ

FASE 6 · Volumen: post-proceso 3D, distancias en mm contra GT, HD95 y ASSD
FASE 7 · SAM con cajas predichas
FASE 8 · Test una sola vez · latencia en CPU y GPU · cierre del notebook 04
```

## 7. Cómputo

El plan explora mucho y eso no cabe en CPU. El `.venv` del repo tiene PyTorch 2.2.2 para
x86_64, lo que indica un Mac Intel sin GPU de Apple. Se confirma con
`python -c "import torch; print(torch.backends.mps.is_available())"`.

Estimación gruesa, a partir de los 116.8 ms por corte que midió Kaggle en CPU a 384 px con
16 canales: una época con extensión completa (unos 16 000 cortes) tomaría del orden de
40 minutos a 256 px y más de una hora y media a 384 px. Ochenta épocas serían días.
Hay que cronometrarlo en la fase 0 antes de decidir.

Reparto propuesto:

| Dónde | Qué |
|---|---|
| Mac | Desarrollo, pruebas, overfit, cachés, evaluación, post-proceso 3D, distancias |
| Kaggle (T4) | Receta base, exploración, ablaciones, SAM y latencia en GPU |

Además, `torch.cuda.amp` es una restricción del enunciado y solo se ejercita con CUDA.

## 8. Notebook 04

Mismo patrón del 03: llama a los scripts de `src/`, `REENTRENAR = False` lee resultados
guardados, y cada sección cierra con lectura y decisión.

0. Entorno y pruebas
1. Qué hereda de la Semana 9 y línea base de la Semana 10 anterior
2. Datos: cachés con extensión completa y negativos
3. Arquitectura de tres cabezas y overfit
4. Pérdida compuesta y calibración de λ
5. Receta base y ruido entre semillas
6. Exploración: tabla de experimentos contra la receta base
7. Modelo candidato y métricas contra objetivos
8. Ablaciones: CBAM y transfer learning
9. Volumen: post-proceso 3D y distancia de separación contra GT
10. Baseline SAM
11. Análisis de error: cinco peores casos, matriz de confusión, fragmentos desplazados
12. Resumen y decisiones

## 9. Pendientes con el equipo

1. ¿Alguien conserva pesos de FundidoraPC (`fundidora_unet_3c.pth`, `reto_s1_<código>.pth`)?
2. Acceso de todos al dataset de Kaggle con los cachés y cuota de GPU disponible.
3. Reparto de frentes: modelo y entrenamiento; distancias y SAM; visualizadores y
   dashboard; documentación.
4. Fechas del avance de la Semana 10 y de la entrega de la Semana 11.

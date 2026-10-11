# Semana 9 Backbone CBAM y deteccion

## Objetivo

Implementar una primera versión verificable de la detección corte a corte: localizar una
bounding box para cada región anatómica visible en el corte, usando un backbone propio,
CBAM y una cabeza por grid.

## Diseño implementado

- `FundidoraBackbone`: cuatro bloques convolucionales con reducción espacial 16x. Acepta
  `use_cbam=False` para la ablación.
- `CBAM`: atención de canal y espacial aplicada antes de la cabeza.
- `GridDetectionHead`: por celda predice objectness, `(cx, cy, w, h)` y una clase entre
  sacro, coxal izquierdo y coxal derecho. Para entrada `256x256` el grid es `16x16`.
- Pérdida multitarea: BCE de objectness, Smooth L1 de caja y entropía cruzada de clase.
- `nms`: supresión de no máximos escrita en PyTorch, sin frameworks de detección.
- Tamaño: 1,77 M de parámetros con 32 canales base (backbone + CBAM + cabeza).

## Preprocesamiento de entrada al modelo (ajuste tras la asesoría)

La imagen que entra al modelo sale de `pengwin_io.window_hu`: clip `[-1024, 3000]` HU →
ventana ósea C400/W1800 → `[0, 1]` → **gamma 1.5**. Gamma reemplazó al CLAHE de la Semana 8
después de la asesoría, con evidencia propia:

- `experimento_normalizacion.py` barrió 166 configuraciones de NGC-CLAHE (gamma, clip
  limit, tamaño de mosaico, con y sin normalización) sobre 5 casos de referencia midiendo
  contraste-ruido hueso/tejido blando (CNR). Resultado: ventana sola CNR 6,6; gamma 1,5
  CNR 8,1; CLAHE 3,2. CLAHE triplicaba el ruido del tejido blando (0,04 → 0,14) y
  perdía en los 5 casos.
- `comparar_gamma.py` fijó 1,5 por criterio de brillo del hueso (rango objetivo 0,22–0,35),
  no por CNR máximo (gamma 2–3 oscurecía el hueso trabecular).
- Pruebas: `tests/test_pengwin_io.py` (9 verificaciones: rango, monotonía, gamma 1 = ventana
  pura, independencia del Visualizador 1).

### Revisión visual de los 100 casos

`notebooks/03_semana9_deteccion.ipynb` (sección 1.1) muestra, para cada caso, el corte con
más hueso tal como entra al modelo con el contorno de la máscara; figura
`outputs/figures/revision_preprocesamiento_100_casos.png`. En los 100 casos el hueso se ve
claro y con textura interna, el tejido blando uniforme y sin granulado, y el contorno sigue
el hueso.

| Medida | Resultado |
|---|---:|
| Brillo medio del hueso (gamma 1,5) | 0,296 |
| Mínimo / máximo | 0,188 (034) / 0,398 (076) |
| Casos atípicos (\|z\| > 2) | 013, 026, 027, 034, 095 (oscuros, 0,19–0,21) y 076 (claro, 0,40) |

Los cinco casos oscuros coinciden con hallazgos previos (026 y 034 hueso de baja densidad,
027 beam hardening por metal, 095 entre los peores casos de la Semana 10). Gamma es una
transformación global y no corrige el histograma corrido de ese grupo; queda documentado
como limitación. Una normalización por caso se evaluó en la Semana 8 (mejora ~17 %) y se
descartó para conservar el significado físico del HU.

## Cómo ejecutar el overfit

Después de instalar las dependencias y descargar `data/raw/`:

```bash
python src/train_overfit_detection.py --case-id 001 --epochs 150
python src/visualize_overfit_detection.py --case-id 001
```

El script selecciona un batch pequeño de dos cortes con máscara, redimensiona imagen y
máscara a `256x256`, construye una caja por región visible, entrena sobre el mismo batch y
guarda `outputs/week9_overfit_detection.pth`. La prueba es de correctitud del pipeline, no
una métrica final. Se considera aprobada si la pérdida final es menor que el 25 % de la
inicial y las predicciones decodifican cajas con NMS.

### Ejecución verificada

Caso `001`, dos cortes con etiqueta visible, `256x256`, 150 épocas:

| Medida | Resultado |
|---|---:|
| Pérdida inicial | 1,687 |
| Pérdida final | 0,002 |
| Reducción | 758x |
| Criterio mínimo | 4x |
| Resultado | Aprobado |

El overlay (`outputs/figures/week9_detection_overfit.png`) confirma que las cajas predichas
tras NMS se superponen a las de referencia (confianza 0,996). Esto prueba que la cadena
máscara → target de grid → backbone/CBAM → cabeza → pérdida → NMS funciona.

## Primer entrenamiento multi paciente

Cache reproducible con cuatro cortes distribuidos axialmente por caso: 24 casos de train
(96 cortes) y 15 de validación (60 cortes, 164 cajas visibles). El cache es local y no se
versiona.

```bash
python src/prepare_week9_detection_data.py
python src/train_week9_detection.py --output outputs/week9_det_cbam32.pth
python src/train_week9_detection.py --no-cbam --output outputs/week9_det_nocbam32.pth
python src/visualize_week9_validation.py --case-id 014 --all-predictions
```

Dos cambios respecto a la primera versión de esta semana, ambos con evidencia:

- **Selección del checkpoint por F1 (IoU ≥ 0,5), no por IoU medio.** El IoU medio solo
  mira las cajas acertadas y premiaba un modelo que producía 2 112 cajas para 164 reales.
  F1 castiga las cajas sobrantes.
- **32 canales base en lugar de 8.** Con 8 canales el modelo sobreajustaba los 96 cortes
  (F1 0,31); con 32, F1 0,52.

El historial por época se guarda en `outputs/week9_det_cbam32.json` y
`outputs/week9_det_nocbam32.json`; curvas en `outputs/figures/week9_curvas_cbam_vs_nocbam.png`.

### Resultados en validación (15 casos, 60 cortes, 164 cajas)

| Modelo | Mejor época | F1 | Precisión | Recall | IoU medio | Falsos positivos | Cajas predichas |
|---|---:|---:|---:|---:|---:|---:|---:|
| Con CBAM (modelo) | 34 | **0,523** | **0,431** | 0,665 | 0,504 | **144** | 253 |
| Sin CBAM (ablación) | 28 | 0,517 | 0,407 | 0,707 | 0,536 | 169 | 285 |

### Calibración del umbral de confianza (validación)

Las cajas sobrantes tienen confianza 0,2–0,5, frente a 0,6–1,0 de la caja principal de cada
hueso. El umbral de decodificación estaba en 0,20; se barrió sobre validación con los dos
modelos ya entrenados (sin reentrenar, sin tocar test):

| Modelo | Umbral | F1 | Precisión | Recall | FP | Cajas predichas |
|---|---:|---:|---:|---:|---:|---:|
| Con CBAM | 0,20 | 0,523 | 0,431 | 0,665 | 144 | 253 |
| Con CBAM | 0,30 | 0,568 | 0,510 | 0,640 | 101 | 206 |
| **Con CBAM** | **0,40** | **0,572** | 0,565 | 0,579 | **73** | 168 |
| Con CBAM | 0,50 | 0,559 | 0,592 | 0,530 | 60 | 147 |
| Sin CBAM | 0,20 | 0,517 | 0,407 | 0,707 | 169 | 285 |
| Sin CBAM | 0,40 | 0,578 | 0,549 | 0,610 | 82 | 182 |
| Sin CBAM | 0,50 | 0,571 | 0,611 | 0,537 | 56 | 144 |

Se fija **0,40**: F1 sube de 0,52 a 0,57 y los falsos positivos caen a la mitad perdiendo 9
puntos de recall; a partir de 0,50 se pierden cajas reales.

Lectura de la ablación: con el umbral calibrado, el modelo sin CBAM queda igual o apenas
mejor (0,578 vs 0,572). Con 15 casos de validación la diferencia no es concluyente en ningún
sentido. CBAM se conserva porque el enunciado lo exige en la arquitectura y la ablación
definitiva se repite con el pipeline completo de la Semana 10, donde hay más señal.

Estas cifras son del detector aislado con 24 casos de entrenamiento y no se comparan con los
objetivos del enunciado: mAP@0,5 y mAP@0,5:0,95 se calculan en la Semana 10 sobre el pipeline
completo (`pengwin_metrics.py`), y el F1 ≥ 0,85 del enunciado es el de clasificación de
presencia, no el de detección de cajas.

### Primeras bounding boxes

`outputs/figures/week9_val_0XX.png` muestra, para los 15 casos de validación, la caja de mayor
confianza por región (una hipótesis legible por hueso). `outputs/figures/week9_val_all_0XX.png`
muestra **todas** las propuestas que sobreviven al NMS, que es lo que la evaluación cuenta: en
promedio ~1,5 cajas por hueso (253 predichas para 164 reales). El NMS (IoU 0,5 por clase,
confianza 0,20) elimina las cajas solapadas; las que quedan son cajas desplazadas del mismo
hueso que no se solapan entre sí. Revisión de las 15 figuras: en 14 casos las tres regiones aparecen con la clase correcta y
la caja principal con confianza 0,6–1,0 (024, 027, 060 y 078 casi exactas); en los coxales
largos (014, 044, 078) la caja queda corta o corrida; en 067 el sacro, una astilla delgada
en el borde superior, no se detectó y aparecen dos cajas de coxal de baja confianza sobre
él; en 077 el sacro predicho es mayor que la GT. Es el estado esperado de "primeras cajas
razonables": clase correcta, imprecisión de escala y posición. En la Semana 10 la caja se
refina con la máscara semántica predicha (`refine_boxes_with_semantic`) y el mAP@0,5 sube a
0,87–0,92.

## Entorno reproducible

Python 3.12, PyTorch 2.2+ (CPU o GPU integrada de Apple). En macOS x86_64, PyTorch 2.2.2
requiere `numpy<2`. El notebook `notebooks/03_semana9_deteccion.ipynb` ejecuta todo lo
anterior en orden y conserva las salidas; los entrenamientos tardan ~4 min cada uno en un
MacBook Air con chip Apple.

## Auditoria de cierre

```bash
python -m pytest tests -q
```

| Verificación | Resultado |
|---|---|
| Backbone produce grid `16x16` para entrada `256x256` | Aprobada |
| CBAM está dentro del backbone, antes de la cabeza; `use_cbam=False` lo retira | Aprobada |
| Máscara → target → caja conserva coordenadas | Aprobada |
| NMS elimina una caja solapada de menor confianza | Aprobada |
| La pérdida es finita y propaga gradientes | Aprobada |
| Separación por paciente entre train y validación | 24 vs 15, intersección vacía |
| Targets del cache | 429 regiones visibles, 429 celdas positivas |
| `window_hu`: rango, monotonía, gamma 1 = ventana, V1 independiente (9 pruebas) | Aprobadas |
| Overfit intencional | Aprobado, reducción 758x |
| Entrenamiento multi-paciente con selección por F1 | F1 0,523 con CBAM, 0,517 sin CBAM |
| Primera predicción en casos no vistos | 14/15 casos con las tres regiones; en 067 el sacro (astilla delgada) no se detectó |
| Calibración del umbral de confianza | 0,20 → 0,40: F1 0,52 → 0,57, FP 144 → 73 |

**Conclusión:** el entregable de Semana 9 está completo. El F1 de detección de 0,57 (umbral
0,40) no se presenta como desempeño final; es el punto de partida del entrenamiento completo de la
Semana 10. Pendiente que arrastra: los cachés y modelos de la Semana 10 se entrenaron con
CLAHE; con gamma 1,5 como preprocesamiento definitivo hay que regenerarlos y reentrenar
para que entrenamiento e inferencia usen la misma función.

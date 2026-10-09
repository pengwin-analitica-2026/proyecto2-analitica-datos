# Bitácora de decisiones — Semana 10

Registro cronológico de lo que se entrenó, lo que se decidió y por qué, desde la línea base hasta
el modelo final (H6). Cerrada el 6 de octubre de 2026; la sección 14 agrega los pedidos del
profesor del 8 de octubre. Las cifras son de **validación** (15 casos,
5047 cortes, subconjunto `todo`, con TTA espejo), salvo que se diga otra cosa. Las dos evaluaciones
en test están en las secciones 11 y 12.5.

Es el documento de la Semana 10. Lo demás está en un solo sitio cada cosa:

| Qué | Dónde |
|---|---|
| El modelo final, sus resultados y sus limitaciones | `MODEL_CARD.md` |
| Comandos para reproducir | `README.md`, sección 3 |
| Tabla de todas las corridas | `outputs/s10/resumen_experimentos.md` |
| Recorrido ejecutable, con tablas y figuras | `notebooks/04_semana10_pipeline_multitarea.ipynb` |
| Uso de IA | `IA_USAGE.md` |
| Por qué cada número del modelo | `docs/hiperparametros.md` |
| Cuantización | `docs/cuantizacion.md` |

### Qué archivo hace cada pieza

| Pieza | Cómo quedó | Archivo |
|---|---|---|
| Backbone | Convolucional propio, 32 canales base, CBAM en una etapa antes de bifurcar. 2.74 M de parámetros en total | `src/week9_detection.py`, `src/week10_multitask.py` |
| Clasificación | Qué regiones hay en el corte (sacro, coxal izquierdo, coxal derecho) | `ClassificationHead` |
| Detección | Grid propio con una celda por región y NMS propio. Se reporta la caja cruda y la refinada con la semántica | `GridDetectionHead`, `src/pengwin_metrics.py` |
| Segmentación semántica | Fondo y las tres regiones | `SegmentationHead` |
| Segmentación de instancia | Por región, fragmento principal o secundario; los fragmentos salen por componentes conexas (`min_size` 12) | `src/instance_postprocess.py` |
| Pérdida | Compuesta, con λ por término y peso ×3 a los secundarios | `src/train_s10.py` |
| Orientación canónica | Endereza los 34 volúmenes guardados girados 180° | `src/orientacion.py` |
| Volumen y distancia | Post-proceso 3D por región y distancia mínima borde a borde en mm | `src/volume_postprocess.py`, `src/fragment_distances.py`, `src/infer_volume.py` |
| Evaluación | Las nueve métricas sobre todos los cortes del caso, con subconjuntos y tabla por caso | `src/evaluate_volumes.py` |
| Análisis de error y tablas | Matriz de confusión, peores casos y tablas del notebook 04 | `src/informe_test.py`, `src/resumen_s10.py` |

## 1. Restricciones del proyecto (no negociables)

- Sistema propio: detección con cabeza de grilla y NMS propios, segmentación de fragmentos por
  región (sacro, coxal izquierdo, coxal derecho), clasificación y medición en mm.
- Sin YOLO, Detectron2 ni Mask R-CNN. CBAM como atención. Pérdida multitarea con λ calibrados.
- Transfer learning solo en el backbone. SAM únicamente como línea base zero-shot.
- Test se evalúa **una sola vez**, con la configuración ya congelada en validación.
- Rama base: `unificado_semana09`. Los commits no llevan firma de Claude.

## 2. Reglas de trabajo que nos dimos

| Regla | Motivo |
|---|---|
| Una sola diferencia por experimento frente a la receta base (mismos datos, épocas, semilla y hardware) | Poder atribuir el efecto a una causa |
| Una mejora solo se acepta si supera el ruido medido con 3 semillas (instancias ±0.004, secundarios ±0.012) | Evitar quedarnos con variaciones de azar |
| Primero cada mejora por separado; después se combinan solo las que ganaron | Las mejoras que no pasan el ruido solas casi nunca lo pasan sumadas |
| Ablaciones obligatorias (CBAM, transfer learning, λ) sobre el candidato final | Requisito del curso |
| Una época = 2500 cortes aleatorios; unas 60 épocas (no 120) | Entrenamientos de ~65–70 min en Kaggle T4; suficiente para converger |
| Kaggle (2 GPU T4 a la vez) y Colab (L4, dos entrenamientos concurrentes) en paralelo | Aprovechar todo el cómputo disponible |

## 3. Línea de tiempo de modelos entrenados

### 3.1 Punto de partida

- `linea_base`: referencia inicial evaluada con el evaluador por volumen completo
  (instancias 0.834, secundarios 0.702, mAP50 0.874). Todo lo demás se compara contra C2.
- Se rehízo la evaluación sobre volumen completo con TTA espejo y se separaron las cajas
  crudas de las refinadas con la semántica.

### 3.2 Familia B (8 canales): atención y pesos

| Corrida | Cambio | Resultado | Decisión |
|---|---|---|---|
| B0 sin CBAM / CBAM 1 / CBAM 3 etapas con γ | Tipo de atención | CBAM 1 etapa la mejor en detección (mAP50 0.926 vs 0.900 sin CBAM) | Quedarse con CBAM de 1 etapa; 3 etapas no ayuda |
| B1 (3 semillas, secundarios ×3, 80 ép.) | Peso de secundarios | Instancias 0.826–0.835, secundarios 0.631–0.661 | Peso 3 |
| B2 secundarios ×6 | Más peso | 0.832 / 0.681, detección peor | Descartado |
| frac2 | Peso por fractura | 0.832 | Descartado |
| IDs | Instancia por IDs en vez de mapa primario/secundario de 7 clases | 0.787 / 0.460 | Descartado: se usa el mapa de 7 clases |

### 3.3 Familia C: ancho del backbone

| Corrida | Resultado | Decisión |
|---|---|---|
| C1 16 canales, 80 ép. | Instancias 0.839, mAP50 0.955 | Mejor que 8 canales |
| **C2 32 canales, 60 ép.** (3 semillas) | Instancias 0.849 / 0.846 / 0.854; secundarios 0.687 / 0.676 / 0.699 | **Receta base** |
| W48 48 canales | Instancias 0.844, secundarios 0.659 | No mejora: el ancho se satura en 32 |

### 3.4 Familia D: una mejora sobre C2 (semilla 42)

| Corrida | Cambio | Inst. | Sec. | Decisión |
|---|---|---|---|---|
| D1 | Contexto 2.5D (3 cortes) | 0.848 | 0.681 | Descartado |
| D2 | Pérdida focal γ=2 | 0.846 | 0.670 | Descartado |
| D3 | Tversky β=0.7 | 0.846 | 0.685 | Descartado |
| D4 | Deep supervision 0.4 | 0.846 | 0.667 | Descartado (solo mejora algo la detección) |
| **D5** | **EMA 0.999** | **0.851** | **0.703** | **Aceptado tras confirmar con 3 semillas** |
| D6 | Sobremuestreo de secundarios ×3 | 0.837 | 0.642 | Descartado: empeora |
| λ×2 | Instancia/borde ×2, detección ×0.5 | 0.849 | 0.676 | Descartado: la detección cruda baja a 0.832 |

**Por qué EMA sí.** Con las semillas 42, 43 y 44 el Dice de secundarios sube +0.016, +0.048 y
+0.009 frente a C2 (media +0.024, por encima del ruido de ±0.012) y el de instancias no empeora
(+0.002, +0.009, 0.000).

**Por qué las otras no.** Ninguna se separa de C2 más allá del ruido. Leído en conjunto, el
techo no parece estar en la pérdida ni en el tamaño de la red sino en el post-proceso y en la
ambigüedad de las etiquetas de fragmentos pequeños.

### 3.5 Familia E: transfer learning (Intel)

- Se preentrenó el backbone propio (`FundidoraBackbone`, 32 canales, CBAM 1) en clasificación de
  Intel Image Classification, 8 épocas, y se cargó con `--backbone-init` (51/51 tensores).
  Solo el backbone se transfiere; las cabezas se entrenan desde cero.
- E1 (Intel sin EMA): instancias 0.847, secundarios 0.683; la detección cruda sube
  (mAP50 0.905 vs 0.875).
- E2 (Intel + EMA): con semilla 42 secundarios 0.714 (+0.011 vs EMA solo); con semilla 43 0.717
  (−0.008 vs EMA solo). **Sin ganancia consistente.**
- **Decisión:** el candidato no usa preentrenamiento; Intel queda documentado como ablación.

### 3.6 Familia A/F: ablaciones obligatorias

| Ablación | Sobre | Resultado |
|---|---|---|
| Sin CBAM (A1) | C2 | Segmentación igual; F1 clasificación 0.976 vs 0.988; mAP50:95 refinada 0.878 vs 0.889 |
| Sin CBAM (F1) | Candidato EMA | Mismo patrón: F1 0.975 vs 0.988; mAP50:95 refinada 0.872 vs 0.883 |
| λ×2 (F2) | Candidato EMA | Instancias 0.855 (ruido); detección cruda mAP50 0.827 vs 0.868 |
| Transfer learning (E2, E2 s43) | Candidato EMA | Neutral (ver 3.5) |

Conclusión: CBAM aporta en clasificación y detección, no en segmentación; λ por defecto; Intel neutral.

## 4. Candidato actual

**C2 + EMA 0.999**: 32 canales, CBAM de 1 etapa, 60 épocas, peso de secundarios ×3, λ por
defecto, sin preentrenamiento, tamaño 384, aumentos, coseno, batch 16.

Validación (media de 3 semillas): instancias ≈ 0.853, secundarios ≈ 0.712 (C2: 0.850 / 0.687).

Checkpoints con Dice de instancias ≥ 0.85 (objetivo del curso; los demás objetivos ya se cumplían):

| Checkpoint | Inst. Dice | Nota |
|---|---|---|
| `d5_ema_w32_e60_s42.pth` | 0.851 | Candidato |
| `d5_ema_w32_e60_s43.pth` | 0.855 | Candidato |
| `d5_ema_w32_e60_s44.pth` | 0.854 | Candidato |
| `e2_intel_ema_w32_e60_s42.pth` | 0.851 | Ablación Intel |
| `e2s43_intel_ema_w32_e60_s43.pth` | 0.856 | Ablación Intel |
| `f1_ema_cbamnone_w32_e60_s42.pth` | 0.851 | Ablación CBAM |
| `f2_ema_lam_w32_e60_s42.pth` | 0.855 | Ablación λ |
| `c2s44_w32_cbam1_sec3_e60_s44.pth` | 0.854 | C2 sin EMA, solo 1 de 3 semillas |

Cautela: los márgenes sobre 0.85 son de 0.001 a 0.006, del orden del ruido (±0.004), y la
validación ya se usó para decidir. La cifra que vale es la de test.

## 5. Post-proceso 3D y medición en milímetros

Se construyó `volume_postprocess.py` + `infer_volume.py`: componentes conectados 3D por región,
el mayor primario es el fragmento principal, los demás son secundarios; distancia en mm entre
cada secundario y el principal, además de Dice/IoU 3D, ASSD y HD95.

Resultados con el post-proceso original (validación, 15 casos):

| Modelo | Dice 3D principal | Dice 3D secundario | Dice 3D todos |
|---|---|---|---|
| C2 | 0.914 | 0.574 | 0.758 |
| D5 (EMA, s42) | 0.911 | 0.605 | 0.771 |

### Hallazgo: la distancia en mm estaba mal medida

- El error medio de distancia salía 1.153 mm en C2 y en D5, idéntico. Al abrir el detalle:
  **la distancia predicha fue 0.0 mm en los 38 fragmentos que la tenían** (de 41 secundarios), y solo 5 de los 41 están
  separados en el GT (entre 3.2 y 19.0 mm). Un predictor que siempre diga 0 obtiene el mismo
  error; la cifra no demostraba capacidad de medir.
- Causa: `postprocess_3d` asignaba al fragmento principal cualquier vóxel de la región sin
  etiqueta, lo que rellenaba el hueco entre fragmentos y los dejaba pegados.
- Corrección: parámetro `fill_unlabeled` (por defecto `True`, para no alterar los Dice) y bandera
  `--distance-nofill` en `infer_volume.py`, que mide las distancias sobre el volumen sin relleno.
  Test añadido en `tests/test_volume_postprocess.py`.
- Verificación (validación, casos 014, 024, 087 y 099, D5 semilla 42, `--distance-nofill`):
  ahora **16 fragmentos tienen distancia predicha > 0**, con error medio 1.23 mm, mediana 0.0 mm,
  p90 4.39 mm y **correlación 0.917** con la distancia real. ASSD medio 5.84 mm, HD95 medio
  25.75 mm. Dice de instancia secundaria en esos casos 0.783, secundarios omitidos 0.065.
  La medición ya distingue fragmentos separados de pegados, a diferencia de la anterior.
- **Validación completa (15 casos, 41 fragmentos secundarios, D5 s42, `--distance-nofill`):**
  38 de 41 con distancia predicha; **error medio 0.518 mm** (mediana 0.0, p90 0.388),
  **correlación 0.925** con la distancia real, ASSD medio 5.11 mm, HD95 medio 22.81 mm, Dice 3D
  de todos los fragmentos 0.771, secundarios omitidos 4.7 %. Referencia: predecir 0 siempre da
  1.153 mm, así que el error baja a menos de la mitad. Solo 5 de los 41 están realmente
  separados en el GT (3.2 a 19.0 mm); en los 4 casos revisados el modelo acertó 19.0 mm
  (error 0.0), midió 4.5 para 3.2, falló 5.3 (lo dio por pegado) y sobreestimó 12.8 como 22.4.
  Cautela: la mediana 0 refleja que la mayoría de fragmentos están pegados; el desempeño real
  en separados es lo que cuenta y hay pocos (5) para una cifra firme.

### Ensamble de 3 semillas EMA (solo inferencia, validación, TTA)

| Modelo | Inst. Dice | Sec. Dice | F1 clasif. | mAP50 refinada |
|---|---|---|---|---|
| D5 s42 (único) | 0.851 | 0.703 | 0.988 | – |
| Media de 3 semillas | 0.853 | 0.712 | – | – |
| **Ensamble s42+s43+s44** | **0.858** | **0.723** | 0.988 | 0.970 |

El ensamble gana +0.005 en instancias y +0.011 en secundarios sobre la media: es una ganancia
modesta, del orden del ruido. Cuesta 3 pasadas por corte; decisión pendiente según la latencia
de la demo.

## 6. Error corregido en la evaluación

`evaluate_volumes.py` fallaba cuando el último lote de un volumen tenía una sola slice
(el post-proceso devuelve 2D). Se corrigió y afecta a 1 de 5047 cortes en evaluaciones previas
(slice final del caso 067), con efecto despreciable en las métricas.

## 7. Infraestructura y operación

- Lanzamiento en Kaggle con `kaggle/lanzar.py` (código embebido), 2 GPU simultáneas, ~65–70 min
  por corrida; datasets privados `pengwin-s10-datos`, `pengwin-s10-codigo`, `pengwin-s10-colab2`.
- Colab Pro (L4): dos entrenamientos a la vez usan ~64 % de GPU; cada época tarda más pero
  el conjunto termina antes que en serie.
- El token de Kaggle lo pega el usuario; no se guarda en archivos ni notebooks.
- La máquina de Colab dura 24 horas como máximo. El 06-oct se cerró a las 24 h con cuatro
  evaluaciones de validación en curso (distancias y estrés de H6), que se perdieron; los
  `.pth` ya estaban copiados. Lección: subir cada resultado al dataset privado apenas sale.
  La máquina nueva no tiene el token de Kaggle, así que lo pendiente se corre en CPU
  (1 modelo + TTA: 0.5 s por corte con 2 núcleos; evaluar los 15 casos toma cerca de 1 hora).
- Los resultados se guardan en `outputs/s10/` (los `.pth` están en `.gitignore`; para compartirlos
  se usa `git add -f` o un dataset privado de Kaggle).

## 8. Rendimiento por corte (demo en vivo)

Script: `src/benchmark_latency.py` (lote 1, corte de 384x384 del caché, 60 cortes de validación
tras calentamiento, incluye decodificación de detección e instancias). Candidato D5.

| Configuración | GPU L4 (mediana / p95) | CPU 12 hilos (mediana / p95) |
|---|---|---|
| 1 modelo | 64 / 66 ms | 162 / 174 ms |
| 1 modelo + TTA espejo | 72 / 77 ms | 250 / 275 ms |
| Ensamble 3 EMA + TTA | 87 / 89 ms | 518 / 629 ms |

Lectura: con GPU la demo corre a ~11-15 cortes por segundo incluso con el ensamble; en CPU
de 12 hilos un modelo con TTA da ~4 cortes/s y el ensamble ~2 cortes/s. El ~2 s por corte de
antes era de la evaluación de volumen, no del modelo.

**Decisión:** para la demo, 1 modelo + TTA como configuración por defecto (la ganancia del
ensamble, +0.005 en instancias, no justifica 2x en CPU); el ensamble queda como opción en GPU.
Pendiente: medir en el Mac del equipo y con un CPU de 2-4 hilos, que es lo que tendrá un
despliegue gratuito; los 12 hilos de Colab no representan ese caso.

## 9. Línea base SAM zero-shot (objetivo 10G)

Script: `src/sam_baseline.py`. SAM ViT-B (pesos oficiales `sam_vit_b_01ec64.pth`, 375 MB, descarga
autorizada por Gus) se usa **solo como referencia**, nunca dentro del sistema ni para crear
etiquetas. Recibe como prompt la caja que predice nuestro detector y devuelve una máscara
binaria; no clasifica fragmentos. Validación, 1 de cada 3 cortes con hueso (1308 cortes, 3393
pares corte-región). Se compara contra la máscara semántica de nuestra red D5 s42 dentro de la
misma caja. Si no hay caja para una región presente, ambos valen 0.

| Región | SAM Dice / IoU | Nuestro Dice / IoU |
|---|---|---|
| Sacro | 0.770 / 0.665 | 0.852 / 0.777 |
| Coxal izquierdo | 0.864 / 0.795 | 0.908 / 0.855 |
| Coxal derecho | 0.883 / 0.815 | 0.919 / 0.867 |
| **Global** | **0.847 / 0.769** | **0.898 / 0.839** |

SAM tarda 259 ms por corte en la L4 (solo la parte de SAM), frente a ~64 ms de todo nuestro
pipeline.

Lectura: el sistema propio supera a SAM zero-shot en las tres regiones (+0.05 Dice global) y
es unas 4 veces más rápido; la mayor brecha es el sacro (+0.08), donde las cajas grandes y el
hueso esponjoso confunden a SAM. Matices que hay que decir en el informe: (1) SAM recibe
nuestras cajas, por lo que la comparación favorece a SAM en localización; (2) la métrica es
sobre la región completa, porque SAM no segmenta ni clasifica fragmentos, que es justamente lo
que aporta nuestro sistema (Dice de instancias 0.85); (3) es validación de una sola semilla,
sin ruido medido para SAM.

### SAM sobre test, con el modelo final (06-oct)

El enunciado pide la comparación sobre test. Se repitió con H6 s42 (las cajas del prompt y la
máscara propia salen de H6; volúmenes en orientación canónica), 1 de cada 3 cortes con hueso:
1 181 cortes y 3 002 pares corte-región en test. Se corrió además sobre validación con H6 para
tener las dos particiones con el mismo modelo. Archivos: `outputs/s10/sam_baseline_test_h6.json`
y `sam_baseline_val_h6.json`.

| Región (test) | SAM Dice / IoU | Nuestro Dice / IoU |
|---|---|---|
| Sacro | 0.792 / 0.690 | 0.884 / 0.820 |
| Coxal izquierdo | 0.915 / 0.859 | 0.947 / 0.905 |
| Coxal derecho | 0.930 / 0.879 | 0.957 / 0.921 |
| **Global test** | **0.889 / 0.823** | **0.934 / 0.889** |
| Global validación (H6) | 0.859 / 0.781 | 0.914 / 0.859 |

Mismo patrón que con D5: el sistema propio supera a SAM en las tres regiones (+0.046 en Dice
global en test) y la mayor brecha está en el sacro (+0.092). Con las cajas de H6, SAM también
sube frente a la medición con D5 (0.847 → 0.859 en validación): mejores cajas, mejor prompt. El
tiempo de SAM de estos dos archivos (unos 500 ms por corte) no vale como latencia: las dos
corridas compartieron la GPU; la cifra medida a solas es 259 ms.

## 10. Congelación de la configuración y evaluación en test

Decisión tomada **antes de ver cualquier cifra de test** (06-oct-2026):

- Modelo: `d5_ema_w32_e60_s42.pth` (receta C2 + EMA 0.999, semilla 42; es la semilla de
  referencia con la que se hicieron todas las comparaciones de una sola mejora).
- Inferencia: 1 modelo con TTA espejo, tamaño 384; sin ensamble (ganancia +0.005 no
  justifica el costo en CPU, ver sección 8) y sin ajuste de umbrales sobre test.
- Distancias: post-proceso 3D con `--distance-nofill`, `min_voxels` por defecto.
- Se corre `evaluate_volumes.py` e `infer_volume.py --surface-metrics` **una sola vez** sobre
  el split `test`. No se repite ni se ajusta nada después de ver el resultado; si sale peor que
  validación, se reporta tal cual.
- Los objetivos del curso son los de `OBJETIVOS` en `evaluate_volumes.py`.

## 11. Resultado en test (única evaluación, configuración de la sección 10)

Modelo D5 s42 + TTA espejo, 15 casos de test, 4664 cortes (subconjunto `todo`). Se evaluó una
sola vez; no se ajustó nada después.

| Métrica | Test | Objetivo | Validación (D5 s42) |
|---|---|---|---|
| F1 clasificación (macro) | 0.984 | 0.85 | 0.988 |
| AUC clasificación (macro) | 0.998 | 0.85 | – |
| IoU medio detección (refinada) | 0.898 | 0.65 | – |
| mAP50 detección (refinada) | 0.920 | 0.65 | 0.959 |
| mAP50:95 detección (refinada) | 0.853 | 0.40 | 0.883 |
| Dice segmentación semántica (macro) | 0.949 | 0.85 | – |
| IoU segmentación semántica (macro) | 0.904 | 0.70 | – |
| **Dice de instancias (emparejadas)** | **0.845** | **0.85** | **0.851** |
| IoU de instancias | 0.781 | 0.70 | – |
| Dice instancia primaria / secundaria | 0.894 / 0.692 | – | 0.900 / 0.703 |
| Detección cruda mAP50 / mAP50:95 | 0.866 / 0.511 | – | 0.868 / 0.477 |

**Lectura honesta.** Todos los objetivos del curso se cumplen salvo el Dice de instancias, que
queda en 0.845 frente a 0.85 (0.005 por debajo, del orden del ruido medido de ±0.004 entre
semillas). En validación el mismo modelo daba 0.851: la caída de 0.006 es compatible con la
variación entre conjuntos y con que el modelo y la configuración se eligieron mirando
validación. El resultado se reporta tal cual; no se vuelve a evaluar ni se afina sobre test.
Los casos con corte en los extremos del volumen bajan el promedio de instancias (0.823) y el
Dice de secundarios (0.692 en conjunto) sigue siendo el punto débil.

### Distancias en mm sobre test (medición sin relleno, 15 casos, una sola pasada)

| Métrica | Test | Validación |
|---|---|---|
| Dice 3D principal / secundario / todos | 0.909 / 0.682 / 0.800 | – / – / 0.771 |
| Fragmentos secundarios con distancia predicha | 36 de 41 | 38 de 41 |
| Error medio de distancia | 2.956 mm | 0.518 mm |
| Error mediana / p90 | 0.000 / 3.453 mm | 0.000 / 0.388 mm |
| Correlación con la distancia real | 0.543 | 0.925 |
| ASSD medio / HD95 medio | 3.02 / 16.33 mm | 5.11 / 22.81 mm |

**Lectura honesta.** La medición en test es claramente peor que en validación (error medio 2.96
mm frente a 0.52 mm; correlación 0.54 frente a 0.93). Mirando los fragmentos con separación real o
predicha (solo diagnóstico, sin cambiar nada):

- Un fragmento secundario falso positivo se predijo a 68.5 mm cuando la distancia real es 0 (un
  componente 3D espurio lejos del hueso): por sí solo explica casi 2 mm del error medio.
- Un fragmento con distancia real de 9.4 mm se predijo a 29.3 mm (error 19.9 mm).
- Dos fragmentos realmente separados (33.7 y 12.3 mm) no recibieron distancia predicha.
- Los demás errores son de 1.6 a 5.0 mm, y el que mejor sale tiene error de 3.6 mm sobre 40.0 mm.

Es decir, hay pocos fragmentos realmente separados (alrededor de 4 a 6 en test, no se contaron todos), y un par de
errores grandes dominan el promedio. La mediana de 0 mm refleja que la mayoría de fragmentos
están pegados al principal. La cifra que se debe citar en el informe es la de test, con esta
salvedad; no se reajusta nada sobre test. Mejora candidata para la semana 11, decidida con
validación y sin tocar test: eliminar componentes 3D secundarios pequeños y lejanos de la
anatomía (filtro por vóxeles mínimos y distancia máxima a la región).

### Evaluación en test de los demás checkpoints que pasaron el umbral (decisión de Gus, 06-oct)

Gus pidió dar test también a los demás checkpoints con Dice de instancias >= 0.85 en validación
(sección 4). Reglas fijadas **antes** de correrlos:

- Se evalúan con la misma configuración (1 modelo + TTA espejo, tamaño 384, `evaluate_volumes.py`),
  sin tocar umbrales; cada uno una sola vez.
- Sirven para **medir la variación de test entre semillas y ablaciones**, no para elegir un modelo
  nuevo. El modelo principal sigue siendo D5 s42 aunque otro salga mejor en test; elegir el mejor
  en test sería seleccionar sobre el conjunto que debe quedar intacto.
- La media y la desviación de las tres semillas EMA (D5 s42, s43, s44) en test es la cifra más
  honesta para el informe, junto a la del modelo congelado.
- Los checkpoints que no estén en Colab (F1 y C2 s44) se llevan allí vía el dataset de código.

Resultado (test, Dice de instancias): D5 s42 0.845, D5 s43 0.841, D5 s44 0.842, E2 s42 0.848,
E2 s43 0.848, F2 0.847. Ninguno llega a 0.85 y todos quedan a menos de 0.007 entre sí: el
0.845 no era mala suerte de una semilla. De E2 s43 y F2 quedó el JSON (`TEST_e2_s43.json`,
`TEST_f2_s42.json`); los de D5 s43, D5 s44 y E2 s42 se leyeron en pantalla y el archivo se
perdió al cerrarse la máquina de Colab. F1 y C2 s44 no se llegaron a correr.

## 12. Diagnóstico después de test y plan para subir el Dice de instancias

Gus pidió sentarnos a entender por qué test no llega y qué hacer para cumplirlo de verdad.
Todo lo de esta sección es **posterior a haber visto test**; se dice así en el informe.

### 12.1 Qué se encontró

**a) En test fallan dos pacientes, no el modelo en general.** Dice de instancias por caso (D5 s42):
13 de 15 casos de test están entre 0.825 y 0.926; los casos **012 (0.692)** y **039 (0.710)**
arrastran el promedio. Sin ellos el promedio sería 0.869. En validación el peor caso es 0.768
y la semántica nunca baja de 0.95; en esos dos casos de test baja a 0.784 y 0.869.

**b) Por qué fallan (análisis de fallos, `src/diagnostico_casos.py`, solo diagnóstico).**

| Caso | Hueso tomado por otra región | Hueso predicho como fondo | Qué tiene de raro |
|---|---|---|---|
| 012 (test) | 13.7 % | 10.4 % | Paciente **girado** en el escáner (pelvis rotada decenas de grados), hueso poco denso (HU máximo 1437, de los más bajos) y pelvis pequeña en la imagen (1.4 % de píxeles de hueso frente a 3.5-4 %) |
| 039 (test) | 8.2 % | 5.7 % | Volumen de 154x322 con espaciado 1.22 mm (el más grueso de los 100): el caché lo amplía 2.5 veces en vertical y 1.2 en horizontal, queda borroso y deformado |
| 087, 078 (peores de validación) | 0.0 % | 0.9-1.4 % | Nada geométrico: la semántica es 0.94-0.98; el error es dónde se parte el hueso en fragmentos |

En 012 el 30 % del coxal izquierdo se predice como derecho. La red decide izquierda/derecha
por la posición en la imagen, y eso se rompe cuando el paciente está girado o descentrado.

**c) La red memoriza.** Historial de D5: la pérdida de entrenamiento baja de 0.86 (época 10) a
0.48 (época 60) mientras el Dice de instancias en validación se queda en 0.849-0.851 desde la
época 30. El aumento de datos actual es solo espejo sagital y un cambio de brillo de +-5 %:
**no hay escala, rotación, traslación ni cambio de proporción**. 36 de los 100 volúmenes no son
512x512 y el caché estira cada uno distinto. Ninguno de los experimentos D1-D6 tocó el aumento.

**d) La métrica.** `instance_dice_matched` promedia un valor por fragmento real y por corte
(0 si no hay pareja) sin pesar por área; los IDs no importan, solo la partición. Los secundarios
son el 25 % de los fragmentos (val 3281 de 13073) con Dice 0.70 y 9 % sin pareja.

**e) Un tercio de los volúmenes entra girado 180 grados.** Al leer la cabecera de los 100 `.mha`:
66 traen matriz de dirección identidad (orientación RAI) y **34 traen `-1 0 0 0 -1 0 0 0 1`
(LPI)**. `SimpleITK.GetArrayFromImage` no aplica esa matriz, así que esos 34 casos (22 de
entrenamiento, 10 de los 15 de validación, 2 de test) llegan al modelo con el sacro arriba y el
coxal izquierdo a la izquierda de la imagen, al revés que el resto. La red tiene que adivinar la
orientación de cada corte para decidir izquierda y derecha; en cortes donde no se ve el sacro
(acetábulo del caso 039) se equivoca. En validación los dos grupos rinden igual (LPI 0.851,
RAI 0.854), así que el modelo actual sí aprendió las dos orientaciones, pero a costa de esa
ambigüedad. Corrección: `src/orientacion.py` y la bandera `--canonical`, que dejan todos los
casos en la misma orientación (un giro de 180 grados no cambia la lateralidad, las etiquetas
no se tocan). El caso 012 es RAI con matriz identidad: ahí el paciente sí estaba acostado de
lado y no hay metadato que corregir.

### 12.2 Hipótesis y experimentos (decididos con validación, no con test)

| Id | Cambio único frente a C2 + EMA | Dónde | Por qué |
|---|---|---|---|
| H1 | `--aug-geo 1.0`: escala 0.80-1.25, proporción 0.85-1.18, rotación +-12 grados, traslación +-8 % | Kaggle | Ataca la memorización (c) |
| H2 | `--mode pad`: conserva la proporción real en vez de estirar | Kaggle | Ataca la deformación de (b) |
| H3 | Paquete de robustez: H1 + rotación libre en 25 % de las muestras + baja resolución simulada 25 % + gamma | Colab | Ataca (b): pacientes girados y volúmenes borrosos |
| H4 | `--canonical`: los 34 casos LPI se giran 180 grados para que todos queden igual | Colab | Ataca (e): quita la ambigüedad izquierda/derecha |
| 512 | Resolución 512 (caché ya construido en Kaggle) | Kaggle | Las líneas de fractura miden 1-2 px a 512 y se pierden a 384 |

**Cómo se valida sin test.** Validación no tiene pacientes girados, así que no puede medir (b).
Se añadió una **prueba de estrés sobre validación** (`evaluate_volumes.py --stress`): los mismos
15 casos de validación girados 60 y 180 grados, deformados y borrosos (`aspect=1.6,lowres=2.5`),
desplazados (`shift=0.15`) y reducidos (`scale=0.7`). Un candidato se acepta si (1) no baja en
validación normal más allá del ruido y (2) mejora claramente en estrés.

**Primer resultado de estrés (confirma la hipótesis sin usar test).** D5 s42 sobre validación
girada 60 grados: Dice de instancias **0.127** (normal 0.851), semántica 0.193, mAP50 0.029.
El modelo actual no tolera la rotación en absoluto.

**Diagnóstico de instancias en validación** (`src/analisis_instancias.py`, D5 s42, 13073
fragmentos; la variante base reproduce la cifra oficial, 0.8508).

*Post-proceso sin reentrenar: no hay nada que ganar.*

| Variante | Dice todos | Primario | Secundario | Fragmentos predichos sobrantes |
|---|---|---|---|---|
| Base (tamaño mínimo 12 px) | 0.851 | 0.900 | 0.703 | 7096 |
| Tamaño mínimo 1 px | 0.853 | 0.901 | 0.710 | 12808 |
| Tamaño mínimo 30 px | 0.845 | 0.899 | 0.684 | 5375 |
| Rellenar huecos | 0.849 | 0.898 | 0.703 | 7096 |
| Absorber fragmentos < 20 px | 0.849 | 0.900 | 0.694 | 6156 |
| Post-proceso 3D | 0.835 | 0.904 | 0.631 | 4756 |
| Corte por la cabeza de borde (0.3 / 0.5 / 0.7) | 0.851 | 0.900 | 0.703 | 7096 |

Bajar el tamaño mínimo a 1 px sube 0.002 pero casi duplica los fragmentos sobrantes, que la
métrica del curso no penaliza: no se adopta, sería ajustar a un punto ciego de la métrica.

*Oráculos: dónde está el techo.*

| Si fuera perfecto... | Dice todos | Primario | Secundario |
|---|---|---|---|
| ...la partición en fragmentos (con nuestra máscara de hueso) | **0.952** | 0.960 | 0.929 |
| ...la máscara de hueso (con nuestra partición) | 0.885 | 0.932 | 0.746 |
| ...el mapa primario/secundario (con nuestra máscara) | 0.928 | 0.934 | 0.909 |
| Techo de la formulación (mapa real + componentes 2D / 3D) | 0.966 / 0.982 | – | – |
| Sin separar fragmentos (uno por región) | 0.699 | 0.902 | 0.093 |

El límite es **la partición** (dónde termina un fragmento y empieza otro), no la máscara de
hueso ni la formulación: con la partición real se llegaría a 0.95.

*Dónde se pierde.* La mitad de la pérdida viene de los secundarios (25 % de los fragmentos,
Dice 0.703, 9 % sin pareja) y la otra mitad de los primarios (0.900). Por tamaño: los
fragmentos de menos de 25 px dan 0.38 pero solo son el 1.6 %; el grueso de la pérdida (41 %)
está en fragmentos de 400 a 1599 px (Dice 0.837). Por región: sacro 0.800, coxal izquierdo
0.851, coxal derecho 0.887. Cuando la región tiene un solo fragmento real el Dice es 0.914;
con dos baja a 0.772. El último 15 % del volumen da 0.806.

### 12.2b Resultados de los experimentos (validación, semilla 42)

| Experimento | Inst. Dice | Primario | Secundario | Sec. sin pareja | mAP50 cruda | Decisión |
|---|---|---|---|---|---|---|
| D5 s42 (referencia) | 0.851 | 0.900 | 0.703 | 9.0 % | 0.868 | – |
| **H1 aumento geométrico** | **0.864** | 0.905 | **0.743** | 7.5 % | **0.915** | Pasa el ruido con holgura (+0.013 inst., +0.040 sec.; ruido 0.004 / 0.012). Se confirma con semilla 43 |
| H2 conservar proporción (`pad`) | 0.857 | 0.902 | 0.723 | 10.4 % | 0.884 | +0.006: en el borde del ruido; se evalúa sobre el caché `pad`, no es comparable al píxel |
| H1 con semilla 43 | 0.865 | 0.903 | 0.750 | 6.5 % | 0.880 | **Confirma H1** (D5 s43: 0.855 / 0.724) |
| H3 paquete de robustez | 0.861 | – | 0.741 | – | 0.837 | Parecido a H1 en validación normal; la detección cruda baja (rotación libre). Su valor se mide en estrés |
| H4 orientación canónica (sin aumento) | 0.861 | – | 0.742 | – | 0.873 | Sola ya aporta +0.010 inst. y +0.039 sec.; con el aumento se suman (H5) |
| **H5 aumento geométrico + orientación canónica** | **0.873** | **0.908** | **0.769** | **6.2 %** | **0.922** | **Mejor resultado. Confirmado con semilla 43** |
| H5 con semilla 43 | 0.869 | 0.904 | 0.762 | 6.4 % | 0.893 | **Confirma H5**: las dos semillas (0.873 / 0.869) quedan por encima de las dos de H1 (0.864 / 0.865) y de las tres de D5 (0.851-0.855) |
| H6 = H5 con rotación de +-35 grados | 0.869 | 0.905 | 0.764 | 6.9 % | 0.887 | Igual que H5 dentro del ruido (−0.004 frente a H5 s42, +0.001 frente a H5 s43). Se decide con la prueba de estrés |
| H6 con semilla 43 | 0.868 | 0.904 | 0.762 | 6.6 % | 0.877 | Confirma H6 |
| H6 con semilla 44 | 0.872 | 0.904 | 0.778 | 5.6 % | 0.902 | **Confirma H6**: las tres semillas dan 0.868-0.872 (media 0.870, desviación 0.002) |

H5 frente a D5: +0.022 en instancias y +0.066 en secundarios (5 veces el ruido). Frente a H1:
+0.009 y +0.026; mejora en 11 de los 15 casos, más en los LPI (+0.010 de media) que en los RAI
(+0.006). También sube todo lo demás: F1 0.992, mAP50 0.976, mAP50:95 0.912. En el centro del
volumen da 0.881 y en los extremos 0.848. Su curva se aplana hacia la época 50 (0.874 → 0.873).

**Prueba de estrés: validación girada 30 grados** (el giro que más se parece al caso 012).

| Modelo | Rotación vista al entrenar | Val. normal | Girada 30 grados | Girada 60 grados |
|---|---|---|---|---|
| D5 (actual) | ninguna | 0.851 | 0.559 | 0.127 |
| H1 | +-12 grados | 0.864 | 0.771 | 0.212 |
| H5 (H1 + canónica) | +-12 grados | 0.873 | 0.807 | 0.218 |
| H3 | +-12 y libre en 25 % | 0.861 | 0.834 | **0.824** |
| **H6 (H5 con +-35 grados)** | +-35 grados | 0.869 | **0.854** | 0.734 |

La tolerancia al giro depende directamente del rango de rotación del aumento: sin rotación el
modelo pierde 0.29 con un giro de 30 grados; con +-12 pierde 0.07-0.09; con rotación libre,
0.03, y aguanta también 60 grados. H5 es el mejor en validación normal pero a 30 grados queda
por debajo de H3. Por eso H6 amplía la rotación de H5 a +-35 grados: se busca conservar el
0.873 de H5 y acercarse al 0.83 de H3 con giro. (D5 con 180 grados da 0.808.)

**Cómputo.** Kaggle agotó la cuota semanal de GPU (30 horas) al intentar lanzar la siguiente
tanda. Desde aquí todo corre en Colab (dos entrenamientos a la vez en la L4).

**H5 con semilla 43 y H6 (Colab, 06-oct).** H5 s43 da 0.869 / 0.762 (centro 0.876, extremos
0.842): confirma el candidato. H6 s42 da 0.869 / 0.764 (centro 0.879, extremos 0.836). Caso por
caso, H6 mejora a H5 s42 en 5 de 15 y H5 s43 mejora a H5 s42 también en 5 de 15: cambiar la
rotación de +-12 a +-35 grados mueve la validación normal lo mismo que cambiar de semilla. El
peor caso sigue siendo el 087 (0.790-0.794 en los tres; D5 0.768). Con +-12 grados el modelo
no aguanta 60 grados (H5 0.218), como era de esperar.

**Estrés de proporción y resolución** (`aspect=1.6,lowres=2.5`, el aspecto del caso 039): H5
da 0.797 y H3, que sí vio baja resolución al entrenar, 0.759. El aumento de baja resolución de
H3 no ayudó; la orientación canónica sí. H6 da 0.797, igual que H5: ampliar la rotación no
cambia la tolerancia a la proporción y la baja resolución.

**Regla para elegir entre H5 y H6** (fijada antes de ver el estrés de H6): se queda H6 si con
giro de 30 grados supera con claridad el 0.807 de H5, porque en validación normal ya están
empatados.

**Decisión (06-oct): el candidato es H6.** Con la validación girada 30 grados H6 da 0.854
(primario 0.892, secundario 0.741, mAP50 0.971, Dice semántico 0.963) frente a 0.807 de H5:
+0.048, unas doce veces el ruido medido entre semillas, y mejora en 13 de los 15 casos. Su peor caso girado es 0.790 (el
087, que da lo mismo sin girar); el de H5 era 0.717. Girado 30 grados, H6 rinde más que D5 sin
girar (0.851). El costo en validación normal es nulo dentro del ruido (0.869 frente a 0.873 /
0.869 de H5). Receta de H6: la de D5 más `--aug-geo 1.0 --aug-rot-deg 35 --canonical`.
Con giro de 60 grados (fuera de lo que vio al entrenar) H6 da 0.734; H5 daba 0.218.

**H6 con tres semillas (validación, TTA).** 0.869 / 0.868 / 0.872 en instancias y 0.764 / 0.762 /
0.778 en secundarios; primario 0.904-0.905; F1 0.990-0.992; mAP50 0.969-0.971; mAP50:95
0.900-0.904; Dice semántico 0.970-0.971. Las tres superan a las tres de D5 (0.851-0.855) por
+0.013 a +0.021. El peor caso sigue siendo el 087 (0.782-0.790).

**Distancias en mm con H6 s42 (validación, 15 casos, `--distance-nofill`).** Mismos 41
fragmentos secundarios, 38 con distancia predicha. Error medio 0.185 mm (D5: 0.518), percentil
90 de 0.0 mm (D5: 0.388), correlación 0.972 (D5: 0.925). Dice 3D del secundario 0.701 (D5:
0.605) y del principal 0.921 (D5: 0.911). Secundarios perdidos en 3D: 5.0 % (D5: 4.7 %).

**Ensamble de las tres semillas de H6 + TTA (validación).** 0.873 en instancias (primario
0.906, secundario 0.773, secundarios sin pareja 5.9 %), F1 0.992, mAP50 0.972, mAP50:95 0.905,
Dice semántico 0.972; centro 0.882, extremos 0.841. Frente a la media de las tres semillas
(0.870 / 0.768) gana +0.003 y +0.006: **dentro del ruido**, igual que pasó con D5. El ensamble
no se justifica por la métrica; queda como opción si se quiere menos varianza.

H1 es el primer cambio de toda la semana que supera el ruido con claridad, y es justo lo que
ningún experimento D había tocado. Mejora en 10 de los 15 casos (hasta +0.057 en el 078) y el
peor retroceso es −0.021 (caso 024). Su curva sigue subiendo en la época 60 (0.850 → 0.858 →
0.860 → 0.863), mientras que D5 ya bajaba desde la 40: con el aumento la red deja de memorizar
y probablemente admite más épocas.

**Hallazgo sobre la orientación (prueba con el notebook 05, caso 044 de validación).** El caso
044 está guardado girado 180 grados (LPI). Tal como viene, D5 da 0.916; si se gira a la
orientación común y se predice con el mismo modelo, cae a 0.778 con 10 % del hueso asignado a
la región equivocada. Es decir, la red no aprendió a reconocer la anatomía en cualquier
orientación: aprendió que los volúmenes "con aspecto de LPI" (campo de visión recortado, no
cuadrado) usan la convención contraria de izquierda/derecha. Eso explica el caso 039 de test:
es RAI pero tiene el aspecto recortado de los LPI (en entrenamiento solo el caso 015 es así),
y la red le aplica la convención equivocada en parte de los cortes. Con orientación canónica
(H4) esa regla espuria deja de hacer falta. Barrido del 044: 0 grados 0.920, 180 grados 0.798,
+-90 grados 0.08-0.10.

Siguientes corridas lanzadas en Kaggle: H1 con semilla 43 y **H5 = aumento geométrico +
orientación canónica** (las dos correcciones atacan causas distintas).

### 12.2c Orientación: qué mostró el notebook 05 y el detector geométrico

**Caso 012 (test, solo diagnóstico).** El paciente está girado en el escáner unos 26 grados
(ángulo del sacro respecto a los coxales medido sobre la etiqueta real). Barrido de giros con
el modelo actual, sin reentrenar:

| Giro aplicado antes del modelo | Dice de instancias | Hueso tomado por otra región |
|---|---|---|
| 0 grados (tal como viene) | 0.690 | 14.0 % |
| −45 grados | **0.831** | 1.2 % |
| −7.5 grados (alineación automática con la primera pasada de la red) | 0.774 | 0.7 % |
| 45 / +-90 / −135 grados | 0.03-0.10 | 49-58 % |

Con solo enderezarlo, el caso pasa de 0.69 a 0.77-0.83 y la confusión de regiones desaparece
(Dice semántico del sacro de 0.756 a 0.948). El problema del 012 es la orientación, no la
calidad de la imagen. La alineación con la primera pasada de la red se queda corta (estima
7.5 grados de los 26) porque parte de una predicción ya equivocada.

**Casos 087 y 078 (validación, LPI).** Girados 180 grados y predichos con el mismo modelo dan
0.770 y 0.808 (antes 0.768 y 0.789): no cambian. Su problema es dónde se parte el hueso.

**Detector geométrico de orientación** (`orientacion.estimate_angle`, idea de Gus: detectar con
código que la imagen está girada y enderezarla). Sin red ni etiquetas: umbral de hueso, eje
principal (la pelvis es más ancha que profunda) y posición del hueso central respecto al lateral.

- Giros sintéticos de 20 a 180 grados en dos casos: recupera el ángulo con error de 0 a 4 grados.
- Caso 012: estima 36.6 grados (real, unos 26).
- Los 100 volúmenes contra la cabecera (`src/validar_orientacion.py`, `outputs/s10/orientacion_100.json`):
  el sentido (cuál lado es la espalda) acierta en 77 de 100, y falla en 22 volúmenes RAI de
  campo completo; el eje, sin contar el sentido, queda a menos de 15 grados en 81 de 100.
- Marca otros 18 casos con 15-37 grados de inclinación (entre ellos el 095 de validación,
  36 grados) sin que se haya comprobado si están realmente inclinados. En el 012 se ven la
  camilla y un fantoma de calibración bajo el paciente, que entran como "hueso" y pueden
  sesgar la estimación.

**Revisión a ojo de los casos marcados** (`src/contacto_orientacion.py`,
`outputs/s10/contacto_orientacion.png`; 16 casos de entrenamiento y validación marcados con más
de 15 grados y 3 de control; no se abrió ningún caso de test). En casi todos los marcados (095,
018, 055, 046, 029, 093, 052, 064, 030, 059, 053, 048, 034) el paciente se ve derecho sobre la
camilla: la flecha del detector apunta hacia adelante en vez de hacia la espalda, y además
torcida 15-35 grados. Es decir, cuando el detector se equivoca de sentido tampoco acierta el
eje, y lo que marcaba como "inclinado" eran fallos suyos, no pacientes girados. Solo el 066 y el
025 muestran una inclinación leve real. En los tres controles (005, 014, 074) acierta.

**Decisión (06-oct).** El detector geométrico **no entra al pipeline**: enderezaría mal a
pacientes que están bien colocados. Queda en el repositorio como experimento documentado.
La tolerancia al giro se consigue con el aumento de rotación (H6) y el sentido con la cabecera.

**Decisión provisional anterior (se conserva como registro).** (1) El sentido se toma siempre de la cabecera (exacto). (2) El
detector geométrico no entra todavía al pipeline: primero hay que revisar a ojo los casos
marcados. Si acierta, se usará solo para giros grandes (más de 25-30 grados) y con aviso en el
dashboard. (3) El aumento de datos sigue siendo necesario para tolerar lo que no se corrija.
El preprocesamiento debe ser el mismo en entrenamiento y en inferencia.

**Estrés de D5 con giro de 180 grados:** 0.808 (normal 0.851). **Test de F2 (lambda x2):** 0.847.

### 12.3 Regla sobre test a partir de ahora

- La cifra oficial de la semana 10 es la de la sección 11 (0.845), obtenida con la
  configuración congelada antes de ver test.
- El candidato nuevo se elige **solo** con validación y estrés. Cuando esté congelado se evalúa
  en test **una vez más**, y esa cifra se reporta como *segunda evaluación, posterior al
  análisis de fallos*, no como estimación limpia.
- No se prueban varias variantes en test para quedarse con la mejor.

### 12.4 Configuración congelada para la segunda evaluación en test

Escrita el 06-oct **antes de correr nada sobre test con H6**. Gus dio el visto bueno el 06-oct
(14:35 UTC) sin cambiarla; la evaluación se lanzó a las 14:40 UTC.

- **Modelo principal:** `h6_auggeo_rot35_canon_ema_w32_e60_s42.pth` (receta de D5 más
  `--aug-geo 1.0 --aug-rot-deg 35 --canonical`; semilla 42, la de referencia). Un solo modelo
  con TTA espejo, tamaño 384: el mismo protocolo de la sección 10, para que D5 → H6 sea una
  comparación de igual a igual. El ensamble no entra como principal porque en validación no
  supera el ruido.
- **Orientación:** los volúmenes cuya cabecera dice `TransformMatrix = -1 0 0 0 -1 0 0 0 1`
  se giran 180 grados en el plano antes de entrar a la red (`orientacion.to_canonical`) y la
  predicción se compara en esa misma orientación. Solo usa la cabecera, no las etiquetas.
- **Distancias:** post-proceso 3D con `--distance-nofill`, `min_voxels` 30.
- **Se reportan además, sin elegir entre ellos:** H6 s43, H6 s44 y el ensamble de los tres
  (misma decisión que se tomó con los checkpoints de D5 y E2). La cifra que cuenta es la del
  modelo principal, salga como salga.
- Se corre una sola vez. No se ajusta ningún umbral ni se reentrena después de ver el
  resultado. La cifra oficial de la primera evaluación (0.845) no se borra ni se reemplaza:
  esta se reporta como segunda evaluación, posterior al análisis de fallos (regla 12.3).

**Qué se espera, dicho antes de mirar.** En validación H6 s42 sube +0.019 sobre D5 s42 (0.869
frente a 0.851). Los dos casos que hundieron el test tienen causas que H6
ataca (012: giro de unos 26 grados; 039: aspecto de LPI con orientación RAI), pero eso no se
ha medido sobre ellos con H6. Si test queda por debajo de 0.85 se reporta así.

### 12.5 Resultado de la segunda evaluación en test (configuración 12.4)

Modelo H6 s42 + TTA espejo, los mismos 15 casos y 4664 cortes de la sección 11. Se corrió una
sola vez el 06-oct (14:40-15:39 UTC), en CPU porque la máquina de Colab se había cerrado; antes
se comprobó con el caso 019 de validación que la CPU reproduce la cifra de la GPU (0.91118 en
los dos). Archivo: `outputs/s10/TEST2_h6_s42_tta.json`.

| Métrica | Test, H6 (segunda) | Objetivo | Test, D5 (primera) | Validación (H6 s42) |
|---|---|---|---|---|
| F1 clasificación (macro) | 0.990 | 0.85 | 0.984 | 0.992 |
| AUC clasificación (macro) | 0.999 | 0.85 | 0.998 | 0.999 |
| IoU medio detección (refinada) | 0.943 | 0.65 | 0.898 | 0.949 |
| mAP50 detección (refinada) | 0.961 | 0.65 | 0.920 | 0.971 |
| mAP50:95 detección (refinada) | 0.902 | 0.40 | 0.853 | 0.904 |
| Dice segmentación semántica (macro) | 0.971 | 0.85 | 0.949 | 0.971 |
| IoU segmentación semántica (macro) | 0.943 | 0.70 | 0.904 | 0.944 |
| **Dice de instancias (emparejadas)** | **0.875** | **0.85** | 0.845 | 0.869 |
| IoU de instancias | 0.816 | 0.70 | 0.781 | 0.808 |
| Dice instancia primaria / secundaria | 0.915 / 0.750 | – | 0.894 / 0.692 | 0.905 / 0.764 |
| Secundarios sin pareja | 7.8 % | – | 10.6 % | 6.9 % |
| Detección cruda mAP50 / mAP50:95 | 0.912 / 0.556 | – | 0.866 / 0.511 | 0.887 / 0.506 |

**Lectura.** Con H6 se cumplen todos los objetivos del curso, incluido el Dice de instancias
(0.875 frente a 0.85). Frente a la primera evaluación sube +0.030 en instancias y +0.058 en
secundarios, y mejora en 13 de los 15 casos. Los dos que hundían el test:

| Caso | D5 (primera) | H6 (segunda) | Causa identificada en 12.1 |
|---|---|---|---|
| 012 | 0.692 | 0.865 | Paciente girado unos 26 grados; H6 entrena con rotación de +-35 |
| 039 | 0.710 | 0.784 | RAI con aspecto de LPI; H6 usa orientación canónica |

El 039 mejora pero sigue siendo el peor caso de test. Bajan dos casos: 054 (−0.005) y 061
(−0.017). El centro del volumen da 0.885 y los extremos 0.842.

**Las otras dos semillas y el ensamble sobre test** (declarados en 12.4 antes de correr; se
reportan sin elegir, la cifra que cuenta es la de H6 s42). Corridos en GPU en una máquina nueva
de Colab el 06-oct.

| Modelo (test, TTA) | Inst. Dice | Primario | Secundario | Sec. sin pareja | F1 | mAP50 | mAP50:95 | Dice semántico |
|---|---|---|---|---|---|---|---|---|
| D5 s42 (primera evaluación) | 0.845 | 0.894 | 0.692 | 10.6 % | 0.984 | 0.920 | 0.853 | 0.949 |
| **H6 s42 (principal)** | **0.875** | 0.915 | 0.750 | 7.8 % | 0.990 | 0.961 | 0.902 | 0.971 |
| H6 s43 | 0.875 | 0.913 | 0.752 | 7.6 % | 0.986 | 0.952 | 0.894 | 0.970 |
| H6 s44 | 0.876 | 0.913 | 0.761 | 7.3 % | 0.989 | 0.960 | 0.902 | 0.970 |
| Ensamble de las tres | 0.879 | 0.916 | 0.761 | 7.5 % | 0.990 | 0.963 | 0.907 | 0.972 |

Las tres semillas dan lo mismo en test (0.875-0.876): el resultado no depende de la semilla.
El ensamble suma +0.003 sobre la media, igual que en validación. En las tres semillas el 012
queda entre 0.865 y 0.892 y el 039 entre 0.775 y 0.784; el 039 sigue siendo el peor caso.

**Distancias en mm sobre test con H6 s42** (`--distance-nofill`, `min_voxels` 30, una pasada;
`outputs/s10/TEST2_h6_s42_nofill.json`). Los mismos 41 fragmentos secundarios de la sección 11.

| Medida (test) | D5 (primera) | H6 (segunda) | H6 en validación |
|---|---|---|---|
| Error medio de distancia | 2.956 mm | 1.277 mm | 0.185 mm |
| Error mediano / percentil 90 | 0.0 / 3.453 mm | 0.0 / 0.556 mm | 0.0 / 0.0 mm |
| Correlación distancia real-predicha | 0.543 | 0.818 | 0.972 |
| Fragmentos con distancia predicha | 36 de 41 | 36 de 41 | 38 de 41 |
| Dice 3D principal / secundario | 0.909 / 0.682 | 0.931 / 0.756 | 0.921 / 0.701 |
| Secundarios perdidos en 3D | 5.6 % | 7.8 % | 5.0 % |
| ASSD / HD95 medios | 3.02 / 16.33 mm | 2.06 / 11.55 mm | 4.08 / 17.99 mm |

El error de distancia en test baja a menos de la mitad y la correlación pasa de 0.54 a 0.82,
pero sigue lejos de la de validación (0.97). El error medio lo domina **un solo fragmento**:
en el caso 007 (coxal derecho) la distancia real es 9.4 mm y se predicen 45.8 mm (error 36.4
mm, el 79 % de todo el error acumulado); sin ese fragmento el error medio sería 0.28 mm. Solo
4 de los 36 fragmentos tienen error mayor a 1 mm (con D5 eran 8) y el error de 68.5 mm que D5
cometía en el caso 012 desaparece. Lo único que empeora es la fracción de secundarios perdidos
en 3D (5.6 % → 7.8 %). Pendiente: mirar el fragmento del 007 (ya fallaba con D5: 19.9 mm).

**Cómo debe citarse.** La cifra limpia, obtenida con la configuración congelada antes de ver
test, sigue siendo 0.845 (sección 11). El 0.875 es una **segunda evaluación, hecha después de
analizar los fallos de la primera**: el diagnóstico usó los casos 012 y 039 de test para
entender qué fallaba, así que no es una estimación independiente. Lo que la respalda es que
el modelo se eligió solo con validación y estrés (reglas escritas antes de ver cada número),
que la validación sube lo mismo (0.851 → 0.869) y que test y validación quedan a 0.006.

### 12.6 Análisis de error en test con H6 (matriz de confusión y peores casos)

Para el informe se repitió la inferencia de H6 s42 sobre test guardando registros y predicciones
(`evaluate_volumes.py --save-records --save-predictions`, misma configuración). La repetición da
exactamente las mismas cifras que la evaluación de 12.5 (diferencia 0.0 en todas las métricas), así
que no es una evaluación nueva. Script: `src/informe_test.py`; salida en
`outputs/s10/informe_h6_test/` (`informe.json` y un panel por caso).

**Clasificación (presencia de cada región por corte, 4 664 cortes).**

| Región | Presente y predicha | Ausente y predicha | Presente y no predicha | Ausente y no predicha | F1 |
|---|---|---|---|---|---|
| Sacro | 2 238 | 37 | 56 | 2 333 | 0.980 |
| Coxal izquierdo | 3 372 | 11 | 12 | 1 269 | 0.997 |
| Coxal derecho | 3 316 | 28 | 4 | 1 316 | 0.995 |

**Segmentación semántica (a qué región va cada píxel real).** Sacro 96.6 % correcto y 3.3 % a
fondo; coxal izquierdo 98.5 % y 1.5 %; coxal derecho 98.5 % y 1.5 %. Solo el 0.06 % del hueso se
asigna a otra región (máximo por caso: 0.18 %, en el 039). Con D5 esa confusión era 13.7 % en el
caso 012 y 8.2 % en el 039: la orientación canónica y el aumento la eliminaron.

**Cinco peores casos (Dice por fragmento).**

| Caso | Dice | Primario | Secundario | Secundarios sin pareja | Hueso a fondo | Centro / extremos |
|---|---|---|---|---|---|---|
| 039 | 0.784 | 0.890 | 0.635 | 7.7 % | 3.2 % | 0.788 / 0.768 |
| 061 | 0.822 | 0.891 | 0.635 | 17.4 % | 1.2 % | 0.806 / 0.877 |
| 007 | 0.831 | 0.913 | 0.545 | 24.5 % | 1.6 % | 0.821 / 0.859 |
| 054 | 0.834 | 0.899 | 0.663 | 16.2 % | 2.9 % | 0.835 / 0.832 |
| 047 | 0.854 | 0.870 | 0.753 | 17.1 % | 5.4 % | 0.888 / 0.745 |

Lectura: lo que queda por resolver ya no es a qué hueso pertenece cada píxel, sino **dónde se
parte el hueso en fragmentos**. En los cinco casos el principal está entre 0.87 y 0.91 y los
secundarios entre 0.54 y 0.75. El 039 tiene muchos fragmentos secundarios por corte (418 en 392
cortes); el 047 pierde en los extremos del volumen y deja 5.4 % del hueso como fondo.

**Dato de las particiones que conviene tener presente.** Los volúmenes guardados girados no están
repartidos por igual: 22 de 70 en entrenamiento, 10 de 15 en validación y 2 de 15 en test. La
partición es por paciente con semilla fija y no se tocó.

## 13. Cierre

Hecho en la Semana 10: las dos evaluaciones en test (secciones 11 y 12.5), el análisis de error
(12.6), la latencia en GPU y CPU (8), SAM zero-shot en validación y en test (9), las distancias sin
relleno (5 y 12.5) y el notebook `04_semana10_pipeline_multitarea.ipynb`.

Lo que quedaba abierto y cómo terminó:

| Punto | Cómo terminó |
|---|---|
| Llevar H6 a la demo | Hecho en la Semana 11: la demo y la API cargan H6 y `demo_export.py` endereza el volumen según la cabecera (`docs/semana11_dashboard.md`) |
| Latencia en el portátil | Medida con `src/measure_latency.py`: 336 ms por corte y 696 ms con TTA (`MODEL_CARD.md`, sección 4.8) |
| Ensamble de tres semillas en la demo | No entra: +0.003 en validación y en test, y triplica el costo. La demo muestra solo el modelo entregado |

No se hizo y queda como trabajo futuro:

- Entender el fragmento del caso 007 (coxal derecho, 9.4 mm reales y 45.8 mm predichos) y por qué
  suben los secundarios perdidos en 3D en test (de 5.6 % a 7.8 %).
- Asignación multi-celda en la detección, segundo pase por región, resolución 256 y 512, umbral de
  clasificación por clase y pesos de FundidoraPC.

## 14. Pedidos del profesor después de la entrega (8 de octubre)

| Pedido | Respuesta | Dónde |
|---|---|---|
| Ver el aporte del CBAM en cada bloque con el valor γ (no el gamma 1.5 de la imagen) y reportar el valor final | `train_s10.py` guarda `gamma_final` y `gamma_checkpoint` en el JSON. Corrida `g1`: receta de H6 con CBAM en los 3 bloques y γ aprendible | `docs/hiperparametros.md`, sección 3 |
| Justificar el EMA, autorizado solo para la segmentación | La ganancia medida del EMA está en la segmentación (secundarios +0.024 de media con 3 semillas). H6 promedia toda la red, así que se agregó `--ema-scope segmentacion` y la corrida `e1` para compararlo | `docs/hiperparametros.md`, sección 5 |
| Elegir la métrica principal | Propuesta: Dice por fragmento (instancias), que ya es la que decide todo el protocolo. Falta la confirmación del grupo | — |
| Justificar cada número de la arquitectura | Una fila por número con su origen: experimento propio, literatura, cómputo o sin barrer | `docs/hiperparametros.md` |
| Limitaciones y mejoras | Ya están en `MODEL_CARD.md` (sección 6). Falta decir, para cada mejora, si se aplica o no y por qué | — |
| Cuantizar el modelo sin perder Dice | `src/cuantizar.py` compara fp32, fp16, bf16 e int8. Hallazgo: las convoluciones transpuestas no toleran int8 y se dejan en fp32 | `docs/cuantizacion.md` |
| HD95 y ASSD por fragmento (extensión opcional del enunciado) | Antes solo se medían en los secundarios emparejados (sección 4.9 del model card). `infer_volume.py --surface-metrics` ahora los da para todos los fragmentos, con los perdidos contados aparte | `src/volume_postprocess.py` |

**γ final ya medido.** Detección sola (semana 9): 0.089 / −0.106 / 0.108 en los bloques 1/4, 1/8 y
1/16. Multitarea B0 (sección 3.2, sacado del checkpoint): −0.630 / 0.716 / −1.319. En el multitarea
γ sale del rango [0, 1], así que no se puede leer como una fracción de atención. `g1` lo repite con
la receta de H6.

**Cómo se corre todo.** En una sola sesión de Colab con L4:
`notebooks/colab/colab_pedidos_profesor.ipynb` (se genera con `python colab/hacer_cuaderno_profesor.py`).
Entrena `g1` y `e1` a la vez y, mientras tanto, calcula HD95/ASSD de H6 en validación y test. Al
final evalúa `g1` y `e1` en validación y cuantiza H6 en CPU. Los kernels equivalentes para Kaggle
están en `kaggle/gpu/g1_*.py` y `kaggle/gpu/e1_*.py`. En test solo se agregan métricas al modelo ya
congelado: nada de esto elige con test.


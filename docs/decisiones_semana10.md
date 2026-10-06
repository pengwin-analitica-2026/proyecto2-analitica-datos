# Bitácora de decisiones — Semana 10

Registro cronológico de lo que se entrenó, lo que se decidió y por qué. Se actualiza a medida que
avanza la semana. Las cifras son de **validación** (15 casos, 5047 cortes, subconjunto `todo`, con
TTA espejo), salvo que se diga otra cosa. Test no se ha tocado.

Documentos relacionados: `outputs/s10/resumen_experimentos.md` (tabla viva de resultados),
`docs/plan_semana10.md` (plan), `IA_USAGE.md` (uso de IA).

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
  (−0.007 vs EMA solo). **Sin ganancia consistente.**
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
  **la distancia predicha fue 0.0 mm en los 41 fragmentos secundarios**, y solo 5 de los 41 están
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

## 10. Pendiente

1. Revisar los fragmentos separados con más detalle (3.5 mm del caso 014 no se vio) y repetir con el ensamble.
2. Decidir si el ensamble entra a la demo según latencia.
3. Asignación multi-celda para la detección y diagnóstico de confusión izquierda/derecha
   (ideas del notebook de la semana 9).
4. Ajustar el post-proceso 3D (voxels mínimos, relleno de huecos).
5. Latencia en CPU de pocos hilos (Mac / despliegue); resolución 512 y más épocas con EMA (opcionales).
6. Evaluación en test, una sola vez, con la configuración congelada.
7. Notebook `04_semana10_pipeline_multitarea.ipynb`, actualizar `plan_semana10.md` e `IA_USAGE.md`.
8. Semana 11: dashboard de tres visores y demo desplegada.

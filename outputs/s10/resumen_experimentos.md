# Semana 10 — resumen de experimentos (actualizado el 6 de octubre de 2026, 19:00 UTC)

**Cerrado.** Modelo final: H6 (sección 2). El registro detallado de cada decisión está en `docs/decisiones_semana10.md`; el notebook `notebooks/04_semana10_pipeline_multitarea.ipynb` lee estos mismos archivos.

Validación, 15 casos completos (5 047 cortes), TTA espejo, métricas sobre el volumen. Objetivos: F1/AUC 0.85; IoU de detección y mAP50 0.65; mAP50:95 0.40; Dice de segmentación 0.85; Dice de instancia 0.85 e IoU 0.70.

## 1. Ya probado

| Experimento | Qué cambia | mAP50 ref. | mAP50:95 ref. | Dice seg. | Dice instancia | Dice secundarios | Lectura |
|---|---|---|---|---|---|---|---|
| Línea base `w10g_384neg` | Semana 10 anterior | 0.874 | 0.781 | 0.956 | 0.834 | 0.702 | Referencia a superar |
| Ablación CBAM (40 ép.) | ninguno / 1 etapa / 3 etapas + γ | 0.900 / 0.926 / 0.907 | 0.796 / 0.835 / 0.814 | — | CBAM no mejora instancias | — | CBAM 1 etapa gana en detección |
| **B1** (3 semillas) | 8 canales, CBAM1, sec3, 80 ép. | 0.933 ± 0.004 | 0.851 ± 0.003 | 0.962 ± 0.002 | 0.829 ± 0.005 | 0.645 ± 0.015 | Receta base; estable entre semillas |
| B2 | sec6, 50 ép. | 0.918 | 0.834 | 0.963 | 0.832 | 0.681 | No supera a B1 |
| v4 Colab | B1 + peso por fractura 2, 5 000 cortes/ép., 60 ép. | 0.945 | 0.867 | 0.965 | 0.832 | 0.642 | Mejora detección, no instancias |
| Colab `ids` | instancia con 31 clases | 0.933 | 0.844 | 0.962 | 0.787 | 0.460 | Peor: se pierden 45 % de secundarios. Descartado |
| c1 | 16 canales, 80 ép. | 0.955 | 0.886 | 0.969 | 0.839 | 0.649 | Mejora detección y clasificación |
| **c2** (3 semillas: 42/43/44) | **32 canales, 60 ép.** | **0.965 ± 0.006** | **0.895 ± 0.010** | **0.969** | **0.850 ± 0.004** | **0.687 / 0.676 / 0.699** | **Mejor receta; confirmada con 3 semillas** |
| w48 | 48 canales, 60 ép. | 0.964 | 0.899 | 0.970 | 0.844 | 0.659 | No supera a c2 en instancias; el ancho se satura en 32 |
| λ ×2 | w32 + λ instancia/borde ×2, detección ×0.5 | 0.964 | 0.896 | 0.967 | 0.849 | 0.676 | Sin mejora; la detección cruda empeora (0.832). Primer paso de F1, descartado |
| D1 | c2 + contexto 2.5D (3 cortes) | 0.963 | 0.896 | 0.970 | 0.848 | 0.681 | Sin mejora; descartado |
| D2 | c2 + pérdida focal γ=2 | 0.962 | 0.889 | 0.970 | 0.846 | 0.670 | Sin mejora; descartado |
| D3 | c2 + Tversky β=0.7 | 0.966 | 0.894 | 0.968 | 0.846 | 0.685 | Sin mejora; descartado |
| D4 | c2 + deep supervision 0.4 | 0.968 | 0.899 | 0.969 | 0.846 | 0.667 | Mejora algo la detección; instancias/secundarios sin mejora; descartado |
| D5 | c2 + EMA 0.999 | 0.959 | 0.883 | 0.968 | 0.851 | 0.703 | Secundarios +0.016 en s42 |
| D5 s43 / s44 | c2 + EMA, semillas 43 y 44 | – | – | – | 0.855 / 0.854 | 0.724 / 0.708 | **Confirmado**: secundarios +0.016/+0.048/+0.009 vs c2 en las 3 semillas (media +0.024), instancias +0.002/+0.009/0.000. EMA entra a la receta |
| E2 | c2 + EMA + backbone Intel | 0.948 | 0.872 | 0.966 | 0.851 | 0.714 | Mejor secundario (0.714), pero la detección refinada baja (mAP50:95 0.872 vs 0.883 de EMA solo) y F1 0.984 |
| A1 | c2 sin CBAM | 0.954 | 0.878 | 0.968 | 0.847 | 0.686 | **Ablación CBAM**: segmentación igual; F1 de clasificación 0.976 vs 0.988 y mAP50:95 refinada 0.878 vs 0.889. CBAM aporta en clasificación y detección |
| F1 | candidato (c2+EMA) sin CBAM | 0.946 | 0.872 | 0.968 | 0.851 | 0.696 | **Ablación CBAM sobre el candidato**: segmentación igual; F1 0.975 vs 0.988, mAP50:95 refinada 0.872 vs 0.883. Mismo patrón que sobre c2: CBAM aporta en clasificación y detección |
| F2 | candidato + λ ×2 (instancia/borde ×2, detección ×0.5) | 0.962 | 0.894 | 0.968 | 0.855 | 0.701 | **Ablación λ**: instancias +0.004 (ruido), secundarios iguales; detección cruda empeora (mAP50 0.827 vs 0.868). Se mantienen los λ por defecto |
| E2 s43 | candidato + backbone Intel, semilla 43 | 0.966 | 0.901 | 0.969 | 0.856 | 0.717 | Con s42 secundarios +0.011 y con s43 −0.008 vs EMA solo: **Intel no da ganancia consistente** (neutral) |
| D6 | c2 + sobremuestreo de secundarios ×3 | 0.960 | 0.892 | 0.969 | 0.837 | 0.642 | Empeora instancias y secundarios; descartado |
| E1 | c2 + backbone preentrenado en Intel (transfer learning) | 0.959 | 0.887 | 0.969 | 0.847 | 0.683 | Igual que c2 en segmentación; la detección cruda sube (mAP50 0.905 vs 0.875, mAP50:95 0.498 vs 0.474) |

Hallazgos:

- Ni 2.5D, ni pérdida focal, ni Tversky (D1–D3) mueven el Dice de instancia (0.846–0.848 frente a 0.849 de c2) ni los secundarios; el techo parece estar en el post-proceso o en la ambigüedad de las etiquetas secundarias, no en la red. Sube la prioridad del post-proceso 3D.
- El ancho del backbone (hasta ahora 8 canales) era la limitación principal. Pasar a 32 sube detección, clasificación y por primera vez los secundarios.
- Lo que no ayuda: más peso a secundarios (sec6), peso por fractura, instancia con IDs, CBAM en tres etapas.
- Detección cruda: mAP50 0.875 y mAP50:95 0.474 en c2; ya cumple el objetivo sin refinar.

## 1b. Serie H (6 de octubre): generalización

Después de la primera evaluación en test (sección 2) se analizaron los fallos y se probaron cambios que atacan la causa, un cambio a la vez sobre D5 y decididos solo con validación. Todas las filas: c2 + EMA, 60 épocas, TTA.

| Experimento | Qué cambia frente a D5 | mAP50 ref. | mAP50:95 ref. | Dice seg. | Dice instancia | Dice secundarios | Lectura |
|---|---|---|---|---|---|---|---|
| D5 s42 | referencia | 0.959 | 0.883 | 0.968 | 0.851 | 0.703 | Modelo de la primera evaluación en test |
| H1 s42 | aumento geométrico (escala, proporción, desplazamiento, rotación ±12°) | 0.969 | 0.902 | 0.970 | 0.864 | 0.743 | Primer cambio de la semana que supera el ruido con claridad |
| H1 s43 | lo mismo, semilla 43 | 0.968 | 0.902 | 0.969 | 0.865 | 0.750 | Confirma H1 |
| H2 | caché que conserva la proporción (`pad`) | 0.961 | 0.890 | 0.969 | 0.857 | 0.723 | En el borde del ruido; no se adopta |
| H3 | aumento con rotación libre en 25 % de las muestras, baja resolución y gamma | – | – | – | 0.861 | 0.741 | Parecido a H1 en validación normal; aguanta giros grandes. No se guardó su JSON de validación |
| H4 | orientación canónica según la cabecera, sin aumento | 0.971 | 0.897 | 0.970 | 0.861 | 0.742 | Sola ya aporta; se suma con el aumento |
| H5 s42 | H1 + orientación canónica | 0.976 | 0.912 | 0.971 | 0.873 | 0.769 | Mejor en validación normal |
| H5 s43 | lo mismo, semilla 43 | 0.969 | 0.903 | 0.971 | 0.869 | 0.762 | Confirma H5 |
| **H6 s42** | **H5 con rotación ±35°** | 0.971 | 0.904 | 0.971 | 0.869 | 0.764 | **Modelo final.** Empata con H5 y aguanta mucho mejor el giro |
| H6 s43 | lo mismo, semilla 43 | 0.969 | 0.900 | 0.971 | 0.868 | 0.762 | Confirma H6 |
| H6 s44 | lo mismo, semilla 44 | 0.971 | 0.901 | 0.970 | 0.872 | 0.778 | Confirma H6 |
| Ensamble H6 | las tres semillas promediadas | 0.972 | 0.905 | 0.972 | 0.873 | 0.773 | +0.003 sobre la media: dentro del ruido |

Prueba de estrés: la misma validación, deformada antes de entrar al modelo (Dice de instancia).

| Modelo | Rotación vista al entrenar | Normal | Girada 30° | Girada 60° | Proporción 1.6 y baja resolución |
|---|---|---|---|---|---|
| D5 | ninguna | 0.851 | 0.559 | 0.127 | – |
| H1 | ±12° | 0.864 | 0.771 | 0.212 | – |
| H3 | ±12° y libre en 25 % | 0.861 | 0.834 | 0.824 | 0.759 |
| H5 | ±12° | 0.873 | 0.807 | 0.218 | 0.797 |
| **H6** | ±35° | 0.869 | **0.854** | 0.734 | 0.797 |

Hallazgos de la serie H:

- El modelo no veía ningún aumento geométrico y memorizaba: la pérdida de entrenamiento bajaba de 0.86 a 0.48 con la validación plana desde la época 30.
- 34 de los 100 volúmenes están guardados girados 180° en el plano (lo dice la cabecera del `.mha`). Sin corregirlo, la red aprendía una regla por el aspecto de la imagen para decidir izquierda y derecha.
- Descartado: detector geométrico de giro (`orientacion.estimate_angle`; al revisar a ojo, casi todo lo que marcaba eran pacientes bien colocados), filtros de post-proceso (≤ +0.002) y `min_size=1` (+0.002, por debajo del ruido).

## 2. Modelo final y resultados

**Modelo final: H6** = c2 + EMA 0.999 (ancho 32, CBAM en 1 etapa, 2.74 M de parámetros, 60 épocas, secundarios ×3, λ por defecto, sin preentrenamiento) + aumento geométrico con rotación de ±35° + orientación canónica según la cabecera. Archivo: `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth`. La arquitectura es la de D5; cambian el entrenamiento y el preprocesamiento.

```bash
python src/train_s10.py --cache-root data/cache/s10 --size 384 --epochs 60 --base-channels 32 --cbam 1 \
    --secondary-weight 3 --augment --cosine --batch-size 16 --per-epoch 2500 --ema 0.999 --seed 42 \
    --aug-geo 1.0 --aug-rot-deg 35 --canonical --output outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth
python src/evaluate_volumes.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split test --tta --output outputs/s10/TEST2_h6_s42_tta.json
```

### Test (15 casos, 4 664 cortes, 1 modelo + TTA)

| Métrica | Objetivo | Test, D5 (primera) | Test, H6 (segunda) | Validación, H6 s42 |
|---|---|---|---|---|
| F1 de clasificación (macro) | 0.85 | 0.984 | 0.990 | 0.992 |
| AUC de clasificación (macro) | 0.85 | 0.998 | 0.999 | 0.999 |
| IoU medio de detección (caja refinada) | 0.65 | 0.898 | 0.943 | 0.949 |
| mAP50 de detección (refinada) | 0.65 | 0.920 | 0.961 | 0.971 |
| mAP50:95 de detección (refinada) | 0.40 | 0.853 | 0.902 | 0.904 |
| Dice de segmentación semántica (macro) | 0.85 | 0.949 | 0.971 | 0.971 |
| IoU de segmentación semántica (macro) | 0.70 | 0.904 | 0.943 | 0.944 |
| **Dice de instancia (por fragmento)** | **0.85** | 0.845 | **0.875** | 0.869 |
| IoU de instancia (por fragmento) | 0.70 | 0.781 | 0.816 | 0.808 |
| Dice primario / secundario | – | 0.894 / 0.692 | 0.915 / 0.750 | 0.905 / 0.764 |
| Detección cruda (cabeza grid sin refinar) mAP50 / mAP50:95 | – | 0.866 / 0.511 | 0.912 / 0.556 | 0.887 / 0.506 |

- **Primera evaluación (D5 s42):** configuración congelada antes de ver test. Es la cifra limpia: 0.845 en Dice de instancia, el único objetivo que no se cumplía.
- **Segunda evaluación (H6 s42):** hecha después de analizar los fallos de la primera; el análisis miró los casos 012 y 039 de test, así que no es una estimación independiente. El modelo se eligió solo con validación y estrés. Cumple los nueve objetivos.
- Las otras semillas de H6 en test dan 0.875 y 0.876; el ensamble de las tres, 0.879. Los demás checkpoints de la familia D/E/F quedaron entre 0.841 y 0.848.
- Casos que hundían el test: 012 (paciente girado unos 26°) pasó de 0.692 a 0.865 y 039 de 0.710 a 0.784. El 039 sigue siendo el peor.

### Ablaciones obligatorias

Medidas sobre c2 y sobre el candidato con EMA (D5), **no repetidas sobre H6**: H6 comparte arquitectura, pérdida y λ con D5.

| Ablación | Resultado |
|---|---|
| CBAM (con/sin) | Aporta en clasificación (F1 +0.013) y detección refinada (mAP50:95 +0.011); segmentación igual |
| Transfer learning (Intel, con/sin) | Sin ganancia consistente en segmentación (secundarios +0.011 y −0.008 con 2 semillas); detección cruda mixta |
| λ (×2 instancia/borde) | Sin ganancia; empeora la detección cruda |

### Volumen: post-proceso 3D y distancia en mm

Medición con `infer_volume.py --tta --distance-nofill --surface-metrics` (la versión original rellenaba los huecos entre fragmentos y dejaba todas las distancias predichas en 0 mm; ver `docs/decisiones_semana10.md`, sección 5).

| Medida | Validación, D5 | Validación, H6 | Test, D5 | Test, H6 |
|---|---|---|---|---|
| Error medio de distancia (mm) | 0.518 | 0.185 | 2.956 | 1.277 |
| Correlación real-predicha | 0.925 | 0.972 | 0.543 | 0.818 |
| Fragmentos secundarios con distancia predicha | 38 de 41 | 38 de 41 | 36 de 41 | 36 de 41 |
| Dice 3D principal / secundario | 0.911 / 0.605 | 0.921 / 0.701 | 0.909 / 0.682 | 0.931 / 0.756 |
| ASSD / HD95 medios (mm) | 5.11 / 22.81 | 4.08 / 17.99 | 3.02 / 16.33 | 2.06 / 11.55 |

En test, un solo fragmento (caso 007, coxal derecho: 9.4 mm reales, 45.8 mm predichos) concentra el 79 % del error; sin él el error medio sería 0.28 mm. En validación solo 5 de los 41 fragmentos están separados en el ground truth, así que esa cifra es indicativa.

### SAM zero-shot y latencia

- **SAM ViT-B con la caja predicha como prompt, sobre test con H6 (1 181 cortes, 3 002 pares corte-región):** Dice 0.889 e IoU 0.823 por región, frente a 0.934 y 0.889 de nuestra máscara semántica dentro de la misma caja. En validación con H6: 0.859 / 0.781 frente a 0.914 / 0.859; con D5: 0.847 / 0.769 frente a 0.898 / 0.839. Archivos `sam_baseline_test_h6.json`, `sam_baseline_val_h6.json` y `sam_baseline_val.json`.
- **Latencia por corte (384², lote 1, medida con D5; H6 tiene la misma arquitectura):** GPU L4 64 / 72 / 87 ms (1 modelo / con TTA / ensamble de 3 con TTA); CPU de 12 hilos 162 / 250 / 518 ms. Falta medirla en el Mac.

## 3. Pendiente

- Latencia en el Mac con `src/measure_latency.py`.
- Llevar H6 y el paso de orientación canónica a la demo y al dashboard de la Semana 11 (hoy usan D5).
- Mirar el fragmento del caso 007 y por qué suben los secundarios perdidos en 3D en test (5.6 % → 7.8 %).
- Opcionales que no se hicieron: asignación multi-celda en la detección, segundo pase por región, resolución 256 y 512, umbral de clasificación por clase, pesos de FundidoraPC.

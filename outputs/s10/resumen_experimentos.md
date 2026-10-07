# Semana 10 — tabla de experimentos

Una fila por corrida, con sus métricas de validación. Cada fila sale de un JSON de esta carpeta. Por qué se probó cada cosa y qué se decidió está en `docs/decisiones_semana10.md`; el modelo final y sus resultados en test, en `MODEL_CARD.md`. El notebook `notebooks/04_semana10_pipeline_multitarea.ipynb` lee estos mismos archivos.

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

Después de la primera evaluación en test se analizaron los fallos y se probaron cambios que atacan la causa, un cambio a la vez sobre D5 y decididos solo con validación. Todas las filas: c2 + EMA, 60 épocas, TTA.

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

## 2. Modelo final

**H6** = c2 + EMA 0.999 (ancho 32, CBAM en 1 etapa, 2.74 M de parámetros, 60 épocas, secundarios ×3, λ por defecto, sin preentrenamiento) + aumento geométrico con rotación de ±35° + orientación canónica según la cabecera. La arquitectura es la de D5; cambian el entrenamiento y el preprocesamiento.

| Qué | Archivo en esta carpeta |
|---|---|
| Pesos (semillas 42, 43 y 44) | `h6_auggeo_rot35_canon_ema_w32_e60_s4{2,3,4}.pth` |
| Validación sobre el volumen completo | `h6_auggeo_rot35_canon_ema_w32_e60_s4{2,3,4}_val.json`, `h6_ensamble3_val.json` |
| Test, primera evaluación (D5) | `TEST_d5_s42_tta.json`, `TEST_d5_s42_nofill.json` |
| Test, segunda evaluación (H6) | `TEST2_h6_s4{2,3,4}_tta.json`, `TEST2_h6_ens3_tta.json`, `TEST2_h6_s42_nofill.json` |
| Otros checkpoints pasados por test (solo para medir la variación) | `TEST_e2_s43.json`, `TEST_f2_s42.json` |
| Pruebas de estrés | `estres2_*.json`, `estres_d5_rot60.json`, `estres_d5_rot180.json` |
| Análisis de error en test | `informe_h6_test/` |
| SAM zero-shot | `sam_baseline_test_h6.json`, `sam_baseline_val_h6.json`, `sam_baseline_val.json` |
| Latencia | `latencia_gpu_L4.json`, `latencia_cpu_colab.json` |

Las cifras de test, las ablaciones, la distancia en mm, SAM y la latencia están en `MODEL_CARD.md` (secciones 4.1 a 4.9); los comandos, en `README.md` (sección 3).

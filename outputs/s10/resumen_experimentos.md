# Semana 10 — resumen de experimentos (5 de octubre de 2026, 18:55 UTC)

Validación, 15 casos completos (16 876 cortes), TTA espejo, métricas sobre el volumen. Objetivos: F1/AUC 0.85; IoU de detección y mAP50 0.65; mAP50:95 0.40; Dice de segmentación 0.85; Dice de instancia 0.85 e IoU 0.70.

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
| E2 s43 | candidato + backbone Intel, semilla 43 | 0.966 | 0.901 | 0.969 | 0.856 | 0.717 | Con s42 secundarios +0.011 y con s43 −0.007 vs EMA solo: **Intel no da ganancia consistente** (neutral) |
| D6 | c2 + sobremuestreo de secundarios ×3 | 0.960 | 0.892 | 0.969 | 0.837 | 0.642 | Empeora instancias y secundarios; descartado |
| E1 | c2 + backbone preentrenado en Intel (transfer learning) | 0.959 | 0.887 | 0.969 | 0.847 | 0.683 | Igual que c2 en segmentación; la detección cruda sube (mAP50 0.905 vs 0.875, mAP50:95 0.498 vs 0.474) |

Hallazgos:

- Ni 2.5D, ni pérdida focal, ni Tversky (D1–D3) mueven el Dice de instancia (0.846–0.848 frente a 0.849 de c2) ni los secundarios; el techo parece estar en el post-proceso o en la ambigüedad de las etiquetas secundarias, no en la red. Sube la prioridad del post-proceso 3D.
- El ancho del backbone (hasta ahora 8 canales) era la limitación principal. Pasar a 32 sube detección, clasificación y por primera vez los secundarios.
- Lo que no ayuda: más peso a secundarios (sec6), peso por fractura, instancia con IDs, CBAM en tres etapas.
- Detección cruda: mAP50 0.875 y mAP50:95 0.474 en c2; ya cumple el objetivo sin refinar.

## 2. Candidato y estado

**Candidato: c2 + EMA 0.999** (w32, CBAM 1, 60 épocas, secundarios ×3, λ por defecto, sin preentrenamiento).
En validación (3 semillas, media): instancias ≈ 0.853, secundarios ≈ 0.712 (c2: 0.850 / 0.687).

Ablaciones obligatorias ya medidas sobre el candidato o sobre c2:

| Ablación | Resultado |
|---|---|
| CBAM (con/sin) | Aporta en clasificación (F1 +0.013) y detección refinada (mAP50:95 +0.011); segmentación igual |
| Transfer learning (Intel, con/sin) | Sin ganancia consistente en segmentación (secundarios +0.011 y −0.007 con 2 semillas); detección cruda mixta |
| λ (×2 instancia/borde) | Sin ganancia; empeora la detección cruda |

Post-proceso 3D (validación, 15 casos, TTA). **Corrección:** la versión original rellenaba los huecos entre fragmentos y dejaba todas las distancias predichas en 0 mm (el 1.15 mm de error era el de predecir 0 siempre; no vale como resultado). Con `--distance-nofill` y D5 s42: Dice 3D principal 0.842 IoU / todos 0.771 Dice; 38 de 41 secundarios con distancia predicha; **error medio 0.518 mm** (mediana 0.0, p90 0.388), **correlación 0.925**, ASSD 5.11 mm, HD95 22.81 mm. Solo 5 de 41 fragmentos están separados en el GT, así que la cifra es indicativa. Detalle en `d5_nofill_val15.json` y en `docs/decisiones_semana10.md` sección 5.

Siguientes: asignación multi-celda para detección (no implementada), post-proceso 3D con el candidato EMA, resolución 256/512, evaluación en test una sola vez, SAM zero-shot, latencia, dashboard y demo.

## 3. Pendiente

Con código nuevo:
- E3 Post-proceso 3D (componentes conectados, primario = el mayor, voto entre cortes) y distancias en mm contra GT, con HD95 y ASSD. Sin GPU; se puede hacer en el Mac ya con c2.
- E2 Segundo pase por región recortada (dos etapas del enunciado); `--two-pass` no está en `train_s10.py`.
- A6 Contexto 2.5D (cortes vecinos); `--context` tampoco está.
- E4 Pérdida focal o Tversky para secundarios.
- C2 Preentrenar el backbone en Intel Image Classification (ablación obligatoria con y sin transfer learning).
- F4 Promedio móvil de pesos.
- D2–D3 Detección: varias celdas positivas por región y pérdida IoU.

Solo con banderas:
- A7 Resolución 256 y 512.
- F1 Calibrar λ en serio (barrido más allá del primer ensayo).
- F3 Calibrar el umbral de clasificación por clase.

Obligatorio del informe y del proyecto:
- Ablaciones sobre el candidato final: con y sin CBAM, con y sin transfer learning, λ.
- Evaluación en test, una sola vez, con todo congelado.
- Latencia por imagen en CPU y GPU; el ancho 32 aumenta el costo y no está medida.
- SAM ViT-B zero-shot con la caja predicha como prompt, sobre test.
- Dashboard con tres visualizadores y demo en vivo; cinco peores casos, matriz de confusión.
- Notebook `04_semana10_pipeline_multitarea.ipynb`; `IA_USAGE.md`; `docs/plan_semana10.md`.

Preguntas abiertas al equipo: pesos de FundidoraPC (C1), acceso al dataset de Kaggle con los cachés, reparto de frentes, fechas del avance y la entrega.

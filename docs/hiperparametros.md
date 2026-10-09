# Por qué cada número: hiperparámetros y umbrales de H6

Este documento responde una sola pregunta: **por qué se eligió cada número del modelo final (H6) y
no otro**. No repite resultados. Las cifras completas están en `MODEL_CARD.md` y la historia de cada
experimento en `docs/decisiones_semana10.md`. Aquí cada valor tiene una fila con su origen.

Cada valor tiene uno de cuatro orígenes:

- **Experimento propio**: lo decidió una comparación nuestra, con el archivo de evidencia en el repo.
- **Literatura**: es el valor del artículo original o el estándar de la técnica. No lo barrimos.
- **Cómputo o protocolo**: lo fija el tiempo de GPU disponible o una regla que nos pusimos.
- **Sin barrer**: se fijó al inicio y no se volvió a probar. Lo decimos tal cual y, si el profesor
  lo pide, es lo primero que se barrería (ver la última sección).

Regla de decisión de todo el proyecto: un cambio se acepta solo si mejora el Dice por fragmento en
validación más que el ruido entre semillas (±0.004 en instancias, ±0.012 en secundarios, medido con
las semillas 42, 43 y 44). El test nunca se usa para elegir.

## 1. Entrada

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| Canales de entrada | 1 (un corte en grises) | Experimento propio | El contexto 2.5D (3 cortes como canales, D1) dio 0.848 contra 0.849 de C2: no se separa del ruido y triplica la entrada |
| Ventana ósea | centro 400 HU, ancho 1800 HU (−500 a 1300 HU) | Diseño | Cubre hueso cortical y trabecular sin saturar. Por debajo de −500 HU solo hay aire y grasa |
| Corrección gamma de la imagen | 1.5 | Experimento propio | Barrido de 166 combinaciones (`experimento_normalizacion.py`, 5 casos): CNR 8.11 contra 6.62 sin gamma y 3.23 con CLAHE. Gamma más alto sube el CNR pero apaga el interior del hueso (`outputs/eda/comparacion_gamma.png`). No confundir con el gamma del CBAM (sección 3) |
| Tamaño | 384 × 384 | Diseño, sin barrer | El curso recomienda 224-256. Se subió a 384 porque hay fragmentos pequeños que a 256 ocupan muy pocos píxeles. No se midió por separado. 512 se planteó pero no se corrió porque se acabó la cuota de GPU de Kaggle |
| Proporción | Estirar (`stretch`), no rellenar | Experimento propio | Rellenar para conservar proporción (H2) dio +0.006, en el borde del ruido, y no es comparable píxel a píxel |
| Orientación | Canónica (34 de 100 volúmenes se giran 180°) | Experimento propio | H4 (solo canónica) +0.010 en instancias y +0.039 en secundarios. Sin ella la red aprendía una regla espuria (el caso 044 cae de 0.916 a 0.778 al girarlo) |

## 2. Arquitectura

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| Ancho inicial del backbone | 32 canales (32 → 64 → 128 → 256) | Experimento propio | Dice por fragmento con 8, 16 y 32 canales: 0.829, 0.839 y **0.850** (media de 3 semillas de C2). Con 48 (W48): 0.844. El ancho se satura en 32. Los 32 son el ancho de la primera capa: el mapa más profundo tiene 256 canales |
| Bloques del backbone | 4, cada uno reduce a la mitad (salida 1/16) | Diseño | Con 384 px el mapa final es 24 × 24. Cada celda ve un campo receptivo que abarca una región ósea completa, que es lo que necesitan la clasificación y la grilla de detección |
| Bloque convolucional | 2 conv 3 × 3 + BatchNorm + SiLU | Literatura | El bloque estándar tipo VGG/U-Net. SiLU en vez de ReLU por ser suave (sin neuronas muertas) |
| Parámetros totales | 2.74 M | Consecuencia | Resultado del ancho 32. Pocos parámetros para pocos pacientes (70 de entrenamiento) |
| CBAM: dónde | Solo al final del backbone (1 bloque) | Experimento propio | Ablación B0: sin CBAM mAP50 0.900, CBAM en 1 bloque 0.926, CBAM en 3 bloques con gamma 0.907 |
| CBAM: reducción del MLP de canal | 16 (256 → 16 → 256) | Literatura | Valor del artículo original de CBAM (Woo et al., 2018) |
| CBAM: kernel de atención espacial | 7 × 7 | Literatura | Valor del artículo original de CBAM; los autores lo comparan contra 3 × 3 y gana 7 × 7 |
| Decodificador | 4 bloques de subida (128, 64, 32, 32) con conexiones de salto aditivas | Diseño | Recupera la resolución completa con detalle de 1/8, 1/4 y 1/2. Las conexiones aditivas suman detalle sin duplicar canales como la concatenación de U-Net |
| Salida de instancia | 7 clases (fondo + principal/secundario por hueso) | Experimento propio | Predecir los 31 IDs directamente dio 0.787 en instancias y 0.460 en secundarios, contra 0.829 y 0.645. Es la formulación del challenge PENGWIN: el ID de un secundario es arbitrario entre pacientes |
| Cabeza de clasificación | promedio global → 128 → 3, dropout 0.1 | Diseño, sin barrer | Multietiqueta (puede haber varios huesos en un corte). Dropout bajo porque la tarea ya da F1 0.99 |
| Cabeza de detección | grilla 24 × 24, 8 salidas por celda (caja + objetividad + 3 clases) | Diseño | Una predicción por celda al estilo YOLO, sin anclas: en un corte hay pocos objetos, en general uno por hueso |

## 3. El gamma del CBAM (no es el gamma 1.5 de la imagen)

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| Fórmula | `salida = x + γ · (CBAM(x) − x)` | Pedido del profesor (asesoría semana 9) | γ = 0 deja pasar la entrada sin atención, γ = 1 es CBAM completo. Es la mezcla aprendible usada en SAGAN (Zhang et al., 2019) |
| γ inicial | 0 | Literatura | La red empieza sin atención y solo la incorpora si le sirve |
| En H6 | No se usa (CBAM en 1 bloque, sin γ) | Experimento propio | Ver la fila de CBAM. La corrida `g1` mide el γ final de cada bloque con la receta de H6 (sección 7) |

Valores finales medidos hasta ahora:

| Corrida | γ bloque 1/4 | γ bloque 1/8 | γ bloque 1/16 |
|---|---|---|---|
| Detección sola, semana 9, 35 épocas (`outputs/week9_det_cbam3g.json`) | 0.089 | −0.106 | 0.108 |
| Multitarea B0, semana 10, 40 épocas (`outputs/s10/b0_cbam3g_s42.pth`) | −0.630 | 0.716 | −1.319 |

En detección, el γ queda cerca de 0.1 en los tres bloques: la red usa poco la atención. En el
multitarea, el γ sale del rango [0, 1]: con tres tareas sobre el mismo backbone, esa perilla deja de
leerse como "cuánto aporta el CBAM". La corrida `g1` lo repite con la receta final.

## 4. Pérdida (pesos λ)

| Término | Peso | Origen | Por qué |
|---|---|---|---|
| Detección | 1.0 | Experimento propio | λ×2 (F2: instancia/borde ×2, detección ×0.5) no mejoró las instancias (ruido) y bajó la detección cruda (mAP50 0.827 contra 0.868). Se quedan los λ por defecto |
| Clasificación | 0.5 | Medida del entrenamiento | Es la tarea que converge primero: en la primera época el F1 ya es 0.907, cuando el mAP50 va en 0.287 y el Dice por fragmento en 0.529. Con más peso competiría con tareas que aún no aprenden. No se barrió |
| Semántica (entropía cruzada) | 2.0 | Diseño, sin barrer | Es la primera de las dos etapas: un error de región arrastra a todos sus fragmentos. También refina las cajas de detección |
| Dice semántico, instancia, borde, centros, desplazamientos, Dice de instancia | 1.0 cada uno | Sin barrer | Peso neutro. F2 movió instancia y borde a ×2 sin ganancia |
| Peso del fondo en la entropía cruzada de instancia | 0.1 | Sin barrer | El fondo ocupa casi todos los píxeles del corte. Con peso 1 dominaría la pérdida de instancia |
| Peso de la clase "secundario" | 3 | Experimento propio | Los secundarios son cerca de la mitad de los fragmentos pero cerca de una décima parte del hueso. ×6 (B2) no superó el ruido y empeoró la detección; el sobremuestreo ×3 (D6) empeoró |
| Pérdida focal | apagada | Experimento propio | γ = 2 (D2): 0.846, descartada |
| Tversky | apagada | Experimento propio | β = 0.7 (D3): 0.846, descartada |
| Supervisión profunda | apagada | Experimento propio | 0.4 (D4): 0.846, descartada |

## 5. Entrenamiento

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| Optimizador | AdamW, weight decay 1e-4 | Literatura, sin barrer | Estándar para redes convolucionales con BatchNorm (Loshchilov y Hutter, 2019) |
| Tasa de aprendizaje | 2e-3 | Sin barrer | Se fijó en la semana 9 y el entrenamiento converge sin divergir. No se hizo barrido |
| Programación de la tasa | Coseno hasta lr/50 | Literatura | Baja la tasa de forma suave al final, que es cuando el EMA y la selección del mejor punto aprovechan pesos estables |
| Lote | 16 | Sin barrer | Se fijó al inicio de la semana 10 y entra en la T4 de Kaggle a 384 px con precisión mixta |
| Épocas | 60 de 2500 cortes con hueso | Cómputo y experimento | Unos 65-70 min por corrida en una T4. D5 empezaba a bajar desde la época 40; con el aumento geométrico la curva se aplana hacia la 50 (H5) |
| Cortes sin hueso | 0.15 por cada corte con hueso, margen de 10 cortes | Diseño, sin barrer | La red debe ver negativos para no inventar hueso en los extremos del volumen, sin que dominen |
| EMA | decaimiento 0.999 | Experimento propio | D5 frente a C2, con 3 semillas: secundarios +0.016, +0.048, +0.009 (media +0.024, el doble del ruido de ±0.012), instancias sin empeorar. Con 2500 cortes y lote 16 hay ~156 pasos por época; 0.999 promedia unos 1000 pasos (~6 épocas) |
| Alcance del EMA | toda la red en H6 | A revisar | El profesor lo autorizó para la segmentación. La ganancia medida es en segmentación (secundarios). La corrida `e1` lo limita a la cabeza de segmentación para comparar |
| Semillas | 42, 43, 44 | Protocolo | Tres semillas para medir el ruido y no confundir azar con mejora |
| Precisión mixta | fp16 automática en GPU | Cómputo | Acelera el entrenamiento en la T4 y reduce la memoria. Los pesos se guardan en fp32 |

## 6. Aumento de datos

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| Espejo izquierda-derecha | 50 % (intercambia las etiquetas de coxal) | Diseño | La pelvis es casi simétrica; duplica los ejemplos sin inventar anatomía |
| Brillo y contraste | ganancia ±5 %, sesgo ±2.5 % | Diseño | Variación leve entre tomógrafos |
| Aumento geométrico | fuerza 1.0 (escala, proporción, traslación) | Experimento propio | H1: +0.013 en instancias, +0.040 en secundarios, confirmado con 2 semillas. Es el primer cambio que superó el ruido con claridad |
| Rotación | ±35° | Experimento propio | Con validación girada 30°: sin rotación 0.559, ±12° 0.807, **±35° 0.854**, sin costo en validación normal. Regla fijada antes de ver el resultado de H6 |
| Baja resolución, gamma aleatorio | apagados | Experimento propio | H3 los incluía. No ayudaron con el estrés de proporción y resolución (0.759 contra 0.797) |

## 7. Inferencia y post-proceso

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| TTA | espejo izquierda-derecha | Experimento propio | Promedia dos lecturas del mismo corte. El ensamble de 3 semillas solo sumó +0.003 (dentro del ruido), así que no se usa |
| Confianza mínima de detección | 0.10 | Sin barrer | En un corte hay pocos objetos. Un umbral bajo, seguido de la NMS, evita perder huesos poco visibles en los extremos del volumen |
| IoU de la NMS | 0.5 | Literatura | Estándar de detección (PASCAL VOC) |
| Tamaño mínimo de fragmento en 2D | 12 px | Experimento propio | 1 px subía 0.002 pero casi duplicaba los fragmentos sobrantes (que la métrica no castiga). Adoptarlo sería ajustarse a un punto ciego de la métrica. 30 px bajaba 0.006 |
| Tamaño mínimo de fragmento en 3D | 30 vóxeles | Diseño | Quita componentes de ruido antes de medir distancias. Los menores se absorben en el principal, así no se pierde hueso |
| Tamaño mínimo de región 3D | 200 vóxeles | Diseño | Descarta un hueso "detectado" en unos pocos vóxeles aislados |
| Fragmentos por hueso | hasta 9 secundarios | Formato PENGWIN | Los IDs van de 1 a 10 por hueso |
| Distancia | borde a borde, EDT en 3D con el espaciado de la cabecera, sin rellenar huecos | Experimento propio | Rellenar huecos tapaba la separación que se quiere medir (sección 5 de decisiones) |

## 8. Selección del mejor punto y protocolo

| Parámetro | Valor | Origen | Por qué |
|---|---|---|---|
| Criterio de mejor época | Dice semántico + mAP50 + Dice de instancias (validación) | Diseño | Una suma de las tres tareas. Ninguna se sacrifica para mejorar otra |
| Evaluación durante el entrenamiento | cada 10 épocas, 1 de cada 6 cortes de validación | Cómputo | Evaluar todo el volumen en cada época doblaría el tiempo |
| Umbral para aceptar una mejora | supera el ruido entre semillas (±0.004 instancias, ±0.012 secundarios) | Protocolo | Medido con 3 semillas de C2 |

## 9. Lo que no se barrió y cómo se haría

Si el profesor pide evidencia de los valores marcados **Sin barrer**, el orden y el costo son estos
(cada corrida de 60 épocas tarda unos 65-70 min de GPU):

1. **Tasa de aprendizaje**: 1e-3, 2e-3 (actual) y 4e-3, con la receta de H6 y semilla 42. Son 2
   corridas nuevas. Es el valor que más influye y el único sin evidencia.
2. **Pesos λ de semántica, clasificación y fondo**: F2 ya mostró que mover λ no ayuda. Una corrida
   con todos en 1.0 confirmaría que el 2.0, el 0.5 y el 0.1 no son decisivos.
3. **Tamaño 256 frente a 384**: una corrida con la receta de H6 a 256 px. El caché de 256 ya existe.
4. **Weight decay y lote**: la literatura y el límite de memoria los fijan. Barrerlos tiene menos
   prioridad.

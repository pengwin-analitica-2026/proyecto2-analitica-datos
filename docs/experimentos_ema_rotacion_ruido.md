# Experimentos tras la asesoría: EMA, rotación y ruido

Proyecto PENGWIN, corte 2. Octubre de 2026.

## Resumen

En la asesoría el profesor pidió dos cosas: no depender del EMA, porque puede fallar con imágenes
"feas", y hacer las rotaciones de forma normalizada, con los metadatos del volumen. Para responder
con medidas se entrenaron cinco modelos que solo difieren en tres opciones (EMA, tipo de rotación y
ruido en el entrenamiento) y se evaluaron todos igual, sobre validación.

Conclusión: **el modelo elegido es H8** (sin EMA, rotación en milímetros y aumento de ruido). En
imágenes limpias pierde 0.008 de Dice por fragmento frente a H7, y a cambio es el único que resiste
el ruido: con ruido fuerte da 0.835, donde los demás quedan entre 0.09 y 0.23. Cumple los nueve
objetivos del curso en validación y en test (Dice por fragmento de 0.874 en test, sección 5).
El documento incluye además el aporte de CBAM por cabeza (sección 6) y las métricas HD95 y ASSD por
fragmento (sección 7).

## 1. Qué se comparó y cómo

Los cinco modelos comparten toda la receta: backbone propio de 32 canales con CBAM de una etapa,
60 épocas de 2 500 cortes, lote 16, tasa de aprendizaje 0.002 con coseno, peso ×3 a los fragmentos
secundarios, orientación canónica según la cabecera, aumento geométrico con rotación de ±35° y
semilla 42. Cada uno entrenó entre 67 y 69 minutos en una GPU T4 de Kaggle y completó las 60 épocas.

| Modelo | EMA | Rotación del aumento | Ruido al entrenar | Origen |
|---|---|---|---|---|
| H6 | 0.999 | en píxeles | no | Modelo entregado en `main` |
| H6 sin EMA | no | en píxeles | no | Nuevo |
| H7 | no | en milímetros | no | Rama `feat/normalizacion-orientacion` |
| H7 con EMA | 0.999 | en milímetros | no | Nuevo |
| H8 | no | en milímetros | sí | Nuevo |

Los cuatro primeros forman un diseño 2 × 2 (EMA sí o no, rotación en píxeles o en milímetros), que
permite medir cada opción por separado. H8 es H7 más el aumento de ruido.

### Qué significa cada opción

- **EMA.** Durante el entrenamiento se guarda un promedio móvil de los pesos y ese promedio es el
  modelo que se evalúa. Sin EMA se usan los pesos de la última época con mejor validación.
- **Rotación en píxeles o en milímetros.** El caché estira cada corte a 384 × 384 aunque el original
  no sea cuadrado, así que un píxel puede medir, por ejemplo, 0.9 mm de ancho y 1.6 mm de alto. Girar
  en esa grilla no es un giro real: deforma la pelvis. La rotación en milímetros (`--aug-mm`) aplica
  el giro en el espacio físico, usando el espaciado de la cabecera, y luego lo lleva a píxeles.
- **Ruido al entrenar** (`--aug-noise 0.1`). En el 30 % de las muestras se suma ruido gaussiano con
  una desviación al azar de hasta 0.1 en la escala 0-1 (hasta unos 25 niveles de gris sobre 255).

### Cómo se evaluó

Las mismas cinco pruebas para los cinco modelos, sobre los 15 casos de validación completos
(5 047 cortes), con TTA de espejo:

| Prueba | Qué hace |
|---|---|
| Sin daño | Validación normal |
| Ruido 10 | Ruido gaussiano de desviación 10 (sobre 255) en la imagen de entrada |
| Ruido 25 | Lo mismo con desviación 25 |
| Giro 60° | Cada volumen girado 60° en milímetros antes de entrar al modelo |
| Giro 60° + enderezado | El mismo giro, corregido por el paso de normalización antes del modelo |

**En esta comparación no se usó test.** El caché de estos entrenamientos solo tiene entrenamiento y
validación. Test se evaluó después, una sola vez y solo con el modelo elegido (sección 5).

## 2. Resultados

### 2.1 Dice por fragmento

Es la métrica más exigente del proyecto (objetivo 0.85).

| Modelo | Sin daño | Ruido 10 | Ruido 25 | Giro 60° | Giro 60° + enderezado |
|---|---|---|---|---|---|
| H6 | 0.869 | 0.690 | 0.135 | 0.608 | 0.851 |
| H6 sin EMA | 0.872 | 0.649 | 0.090 | 0.570 | 0.855 |
| H7 | **0.873** | 0.654 | 0.228 | **0.636** | **0.855** |
| H7 con EMA | 0.871 | 0.720 | 0.208 | 0.635 | 0.853 |
| **H8** | 0.865 | **0.860** | **0.835** | 0.612 | 0.849 |

### 2.2 Las demás métricas, sin daño

| Modelo | F1 | AUC | IoU det. | mAP50 | mAP50:95 | Dice seg. | IoU seg. | Dice frag. | IoU frag. | Dice secundarios |
|---|---|---|---|---|---|---|---|---|---|---|
| H6 | 0.992 | 0.999 | 0.949 | 0.971 | 0.904 | 0.971 | 0.944 | 0.869 | 0.808 | 0.764 |
| H6 sin EMA | 0.991 | 0.999 | 0.947 | 0.974 | 0.903 | 0.971 | 0.944 | 0.872 | 0.811 | 0.771 |
| H7 | 0.992 | 0.999 | 0.948 | 0.974 | 0.903 | 0.971 | 0.944 | 0.873 | 0.812 | 0.774 |
| H7 con EMA | 0.990 | 0.999 | 0.946 | 0.969 | 0.898 | 0.970 | 0.943 | 0.871 | 0.808 | 0.780 |
| H8 | 0.991 | 0.999 | 0.946 | 0.969 | 0.900 | 0.970 | 0.943 | 0.865 | 0.800 | 0.769 |
| Objetivo | 0.85 | 0.85 | 0.65 | 0.65 | 0.40 | 0.85 | 0.70 | 0.85 | 0.70 | — |

Los cinco modelos cumplen los nueve objetivos en validación.

### 2.3 Con ruido: detección y segmentación

| Modelo | F1 (ruido 10) | mAP50 (ruido 10) | Dice seg. (ruido 10) | F1 (ruido 25) | mAP50 (ruido 25) | Dice seg. (ruido 25) |
|---|---|---|---|---|---|---|
| H6 | 0.881 | 0.690 | 0.889 | 0.691 | 0.188 | 0.249 |
| H6 sin EMA | 0.883 | 0.609 | 0.821 | 0.857 | 0.096 | 0.177 |
| H7 | 0.879 | 0.730 | 0.825 | 0.517 | 0.224 | 0.383 |
| H7 con EMA | 0.886 | 0.805 | 0.887 | 0.401 | 0.200 | 0.364 |
| **H8** | **0.990** | **0.968** | **0.967** | **0.987** | **0.957** | **0.955** |

### 2.4 Con giro de 60°

| Modelo | mAP50 girado | Dice seg. girado | Dice frag. girado | mAP50 enderezado | Dice frag. enderezado |
|---|---|---|---|---|---|
| H6 | 0.311 | 0.715 | 0.608 | 0.963 | 0.851 |
| H6 sin EMA | 0.293 | 0.692 | 0.570 | 0.961 | 0.855 |
| H7 | 0.347 | 0.725 | 0.636 | 0.963 | 0.855 |
| H7 con EMA | 0.356 | 0.725 | 0.635 | 0.961 | 0.853 |
| H8 | 0.362 | 0.721 | 0.612 | 0.960 | 0.849 |

La tabla completa, con las doce métricas de cada corrida, está en `outputs/h8/comparacion/tabla_comparativa.csv`.

## 3. Análisis

Referencia para leer las diferencias: el ruido entre semillas medido en la Semana 10 es de ±0.004 en
Dice por fragmento. Aquí hay una sola semilla por modelo, así que una diferencia de ese tamaño no se
puede interpretar.

### 3.1 Efecto del EMA

Se compara cada pareja que solo difiere en el EMA.

| Rotación | Prueba | Con EMA | Sin EMA | Diferencia al quitarlo |
|---|---|---|---|---|
| Píxeles (H6 y H6 sin EMA) | Sin daño | 0.869 | 0.872 | +0.003 |
| | Ruido 10 | 0.690 | 0.649 | −0.041 |
| | Giro 60° | 0.608 | 0.570 | −0.038 |
| Milímetros (H7 con EMA y H7) | Sin daño | 0.871 | 0.873 | +0.002 |
| | Ruido 10 | 0.720 | 0.654 | −0.066 |
| | Giro 60° | 0.635 | 0.636 | +0.001 |

- En imágenes limpias, quitar el EMA no cambia nada (+0.002 y +0.003, dentro del ruido).
- Con ruido leve, el EMA ayudaba en las dos parejas (0.04 y 0.07). El riesgo que señaló el profesor
  no aparece en estas pruebas: con imágenes dañadas el modelo con EMA no se comporta peor.
- Con ruido fuerte todos los modelos sin aumento de ruido colapsan, con EMA o sin él (0.09 a 0.23).

Quitar el EMA es defendible porque no cuesta nada en condiciones normales, y lo que se pierde con
ruido leve lo recupera con creces el aumento de ruido de H8 (sección 3.3).

### 3.2 Efecto de rotar en milímetros

Se compara cada pareja que solo difiere en el tipo de rotación.

| EMA | Prueba | En píxeles | En milímetros | Diferencia |
|---|---|---|---|---|
| Sí (H6 y H7 con EMA) | Sin daño | 0.869 | 0.871 | +0.002 |
| | Giro 60° | 0.608 | 0.635 | +0.027 |
| | Ruido 25 | 0.135 | 0.208 | +0.073 |
| No (H6 sin EMA y H7) | Sin daño | 0.872 | 0.873 | +0.001 |
| | Giro 60° | 0.570 | 0.636 | +0.066 |
| | Ruido 25 | 0.090 | 0.228 | +0.138 |

- No cuesta nada en imágenes limpias.
- Mejora la tolerancia al giro en las dos parejas (+0.03 y +0.07). Es coherente con la causa: el
  modelo entrenado en píxeles veía pelvis deformadas, no giradas.
- Aun así, ningún modelo aguanta por sí solo un giro de 60° (0.57 a 0.64). Lo que resuelve el giro es
  el **enderezado**: con él los cinco modelos quedan entre 0.849 y 0.855, sin importar el EMA ni el
  tipo de rotación.

### 3.3 Efecto del aumento de ruido (H7 frente a H8)

| Prueba | H7 | H8 | Diferencia |
|---|---|---|---|
| Sin daño | 0.873 | 0.865 | −0.008 |
| Ruido 10 | 0.654 | 0.860 | +0.206 |
| Ruido 25 | 0.228 | 0.835 | +0.607 |
| Giro 60° | 0.636 | 0.612 | −0.024 |
| Giro 60° + enderezado | 0.855 | 0.849 | −0.006 |

- Con ruido, H8 mejora en los 15 casos de validación. Con ruido leve mantiene el objetivo (0.860) y
  con ruido fuerte queda en 0.835, mientras la detección y la clasificación casi no se mueven
  (mAP50 0.957 y F1 0.987 con ruido 25).
- El costo en imágenes limpias es 0.008, el doble del ruido entre semillas, y H8 queda por debajo de
  H7 en 12 de los 15 casos (la mayor baja en un caso es 0.019). Es un costo real, aunque pequeño.
- Comparado con la alternativa de la rama, suavizar la imagen antes del modelo: con H7, el suavizado
  recupera el ruido leve a 0.851 y el fuerte solo a 0.658, cuesta 0.014 en imágenes limpias si se
  aplica siempre, y obliga a tener un detector de ruido para decidir cuándo usarlo. H8 da más con
  ruido (0.860 y 0.835), cuesta menos en limpio (0.008) y no necesita detector.

## 4. Modelo elegido: H8

**H8** = receta de H7 (sin EMA, rotación en milímetros, orientación canónica) más aumento de ruido
gaussiano en el entrenamiento. Archivo: `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth`.

Justificación:

1. **Responde a las dos observaciones del profesor.** No usa EMA y el giro del entrenamiento es
   físico, en milímetros y a partir de los metadatos.
2. **Es el único robusto a imágenes dañadas.** Era la preocupación de fondo de la asesoría. Con
   ruido leve y fuerte da 0.860 y 0.835 de Dice por fragmento; el mejor de los otros da 0.720 y 0.228.
3. **Cumple los nueve objetivos del curso** en validación, sin daño (0.865) y con ruido leve (0.860).
4. **El costo es pequeño y está medido:** 0.008 en imágenes limpias frente a H7.
5. **Simplifica el pipeline.** No hace falta el suavizado previo ni un detector de ruido en la demo.

Si la prioridad fuera solo la cifra en imágenes limpias, el modelo sería H7 (0.873). Se prefiere H8
porque 0.008 en limpio vale menos que no colapsar cuando la imagen llega con ruido.

### Limitaciones

- **Una sola semilla por modelo.** Las diferencias pequeñas en imágenes limpias (±0.004) no son
  concluyentes. Las de ruido y giro son mucho mayores que ese margen.
- **Ruido y giro se midieron solo en validación.** En test se evaluó H8 una vez, sin pruebas de
  estrés (sección 5). Fue la sexta consulta a test del proyecto.
- **El ruido es sintético.** Se probó ruido gaussiano, el mismo tipo con que se entrenó. No se midió
  con artefactos reales de CT (metal, movimiento, dosis baja).
- **Giro de 60° con enderezado: 0.849**, una milésima por debajo del objetivo. El enderezado sigue
  siendo necesario para pacientes girados y solo corrige el giro en el plano axial.
- **Ruido fuerte: 0.835**, por debajo de 0.85. La demo debería avisar cuando la imagen es muy ruidosa.

## 5. H8 en test

Evaluado una sola vez el 13 de octubre, con la configuración escrita antes de correr
(`outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md`): un modelo con TTA de espejo, tamaño 384, sin enderezado ni
suavizado. Son los mismos 15 casos y 4 664 cortes de las evaluaciones anteriores.

| Métrica | Objetivo | H8 en test | H8 en validación |
|---|---|---|---|
| F1 de clasificación | 0.85 | 0.989 | 0.991 |
| AUC de clasificación | 0.85 | 0.999 | 0.999 |
| IoU de detección | 0.65 | 0.938 | 0.946 |
| mAP50 | 0.65 | 0.953 | 0.969 |
| mAP50:95 | 0.40 | 0.898 | 0.900 |
| Dice de segmentación | 0.85 | 0.968 | 0.970 |
| IoU de segmentación | 0.70 | 0.939 | 0.943 |
| **Dice por fragmento** | 0.85 | **0.874** | 0.865 |
| IoU por fragmento | 0.70 | 0.812 | 0.800 |

**H8 cumple los nueve objetivos en test.** Desglose del Dice por fragmento: principal 0.910,
secundario 0.760, con 7.2 % de secundarios sin pareja. La cabeza grid sola, sin refinar las cajas
con la máscara, da mAP50 0.909 y mAP50:95 0.566.

Peores casos: 039 (0.797), 054 (0.831), 061 (0.833), 007 (0.839) y 047 (0.841). Son los mismos
cinco que con H6. El 012, el paciente girado que hundía el test con D5, queda en 0.898.

Evaluaciones en test del proyecto, Dice por fragmento:

| Evaluación | Modelo | Dice por fragmento |
|---|---|---|
| 1.ª | D5 | 0.845 |
| 2.ª | H6 | 0.875 |
| 3.ª | H7 | 0.879 |
| 4.ª | **H8** | **0.874** |

Contando además las dos consultas anteriores a D5 (c2, 0.848, y EMA + CASR, 0.847), es la **sexta
consulta a test**. El resultado coincide con lo que se esperaba antes de mirar: cerca de 0.87 y algo
por debajo de H7. La diferencia con H7 (0.005) es la misma que en validación (0.008): el costo del
aumento de ruido en imágenes limpias se mantiene, y no se midió ruido sobre test.

**Cómo debe citarse.** Los casos 012 y 039 de test se usaron para diagnosticar fallos antes de H6,
así que ninguna cifra posterior a D5 es una estimación independiente. H8 se eligió solo con
validación y no se ajustó nada después de ver test.

## 6. Aporte de CBAM por cabeza

Se entrenó un modelo con la receta exacta de H8 pero sin CBAM (`--cbam none`): 60 épocas, semilla
42, 77 minutos. CBAM son 8 290 de los 2.74 millones de parámetros. Se evaluó igual que H8 en
validación y en test.

### 6.1 Validación

| Cabeza | Métrica | Con CBAM (H8) | Sin CBAM | Diferencia |
|---|---|---|---|---|
| Clasificación | F1 | 0.991 | 0.987 | +0.004 |
| | AUC | 0.999 | 0.999 | 0.000 |
| Detección | IoU | 0.946 | 0.943 | +0.003 |
| | mAP50 | 0.969 | 0.964 | +0.005 |
| | mAP50:95 | 0.900 | 0.895 | +0.005 |
| Segmentación | Dice semántico | 0.970 | 0.970 | +0.001 |
| | IoU semántico | 0.943 | 0.941 | +0.001 |
| | Dice por fragmento | 0.865 | 0.863 | +0.002 |
| | IoU por fragmento | 0.800 | 0.799 | +0.001 |

### 6.2 Test

| Cabeza | Métrica | Con CBAM (H8) | Sin CBAM | Diferencia |
|---|---|---|---|---|
| Clasificación | F1 | 0.989 | 0.981 | +0.007 |
| | AUC | 0.999 | 0.998 | +0.001 |
| Detección | IoU | 0.938 | 0.931 | +0.007 |
| | mAP50 | 0.953 | 0.943 | +0.011 |
| | mAP50:95 | 0.898 | 0.879 | +0.019 |
| Segmentación | Dice semántico | 0.968 | 0.970 | −0.002 |
| | IoU semántico | 0.939 | 0.942 | −0.003 |
| | Dice por fragmento | 0.874 | 0.874 | 0.000 |
| | IoU por fragmento | 0.812 | 0.811 | +0.001 |

### 6.3 Aporte por cabeza y global

Media de las diferencias de cada cabeza y de las nueve métricas (entre paréntesis, en porcentaje
relativo al modelo sin CBAM).

| | Validación | Test |
|---|---|---|
| Clasificación | +0.002 (+0.2 %) | +0.004 (+0.4 %) |
| Detección | +0.004 (+0.5 %) | +0.012 (+1.4 %) |
| Segmentación | +0.001 (+0.2 %) | −0.001 (−0.1 %) |
| **Global (nueve métricas)** | **+0.003 (+0.3 %)** | **+0.005 (+0.5 %)** |

- **CBAM aporta en detección y en clasificación, y no en segmentación.** Es el mismo patrón de la
  ablación de la Semana 10 sobre D5 (F1 0.988 contra 0.975, mAP50:95 0.883 contra 0.872, Dice por
  fragmento igual), ahora repetido sobre H8 y en las dos particiones.
- La mayor diferencia es el mAP50:95 en test (+0.019): las cajas quedan mejor ajustadas. El F1 de
  clasificación es mejor con CBAM en 11 de 15 casos de validación y en 13 de 15 de test.
- En segmentación la diferencia es nula o cambia de signo entre particiones (Dice por fragmento
  +0.002 y 0.000): no hay aporte medible.
- El aporte global es pequeño, de +0.3 % a +0.5 %, con un costo de 8 290 parámetros.
- Con una sola semilla, las diferencias de menos de 0.004 no se pueden separar del ruido. Solo las
  de detección en test superan ese margen con claridad.

### 6.4 CBAM apagado al predecir (dependencia, no aporte)

Aparte, se tomó H8 ya entrenado y se reemplazó su CBAM por la identidad en la inferencia
(`--sin-cbam`).

| Métrica | H8 | H8 con CBAM apagado (validación) | H8 con CBAM apagado (test) |
|---|---|---|---|
| F1 de clasificación | 0.991 / 0.989 | 0.392 | 0.468 |
| mAP50 | 0.969 / 0.953 | 0.340 | 0.390 |
| Dice de segmentación | 0.970 / 0.968 | 0.679 | 0.725 |
| Dice por fragmento | 0.865 / 0.874 | 0.459 | 0.481 |

Las caídas son enormes porque las cabezas de H8 aprendieron a leer las características ya
ponderadas por la atención; quitársela de golpe las descuadra. Esto mide cuánto **depende** de CBAM
un modelo entrenado con él, no cuánto **gana** por tenerlo. La cifra que responde a "¿qué aporta
CBAM?" es la de las secciones 6.1 a 6.3.

## 7. HD95 y ASSD por fragmento

Métricas de superficie del reto PENGWIN, en milímetros, calculadas con `src/metricas_superficie.py`
sobre el volumen 3D completo de cada caso.

- **Cómo se arma la predicción.** Se predicen todos los cortes y el post-proceso 3D une las máscaras
  en un volumen de IDs de fragmento.
- **Emparejamiento.** Cada fragmento real se empareja con el predicho de mayor IoU 3D de su misma
  región. Un fragmento predicho se usa una sola vez; se asignan primero los reales más grandes.
- **No detectado.** Un fragmento real sin ningún predicho que lo toque en su región.
- **Distancias.** HD95 es el percentil 95 de las distancias entre las dos superficies, en ambos
  sentidos; ASSD es la media simétrica. Se usa `distance_transform_edt` con el espaciado de la
  cabecera, ajustado a la resolución del caché.
- **Cobertura.** Fracción de fragmentos reales detectados. Se da también con IoU de al menos 0.5.

### 7.1 Resultados de H8

| Partición | Grupo | Fragmentos | Detectados | Cobertura | Cobertura IoU ≥ 0.5 | HD95 media | HD95 mediana | ASSD media | ASSD mediana |
|---|---|---|---|---|---|---|---|---|---|
| Validación | Todos | 86 | 83 | 96.5 % | 84.9 % | 12.29 | 4.05 | 2.40 | 0.86 |
| | Principales | 45 | 45 | 100 % | 100 % | 7.05 | 2.31 | 1.11 | 0.50 |
| | Secundarios | 41 | 38 | 92.7 % | 68.3 % | 18.50 | 7.72 | 3.93 | 1.45 |
| Test | Todos | 86 | 82 | 95.3 % | 88.4 % | 9.40 | 2.73 | 1.53 | 0.67 |
| | Principales | 45 | 45 | 100 % | 100 % | 7.23 | 2.01 | 1.13 | 0.51 |
| | Secundarios | 41 | 37 | 90.2 % | 75.6 % | 12.03 | 8.09 | 2.02 | 1.22 |

Distancias en mm, sobre los fragmentos detectados.

Por región:

| Partición | Región | Fragmentos | Cobertura | HD95 mediana | ASSD mediana |
|---|---|---|---|---|---|
| Validación | Sacro | 25 | 100 % | 4.72 | 0.90 |
| | Coxal izquierdo | 31 | 100 % | 2.36 | 0.50 |
| | Coxal derecho | 30 | 90.0 % | 2.44 | 0.50 |
| Test | Sacro | 22 | 100 % | 2.78 | 0.61 |
| | Coxal izquierdo | 34 | 97.1 % | 3.73 | 0.90 |
| | Coxal derecho | 30 | 90.0 % | 2.09 | 0.56 |

Percentil 90 del HD95: 40.3 mm en validación y 23.9 mm en test.

### 7.2 Fragmentos no detectados

| Partición | Caso | Fragmento | Región |
|---|---|---|---|
| Validación | 014 | 24 | Coxal derecho |
| Validación | 024 | 23 | Coxal derecho |
| Validación | 060 | 23 | Coxal derecho |
| Test | 007 | 13 | Coxal izquierdo |
| Test | 039 | 23 y 24 | Coxal derecho |
| Test | 054 | 23 | Coxal derecho |

Todos son secundarios, y seis de los siete están en el coxal derecho. Los 90 fragmentos principales
de las dos particiones se detectan.

### 7.3 Lectura

- **Los fragmentos principales se localizan bien:** se detectan todos, con una mediana de 2 mm de
  HD95 y 0.5 mm de ASSD, del orden de uno a tres vóxeles.
- **Los secundarios son el punto débil,** igual que en el Dice: mediana de HD95 cercana a 8 mm y uno
  de cada diez sin detectar.
- **La media es mucho mayor que la mediana** porque unos pocos fragmentos tienen errores muy
  grandes. El mayor en test es el principal del coxal derecho del caso 089: 102.8 mm de HD95 con un
  IoU de 0.84. El fragmento está bien segmentado, pero la predicción le suma un componente pequeño
  lejos del hueso, y el HD95 castiga ese trozo aislado. Por eso la mediana es la cifra que mejor
  resume el comportamiento típico.
- **Fragmentos predichos de más:** 297 en validación y 258 en test no se emparejan con ningún
  fragmento real. Son componentes pequeños que el post-proceso deja como secundarios. Ni el Dice por
  fragmento ni estas métricas los penalizan; es una limitación conocida del modelo.

## 8. Siguientes pasos

1. Entrenar H8 con las semillas 43 y 44 para medir su variación (unas 2.3 horas de GPU cada una).
2. Probar H8 con desenfoque y baja resolución, las otras dos pruebas de robustez de la rama.
3. Conectar la normalización de orientación y el enderezado a la demo y a la API, que hoy siguen con
   la regla anterior.
4. Reducir los fragmentos predichos de más y los componentes aislados lejos del hueso, que explican
   los valores extremos de HD95. Debe decidirse con validación.
5. Actualizar el model card, el README y el informe con el modelo elegido.

## 9. Archivos y reproducción

| Qué | Dónde |
|---|---|
| Pesos de H8 | `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth` |
| Historial de entrenamiento de H8 | `outputs/h8/h8_aumm_noise01_noema_w32_e60_s42.json` |
| Validación de H8 (sin daño, ruido y giro) | `outputs/h8/validacion/val_h8_*.json` |
| Test de H8 y su configuración congelada | `outputs/h8/test/` |
| Modelo sin CBAM y CBAM apagado | `outputs/h8/cbam/` |
| HD95 y ASSD | `outputs/h8/{validacion,test}/superficie_*_h8.json` |
| H6 sin EMA y H7 con EMA | `outputs/h8/comparacion/h6noema/`, `outputs/h8/comparacion/h7ema/` |
| Resultados de H6 y H7 | `outputs/normalizacion/` y `outputs/h7/` |
| Tabla con todas las métricas de la comparación | `outputs/h8/comparacion/tabla_comparativa.csv` |
| Kernels de Kaggle | `kaggle/h8/` (`armar.py` los genera a partir de las plantillas) |
| Notebook con los resultados | `notebooks/11_modelo_h8.ipynb` |

Los pesos de H6 sin EMA, H7 con EMA y el modelo sin CBAM no se versionan; solo sus resultados.

El caché se construyó con `src/build_case_cache.py --sizes 384 --modes stretch` sobre los 85 casos de
entrenamiento y validación. Los 15 de validación son idénticos, valor por valor, a los que se usaron
para medir H6 y H7, y con ese caché se reprodujo una cifra de H7 con cinco decimales (caso 024, giro
de 60° con enderezado: 0.8313).

Comando de entrenamiento. Lo que cambia entre modelos es el final:

```bash
python src/train_s10.py --cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 \
    --batch-size 16 --lr 0.002 --base-channels 32 --cbam 1 --instance-mode primary --augment --cosine \
    --margin 10 --neg-fraction 0.15 --per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 \
    --secondary-weight 3.0 --canonical --aug-geo 1.0 --aug-rot-deg 35 --seed 42 \
    <opciones del modelo> --output <archivo.pth>
```

| Modelo | Opciones del modelo |
|---|---|
| H6 | `--ema 0.999` |
| H6 sin EMA | `--ema 0` |
| H7 | `--aug-mm --ema 0` |
| H7 con EMA | `--aug-mm --ema 0.999` |
| H8 | `--aug-mm --ema 0 --aug-noise 0.1` |

Evaluación de cada prueba:

```bash
python src/evaluate_volumes.py --checkpoint <archivo.pth> --cache-dir data/cache/s10/cache_384 \
    --split val --tta [--stress noise=10 | --stress noise=25 | --stress rotmm=60 | --stress rotmm=60 --normalizar]
```

HD95 y ASSD por fragmento, y el modelo sin CBAM:

```bash
python src/metricas_superficie.py --checkpoint <archivo.pth> --cache-dir data/cache/s10/cache_384     --split val --tta --output <salida.json>
python src/train_s10.py <receta de H8> --cbam none --output <archivo.pth>
python src/evaluate_volumes.py --checkpoint <archivo.pth> ... --sin-cbam    # CBAM apagado al predecir
```

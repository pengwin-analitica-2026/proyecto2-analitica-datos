# Model card — H6 (detección, segmentación y clasificación de fragmentos de fractura pélvica)

> **Pendientes antes de entregar (borrar este bloque al cerrar):**
> - Licencia exacta del dataset PENGWIN (sección 2).
> - Limitación 1: actualizar cuando la demo lea la orientación de la cabecera de cada archivo.
> - Despliegue: confirmar la URL del túnel de Cloudflare y la prueba desde celular.

## 1. Resumen

| | |
|---|---|
| **Modelo** | H6: `h6_auggeo_rot35_canon_ema_w32_e60_s42.pth` (semilla 42) |
| **Tarea** | En cortes axiales de CT pélvica: clasificar las regiones presentes, detectarlas con cajas, segmentarlas (sacro, coxal izquierdo, coxal derecho) y separar cada región en fragmento primario y secundarios |
| **Equipo** | Gustavo Adolfo Chipantiza, Federico Gonzales Cárdenas, Juan Jacobo Delgado Melo, María de los Ángeles Valencia Hurtado |
| **Curso** | Analítica de Datos, Proyecto Integrador Corte 2, UAO 2026-2. Profesor Carlos Ferro |
| **Fecha** | Octubre de 2026 |
| **Uso previsto** | Académico. **No es un dispositivo médico ni debe usarse para decisiones clínicas.** |

## 2. Datos

- **Fuente:** PENGWIN 2024, Task 1 (CT), 100 volúmenes con su máscara de fragmentos, en formato
  `.mha`. El enunciado menciona NIfTI; el profesor aprobó trabajar con `.mha`, cuya cabecera trae
  la misma información de espaciado y orientación. Licencia: *(completar)*.
- **Partición:** 70 / 15 / 15 por paciente (semilla 42), sin solapamiento. El volumen óseo medio es
  parecido entre splits (770 476, 780 222 y 795 543 mm³ en test, train y val).
- **Etiquetas:** las del dataset, sin etiquetas nuevas. Tres regiones (sacro, coxal izquierdo,
  coxal derecho). Dentro de cada región, el ID 1/11/21 es siempre el fragmento de mayor volumen
  (verificado en 300 de 300 combinaciones caso-región), y se usa como primario; el resto son
  secundarios.
- **Características que afectan al modelo:**
  - la mediana HU del hueso varía de 81 a 431 HU entre pacientes (CV 26.5 %);
  - 25 casos tienen metal (19 en train, 3 en val y 3 en test), sin efecto medible sobre los HU del
    hueso a nivel de volumen;
  - 34 de los 100 volúmenes están almacenados en orientación LPI, girados 180° en el plano axial
    respecto a los demás;
  - hay fragmentos desde 9 cortes axiales; la mediana es de 136.

## 3. Arquitectura y entrenamiento

- **Backbone:** `FundidoraBackbone`, extensión propia de FundidoraPC con 32 canales base y CBAM de
  una etapa antes de la bifurcación. Unos 2.74 M de parámetros en total.
- **Cabezas, entrenadas desde cero:** clasificación multietiqueta de regiones; detección con grid y
  NMS propios; segmentación semántica (3 regiones más fondo) y de instancia
  (primario/secundario por región), con lógica de dos etapas.
- **Sin frameworks de detección o segmentación** (ni YOLO, ni Detectron2, ni Mask R-CNN).
- **Entrada:** cortes axiales 2D a 384 px. Ventana C400/W1800 con corrección gamma 1.5.
- **Orientación canónica:** los volúmenes LPI se giran 180° en el plano antes del modelo, a partir
  de su cabecera. Se aplica igual a imagen y máscara, en todos los splits, sin usar etiquetas.
- **Entrenamiento:** 60 épocas, batch 16, tasa de aprendizaje con coseno, precisión mixta
  (`torch.cuda.amp`), sin preentrenamiento.
- **Pérdida compuesta:** detección 1.0, clasificación 0.5, semántica 2.0 (CE) y 1.0 (Dice), instancia
  1.0 (CE, con secundarios ×3 y fondo ×0.1) y 1.0 (Dice). Cada peso se justifica con una medida del
  dataset o un barrido; la tabla de calibración está en el informe (sección 4).
- **Aumento de datos (solo en entrenamiento):** aumento geométrico (escala, proporción, traslación
  y rotación de ±35°) e intensidad. No se aplica en validación ni en test.
- **Pesos:** promedio exponencial (EMA 0.999).
- **Inferencia:** TTA de espejo (se promedia con la predicción de la imagen reflejada) y
  post-proceso 3D con `min_size=12`, sin CASR.

## 4. Resultados

### 4.1 Test (15 pacientes)

| Métrica | H6 s42 (entregado) | Media de 3 semillas |
|---|---|---|
| Dice por fragmento | 0.875 | 0.875 ± 0.001 |
| IoU por fragmento | 0.816 | 0.815 |
| F1 de clasificación | 0.990 | 0.988 |
| AUC de clasificación | 0.999 | 0.999 |
| IoU de detección | 0.943 | 0.941 |
| mAP50 | 0.961 | 0.958 |
| mAP50:95 | 0.902 | 0.899 |
| Dice semántico | 0.971 | 0.970 |
| IoU semántico | 0.943 | 0.943 |

El ensamble de las tres semillas llega a 0.879 de Dice por fragmento, pero triplica el costo de
inferencia y no se usa.

**Cómo leer estas cifras.** Son una **segunda evaluación en test, posterior al análisis de
fallos**. La primera evaluación oficial fue la de D5 (Dice por fragmento 0.845), que no se borra ni
se reemplaza. H6 se diseñó después de analizar por qué fallaban dos casos de test (012, paciente
girado unos 26°; 039, aspecto LPI con orientación RAI), pero se eligió solo con validación y
pruebas de estrés, y su configuración se congeló por escrito antes de evaluarla. Test se corrió una
sola vez con H6, sin ajustar nada después (`docs/decisiones_semana10.md`, sección 12).

En total, test se consultó cuatro veces durante el proyecto: receta c2 (0.848), EMA + CASR (0.847),
D5 (0.845) y H6 (0.875).

**Sobre la detección.** El mAP se mide sobre cajas refinadas con la máscara semántica. Con la
cabeza grid sola, el mAP50 en validación es 0.887 (0.971 con refinamiento).

### 4.2 Validación (15 pacientes)

| Semilla | Dice por fragmento |
|---|---|
| s42 (entregada) | 0.869 |
| s43 | 0.868 |
| s44 | 0.872 |

Desglose de s42: primario 0.905, secundario 0.764; secundarios no detectados 6.9 %, primarios no
detectados 0.2 %.

### 4.3 Ablación de H6 (validación, semilla 42)

| Corrida | Orientación canónica | Aumento geométrico | Dice fragmento | Secundario | Sec. perdidos | mAP50 |
|---|---|---|---|---|---|---|
| D5 (base) | no | no | 0.851 | 0.703 | 9.0 % | 0.959 |
| H1 | no | sí, ±12° | 0.864 | 0.743 | 7.5 % | 0.969 |
| H4 | sí | no | 0.861 | 0.742 | 8.8 % | 0.971 |
| H5 | sí | sí, ±12° | 0.873 | 0.769 | 6.2 % | 0.976 |
| H6 | sí | sí, ±35° | 0.869 | 0.764 | 6.9 % | 0.971 |

La orientación y el aumento aportan por separado (+0.010 y +0.014) y casi de forma aditiva
(+0.023 juntos). La ganancia se concentra en los fragmentos secundarios. H5 y H6 empatan en
validación (medias de 0.871 y 0.870, dentro del ruido de ±0.004).

### 4.4 Robustez a pacientes girados (Dice por fragmento)

| Entrada girada | D5 | H5 (±12°) | H6 (±35°) |
|---|---|---|---|
| 30° | 0.559 | 0.807 | 0.854 |
| 60° | — | 0.218 | 0.734 |

Esta prueba decidió entre H5 y H6: con el mismo rendimiento en validación, H6 resiste mucho mejor a
pacientes girados en el escáner, que existen en el dataset (caso 012).

### 4.5 Ablaciones del enunciado (sobre D5)

| Ablación | Resultado | Conclusión |
|---|---|---|
| Sin CBAM | F1 0.975 contra 0.988; mAP50:95 0.872 contra 0.883; Dice por fragmento igual (0.851) | CBAM aporta en clasificación y detección, no en segmentación |
| Transfer learning (backbone preentrenado en Intel Image Classification, 2 semillas) | Dice por fragmento 0.851 y 0.856; secundarios +0.011 y −0.007 | Neutral; el modelo final no usa preentrenamiento |
| λ × 2 | Dice por fragmento 0.855 (ruido); mAP50 crudo 0.827 contra 0.868 | Se mantiene λ por defecto |

Las ablaciones se hicieron sobre D5. H6 es D5 más la orientación canónica y el aumento geométrico,
cuya ablación es la de la sección 4.3.

### 4.6 Comparación con SAM (test)

SAM ViT-B en modo zero-shot, con la caja predicha por nuestro detector como prompt, comparado con
la máscara semántica de H6 dentro de la misma caja (1 181 cortes de test, uno de cada tres; 3 002
pares corte-región). SAM no forma parte del pipeline.

| Región | Dice SAM | Dice H6 | IoU SAM | IoU H6 |
|---|---|---|---|---|
| Sacro | 0.792 | 0.884 | 0.690 | 0.820 |
| Coxal izquierdo | 0.915 | 0.947 | 0.859 | 0.905 |
| Coxal derecho | 0.930 | 0.957 | 0.879 | 0.921 |
| **Global** | **0.889** | **0.935** | **0.823** | **0.889** |

El modelo propio supera a SAM en las tres regiones; la mayor diferencia está en el sacro. SAM tarda
unos 259 ms por corte en la GPU L4 solo en su parte, frente a unos 64 ms de todo el modelo propio.

### 4.7 Clasificación y peores casos (test)

- **Clasificación por región:** F1 de 0.980 (sacro), 0.997 (coxal izquierdo) y 0.995 (coxal
  derecho). A nivel de píxel, casi todo el error es hueso predicho como fondo (3.3 % del sacro) y
  la confusión entre coxal izquierdo y derecho es menor al 0.1 %.
- **Peores casos:** 039 (0.784), 061 (0.822), 007 (0.831), 054 (0.834) y 047 (0.854). En todos el
  fragmento principal está entre 0.87 y 0.91 y el error se concentra en los secundarios: volumen
  de baja resolución que se sobrefragmenta (039), intercambio principal/secundario entre fragmentos
  de tamaño parecido (061, 054), fragmentos pequeños en contacto (007) y hueso fino en los extremos
  del volumen (047). La galería y el análisis están en el informe, sección 8.

### 4.8 Latencia (mediana por corte de 384 px, lote 1)

| Configuración | GPU (NVIDIA L4) | CPU (12 hilos) |
|---|---|---|
| 1 modelo | 63.7 ms | 161.7 ms |
| 1 modelo + TTA de espejo (entregado) | 71.6 ms | 249.6 ms |
| Ensamble de 3 semillas + TTA | 87.4 ms | 518.0 ms |

Medido sobre 60 cortes reales de validación tras 5 pasadas de calentamiento. D5 y H6 comparten
arquitectura, así que su latencia es la misma.

### 4.9 Distancia de separación y métricas de superficie (41 secundarios)

| Métrica | Val. H6 | Test D5 | Test H6 |
|---|---|---|---|
| Error medio de distancia | 0.185 mm | 2.956 mm | 1.277 mm |
| Correlación con la distancia real | 0.972 | 0.543 | 0.818 |
| Dice 3D principal / secundario | 0.921 / 0.701 | 0.909 / 0.682 | 0.931 / 0.756 |
| ASSD / HD95 | 4.08 / 17.99 mm | 3.02 / 16.33 mm | 2.06 / 11.55 mm |

La distancia se mide borde a borde en 3D con `distance_transform_edt` y el espaciado de la
cabecera, en la predicción y en el ground truth. En test, un solo fragmento del caso 007 concentra el
79 % del error: la predicción le agrega un componente aislado lejos del hueso.

## 5. Protocolo de evaluación

- Todo se elige en validación. Una variante se adopta solo si mejora al menos 0.003, con bootstrap
  por paciente (P ≥ 0.90) y mejora en las dos mitades de validación.
- Las cifras de test se reportan todas, incluidas las que no se usaron, y no se ajusta ningún
  umbral después de verlas.
- Se entregó una sola semilla (s42); s43 y s44 sirven para medir la variabilidad.

## 6. Limitaciones

1. **Orientación en la demo.** La orientación canónica se identifica con una lista de los 34 casos
   LPI del dataset, derivada de sus cabeceras. Para un volumen nuevo, la demo debe leer la
   orientación de la cabecera de cada archivo. *(Actualizar cuando esté corregido.)*
2. **Fragmentos en contacto en la superficie de fractura.** Cuando dos fragmentos se tocan, la
   segmentación puede fusionarlos en uno solo. En ese caso se pierde el fragmento secundario y su
   distancia de separación al principal no es confiable (puede salir 0 mm o no existir).
3. **Fragmentos secundarios.** Son el punto débil: Dice de 0.764 frente a 0.905 del primario, y el
   6.9 % no se detecta. Representan el 47.9 % de los fragmentos pero solo el 10.1 % de los vóxeles
   de hueso.
4. **Fragmentos sobrantes.** El Dice por fragmento empareja fragmentos reales con predichos, pero
   no penaliza los fragmentos predichos que sobran. El dashboard oculta los menores de 1000 mm³; es
   un filtro de presentación que no cambia la métrica.
5. **Modelo 2D.** Procesa corte a corte. La coherencia entre cortes depende del post-proceso 3D.
6. **Tamaño de la muestra.** Test tiene 15 pacientes de un solo dataset. Las diferencias de ±0.004
   son del orden del ruido, y no se evaluó en otros centros ni escáneres.
7. **Segunda evaluación en test.** La cifra de H6 no es una estimación limpia (ver la sección 4.1).
8. **Métrica de detección.** El mAP reportado usa cajas refinadas con la máscara semántica; la
   cabeza grid sola rinde menos.

## 7. Consideraciones éticas

- Los datos son públicos y anonimizados, del challenge PENGWIN 2024.
- El modelo no debe usarse para diagnóstico ni planificación quirúrgica. Un error en la separación
  de fragmentos o en la lateralidad tendría consecuencias clínicas.
- El rendimiento puede bajar con escáneres, protocolos o poblaciones distintas a las del dataset.

## 8. Despliegue

El dashboard se sirve con Streamlit y se publica con un túnel de Cloudflare, como pide el enunciado,
accesible desde el navegador de un computador y de un celular. La advertencia de uso no clínico está
siempre visible en el dashboard.

## 9. Reproducción

Los comandos para regenerar los datos, entrenar H6 y evaluarlo están en el `README.md`. La
configuración congelada de la evaluación en test está en `docs/decisiones_semana10.md`
(sección 12.4) y los resultados en `outputs/s10/`.

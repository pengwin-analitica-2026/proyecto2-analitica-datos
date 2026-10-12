# Model card — H7 (detección, segmentación y clasificación de fragmentos de fractura pélvica)

## 1. Resumen

| | |
|---|---|
| **Modelo** | H7: `h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42.pth` (semilla 42). Es la receta de H6 sin EMA y con el giro de entrenamiento en milímetros, como pidió el profesor |
| **Tarea** | En cortes axiales de CT pélvica: clasificar las regiones presentes, detectarlas con cajas, segmentarlas (sacro, coxal izquierdo, coxal derecho) y separar cada región en fragmento primario y secundarios |
| **Equipo** | Gustavo Adolfo Chipantiza, Federico Gonzales Cárdenas, Juan Jacobo Delgado Melo, María de los Ángeles Valencia Hurtado |
| **Curso** | Analítica de Datos, Proyecto Integrador Corte 2, UAO 2026-2. Profesor Carlos Ferro |
| **Fecha** | Octubre de 2026 |
| **Uso previsto** | Académico. **No es un dispositivo médico ni debe usarse para decisiones clínicas.** |

## 2. Datos

- **Fuente:** PENGWIN 2024, Task 1 (CT), 100 volúmenes con su máscara de fragmentos, en formato
  `.mha`. El enunciado menciona NIfTI; el profesor aprobó trabajar con `.mha`, cuya cabecera trae
  la misma información de espaciado y orientación.
- **Licencia de los datos:** Creative Commons Atribución 4.0 Internacional (**CC BY 4.0**), según el
  registro del dataset en Zenodo (*PENGWIN Task 1: Pelvic Fracture Segmentation on CT*, versión 1,
  7 de abril de 2024, [doi:10.5281/zenodo.10927452](https://doi.org/10.5281/zenodo.10927452)).
  Permite usar, compartir y adaptar los datos siempre que se cite a sus autores. Los volúmenes
  originales no se versionan en este repositorio; los cuatro casos exportados para la demo
  (`outputs/demo/`) y los pesos son material derivado de PENGWIN y llevan la misma atribución.
- **Cómo citar los datos** (referencias que pide el sitio del reto):
  - Y. Sang, Y. Liu, S. Yibulayimu *et al.*, «Benchmark of Segmentation Techniques for Pelvic
    Fracture in CT and X-Ray: Summary of the PENGWIN 2024 Challenge», *IEEE Transactions on Medical
    Imaging*, 45(5):2212-2228, 2026. doi:10.1109/TMI.2025.3650126.
  - Y. Liu, S. Yibulayimu, G. Zhu *et al.*, «Automatic pelvic fracture segmentation: a deep
    learning approach and benchmark dataset», *Frontiers in Medicine*, 12:1511487, 2025.
- **Partición:** 70 / 15 / 15 por paciente (semilla 42), sin solapamiento. El volumen óseo medio es
  parecido entre splits (770 476, 780 222 y 795 543 mm³ en test, train y val).
- **Etiquetas:** las del dataset, sin etiquetas nuevas. Tres regiones (sacro, coxal izquierdo,
  coxal derecho). Dentro de cada región, el ID 1/11/21 es siempre el fragmento de mayor volumen
  (verificado en 300 de 300 combinaciones caso-región), y se usa como primario; el resto son
  secundarios.
- **Características que afectan al modelo:**
  - la mediana HU del hueso varía de 81 a 431 HU entre pacientes (CV 26.5 %);
  - 25 casos tienen metal de más de 100 mm³ (19 en train, 3 en val y 3 en test) y 41 tienen algún
    vóxel sobre 3000 HU, sin efecto medible sobre los HU del hueso a nivel de volumen;
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
- **Orientación canónica con los metadatos:** la orientación se calcula con productos punto y cruz
  sobre la matriz de dirección de la cabecera, siguiendo el método de Ferro Sánchez *et al.*
  (*MethodsX* 13:103073, 2024, doi:10.1016/j.mex.2024.103073). Producto punto: con qué eje del
  paciente coincide cada eje de la imagen y si son perpendiculares. Producto cruz (filas × columnas):
  hacia dónde van los cortes y si la imagen está en espejo. Todo queda en **RAI**; los 34 volúmenes
  LPI se giran 180° en el plano (`src/normalizacion.py`). En los 100 casos da lo mismo que la versión
  anterior (`to_canonical`). Se aplica igual a imagen y máscara, en todos los splits, sin usar
  etiquetas.
- **Enderezado del paciente (opcional, al predecir):** estima el giro del paciente en la camilla por
  la simetría izquierda-derecha de la silueta del hueso vista desde arriba, y lo corrige en
  milímetros solo si es claro (más de 5° y mejora la simetría) (`--normalizar`). No requiere
  reentrenar. Solo corrige el giro en el plano axial.
- **Entrenamiento:** 60 épocas, batch 16, tasa de aprendizaje con coseno, precisión mixta
  (`torch.cuda.amp`), sin preentrenamiento.
- **Pérdida compuesta:** detección 1.0, clasificación 0.5, semántica 2.0 (CE) y 1.0 (Dice), instancia
  1.0 (CE, con secundarios ×3 y fondo ×0.1) y 1.0 (Dice). Cada peso se justifica con una medida del
  dataset o un barrido; la tabla de calibración está en el informe (sección 4).
- **Aumento de datos (solo en entrenamiento):** aumento geométrico (escala, proporción, traslación
  y rotación de ±35°) e intensidad. En H7 la rotación se hace **en milímetros** (`--aug-mm`): en H6
  se hacía en píxeles y, con píxeles no cuadrados (10 de los 15 casos de validación), deformaba la
  pelvis. No se aplica en validación ni en test.
- **Pesos:** **sin EMA** (`--ema 0`). El profesor pidió no usarlo porque, al promediar los pesos sobre
  las condiciones de entrenamiento, puede hacer al modelo frágil con imágenes "feas". H6 usaba EMA
  0.999 (sección 4.10).
- **CBAM, γ aprendible por bloque** (análisis de la Semana 9 con CBAM en tres etapas): 1/4 → 0.089,
  1/8 → −0.106, 1/16 → 0.108. El bloque de 1/4 casi no aporta; por eso el modelo final usa CBAM en
  una sola etapa, antes de la bifurcación hacia las cabezas.
- **Hiperparámetros de H7:** 384 px, `--mode stretch`, 60 épocas, batch 16, lr 0.002 con coseno,
  32 canales base, CBAM 1, `--instance-mode primary`, secundarios ×3, `--margin 10`,
  `--neg-fraction 0.15`, 2500 cortes por época, `--aug-geo 1.0`, `--aug-rot-deg 35`, `--aug-mm`,
  `--canonical`, `--ema 0`, semilla 42. Entrenado en Kaggle (T4) en 67 minutos.
- **Inferencia:** TTA de espejo (se promedia con la predicción de la imagen reflejada) y
  post-proceso 3D con `min_size=12`, sin CASR.

## 4. Resultados

### 4.1 Test (15 pacientes)

| Métrica | Objetivo | **H7 s42 (entregado, 3.ª evaluación)** | H6 s42 (2.ª evaluación) | H6, media de 3 semillas | D5 s42 (1.ª evaluación) |
|---|---|---|---|---|---|
| Dice por fragmento | 0.85 | **0.879** | 0.875 | 0.875 ± 0.001 | 0.845 |
| IoU por fragmento | 0.70 | 0.819 | 0.816 | 0.815 | 0.781 |
| F1 de clasificación | 0.85 | 0.989 | 0.990 | 0.988 | 0.984 |
| AUC de clasificación | 0.85 | 0.999 | 0.999 | 0.999 | 0.998 |
| IoU de detección | 0.65 | 0.940 | 0.943 | 0.941 | 0.898 |
| mAP50 | 0.65 | 0.962 | 0.961 | 0.958 | 0.920 |
| mAP50:95 | 0.40 | 0.899 | 0.902 | 0.899 | 0.853 |
| Dice semántico | 0.85 | 0.971 | 0.971 | 0.970 | 0.949 |
| IoU semántico | 0.70 | 0.944 | 0.943 | 0.943 | 0.904 |

H7 y H6 cumplen los nueve objetivos; D5 cumplía ocho (le faltaba el Dice por fragmento). La fila de
H7 es **sin enderezar**, la configuración validada. En la misma consulta se corrió también con
enderezado: 0.874 de Dice por fragmento (sección 4.4).

El ensamble de las tres semillas llega a 0.879 de Dice por fragmento, pero triplica el costo de
inferencia y no se usa.

**Cómo leer estas cifras.** Son una **segunda evaluación en test, posterior al análisis de
fallos**. La primera evaluación oficial fue la de D5 (Dice por fragmento 0.845), que no se borra ni
se reemplaza. H6 se diseñó después de analizar por qué fallaban dos casos de test (012, paciente
girado unos 26°; 039, aspecto LPI con orientación RAI), pero se eligió solo con validación y
pruebas de estrés, y su configuración se congeló por escrito antes de evaluarla. Test se corrió una
sola vez con H6, sin ajustar nada después (`docs/decisiones_semana10.md`, sección 12).

**H7 (tercera evaluación oficial).** Corrige lo que pidió el profesor (orientación con metadatos,
sin EMA, giro en milímetros). Se eligió solo con validación (sección 4.2), y test se corrió una vez,
con las dos variantes (sin y con enderezado) decididas de antemano. La principal es sin enderezar.

En total, test se consultó cinco veces durante el proyecto: receta c2 (0.848), EMA + CASR (0.847),
D5 (0.845), H6 (0.875) y H7 (0.879). En la evaluación de D5 también se pasaron por test otros cinco checkpoints
(dos semillas más de D5 y las ablaciones E2 y F2), solo para medir la variación y con la regla
escrita de antemano de que no servían para elegir modelo: quedaron entre 0.841 y 0.848.

**Sobre la detección.** El mAP se mide sobre cajas refinadas con la máscara semántica. Con la
cabeza grid sola, el mAP50 es 0.912 en test (0.961 con refinamiento) y 0.887 en validación (0.971);
el mAP50:95 de la cabeza sola es 0.556 en test.

### 4.2 Validación (15 pacientes)

**H7 s42 (entregado), nueve métricas:**

| Métrica | H7 | Objetivo |
|---|---|---|
| F1 / AUC de clasificación | 0.992 / 0.999 | 0.85 / 0.85 |
| IoU de detección / mAP50 / mAP50:95 | 0.948 / 0.974 / 0.903 | 0.65 / 0.65 / 0.40 |
| Dice / IoU semántico | 0.971 / 0.944 | 0.85 / 0.70 |
| Dice / IoU por fragmento | **0.873** / 0.812 | 0.85 / 0.70 |

H7 se entrenó con una sola semilla (s42). Curva de validación (Dice por fragmento): época 20 →
0.827 · 40 → 0.863 · 60 → 0.875.

**H6, tres semillas (modelo anterior):**

| Semilla | Dice por fragmento |
|---|---|
| s42 | 0.869 |
| s43 | 0.868 |
| s44 | 0.872 |

Desglose de H6 s42: primario 0.905, secundario 0.764; secundarios no detectados 6.9 %, primarios no
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
pacientes girados en el escáner, que existen en el dataset (caso 012). **Nota:** esta prueba giraba
la imagen en píxeles (`rot=`), lo que deforma la pelvis cuando los píxeles no son cuadrados. Desde
H7 se usa `rotmm=`, que gira al paciente en milímetros reales:

| Validación, `rotmm` | H6 | H6 + enderezado | H7 | H7 + enderezado |
|---|---|---|---|---|
| Sin giro | 0.869 | — | 0.873 | 0.872 |
| 30° | 0.853 | 0.854 | — | — |
| 60° | 0.608 (mAP50 0.311) | 0.851 (mAP50 0.963) | 0.636 (mAP50 0.347) | 0.855 (mAP50 0.963) |

El enderezado casi no cuesta en pacientes derechos (0.872 contra 0.873; corrigió 2 de 15 casos, con
6°) y recupera casi todo con 60°. En test (consulta ya hecha) corrigió 6 de 15 casos: el 012
(+23°), que sí lo necesitaba, y cinco con giros de 5° a 7°, que explican la baja de 0.879 a 0.874.
Por eso se propone subir el umbral de corrección a unos 10°, decidido en validación.

### 4.5 Ablaciones del enunciado (sobre D5)

| Ablación | Resultado | Conclusión |
|---|---|---|
| Sin CBAM | F1 0.975 contra 0.988; mAP50:95 0.872 contra 0.883; Dice por fragmento igual (0.851) | CBAM aporta en clasificación y detección, no en segmentación |
| Transfer learning (backbone preentrenado en Intel Image Classification, 2 semillas) | Dice por fragmento 0.851 y 0.856; secundarios +0.011 y −0.007 | Neutral; el modelo final no usa preentrenamiento |
| λ × 2 | Dice por fragmento 0.855 (ruido); mAP50 crudo 0.827 contra 0.868 | Se mantiene λ por defecto |

Las ablaciones se hicieron sobre D5. H6 es D5 más la orientación canónica y el aumento geométrico,
cuya ablación es la de la sección 4.3. H7 es H6 sin EMA y con el giro en milímetros; los dos cambios
se hicieron a la vez, así que esta comparación no aísla el aporte de cada uno (experimentos 23 y 25
del seguimiento).

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

Medido con H6. El modelo propio supera a SAM en las tres regiones; la mayor diferencia está en el sacro. SAM tarda
unos 259 ms por corte en la GPU L4 solo en su parte, frente a unos 64 ms de todo el modelo propio.

### 4.7 Clasificación y peores casos (test, H6)

- **Clasificación por región:** F1 de 0.980 (sacro), 0.997 (coxal izquierdo) y 0.995 (coxal
  derecho). A nivel de píxel, casi todo el error es hueso predicho como fondo (3.3 % del sacro) y
  la confusión entre coxal izquierdo y derecho es menor al 0.1 %.
- **Peores casos:** 039 (0.784), 061 (0.822), 007 (0.831), 054 (0.834) y 047 (0.854). En todos el
  fragmento principal está entre 0.87 y 0.91 y el error se concentra en los secundarios: volumen
  de baja resolución que se sobrefragmenta (039), intercambio principal/secundario entre fragmentos
  de tamaño parecido (061, 054), fragmentos pequeños en contacto (007) y hueso fino en los extremos
  del volumen (047). La galería y el análisis están en el informe, sección 8.

### 4.8 Latencia (mediana por corte de 384 px, lote 1)

| Configuración | GPU (NVIDIA L4) | CPU (12 hilos, Colab) | CPU (portátil, i9-9980HK, 8 hilos) |
|---|---|---|---|
| 1 modelo | 63.7 ms | 161.7 ms | 336.1 ms |
| 1 modelo + TTA de espejo (entregado) | 71.6 ms | 249.6 ms | 695.5 ms |
| Ensamble de 3 semillas + TTA | 87.4 ms | 518.0 ms | — |

GPU y CPU de Colab: 60 cortes reales de validación tras 5 pasadas de calentamiento
(`outputs/s10/latencia_gpu_L4.json`, `latencia_cpu_colab.json`). Portátil: `src/measure_latency.py`
con H6 (`outputs/demo/latency.json`). D5 y H6 comparten arquitectura, así que su latencia es la
misma; H7 también comparte arquitectura.

### 4.9 Distancia de separación y métricas de superficie (41 secundarios; medidas con H6, no se repitieron con H7)

| Métrica | Val. H6 | Test D5 | Test H6 |
|---|---|---|---|
| Error medio de distancia | 0.185 mm | 2.956 mm | 1.277 mm |
| Correlación con la distancia real | 0.972 | 0.543 | 0.818 |
| Dice 3D principal / secundario | 0.921 / 0.701 | 0.909 / 0.682 | 0.931 / 0.756 |
| ASSD / HD95 | 4.08 / 17.99 mm | 3.02 / 16.33 mm | 2.06 / 11.55 mm |

La distancia se mide borde a borde en 3D con `distance_transform_edt` y el espaciado de la
cabecera, en la predicción y en el ground truth. En test, un solo fragmento del caso 007 concentra el
79 % del error: la predicción le agrega un componente aislado lejos del hueso.

### 4.10 Imágenes "feas": H6 (EMA) contra H7 (sin EMA), validación

| Daño | H6 | H7 | H7 + suavizado σ = 1 |
|---|---|---|---|
| Ninguno | 0.869 | 0.873 | 0.859 |
| Baja resolución (×2.5) | 0.857 | 0.861 | — |
| Desenfoque (σ = 1.5) | 0.818 | 0.819 | — |
| Ruido leve (σ = 10 niveles) | 0.690 | 0.654 | **0.851** |
| Ruido fuerte (σ = 25 niveles) | 0.135 | 0.228 | 0.658 |

Dice por fragmento. El EMA no era la causa de la fragilidad: **el ruido es el punto débil de ambos
modelos** (con ruido leve H6 aguanta algo mejor; con ruido fuerte, H7). Suavizar la imagen antes del
modelo (`suav=`) recupera el ruido leve sin reentrenar, pero cuesta 0.014 en imágenes limpias, así
que se aplica solo cuando la imagen viene ruidosa, con un aviso de calidad.

### 4.11 Dependencia de CBAM en H7 (CBAM apagado al predecir, validación)

| Cabeza | Con CBAM | CBAM apagado |
|---|---|---|
| F1 de clasificación | 0.992 | 0.444 |
| mAP50 | 0.974 | 0.401 |
| Dice semántico | 0.971 | 0.678 |
| Dice por fragmento | 0.873 | 0.461 |

Mide cuánto **depende** H7 de CBAM, no cuánto gana por tenerlo: H7 se entrenó con CBAM y quitárselo
de golpe lo descuadra. El aporte entrenando sin CBAM es la ablación de la sección 4.5.

### 4.12 Cuantización / modelo compacto

*(pendiente: Gus — tamaño antes y después y Dice por fragmento en validación; regla: caída ≤ 0.003.
Medida con H6; confirmar si se repite con H7.)*

## 5. Protocolo de evaluación

- Todo se elige en validación. Una variante se adopta solo si mejora al menos 0.003, con bootstrap
  por paciente (P ≥ 0.90) y mejora en las dos mitades de validación.
- Las cifras de test se reportan todas, incluidas las que no se usaron, y no se ajusta ningún
  umbral después de verlas.
- Se entregó una sola semilla (s42); s43 y s44 sirven para medir la variabilidad.

## 6. Limitaciones

1. **Orientación de la entrada.** La orientación se lleva a RAI con los metadatos (productos punto y
   cruz), así que sirve para volúmenes nuevos alineados a los ejes. Un volumen **oblicuo** (inclinado
   respecto a los ejes) necesitaría remuestrearse con la matriz completa de la cabecera; no hay casos
   así en PENGWIN y no se probó. El enderezado del paciente solo corrige el giro en el plano axial: el
   cabeceo y la inclinación lateral no se corrigen. Falla en volúmenes con campo de visión corto
   (caso 044, ~16 cm de profundidad: al girarlo, la pelvis se sale del recuadro).
2. **Ruido.** Con ruido leve el Dice por fragmento baja de 0.873 a 0.654; el suavizado lo recupera
   (0.851), pero con ruido fuerte no llega al objetivo (0.658). La demo debe avisar cuando la imagen
   viene ruidosa (sección 4.10).
3. **Fragmentos en contacto en la superficie de fractura.** Cuando dos fragmentos se tocan, la
   segmentación puede fusionarlos en uno solo. En ese caso se pierde el fragmento secundario y su
   distancia de separación al principal no es confiable (puede salir 0 mm o no existir).
4. **Fragmentos secundarios.** Son el punto débil: Dice de 0.764 frente a 0.905 del primario, y el
   6.9 % no se detecta. Representan el 47.9 % de los fragmentos pero solo el 10.1 % de los vóxeles
   de hueso.
5. **Fragmentos sobrantes.** El Dice por fragmento empareja fragmentos reales con predichos, pero
   no penaliza los fragmentos predichos que sobran. El dashboard oculta los menores de 1000 mm³; es
   un filtro de presentación que no cambia la métrica.
6. **Modelo 2D.** Procesa corte a corte. La coherencia entre cortes depende del post-proceso 3D.
7. **Tamaño de la muestra.** Test tiene 15 pacientes de un solo dataset. Las diferencias de ±0.004
   son del orden del ruido, y no se evaluó en otros centros ni escáneres.
8. **Evaluaciones repetidas en test.** Test se consultó cinco veces; las cifras de H6 y H7 no son estimaciones limpias (ver la sección 4.1).
9. **Métrica de detección.** El mAP reportado usa cajas refinadas con la máscara semántica; la
   cabeza grid sola rinde menos.

## 7. Consideraciones éticas

- Los datos son públicos y anonimizados, del challenge PENGWIN 2024.
- El modelo no debe usarse para diagnóstico ni planificación quirúrgica. Un error en la separación
  de fragmentos o en la lateralidad tendría consecuencias clínicas.
- El rendimiento puede bajar con escáneres, protocolos o poblaciones distintas a las del dataset.

## 8. Despliegue

El dashboard (Streamlit) está publicado en Streamlit Community Cloud y la inferencia en vivo de un
corte corre en una API en AWS Lambda con H6 en CPU *(pendiente: pasar la demo a H7 y agregar el
enderezado y el aviso de ruido)*; los dos se vuelven a desplegar en cada push a
`main` (ver `docs/despliegue_aws_streamlit.md`). Se probó desde el navegador de un computador y de
un celular. El profesor aprobó este despliegue en lugar del túnel de Cloudflare que nombra el
enunciado. La advertencia de uso no clínico está siempre visible en el dashboard.

## 9. Modelos entrenados y pesos en el repositorio

El sistema es un solo modelo multitarea, así que hay un model card. Los pesos versionados son:

| Archivo | Qué es | En qué difiere del entregado |
|---|---|---|
| `outputs/h7/h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42.pth` | **Modelo entregado** (H7, semilla 42) | — |
| `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth` | H6, modelo de la segunda evaluación | Con EMA 0.999 y giro de entrenamiento en píxeles |
| `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s43.pth` y `_s44.pth` | H6 con otras semillas | Solo la semilla; sirven para medir la variación (secciones 4.1 y 4.2) |
| `outputs/s10/d5_ema_w32_e60_s42.pth` | D5, el modelo de la primera evaluación en test | Misma arquitectura; sin orientación canónica y sin aumento geométrico. Espera los volúmenes tal como están guardados |
| `outputs/gamma/w10g_384neg.pth` y `w10g_sec3.pth` | Primera versión de la Semana 10, usada como línea base | Backbone de 16 canales y entrenamiento en etapas; evaluada sobre una muestra de cortes, no sobre el volumen completo |

Datos de entrenamiento, alcances, limitaciones y advertencia de uso son los mismos para todos:
los 70 casos de entrenamiento de PENGWIN (sección 2), las limitaciones de la sección 6 y el uso
exclusivamente académico.

## 10. Reproducción

Los comandos para regenerar los datos, entrenar H6 y evaluarlo están en el `README.md`. H7 se
entrena y evalúa con `outputs/h7/notebooks_ejecutados/09_H7_entrenamiento_validacion_test.ipynb`
(Kaggle); las pruebas de rotación, ruido, suavizado y CBAM están en los notebooks 07 y 10 de la misma
carpeta, y el resumen en `outputs/h7/RESUMEN_COMPLETO.pdf`. La
configuración congelada de la evaluación en test está en `docs/decisiones_semana10.md`
(sección 12.4) y los resultados en `outputs/s10/`.

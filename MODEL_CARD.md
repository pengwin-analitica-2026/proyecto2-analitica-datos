# Model card — H8 (detección, segmentación y clasificación de fragmentos de fractura pélvica)

## 1. Resumen

| | |
|---|---|
| **Modelo** | H8: `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth` (semilla 42) |
| **Tarea** | En cortes axiales de CT pélvica: clasificar las regiones presentes, detectarlas con cajas, segmentarlas (sacro, coxal izquierdo, coxal derecho) y separar cada región en fragmento primario y secundarios |
| **Equipo** | Gustavo Adolfo Chipantiza, Federico Gonzales Cárdenas, Juan Jacobo Delgado Melo, María de los Ángeles Valencia Hurtado |
| **Curso** | Analítica de Datos, Proyecto Integrador Corte 2, UAO 2026-2. Profesor Carlos Ferro |
| **Fecha** | Octubre de 2026 |
| **Uso previsto** | Académico. **No es un dispositivo médico ni debe usarse para decisiones clínicas.** |

H8 reemplaza a H6 como modelo entregado. Tiene la misma arquitectura y cambia el entrenamiento en
tres puntos: sin EMA, giro del aumento en milímetros reales y aumento de ruido gaussiano. Los dos
primeros responden a la asesoría del profesor. El detalle de los cambios está en `docs/modelo_h8.md`
y la comparación con los demás modelos en `docs/experimentos_ema_rotacion_ruido.md`.

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
  una etapa antes de la bifurcación. Unos 2.74 M de parámetros en total, 8 290 de ellos en CBAM.
- **Cabezas, entrenadas desde cero:** clasificación multietiqueta de regiones; detección con grid y
  NMS propios; segmentación semántica (3 regiones más fondo) y de instancia
  (primario/secundario por región), con lógica de dos etapas.
- **Sin frameworks de detección o segmentación** (ni YOLO, ni Detectron2, ni Mask R-CNN).
- **Entrada:** cortes axiales 2D a 384 px. Ventana C400/W1800 con corrección gamma 1.5.
- **Orientación canónica:** los volúmenes LPI se giran 180° en el plano antes del modelo, a partir
  de su cabecera. La matriz de dirección se analiza con productos punto y cruz
  (`src/normalizacion.py`); en los 100 casos da lo mismo que la regla anterior (`src/orientacion.py`).
  Se aplica igual a imagen y máscara, en todos los splits, sin usar etiquetas.
- **Entrenamiento:** 60 épocas de 2 500 cortes, lote 16, tasa de aprendizaje 0.002 con coseno,
  precisión mixta (`torch.cuda.amp`), sin preentrenamiento. 67 minutos en una GPU T4 de Kaggle.
- **Pérdida compuesta:** detección 1.0, clasificación 0.5, semántica 2.0 (CE) y 1.0 (Dice), instancia
  1.0 (CE, con secundarios ×3 y fondo ×0.1) y 1.0 (Dice). Cada peso se justifica con una medida del
  dataset o un barrido; la tabla de calibración está en el informe (sección 4).
- **Aumento de datos (solo en entrenamiento):**
  - geométrico: escala, proporción, traslación y rotación de ±35°, **definido en milímetros** con el
    espaciado de la cabecera (`--aug-mm`, `src/aumento_fisico.py`), para no deformar la pelvis cuando
    los píxeles del caché no son cuadrados;
  - intensidad: espejo sagital, ganancia y sesgo;
  - **ruido gaussiano** en el 30 % de las muestras, con desviación al azar de hasta 0.1 en la escala
    0-1 (`--aug-noise 0.1`).

  No se aplica en validación ni en test.
- **Pesos:** los del propio entrenamiento, **sin EMA** (`--ema 0`).
- **Inferencia:** TTA de espejo (se promedia con la predicción de la imagen reflejada) y
  post-proceso 3D con `min_size=12`, sin CASR, sin enderezado del giro axial y sin suavizado. Es la
  configuración de todas las cifras de la sección 4. La demo agrega dos pasos que no cambian el
  modelo: el enderezado del giro axial (sección 4.11) y los umbrales del post-proceso 3D en mm³
  (sección 4.12).

## 4. Resultados

### 4.1 Test (15 pacientes, 4 664 cortes)

| Métrica | Objetivo | **H8 (entregado)** | H7 | H6 | D5 (primera evaluación) |
|---|---|---|---|---|---|
| Dice por fragmento | 0.85 | **0.874** | 0.879 | 0.875 | 0.845 |
| IoU por fragmento | 0.70 | 0.812 | 0.819 | 0.816 | 0.781 |
| F1 de clasificación | 0.85 | 0.989 | 0.989 | 0.990 | 0.984 |
| AUC de clasificación | 0.85 | 0.999 | 0.999 | 0.999 | 0.998 |
| IoU de detección | 0.65 | 0.938 | 0.940 | 0.943 | 0.898 |
| mAP50 | 0.65 | 0.953 | 0.962 | 0.961 | 0.920 |
| mAP50:95 | 0.40 | 0.898 | 0.899 | 0.902 | 0.853 |
| Dice semántico | 0.85 | 0.968 | 0.971 | 0.971 | 0.949 |
| IoU semántico | 0.70 | 0.939 | 0.944 | 0.943 | 0.904 |

H8 cumple los nueve objetivos. Desglose del Dice por fragmento: principal 0.910, secundario 0.760,
con 7.2 % de secundarios sin pareja; centro del volumen 0.885 y extremos 0.836.

**Cómo leer estas cifras.** Test se consultó seis veces durante el proyecto: receta c2 (0.848),
EMA + CASR (0.847), D5 (0.845), H6 (0.875), H7 (0.879) y H8 (0.874). La única cifra limpia es la de
D5, obtenida con la configuración congelada antes de ver test. Los casos de test 012 y 039 se
usaron para diagnosticar fallos antes de H6, así que las evaluaciones posteriores no son
estimaciones independientes. H8 se eligió solo con validación, su configuración se escribió antes de
correr (`outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md`) y test se evaluó una sola vez, sin ajustar
nada después.

**Sobre la detección.** El mAP se mide sobre cajas refinadas con la máscara semántica. Con la
cabeza grid sola, el mAP50 es 0.909 en test y 0.903 en validación, y el mAP50:95 0.566 y 0.534.

### 4.2 Validación (15 pacientes, 5 047 cortes)

| Métrica | Objetivo | H8 |
|---|---|---|
| F1 / AUC de clasificación | 0.85 / 0.85 | 0.991 / 0.999 |
| IoU de detección / mAP50 / mAP50:95 | 0.65 / 0.65 / 0.40 | 0.946 / 0.969 / 0.900 |
| Dice / IoU semántico | 0.85 / 0.70 | 0.970 / 0.943 |
| Dice / IoU por fragmento | 0.85 / 0.70 | 0.865 / 0.800 |

Desglose: principal 0.897, secundario 0.769; secundarios no detectados 6.3 %, primarios 0.2 %.

### 4.3 Por qué H8: comparación de cinco modelos (validación, semilla 42)

Mismos datos, receta, épocas y semilla; solo cambian EMA, tipo de giro y ruido al entrenar. Dice por
fragmento.

| Modelo | EMA | Giro del aumento | Ruido al entrenar | Sin daño | Ruido 10 | Ruido 25 | Giro 60° | Giro 60° + enderezado |
|---|---|---|---|---|---|---|---|---|
| H6 | sí | píxeles | no | 0.869 | 0.690 | 0.135 | 0.608 | 0.851 |
| H6 sin EMA | no | píxeles | no | 0.872 | 0.649 | 0.090 | 0.570 | 0.855 |
| H7 | no | mm | no | 0.873 | 0.654 | 0.228 | 0.636 | 0.855 |
| H7 con EMA | sí | mm | no | 0.871 | 0.720 | 0.208 | 0.635 | 0.853 |
| **H8** | no | mm | sí | 0.865 | **0.860** | **0.835** | 0.612 | 0.849 |

El ruido es gaussiano, con desviación σ de 10 y 25 niveles de gris sobre 255 en la imagen de
entrada (no es un porcentaje); el giro se aplica en milímetros.

- **EMA:** quitarlo no cuesta nada en imágenes limpias (+0.002 y +0.003, dentro del ruido entre
  semillas de ±0.004). Con ruido leve el EMA ayudaba algo (0.04 a 0.07).
- **Giro en milímetros:** no cuesta nada en limpio y mejora la tolerancia a un giro de 60° (+0.03 y
  +0.07).
- **Ruido al entrenar:** H8 es el más robusto de los cinco frente al ruido gaussiano. Cuesta 0.008
  en imágenes limpias frente a H7 y gana +0.21 y +0.61 con ruido de σ = 10 y 25; mejora a H7 con ruido
  en los 15 casos.
- **Giro:** ningún modelo aguanta 60° por sí solo; con el enderezado previo todos quedan entre
  0.849 y 0.855.

Con ruido, las demás cabezas de H8 casi no se mueven: con ruido 25, F1 0.987, mAP50 0.957 y Dice
semántico 0.955.

**Alcance de la robustez.** H8 es el más robusto de los cinco frente a ruido gaussiano sintético,
pero no garantiza superar el objetivo bajo cualquier daño:

| Condición (validación) | Dice por fragmento de H8 | ¿Supera 0.85? |
|---|---|---|
| Sin daño | 0.865 | sí |
| Ruido gaussiano σ = 10 sobre 255 (≈ 4 % del rango) | 0.860 | sí |
| Ruido gaussiano σ = 25 sobre 255 (≈ 10 % del rango) | 0.835 | no |
| Giro de 60° sin enderezar | 0.612 | no |
| Giro de 60° con enderezado | 0.849 | no, por 0.001 |

El ruido probado es el mismo tipo que se usó para entrenar; no se midió con artefactos reales de CT.
El giro grande depende del enderezado previo, que la demo aplica desde esta versión
(`src/demo_export.py`) y la evaluación oficial no.

### 4.4 Ablación de CBAM (sobre H8)

Modelo con la receta exacta de H8 entrenado sin CBAM (`--cbam none`). Diferencia a favor de CBAM,
media por cabeza:

| Cabeza | Validación | Test |
|---|---|---|
| Clasificación (F1, AUC) | +0.002 | +0.004 |
| Detección (IoU, mAP50, mAP50:95) | +0.004 | +0.012 |
| Segmentación (Dice e IoU semánticos y por fragmento) | +0.001 | −0.001 |
| Global, nueve métricas | +0.003 (+0.3 %) | +0.005 (+0.5 %) |

CBAM aporta en detección y clasificación, no en segmentación; la mayor diferencia es el mAP50:95 en
test (0.898 contra 0.879). Es el mismo patrón que la ablación sobre D5 (F1 0.988 contra 0.975,
mAP50:95 0.883 contra 0.872, Dice por fragmento igual). Apagar CBAM en un H8 ya entrenado hunde el
modelo (F1 0.39 a 0.47): eso mide dependencia, no aporte.

### 4.5 Otras ablaciones del enunciado (sobre D5)

| Ablación | Resultado | Conclusión |
|---|---|---|
| Transfer learning (backbone preentrenado en Intel Image Classification, 2 semillas) | Dice por fragmento 0.851 y 0.856; secundarios +0.011 y −0.007 | Neutral; el modelo final no usa preentrenamiento |
| λ × 2 | Dice por fragmento 0.855 (ruido); mAP50 crudo 0.827 contra 0.868 | Se mantiene λ por defecto |

### 4.6 Comparación con SAM zero-shot

SAM ViT-B en modo zero-shot, con la caja que predice H8 como prompt, comparado con la máscara
semántica de H8 dentro de la misma caja. Uno de cada tres cortes con hueso: 1 181 cortes y 3 002
pares corte-región en test; 1 308 cortes y 3 393 pares en validación. SAM no forma parte del
pipeline ni genera etiquetas (`src/sam_baseline.py`).

| Región (test) | Dice SAM | Dice H8 | IoU SAM | IoU H8 |
|---|---|---|---|---|
| Sacro | 0.794 | 0.886 | 0.690 | 0.824 |
| Coxal izquierdo | 0.914 | 0.943 | 0.860 | 0.901 |
| Coxal derecho | 0.930 | 0.952 | 0.880 | 0.913 |
| **Global test** | **0.889** | **0.931** | **0.824** | **0.886** |
| Global validación | 0.866 | 0.919 | 0.791 | 0.866 |

H8 supera a SAM en las tres regiones y en las dos particiones (+0.042 de Dice global en test). La
mayor diferencia está en el sacro (+0.092), donde la caja es grande y el hueso esponjoso confunde a
SAM. Con H6 la comparación había dado lo mismo (SAM 0.889 contra 0.935 en test). Matices: SAM
recibe las cajas de nuestro detector, lo que lo favorece en localización, y se compara la región
completa porque SAM no separa ni clasifica fragmentos, que es lo que aporta el modelo propio. SAM
tarda unos 450 a 475 ms por corte en una GPU T4 solo en su parte (en una L4 se habían medido
259 ms), frente a unos 64 ms del modelo propio completo en una L4.

### 4.7 Peores casos (test)

039 (0.797), 054 (0.831), 061 (0.833), 007 (0.839) y 047 (0.841): los mismos cinco que con H6. En
todos el fragmento principal se segmenta bien y el error se concentra en los secundarios (0.60 a
0.77): volumen de baja resolución que se sobrefragmenta (039), intercambio principal/secundario
entre fragmentos de tamaño parecido (061, 054), fragmentos pequeños en contacto (007) y hueso fino en
los extremos del volumen (047). El caso 012, el paciente girado que hundía el test con D5 (0.692),
queda en 0.898. La galería de los cinco peores casos del informe es de H6.

### 4.8 Latencia (mediana por corte de 384 px, lote 1)

| Configuración | GPU (NVIDIA L4) | CPU (12 hilos, Colab) | CPU (equipo del grupo, 6 hilos) |
|---|---|---|---|
| 1 modelo | 63.7 ms | 161.7 ms | 154.6 ms |
| 1 modelo + TTA de espejo (entregado) | 71.6 ms | 249.6 ms | 375.4 ms |

La columna del equipo del grupo se midió con H8 (`src/measure_latency.py`,
`outputs/demo/latency.json`). Las de Colab se midieron con H6; H8 tiene exactamente la misma
arquitectura, así que su latencia es la misma.

### 4.8.1 Cuantización

El mismo H8 en menos bits, sin reentrenar (detalle en `docs/cuantizacion.md`). Validación con TTA y
test una vez por variante elegida, en Kaggle:

| Precisión | Para | Tamaño | Latencia por corte | Dice por fragmento (val / test) |
|---|---|---|---|---|
| fp32 (entrenado) | Referencia | 11.0 MB | 183 ms CPU · 11 ms GPU T4 | 0.8652 / 0.8739 |
| **int8 híbrido** (4 transpuestas en fp32) | **CPU (la Lambda)** | 5.0 MB | **81 ms** CPU | 0.8653 / 0.8745 |
| **fp16** | GPU | 5.5 MB | **6 ms** GPU T4 | 0.8651 / 0.8740 |

Ninguna pierde Dice. La API usa int8 en CPU y fp16 en GPU (`PRECISION=auto`); las métricas oficiales
de este documento son las de fp32.

### 4.9 Distancia de separación y métricas de superficie (41 secundarios)

Distancia mínima borde a borde en 3D entre cada fragmento secundario y el principal de su hueso, con
`distance_transform_edt` y el espaciado de la cabecera, medida en la predicción y en el ground truth
(`src/infer_volume.py --distance-nofill`).

| Métrica | Validación H8 | Test H8 | Test H6 |
|---|---|---|---|
| Fragmentos con distancia predicha | 38 de 41 | 37 de 41 | 36 de 41 |
| Error medio de distancia | 0.312 mm | 1.516 mm | 1.277 mm |
| Error mediano / percentil 90 | 0.0 / 0.0 mm | 0.0 / 2.56 mm | 0.0 / 0.56 mm |
| Correlación con la distancia real | 0.956 | 0.811 | 0.818 |
| Dice 3D principal / secundario | 0.918 / 0.722 | 0.927 / 0.777 | 0.931 / 0.756 |
| Secundarios perdidos en 3D | 5.0 % | 5.6 % | 7.8 % |

Frente a H6, H8 encuentra más secundarios en 3D (5.6 % perdidos contra 7.8 % en test) y los
segmenta mejor (Dice 0.777 contra 0.756), pero su error medio de distancia es algo mayor. En test, un
solo fragmento del caso 007 concentra el 64 % del error (9.4 mm reales, 45.5 mm predichos: la
predicción le agrega un componente aislado lejos del hueso, igual que con H6); sin él, el error medio
sería de 0.56 mm. En validación, los dos mayores errores son el fragmento del caso 099 (12.8 mm
reales, 18.6 predichos) y el del 087 (5.3 mm reales, dado por pegado). Solo 5 de los 41 secundarios
de validación y 4 de test están realmente separados del principal, así que estas cifras se apoyan
en pocos casos.

### 4.10 HD95 y ASSD por fragmento

Métricas de superficie del reto, en mm, sobre el volumen 3D (`src/metricas_superficie.py`). Cada
fragmento real se empareja con el predicho de mayor IoU de su región; sin pareja, cuenta como no
detectado.

| Partición | Grupo | Fragmentos | Cobertura | HD95 media / mediana | ASSD media / mediana |
|---|---|---|---|---|---|
| Validación | Todos | 86 | 96.5 % | 12.29 / 4.05 | 2.40 / 0.86 |
| | Principales | 45 | 100 % | 7.05 / 2.31 | 1.11 / 0.50 |
| | Secundarios | 41 | 92.7 % | 18.50 / 7.72 | 3.93 / 1.45 |
| Test | Todos | 86 | 95.3 % | 9.40 / 2.73 | 1.53 / 0.67 |
| | Principales | 45 | 100 % | 7.23 / 2.01 | 1.13 / 0.51 |
| | Secundarios | 41 | 90.2 % | 12.03 / 8.09 | 2.02 / 1.22 |

No detectados: tres secundarios en validación y cuatro en test. La media supera a la mediana por
unos pocos fragmentos con un trozo aislado lejos del hueso, que el HD95 castiga.

### 4.11 Otras rotaciones (medidas con H7)

Volúmenes oblicuos e inclinados fuera del plano axial, en validación. Se midieron con H7, que
comparte con H8 la orientación por metadatos y el giro del aumento en milímetros.

- **Volumen oblicuo** (casos 014, 044 y 099, con cabeceo de 10°, inclinación lateral de 20°, ambos
  combinados y giro de 15° en el plano): la normalización lee la inclinación de la matriz de
  dirección (de 10.0° a 22.3°) y la corrige. El Dice de la orientación pasa de 0.24-0.55 sin
  corregir a 0.93-0.99 corregido (`outputs/h7/otras_rotaciones/prueba1_oblicuo.csv`).
- **Paciente inclinado dentro del escáner** (no está en los metadatos y no se corrige): con 10° y
  20° de cabeceo, el Dice por fragmento de H7 baja de 0.873 a 0.864 y 0.858; con 10° y 20° de
  inclinación lateral, a 0.869 y 0.858 (`outputs/h7/otras_rotaciones/prueba2_inclinacion.csv`).

### 4.12 Grosor de corte y umbrales del post-proceso en mm³

Volúmenes con cortes más gruesos, en validación. El modelo es 2D y recibe cada corte llevado a
384 × 384, así que acepta cualquier número de cortes y cualquier tamaño de corte; lo que depende de
la dimensión es el post-proceso 3D, cuyos tamaños mínimos (30 vóxeles por fragmento y 200 por región)
se contaban en vóxeles. Desde esta versión se pueden fijar en mm³ (`--umbral-mm`, 26.1 y 173.8 mm³,
el equivalente con el vóxel mediano del caché de PENGWIN). La prueba conserva 1 de cada k cortes
(`--zstep k`) y multiplica el espaciado z por k (`outputs/h8/validacion/grosor_corte/`).

| Cortes | Grosor aprox. | Umbrales | Error medio de distancia | Correlación | Dice 3D principal / secundario | Cobertura | HD95 / ASSD medianos | Predichos sin pareja |
|---|---|---|---|---|---|---|---|---|
| todos | 0.8 mm | vóxeles | 0.312 mm | 0.956 | 0.918 / 0.722 | 96.5 % | 4.05 / 0.86 mm | 297 |
| todos | 0.8 mm | mm³ | 0.312 mm | 0.956 | 0.918 / 0.722 | 96.5 % | 4.05 / 0.86 mm | 278 |
| 1 de 2 | 1.6 mm | vóxeles | 0.328 mm | 0.945 | 0.918 / 0.721 | 96.5 % | 4.23 / 0.83 mm | 257 |
| 1 de 2 | 1.6 mm | mm³ | 0.328 mm | 0.945 | 0.918 / 0.721 | 96.5 % | 4.23 / 0.83 mm | 278 |
| 1 de 3 | 2.4 mm | vóxeles | 0.319 mm | 0.945 | 0.917 / 0.726 | 96.5 % | 4.25 / 0.75 mm | 238 |
| 1 de 3 | 2.4 mm | mm³ | 0.319 mm | 0.945 | 0.917 / 0.726 | 96.5 % | 4.25 / 0.75 mm | 284 |

Lectura:

- **El sistema tolera cortes hasta tres veces más gruesos.** El error de distancia pasa de 0.31 a
  0.33 mm, la cobertura se mantiene en 96.5 % y el Dice 3D del principal en 0.92. Las nueve métricas
  del curso se calculan corte a corte, así que no dependen del número de cortes.
- **Los umbrales en mm³ hacen el comportamiento estable.** Dan exactamente las mismas distancias,
  Dice y cobertura que los de vóxeles en los tres grosores. La diferencia está en los fragmentos
  espurios: en vóxeles su número cambia con el grosor (297, 257, 238), porque el mismo umbral equivale
  a más milímetros cúbicos; en mm³ se mantiene entre 278 y 284. La demo usa los umbrales en mm³ desde
  esta versión; en los cuatro casos exportados la comparación con el ground truth no cambió y solo
  desaparecieron fragmentos predichos pequeños (de 30 a 26 en el caso 044, de 27 a 26 en el 099).
- **La propia medida cambia con el grosor.** La distancia real de un mismo fragmento, medida sobre
  el ground truth submuestreado, se mueve hasta 1.4 mm (5.3 → 6.7 mm en el caso 087 y 3.2 → 4.2 mm en
  el 024 con la mitad de los cortes): con cortes gruesos la separación mínima se resuelve peor. El
  error del modelo frente al ground truth del mismo grosor sigue en unos 0.3 mm.
- **Alcance.** Se probó hasta unos 2.4 mm. No se probaron cortes de 3 a 5 mm, comunes en clínica, ni
  cortes gruesos reales: quedarse con uno de cada k no reproduce el promediado de volumen parcial de
  un corte grueso adquirido así.

### 4.13 Enderezado del giro axial en la demo

La demo endereza el giro axial del paciente antes de predecir (`src/demo_export.py`, con
`normalizacion.normalizar`): estima el ángulo con la simetría izquierda-derecha de la pelvis y solo
corrige giros claros de 10° o más. En los cuatro casos exportados el giro estimado fue de −1.5°,
+0.5°, −0.5° y +4.8°, así que ninguno se corrigió y su resultado no cambió. Sobre el caso 024 girado
a propósito, estimó 28.5° para 30° y 58.8° para 60° y los corrigió. La evaluación oficial no aplica
este paso.

## 5. Protocolo de evaluación

- Todo se elige en validación. Una variante se adopta solo si mejora al menos 0.003, con bootstrap
  por paciente (P ≥ 0.90) y mejora en las dos mitades de validación. Para la elección de H8 el
  criterio fue la robustez a imágenes dañadas, que supera ese margen con mucha holgura.
- Las cifras de test se reportan todas, incluidas las que no se usaron, y no se ajusta ningún
  umbral después de verlas.
- Ruido y giro se miden solo en validación (`evaluate_volumes.py --stress`).
- HD95, ASSD, la distancia de separación y la comparación con SAM en test usan el mismo modelo y
  la misma configuración congelada de la evaluación de H8; son mediciones adicionales, no una nueva
  elección.

## 6. Limitaciones

1. **Una sola semilla.** H8 se entrenó solo con la semilla 42; no se conoce su variación propia
   (en el proyecto el ruido entre semillas es de ±0.004 en Dice por fragmento).
2. **Costo en imágenes limpias.** 0.008 menos que H7 en validación y 0.005 menos en test.
3. **Ruido sintético.** Se probó ruido gaussiano, el mismo tipo con que se entrenó, y solo en
   validación. No se midió con artefactos reales de CT (metal, movimiento, dosis baja).
4. **Orientación de la entrada.** El modelo espera los volúmenes en orientación canónica; la demo
   (`src/demo_export.py`) lee la cabecera de cada archivo y endereza el giro axial del paciente si es
   de 10° o más (sección 4.13). Sin ese paso, un giro de 60° baja el Dice a 0.61; con él, a 0.849. El
   enderezado solo corrige el giro en el plano axial, no el cabeceo ni la inclinación lateral, y la
   API recibe cortes ya enderezados porque estimar el giro necesita el volumen completo.
5. **Fragmentos en contacto en la superficie de fractura.** Cuando dos fragmentos se tocan, la
   segmentación puede fusionarlos en uno solo. En ese caso se pierde el fragmento secundario y su
   distancia de separación al principal no es confiable (puede salir 0 mm o no existir).
6. **Fragmentos secundarios.** Son el punto débil: Dice de 0.760 frente a 0.910 del primario en test,
   HD95 mediano cercano a 8 mm y uno de cada diez sin detectar en 3D.
7. **Fragmentos sobrantes.** El Dice por fragmento no penaliza los fragmentos predichos que sobran
   (258 en test sin pareja real). El dashboard oculta los menores de 1000 mm³; es un filtro de
   presentación que no cambia la métrica.
8. **Modelo 2D.** Procesa corte a corte. La coherencia entre cortes depende del post-proceso 3D.
   Acepta cualquier número de cortes; con cortes más gruesos el post-proceso se probó hasta unos
   2.4 mm (sección 4.12), no con los 3 a 5 mm habituales en clínica.
9. **Tamaño de la muestra.** Test tiene 15 pacientes de un solo dataset.
10. **Evaluaciones repetidas en test.** La cifra de H8 no es una estimación limpia (sección 4.1).
11. **SAM se compara solo por región.** SAM no separa fragmentos, así que la comparación de la
    sección 4.6 es de la segmentación semántica dentro de la caja, no del Dice por fragmento.
12. **El pulido 3D es solo visual.** En el visualizador 1 la malla "pulida" agrega la recuperación de
    hueso tenue (histéresis 150/250 HU, cierre y relleno de cavidades), y en el visualizador 3 cada
    fragmento se dibuja con sus cavidades cerradas rellenas. Nada de eso entra al modelo, a las
    métricas ni a las distancias, que usan los datos sin modificar. Aun pulidos, los casos 025, 026 y
    066, con hueso de muy baja densidad, siguen incompletos: la recuperación solo llega al 51 %, 48 % y
    54 % del hueso etiquetado (`docs/resumen_semana8.md`, sección 8.5).

## 7. Consideraciones éticas

- Los datos son públicos y anonimizados, del challenge PENGWIN 2024.
- El modelo no debe usarse para diagnóstico ni planificación quirúrgica. Un error en la separación
  de fragmentos o en la lateralidad tendría consecuencias clínicas.
- El rendimiento puede bajar con escáneres, protocolos o poblaciones distintas a las del dataset.

## 8. Despliegue

El dashboard (Streamlit) está publicado en Streamlit Community Cloud y la inferencia en vivo de un
corte corre en una API en AWS Lambda en CPU; los dos se vuelven a desplegar en cada push a `main`
(ver `docs/despliegue_aws_streamlit.md`). La imagen de la API copia el checkpoint de H8 y al
construirse genera su versión int8, que es la que sirve la Lambda (CPU); con GPU usaría fp16 (sección
4.8.1). El profesor aprobó este despliegue en lugar del túnel de Cloudflare que nombra el
enunciado. La advertencia de uso no clínico está siempre visible en el dashboard.

## 9. Modelos entrenados y pesos en el repositorio

El sistema es un solo modelo multitarea, así que hay un model card. Los pesos versionados son:

| Archivo | Qué es | En qué difiere del entregado |
|---|---|---|
| `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth` | **Modelo entregado** (H8, semilla 42) | — |
| `outputs/h7/h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42.pth` | H7, modelo de la tercera evaluación en test | Sin aumento de ruido |
| `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s4{2,3,4}.pth` | H6, el modelo anterior, con tres semillas | Con EMA, giro del aumento en píxeles y sin aumento de ruido |
| `outputs/s10/d5_ema_w32_e60_s42.pth` | D5, el modelo de la primera evaluación en test | Además, sin orientación canónica ni aumento geométrico |
| `outputs/gamma/w10g_384neg.pth` y `w10g_sec3.pth` | Primera versión de la Semana 10, usada como línea base | Backbone de 16 canales y entrenamiento en etapas |

Los modelos de la comparación de la sección 4.3 (H6 sin EMA, H7 con EMA) y el de la ablación de
CBAM no se versionan; sus resultados están en `outputs/h8/`. Los pesos y resultados de H7 están en
`outputs/h7/`.

Datos de entrenamiento, alcances, limitaciones y advertencia de uso son los mismos para todos:
los 70 casos de entrenamiento de PENGWIN (sección 2), las limitaciones de la sección 6 y el uso
exclusivamente académico.

## 10. Reproducción

Los comandos para regenerar los datos, entrenar H8 y evaluarlo están en el `README.md` y en
`docs/modelo_h8.md`. Los kernels de Kaggle con que se entrenó y evaluó están en `kaggle/h8/`. Los
resultados están en `outputs/h8/` y el recorrido con tablas y figuras, en
`notebooks/11_modelo_h8.ipynb`.

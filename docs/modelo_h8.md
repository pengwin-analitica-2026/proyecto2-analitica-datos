# Modelo H8 — detección, segmentación y clasificación de fragmentos de fractura pélvica

Proyecto PENGWIN, corte 2. Octubre de 2026.

> **Uso exclusivamente académico.** No es un dispositivo médico, no ha sido validado clínicamente y
> no debe usarse para apoyar decisiones quirúrgicas.

## 1. Resumen

| | |
|---|---|
| **Archivo** | `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth` (11 MB) |
| **Qué es** | El modelo elegido después de la asesoría con el profesor. Reemplaza a H6 |
| **Qué cambia frente a H6** | Sin EMA, rotación del aumento en milímetros reales y aumento de ruido gaussiano |
| **Arquitectura** | La misma de H6: backbone propio de 32 canales, CBAM de una etapa y tres cabezas. 2.74 M de parámetros |
| **Entrenamiento** | 60 épocas, semilla 42, 67 minutos en una GPU T4 de Kaggle |
| **Dice por fragmento** | 0.865 en validación y 0.874 en test |
| **Objetivos del curso** | Cumple los nueve, en validación y en test |

Por qué se eligió: es el único de los cinco modelos comparados que resiste imágenes con ruido (0.860
con ruido leve y 0.835 con ruido fuerte, donde los demás dan como mucho 0.720 y 0.228), y lo hace
sin EMA, que era la observación del profesor. El costo es de 0.008 de Dice por fragmento en
imágenes limpias frente a H7. La comparación completa está en `docs/experimentos_ema_rotacion_ruido.md`
y el recorrido con tablas y figuras, en `notebooks/11_modelo_h8.ipynb`.

## 2. De dónde viene: la cadena de cambios

Cada modelo es el anterior con cambios puntuales. La arquitectura no cambia desde D5.

| Modelo | Qué agrega al anterior | Por qué | Dice por fragmento (validación / test) |
|---|---|---|---|
| D5 | 32 canales, CBAM de una etapa, EMA 0.999 | Mejor receta de la Semana 10 | 0.851 / 0.845 |
| H6 | Orientación canónica y aumento geométrico con rotación de ±35° | 34 de los 100 volúmenes están guardados girados 180°; el modelo no veía ningún aumento geométrico | 0.869 / 0.875 |
| H7 | Quita el EMA; la rotación del aumento pasa de píxeles a milímetros | Asesoría del profesor: no depender del EMA y normalizar las rotaciones con los metadatos | 0.873 / 0.879 |
| **H8** | **Aumento de ruido gaussiano** | El ruido era el punto débil de todos los modelos anteriores | **0.865 / 0.874** |

## 3. Los tres cambios frente a H6, en detalle

### 3.1 Sin EMA

- **Antes (H6).** Durante el entrenamiento se mantenía un promedio móvil exponencial de los pesos
  (decaimiento 0.999) y ese promedio era el modelo que se evaluaba y se guardaba.
- **Ahora (H8).** `--ema 0`. Se guardan los pesos del propio entrenamiento, en la época con mejor
  validación.
- **Motivo.** El profesor señaló que el EMA promedia lo aprendido en las condiciones "bonitas" del
  entrenamiento y puede fallar con imágenes distintas.
- **Qué se midió.** Quitar el EMA no cuesta nada en imágenes limpias: +0.003 con rotación en píxeles
  y +0.002 con rotación en milímetros, dentro del ruido entre semillas (±0.004). Con ruido leve el
  EMA ayudaba algo (0.04 a 0.07); el aumento de ruido de H8 recupera mucho más que eso.

### 3.2 Rotación del aumento en milímetros reales

- **El problema.** El caché lleva cada corte a 384 × 384 píxeles aunque el original no sea cuadrado.
  Un píxel del caché puede medir, por ejemplo, 0.9 mm de ancho y 1.6 mm de alto. Girar en esa
  grilla no es un giro: gira y deforma la pelvis a la vez. En 10 de los 15 casos de validación los
  píxeles no son cuadrados.
- **Antes (H6).** `random_affine` aplicaba el giro directamente en la grilla de píxeles.
- **Ahora (H8).** `--aug-mm`. La transformación se define en milímetros y se lleva a píxeles:

  ```
  theta_píxeles = D⁻¹ · A_mm · D,     D = diag(ancho · mm_x, alto · mm_y)
  ```

  `A_mm` es la misma combinación de giro, escala y proporción de antes. Los milímetros por píxel de
  cada corte salen del espaciado de la cabecera y de la forma original del volumen. Con píxeles
  cuadrados el resultado es idéntico al anterior.
- **Dónde está.** `src/aumento_fisico.py` (`random_affine_mm`, `theta_mm`, `mm_por_pixel`) y la
  opción `--aug-mm` de `src/train_s10.py`. Pruebas en `tests/test_aumento_fisico.py`.
- **Qué se midió.** No cuesta nada en imágenes limpias y mejora la tolerancia a un giro de 60° en
  +0.03 con EMA y +0.07 sin EMA.

Este cambio va junto con la normalización de la orientación con los metadatos
(`src/normalizacion.py`): la matriz de dirección de la cabecera se analiza con productos punto
(perpendicularidad de los ejes y con qué eje del paciente se alinea cada uno) y producto cruz
(normal a los cortes y si el volumen está espejado), y el volumen se lleva a una orientación común.
En los 100 casos de PENGWIN da el mismo resultado que la regla anterior de H6.

### 3.3 Aumento de ruido gaussiano

- **El problema.** Con ruido gaussiano de desviación 10 sobre 255 en la entrada, H6 y H7 bajan de
  0.87 a 0.69 y 0.65 de Dice por fragmento. Con desviación 25 colapsan (0.14 y 0.23).
- **Ahora (H8).** `--aug-noise 0.1`. En el 30 % de las muestras de cada lote se suma ruido gaussiano
  con una desviación al azar entre 0 y 0.1 en la escala 0-1 (hasta unos 25 niveles de gris):

  ```python
  if args.aug_noise > 0:
      on = torch.rand(x.shape[0]) < 0.3
      sigma = torch.rand(x.shape[0], 1, 1, 1) * args.aug_noise
      x[on] = (x[on] + torch.randn_like(x[on]) * sigma[on]).clamp(0.0, 1.0)
  ```

  Se aplica después del aumento geométrico y de intensidad, solo en entrenamiento. El valor queda
  guardado en el checkpoint (`aug_noise`).
- **Dónde está.** Opción `--aug-noise` de `src/train_s10.py`.
- **Qué se midió.** Sección 5.3.

### 3.4 Lo que no cambió

Backbone, CBAM, las tres cabezas, la pérdida y sus pesos, el tamaño de entrada, el preprocesamiento
de intensidad, la orientación canónica, el post-proceso y la inferencia son los de H6.

## 4. Ficha técnica

### 4.1 Arquitectura

- **Backbone:** `FundidoraBackbone`, cuatro bloques convolucionales con 32 canales base y reducción
  espacial de 16. CBAM (atención de canal y espacial) en una etapa, antes de bifurcar.
- **Tres cabezas, entrenadas desde cero:**
  1. clasificación multietiqueta de las regiones presentes en el corte;
  2. detección con grid de 24 × 24 celdas y supresión de no máximos propios;
  3. segmentación, con una salida semántica (fondo y tres regiones) y una de instancia (fragmento
     principal o secundario por región).
- Sin YOLO, Detectron2, Mask R-CNN ni otro framework de detección o segmentación. Sin
  preentrenamiento.

### 4.2 Datos y preprocesamiento

- PENGWIN 2024, Task 1 (CT): 100 volúmenes `.mha`, partición 70 / 15 / 15 por paciente, semilla 42.
- Recorte a [−1024, 3000] HU, ventana ósea C400/W1800, gamma 1.5 y cortes axiales a 384 × 384.
- Orientación canónica: los 34 volúmenes guardados girados 180° se enderezan según su cabecera.

### 4.3 Entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas | 60, de 2 500 cortes cada una |
| Lote | 16 |
| Tasa de aprendizaje | 0.002, con coseno |
| Precisión mixta | Sí |
| Peso de los secundarios | ×3 |
| Pérdida | Detección 1.0, clasificación 0.5, semántica 2.0 (CE) y 1.0 (Dice), instancia 1.0 (CE) y 1.0 (Dice) |
| Aumento geométrico | Escala 0.80-1.25, proporción 0.85-1.18, rotación ±35°, traslación ±8 %, en milímetros |
| Aumento de intensidad | Espejo sagital, ganancia ±5 % y sesgo |
| Aumento de ruido | Gaussiano, 30 % de las muestras, desviación hasta 0.1 |
| EMA | No |
| Semilla | 42 |

Comando:

```bash
python src/train_s10.py --cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 \
    --batch-size 16 --lr 0.002 --base-channels 32 --cbam 1 --instance-mode primary --augment --cosine \
    --margin 10 --neg-fraction 0.15 --per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 \
    --secondary-weight 3.0 --canonical --aug-geo 1.0 --aug-rot-deg 35 --aug-mm --ema 0 \
    --aug-noise 0.1 --seed 42 --output outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth
```

Curva de entrenamiento (validación rápida, un corte de cada seis):

| Época | 1 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| Pérdida de entrenamiento | 4.99 | 0.99 | 0.85 | 0.76 | 0.70 | 0.66 | 0.64 |
| Dice por fragmento | 0.371 | 0.790 | 0.813 | 0.848 | 0.862 | 0.860 | 0.862 |

Completó las 60 épocas en 67 minutos, sin cortarse por tiempo.

### 4.4 Inferencia

Un modelo con TTA de espejo, tamaño 384, y post-proceso con tamaño mínimo de fragmento de 12
píxeles. Sin ensamble, sin enderezado del giro axial y sin suavizado.

## 5. Resultados

Todas las cifras son sobre el volumen completo de cada caso, con TTA de espejo.

### 5.1 Las nueve métricas del curso

| Métrica | Objetivo | Validación (15 casos, 5 047 cortes) | Test (15 casos, 4 664 cortes) |
|---|---|---|---|
| F1 de clasificación | 0.85 | 0.991 | 0.989 |
| AUC de clasificación | 0.85 | 0.999 | 0.999 |
| IoU de detección | 0.65 | 0.946 | 0.938 |
| mAP50 | 0.65 | 0.969 | 0.953 |
| mAP50:95 | 0.40 | 0.900 | 0.898 |
| Dice de segmentación | 0.85 | 0.970 | 0.968 |
| IoU de segmentación | 0.70 | 0.943 | 0.939 |
| Dice por fragmento | 0.85 | 0.865 | 0.874 |
| IoU por fragmento | 0.70 | 0.800 | 0.812 |

H8 cumple los nueve objetivos en las dos particiones.

### 5.2 Desglose

| | Validación | Test |
|---|---|---|
| Dice del fragmento principal | 0.897 | 0.910 |
| Dice de los fragmentos secundarios | 0.769 | 0.760 |
| Secundarios sin pareja | 6.3 % | 7.2 % |
| Principales sin pareja | 0.2 % | 0.2 % |
| Dice por fragmento, centro del volumen | 0.874 | 0.885 |
| Dice por fragmento, extremos del volumen | 0.834 | 0.836 |
| Dice semántico: sacro / coxal izquierdo / coxal derecho | 0.956 / 0.978 / 0.978 | 0.955 / 0.975 / 0.974 |
| Detección con la cabeza grid sola: mAP50 / mAP50:95 | 0.903 / 0.534 | 0.909 / 0.566 |

El mAP de la tabla 5.1 se mide sobre cajas refinadas con la máscara semántica; la última fila es la
cabeza de detección sin ese refinamiento.

Dice por fragmento por caso:

| Partición | Peores cinco | Mejores tres |
|---|---|---|
| Validación | 087 (0.779), 024 (0.831), 095 (0.839), 027 (0.846), 078 (0.847) | 099 (0.931), 050 (0.916), 044 (0.901) |
| Test | 039 (0.797), 054 (0.831), 061 (0.833), 007 (0.839), 047 (0.841) | 043 (0.925), 020 (0.914), 086 (0.908) |

Los cinco peores de test son los mismos que con H6. En todos el fragmento principal se segmenta bien
y el error está en los secundarios (entre 0.60 y 0.77). El caso 012, el paciente girado que hundía
el test con D5 (0.692), queda en 0.898.

### 5.3 Robustez (solo validación)

Dice por fragmento. El ruido es gaussiano, con la desviación indicada sobre 255; el giro se aplica
en milímetros.

| Prueba | H6 | H7 | **H8** |
|---|---|---|---|
| Sin daño | 0.869 | 0.873 | 0.865 |
| Ruido 10 | 0.690 | 0.654 | **0.860** |
| Ruido 25 | 0.135 | 0.228 | **0.835** |
| Giro de 60° | 0.608 | 0.636 | 0.612 |
| Giro de 60° con enderezado | 0.851 | 0.855 | 0.849 |

Con ruido, las otras cabezas de H8 casi no se mueven:

| Prueba | F1 | mAP50 | Dice de segmentación | Dice por fragmento |
|---|---|---|---|---|
| Sin daño | 0.991 | 0.969 | 0.970 | 0.865 |
| Ruido 10 | 0.990 | 0.968 | 0.967 | 0.860 |
| Ruido 25 | 0.987 | 0.957 | 0.955 | 0.835 |

- H8 mejora a H7 con ruido en los 15 casos de validación.
- Un giro de 60° sigue hundiendo al modelo por sí solo; lo resuelve el enderezado previo
  (`--normalizar`), que no forma parte de la inferencia por defecto.

### 5.4 Aporte de CBAM

Comparación con un modelo de la misma receta entrenado sin CBAM (`--cbam none`). Diferencia a favor
de CBAM, media por cabeza:

| Cabeza | Validación | Test |
|---|---|---|
| Clasificación (F1, AUC) | +0.002 | +0.004 |
| Detección (IoU, mAP50, mAP50:95) | +0.004 | +0.012 |
| Segmentación (Dice e IoU semánticos y por fragmento) | +0.001 | −0.001 |
| Global, nueve métricas | +0.003 (+0.3 %) | +0.005 (+0.5 %) |

CBAM aporta en detección y en clasificación, y no en segmentación. La mayor diferencia es el
mAP50:95 en test (0.898 contra 0.879). El detalle métrica por métrica está en
`docs/experimentos_ema_rotacion_ruido.md`, sección 6.

### 5.5 HD95 y ASSD por fragmento

En milímetros, sobre el volumen 3D. Cada fragmento real se empareja con el predicho de mayor IoU de
su región; sin pareja, cuenta como no detectado.

| Partición | Grupo | Fragmentos | Cobertura | HD95 media / mediana | ASSD media / mediana |
|---|---|---|---|---|---|
| Validación | Todos | 86 | 96.5 % | 12.29 / 4.05 | 2.40 / 0.86 |
| | Principales | 45 | 100 % | 7.05 / 2.31 | 1.11 / 0.50 |
| | Secundarios | 41 | 92.7 % | 18.50 / 7.72 | 3.93 / 1.45 |
| Test | Todos | 86 | 95.3 % | 9.40 / 2.73 | 1.53 / 0.67 |
| | Principales | 45 | 100 % | 7.23 / 2.01 | 1.13 / 0.51 |
| | Secundarios | 41 | 90.2 % | 12.03 / 8.09 | 2.02 / 1.22 |

No detectados: tres en validación (casos 014, 024 y 060) y cuatro en test (007, dos del 039 y 054).
Todos son secundarios. La media es mucho mayor que la mediana por unos pocos valores extremos; la
mediana describe mejor el caso típico.

### 5.6 Distancia de separación y métricas de superficie (41 secundarios)

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

## 6. Protocolo de evaluación

- H8 se eligió **solo con validación**, comparándolo con H6, H6 sin EMA, H7 y H7 con EMA.
- La configuración de test se escribió antes de correr (`outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md`) y se evaluó
  una sola vez, sin ajustar nada después.
- Es la **sexta consulta a test** del proyecto: c2 (0.848), EMA + CASR (0.847), D5 (0.845),
  H6 (0.875), H7 (0.879) y H8 (0.874). En la misma sesión pasaron por test el modelo sin CBAM y H8
  con el CBAM apagado, como medición del aporte de CBAM y no para elegir modelo.
- Los casos 012 y 039 de test se usaron para diagnosticar fallos antes de H6. Ninguna cifra de test
  posterior a D5 es una estimación independiente.

## 7. Limitaciones

1. **Una sola semilla.** H8 se entrenó solo con la semilla 42. El ruido entre semillas medido en el
   proyecto es de ±0.004 en Dice por fragmento; no se conoce la variación propia de H8.
2. **Costo en imágenes limpias.** 0.008 menos que H7 en validación y 0.005 menos en test.
3. **El ruido probado es sintético.** Gaussiano, el mismo tipo con que se entrenó, y solo en
   validación. No se midió con artefactos reales de CT (metal, movimiento, dosis baja).
4. **Pacientes girados.** Un giro grande sigue necesitando el enderezado previo, que solo corrige el
   giro en el plano axial y todavía no está conectado a la demo ni a la API.
5. **Fragmentos secundarios.** Dice de 0.76 a 0.77, uno de cada diez sin detectar en 3D y HD95
   mediano cercano a 8 mm.
6. **Fragmentos en contacto.** Cuando dos fragmentos se tocan en la superficie de fractura, la
   segmentación puede fusionarlos; el secundario se pierde y su distancia al principal no es fiable.
7. **Fragmentos predichos de más.** 297 en validación y 258 en test no corresponden a ningún
   fragmento real. Las métricas del curso no los penalizan.
8. **Modelo 2D.** Procesa corte a corte; la coherencia entre cortes depende del post-proceso 3D.
9. **Muestra pequeña.** 15 pacientes por partición, de un solo dataset.
10. **SAM medido con H6.** La comparación con SAM zero-shot no se repitió con H8. Su Dice semántico
    en test es casi igual al de H6 (0.968 contra 0.971), así que es de esperar la misma diferencia.

## 8. Integración en el repositorio

| Qué | Dónde |
|---|---|
| Pesos | `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth` (con excepción en `.gitignore`) |
| `--aug-noise` | `src/train_s10.py` |
| HD95 y ASSD por fragmento | `src/metricas_superficie.py` y `tests/test_metricas_superficie.py` |
| Demo | Casos 024, 044, 099 y 039 exportados con H8 en `outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/`; los de H6 se retiraron |
| Dashboard | Lee los casos de H8 y muestra las cuatro evaluaciones en test (D5, H6, H7 y H8), HD95/ASSD y la distancia |
| API | `api/server.py` y `api/Dockerfile` cargan H8. La Lambda publicada se actualiza cuando la rama llegue a `main` |
| Latencia | Medida con H8 en CPU (`outputs/demo/latency.json`) |
| Model card y README | Describen H8 |
| Kernels de Kaggle | `kaggle/h8/` |

## 9. Archivos

| Qué | Dónde |
|---|---|
| Pesos | `outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth` |
| Historial de entrenamiento | `outputs/h8/h8_aumm_noise01_noema_w32_e60_s42.json` |
| Validación: sin daño, ruido y giro | `outputs/h8/validacion/val_h8_*.json` |
| Test y su configuración congelada | `outputs/h8/test/` |
| HD95 y ASSD | `outputs/h8/{validacion,test}/superficie_*_h8.json` |
| Distancia de separación | `outputs/h8/{validacion,test}/distancias_*_h8.json` |
| Modelo sin CBAM y CBAM apagado | `outputs/h8/cbam/` |
| Comparación con los otros cuatro modelos | `outputs/h8/comparacion/` y `docs/experimentos_ema_rotacion_ruido.md` |
| Figuras | `outputs/h8/figuras/` (las genera el notebook 11) |

Evaluación:

```bash
python src/evaluate_volumes.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta
python src/metricas_superficie.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta
python src/infer_volume.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --distance-nofill --surface-metrics
```

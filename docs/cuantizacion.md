# Cuantización de H8

El profesor sugirió compactar el modelo con una condición: se adopta solo si mantiene las métricas y
no baja el Dice. Este documento explica qué es cuantizar, qué se probó con **H8**
(`h8_aumm_noise01_noema_w32_e60_s42.pth`) y qué se eligió. Todo se corrió en Kaggle (cuenta
`gustaveloper`), sin tocar el modelo entrenado.

## Qué es cuantizar

H8 se entrenó en **fp32**: cada peso y cada valor intermedio es un número de punto flotante de 32
bits. Cuantizar es guardar y calcular esos mismos números con menos bits. No se reentrena: es el
mismo modelo con otra representación numérica.

| Tipo | Bits | Cómo funciona | Dónde rinde |
|---|---|---|---|
| fp16 | 16 | Punto flotante más corto (más decimales, menos rango) | GPU (Tensor Cores) |
| bf16 | 16 | Punto flotante con el rango de fp32 y menos decimales | GPU recientes (Ampere en adelante) y algunas CPU |
| int8 estático (PTQ) | 8 | Enteros con una escala por canal: `valor ≈ escala × entero`. El rango de las activaciones se **calibra** con cortes reales | CPU (instrucciones vectoriales int8). Es lo estándar para CNN en CPU |
| int8 con QAT | 8 | Se reentrena simulando la cuantización | Solo si PTQ pierde precisión (aquí no hizo falta) |
| int4 | 4 | 16 niveles | Pensado para la memoria de modelos de lenguaje enormes; PyTorch no tiene convoluciones int4 en CPU. No se probó |

int8 se calibra con 128 cortes con hueso de **8 casos de entrenamiento** (16 por caso), nunca de
validación ni de test. Los pesos se cuantizan por canal (FX de PyTorch, backend x86).

## Las capas de H8 y por qué las transpuestas quedan en fp32

Una capa es un paso de cálculo con sus propios pesos aprendidos. H8 tiene 31 capas con pesos (más
normalizaciones y activaciones, que no tienen pesos que cuantizar):

| Parte | Capas | Qué hace |
|---|---|---|
| Encoder (backbone) | 8 convoluciones, en 4 etapas | Extrae rasgos; cada etapa reduce la imagen a la mitad (384 → 24) |
| CBAM | 3 convoluciones pequeñas | Atención: qué canales y qué zonas importan |
| Cabeza de clasificación | 2 capas lineales | Qué regiones hay en el corte |
| Cabeza de detección | 2 convoluciones | Cajas |
| Decoder de segmentación | **4 convoluciones transpuestas** + 4 convoluciones + 3 de conexión | Vuelve a agrandar la imagen (24 → 48 → 96 → 192 → 384) para pintar la máscara |
| Salidas de segmentación | 5 convoluciones 1×1 | Hueso, fragmento, bordes, centros y desplazamientos |

Una **convolución transpuesta** hace lo contrario de una convolución con paso 2: cada valor de entrada
se «estampa» en un parche de 4×4 de la salida, que queda del doble de tamaño. Las 4 transpuestas
(`segmentation_head.decoder.0` a `decoder.3`) llevan el 26 % de los parámetros (704 mil de 2.7
millones) y trabajan con las imágenes más grandes del decoder.

La variante elegida, **int8 híbrido**, cuantiza las otras 27 capas y deja esas 4 en fp32: al llegar a
una transpuesta el valor pasa a fp32, la capa calcula como antes y vuelve a int8.

**Con H8 el motivo es solo la velocidad, no la precisión.** Cuantizándolas también (`int8_todo`), el
Dice es el mismo: 0.8654 en validación en Kaggle (torch 2.11), y la misma segmentación que el híbrido
en el torch 2.2 del Mac (Dice frente a fp32 de 0.993 / 0.998 / 0.998 en el caso 024). Lo que pasa es
que PyTorch tiene código int8 muy optimizado para las convoluciones normales pero no para las
transpuestas en CPU: cuantizadas, esas 4 capas van por una ruta lenta y el modelo completo termina
**más lento que el original** en la CPU de Kaggle (197 frente a 183 ms por corte), mientras que el
híbrido es 2.3 veces más rápido (81 ms). `int8_todo` pesa menos (2.8 frente a 5.0 MB): sería la opción
si importara la memoria y no el tiempo.

(Con **H6** y torch 2.2 cuantizar las transpuestas sí rompía la segmentación, Dice 0.03; por eso se
excluyeron al principio. Con H8 eso ya no ocurre.)

### Por qué la ganancia depende de la CPU

| | CPU | Instrucciones int8 (VNNI) | fp32 | int8 híbrido | int8 todo |
|---|---|---|---|---|---|
| Kaggle | Xeon de servidor | Sí (AVX-512 VNNI) | 183 ms | **81 ms** | 197 ms |
| Mac de Gustavo | i9-9980HK de portátil (2019) | No (AVX2) | 205 ms | 130 ms | 149 ms |

Los Xeon de servidor multiplican enteros de 8 bits con instrucciones dedicadas; un portátil sin ellas
gana menos. Las Lambda x86 de AWS corren en Xeon, así que se espera un comportamiento como el de Kaggle.
(La medición del Mac es corta, 16 cortes, e indicativa.)

## Resultados en validación (15 casos, con TTA)

Kaggle, CPU de 4 núcleos (`fp32`, `int8`, `int8_todo`, un kernel por variante) y GPU Tesla T4
(`fp32`, `fp16`, `bf16`, un kernel). La diferencia de Dice se mide contra fp32 en el mismo
dispositivo. Archivos en `outputs/h8/cuantizacion/`.

| Variante | Dispositivo | Tamaño | Latencia por corte (mediana / p90) | Dice por fragmento | Δ vs fp32 | Dice semántico | F1 | mAP50 | Dice 3D | Error de distancia |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | CPU | 11.0 MB | 183 / 195 ms | 0.8652 | — | 0.9704 | 0.9909 | 0.9689 | 0.8232 | 0.31 mm |
| **int8** | CPU | **5.0 MB** | **81 / 84 ms** | 0.8653 | +0.0002 | 0.9698 | 0.9896 | 0.9689 | 0.8237 | 0.31 mm |
| int8_todo | CPU | 2.8 MB | 197 / 201 ms | 0.8654 | +0.0002 | 0.9697 | 0.9896 | 0.9692 | 0.8233 | 0.29 mm |
| fp32 | GPU T4 | 11.0 MB | 11 / 11 ms | 0.8652 | — | 0.9704 | 0.9909 | 0.9689 | 0.8232 | 0.31 mm |
| **fp16** | GPU T4 | **5.5 MB** | **6 / 6 ms** | 0.8651 | −0.0000 | 0.9703 | 0.9909 | 0.9687 | 0.8233 | 0.31 mm |
| bf16 | GPU T4 | 5.5 MB | 12 / 19 ms | 0.8652 | +0.0001 | 0.9704 | 0.9909 | 0.9693 | 0.8233 | 0.34 mm |

- fp32 reproduce la validación oficial de H8 (0.8652, `outputs/h8/validacion/val_h8_base.json`).
- Todas cumplen la regla del protocolo (el Dice por fragmento no baja más de 0.003): la diferencia en
  precisión es nula. Lo que cambia es el tamaño y la velocidad.
- En CPU, int8 es 2.3 veces más rápido que fp32 y pesa menos de la mitad.
- En GPU, fp16 es 1.8 veces más rápido. bf16 no acelera en la T4 (sin soporte nativo).
- Las latencias de cada kernel se midieron en máquinas distintas de Kaggle con el mismo tipo de CPU;
  la diferencia entre int8 y fp32 es mucho mayor que esa variación.

## Elección y test

Se eligió en validación una variante por escenario de despliegue, y solo esas se reportaron en test,
una vez cada una:

- **CPU (la demo en AWS Lambda): int8** con las transpuestas en fp32.
- **GPU: fp16.**

No se mandaron a test `int8_todo` ni `bf16`: no ganan en ningún escenario.

| Variante | Dispositivo | Tamaño | Latencia por corte | Dice por fragmento | Dice semántico | F1 | mAP50 | Dice 3D | Error de distancia |
|---|---|---|---|---|---|---|---|---|---|
| fp32 (oficial, `outputs/h8/test/test_h8.json`) | — | 11.0 MB | — | 0.8739 | 0.9682 | 0.9885 | 0.9533 | — | — |
| fp32 (reproducción) | GPU T4 | 11.0 MB | 11 ms | 0.8739 | 0.9682 | 0.9885 | 0.9533 | 0.8503 | 1.52 mm |
| **int8** | CPU | 5.0 MB | 109 ms | **0.8745** | 0.9676 | 0.9890 | 0.9513 | 0.8501 | 1.28 mm |
| **fp16** | GPU T4 | 5.5 MB | 5 ms | **0.8740** | 0.9682 | 0.9886 | 0.9530 | 0.8503 | 1.52 mm |

En test, las dos variantes elegidas dan el mismo Dice que fp32 (+0.0006 y +0.0001). La latencia de
int8 en test (109 ms) es de otra máquina de Kaggle; la comparación de velocidad válida es la de
validación, pareada.

## Qué significa en la práctica

Para un volumen de unos 400 cortes, solo la red:

| Escenario | Tiempo por volumen |
|---|---|
| fp32 en CPU (la Lambda hoy) | ~73 s |
| int8 en CPU | ~32 s |
| fp32 en GPU | ~4.4 s |
| fp16 en GPU | ~2.4 s |

La precisión es la misma en todos. En GPU la red deja de ser lo lento del pipeline: las distancias en
mm, las mallas y la vista cruda (CPU) tardan más que todos los cortes juntos.

## Cómo se usa

`src/precision.py` aplica la precisión al cargar el modelo (`predict_volume.load_pipeline(...,
precision=...)`):

| Precisión | Dónde | Cómo se obtiene |
|---|---|---|
| `fp32` | Cualquiera | El checkpoint tal cual (`h8_…_s42.pth`, 11 MB) |
| `int8` | Solo CPU | `h8_…_s42_int8.pt` (TorchScript, 4.9 MB), ya cuantizado: se carga sin calibrar. Si no existe o es de otra versión de torch, se cuantiza al cargar (unos segundos) con `outputs/s10/calibracion_int8_h8.npz`: los 128 cortes de entrenamiento con que se calibró en Kaggle |
| `fp16` | GPU | Conversión directa al cargar |
| `auto` | — | fp16 si hay GPU, int8 si no |

```bash
python src/precision.py exportar --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth   # crea el _int8.pt
```

El `_int8.pt` no se versiona (depende de la versión de torch); la imagen de la API lo genera al
construirse (`api/Dockerfile`).

- **API**: variable `PRECISION` (por defecto `auto`): en la Lambda (CPU) usa int8; en un servidor con
  GPU, fp16. `/health?deep=1` informa `precision` y si salió del archivo o se cuantizó al cargar.
- **Dashboard**: bloque «Procesamiento» de la barra lateral: dónde corre el modelo (este equipo, AWS
  CPU o AWS GPU si se configura `INFERENCE_URL_GPU`) y, en este equipo, la precisión (H8 normal fp32,
  int8, fp16 o automática). Aplica a la inferencia en vivo, «Procesar ahora» y los CT subidos; cada
  combinación se guarda aparte para comparar.
- **Casos precalculados y exportaciones oficiales** (`demo_export.py`, `deploy/precalcular_casos.py`):
  siguen en fp32, como las métricas oficiales.

## Cómo reproducirlo

```bash
python src/build_case_cache.py --sizes 384 --cases <15 de val> 074 006 002 045 028 023 003 055   # caché de val + calibración
python kaggle/h8/armar_cuantizacion.py --salida build/kaggle_cuant --usuario <usuario_kaggle>
kaggle kernels push -p build/kaggle_cuant/<kernel>      # fp32, int8, int8-todo, gpu-val, test-int8, test-fp32-gpu, test-fp16-gpu
kaggle kernels output <usuario>/pengwin-cuant-h8-<kernel> -p outputs/h8/cuantizacion/<kernel>
```

Los 8 casos de calibración son los que elige `cuantizar.calibration_slices` con semilla 0. Los datos
viajan en dos datasets privados de Kaggle: `pengwin-cuantizacion-h8` (validación) y
`pengwin-cuantizacion-h8-test` (test).

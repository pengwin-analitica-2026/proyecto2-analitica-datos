# Semana 11 — dashboard con tres visualizadores

Estado al 13 de octubre de 2026. Modelo de la demo: **H8** (`h8_aumm_noise01_noema_w32_e60_s42.pth`), que reemplazó a
H6 tras la asesoría (ver `docs/modelo_h8.md`). La demo, la API y el dashboard funcionan igual con
los dos, porque la arquitectura y la orientación canónica son las mismas.
Es el documento de la Semana 11 para los visualizadores. El despliegue está en
`docs/despliegue_aws_streamlit.md` y las pruebas y el despliegue automáticos en
`docs/cicd_github_actions.md`.

## Cómo correrlo

Publicado: la app de Streamlit Community Cloud (rama `main`, ver `docs/despliegue_aws_streamlit.md`).

En local, desde la raíz del repo, con el `.venv` activo:

```bash
pip install -r requirements.txt
python src/demo_export.py --case 024 044 --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth --tta
streamlit run app/dashboard.py
```

- `demo_export.py` corre el pipeline completo una vez por caso y deja todo en
  `outputs/demo/<modelo>/<caso>/`. En la CPU del Mac de Gustavo tarda unos 4 s por corte con TTA,
  es decir, de 15 a 25 minutos por caso. El dashboard solo lee esos archivos, así que abre al
  instante.
- `--reuse` rehace solo el post-proceso, las distancias y las mallas, sin volver a correr el modelo
  (necesita el `predictions.npz` local, que no se versiona).
- `DEMO_ROOT=<carpeta>` cambia dónde busca el dashboard los casos exportados.
- El botón «Inferir el corte seleccionado» del visualizador 2 usa la API de AWS si están
  `INFERENCE_URL` e `INFERENCE_KEY` (variables de entorno o secretos de Streamlit); si no, carga el
  modelo local, que necesita `torch` y el checkpoint en `outputs/s10/`.

Casos versionados en `outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/`, todos con H8 y TTA:

| Caso | Split | Guardado girado 180° | Para qué |
|---|---|---|---|
| 024 | validación | no | Caso de referencia, con dos fragmentos separados (19 y 3 mm) |
| 044 | validación | sí | Comprobar el enderezado |
| 099 | validación | sí | Comprobar el enderezado |
| 039 | test | no | Uno de los dos casos que hundían test con D5; se exporta porque el modelo ya está congelado |

## Los 100 casos de PENGWIN (desde S3)

En la barra lateral, **Origen → Casos PENGWIN (100)**: un selector con los 100 casos, su split
(train/val/test) y si la cabecera dice que vienen girados 180°. Cada caso tiene dos modos:

- **Precalculado**: el resultado completo (con ground truth y métricas) ya está en S3; se descarga en
  segundos y queda en caché local (`S3_CACHE`).
- **Procesar ahora**: descarga el `.mha` original de S3 (con su etiqueta) y corre el pipeline en vivo
  (`demo_export.preparar_caso` + `exportar_volumen`), con el modelo local o la API de AWS. Tarda unos
  minutos.

Los archivos están en el bucket privado `pengwin-ct-050752608083` (cuenta de Almia, us-east-2):

| Prefijo | Qué hay | Cómo se llena |
|---|---|---|
| `crudos/imagenes/<caso>.mha`, `crudos/etiquetas/<caso>.mha` | Los 100 casos originales, comprimidos sin pérdida (18.5 GB → ~12 GB; etiquetas 13 GB → unos MB). Vóxeles, dirección, spacing y `TransformMatrix` idénticos | `deploy/aws/subir_crudos_s3.py` |
| `exportados/<modelo>/<caso>/` | `meta.json`, `volume.npz`, `meshes.npz`, `raw_view.npz` y `pasos.npz` de cada caso | `deploy/precalcular_casos.py` |

El precálculo corre en el Mac con 4 procesos en paralelo (2 hilos de torch cada uno), con H8 y TTA,
unos 6 minutos de CPU por caso. Se hizo en el Mac y no en Colab porque casi la mitad del tiempo de
cada caso es CPU (distancias, mallas, vista cruda), que una GPU no acelera, y los datos ya estaban en
disco. Los dos scripts saltan lo que ya está en S3, así que se pueden cortar y volver a correr.

Comprobado: el caso 044 precalculado da exactamente lo mismo que el versionado en `outputs/demo/`
(solo se agregan los pasos), y el caso 014 bajado de S3 (comprimido) se prepara idéntico al `.mha`
original de `data/raw`.

## Preprocesamiento paso a paso

Pestaña **Preprocesamiento** (la primera). Muestra, para el corte axial con más hueso y para la MIP
axial, qué le pasa al volumen antes de la red (`src/pasos_preproceso.py`, guardado en `pasos.npz`):

1. **Crudo, como viene en el archivo**, con la matriz de dirección de la cabecera y lo que significa
   (identidad = orientación estándar; `-1 0 0 0 -1 0 0 0 1` = guardado girado 180°, el sacro arriba).
2. **Orientado con los metadatos**: «El metadato dice que viene girado 180°: se giró» o «El metadato no
   indica ningún giro: no se cambió nada».
3. **Giro axial del paciente**: el ángulo estimado con la simetría de la pelvis y si se corrigió
   (solo giros claros de 10° o más).
4. **Entrada al modelo**: ventana ósea, gamma 1.5 y 384×384, tal cual la recibe la red.

Las etapas 1 a 3 se muestran con la ventana ósea sin gamma y con la proporción real del vóxel, para que
solo cambie lo que hace cada etapa. Está en los casos de PENGWIN y en los CT subidos; los casos de
ejemplo versionados se exportaron antes y no la tienen.

## Subir un CT propio

Pedido del profesor: cargar un CT que no es de los casos de ejemplo y ver al instante la predicción
en los tres visualizadores. En la barra lateral, **Origen → Subir un CT**:

| Qué se sube | Cómo se lee |
|---|---|
| Un archivo `.mha`, `.nii`, `.nii.gz` o `.nrrd` (o un `.mhd`/`.nhdr` junto con su archivo de datos) | SimpleITK, con su cabecera |
| Una carpeta con los cortes DICOM de una serie (botón «Upload directories») | GDCM arma la serie: ordena los cortes por posición y aplica `RescaleSlope`/`RescaleIntercept`, así queda en HU. Si hay varias series, usa la de más cortes |

Una carpeta de PNG o JPG se rechaza con un mensaje: esas imágenes no guardan las unidades Hounsfield
ni el tamaño del vóxel, y el modelo (ventana ósea en HU) y la distancia en mm los necesitan.

Qué pasa al pulsar **Procesar** (el mismo pipeline de los casos de ejemplo, `demo_export.exportar_volumen`):

1. `volumen_entrada.leer_subida` lee el volumen y lo lleva a la orientación estándar con la matriz de
   dirección de la cabecera (`normalizacion.a_orientacion_estandar`, productos punto y cruz). Es la
   orientación canónica de H8; en los 100 casos de PENGWIN da lo mismo que `orientacion.to_canonical`.
2. Enderezado del giro axial del paciente si es claro (opcional, activado por defecto).
3. Modelo sobre todos los cortes: **local** si hay torch y el checkpoint (en el Mac); si no
   (Streamlit Cloud), la **API de AWS** con `POST /predict_batch`, en lotes de 8 cortes y 4 pedidos en
   paralelo (`src/inferencia_remota.py`). TTA opcional, solo con el modelo local.
4. Post-proceso 3D, distancias en mm, mallas por fragmento y vista cruda. Sin ground truth: no hay
   métricas ni contorno del GT.

El caso queda en `UPLOAD_ROOT` (por defecto una carpeta temporal), identificado por un hash del
contenido: si se vuelve a subir el mismo CT con las mismas opciones, se abre sin recalcular. Límite de
1 GB por archivo.

Comprobado:

- `044.mha` subido por el navegador (55 MB, guardado girado 180°): se orientó con la cabecera y la
  entrada al modelo quedó idéntica, byte a byte, a la del caso 044 de la demo. Tardó unos 6 minutos en
  la CPU del Mac sin TTA (349 ms por corte). Dice por hueso contra el ground truth: 0.955 / 0.969 /
  0.966 (la demo con TTA: 0.955 / 0.970 / 0.966).
- El caso 024 convertido a 350 archivos DICOM se reconstruye idéntico al `.mha` original, con el mismo
  spacing.
- En un contenedor con Python 3.14 y sin torch (como Streamlit Cloud) el dashboard abre la página de
  subida y el pipeline corre sin importar torch.
- `tests/test_volumen_entrada.py` (lectura de `.mha`, NIfTI y DICOM desordenado, errores) y
  `tests/test_api_server.py` (el cliente remoto contra la API da lo mismo que el modelo local).

Límites: en Streamlit Cloud la memoria es limitada (del orden de 2-3 GB), así que un volumen muy
grande (por ejemplo 512×512×800) puede no caber. La primera llamada a la API después de un rato sin
uso tarda hasta ~90 s por el arranque en frío de Lambda.

## Procesamiento: dónde corre el modelo y con qué precisión

Bloque **Procesamiento** de la barra lateral. Aplica a la inferencia en vivo del visualizador 2, a
«Procesar ahora» (casos de PENGWIN) y a los CT subidos; los casos precalculados no se vuelven a procesar.

- **Dónde corre el modelo**: «Este equipo» (si hay torch y el checkpoint), «AWS · CPU (Lambda)» (si
  está `INFERENCE_URL`) y «AWS · GPU» (si está `INFERENCE_URL_GPU`). Solo aparecen los disponibles.
- **Precisión** (en este equipo): H8 normal (fp32), H8 cuantizado int8 (CPU), fp16 (GPU) o automática
  (int8 en CPU, fp16 en GPU). En AWS la fija el servicio (`PRECISION`) y se informa al procesar.

Es el mismo H8 con el mismo Dice; cambia la velocidad (`docs/cuantizacion.md`). Cada combinación de
destino y precisión se guarda aparte, para comparar las dos versiones del mismo caso; la barra lateral
dice con qué precisión se predijo el caso abierto.

## Orientación

H8, como H6, se entrenó con los volúmenes en orientación canónica: 34 de los 100 `.mha` de PENGWIN están
guardados girados 180° en el plano (`TransformMatrix = -1 0 0 0 -1 0 0 0 1`, orientación LPI) y
`SimpleITK.GetArrayFromImage` no aplica esa dirección.

`demo_export.py` lo resuelve al cargar el caso:

1. Lee la marca `canonical` del checkpoint (`True` en H6 y H8, `False` en D5). Con D5 no se endereza nada.
2. Si el modelo la pide, lee la `TransformMatrix` de la cabecera del `.mha`
   (`orientacion.header_is_rotated`). Si la cabecera no la trae, usa la lista
   `orientacion.LPI_CASES`.
3. Si el volumen está girado, gira 180° en el plano la imagen en HU y el ground truth, una sola vez.
   Todo lo que sigue sale de ese volumen enderezado: la vista cruda (visualizador 1), los cortes del
   modelo (visualizador 2), las mallas (visualizador 3) y los cortes que se mandan a la API en vivo.
4. Después, `normalizacion.normalizar` estima el giro axial del paciente dentro del escáner, que la
   cabecera no trae, con la simetría izquierda-derecha de la pelvis, y lo corrige si es de 10° o más
   (`GIRO_MINIMO_DEMO`). En los cuatro casos versionados el giro fue menor (de −1.5° a +4.8°) y no se
   corrigió. `--sin-enderezar` omite este paso.
5. `meta.json` guarda `canonical`, `flipped` y `enderezado` (ángulo y si se corrigió); la barra
   lateral del dashboard dice si el caso se enderezó.

No se devuelve el resultado a la orientación original: en la canónica todos los casos se ven igual
(sacro abajo) y la distancia en mm no cambia con un giro de 180°.

## Qué hay

| Archivo | Para qué |
|---|---|
| `src/predict_volume.py` | Inferencia sobre un volumen, sin ground truth: preprocesa como el entrenamiento, corre las tres cabezas con NMS propio y devuelve semántica, mapa primario/secundario, cajas y clasificación por corte. `load_pipeline` informa `canonical` |
| `src/demo_export.py` | Exportador por caso: orientación, volumen de IDs, distancias en mm, comparación con el ground truth, mallas por fragmento y vista cruda |
| `src/orientacion.py` | `header_is_rotated` (cabecera del `.mha`), `flip_inplane`, `to_canonical` y la lista de los 34 casos LPI |
| `src/demo_render.py` | Dibujo del visualizador 2 (corte con cajas, máscaras y etiquetas), del 3 (mallas Plotly) y de la isosuperficie cruda |
| `src/raw_view.py` | Visualizador 1: MIP en tres ejes e isosuperficie del hueso, sobre el volumen sin procesar |
| `src/measure_latency.py` | Latencia por imagen en CPU y GPU (mediana y p90); escribe `outputs/demo/latency.json` |
| `app/dashboard.py` | Dashboard Streamlit con tres pestañas, métricas y latencia, y la advertencia de uso no clínico siempre visible |
| `api/server.py` | API de inferencia por corte (AWS Lambda); precisión con `PRECISION` |
| `src/precision.py` | Precisión de H8: fp32, int8 híbrido (CPU, exportable ya cuantizado) o fp16 (GPU); `auto` según el hardware |

## Lo que se ve en cada visualizador

1. **Volumen crudo.** Umbral de 250 HU sobre el volumen sin procesar y limpieza de camilla y cables.
   MIP axial, coronal y sagital con su proporción física, e isosuperficie 3D opcional. No interviene
   ningún modelo.
2. **Corte por corte.** Slider de cortes; máscaras por macro-hueso (sacro naranja, coxal izquierdo
   celeste, coxal derecho verde azulado), cajas refinadas o crudas, contorno opcional del ground
   truth, y junto a cada fragmento secundario su ID y la distancia en mm (`F4 · 19.0 mm`).
3. **Reconstrucción 3D.** Una malla por fragmento con marching cubes y `Mesh3d`, mismos colores, y
   rótulo `ID · mm` con flecha en cada fragmento. Interruptores para el fragmento principal, su
   opacidad y el ground truth superpuesto.

Pestaña **Métricas y latencia**: distancia predicha contra la real en el caso elegido; las cuatro
evaluaciones en test del proyecto (D5 con la configuración congelada antes de ver test, 0.845 de
Dice por fragmento; H6, 0.875; H7, 0.879; y H8, 0.874, el modelo de la demo), HD95 y ASSD por
fragmento, el error de distancia y SAM sobre test, leídos de `outputs/`; y la latencia por imagen.

## Distancia en mm

Los tamaños mínimos del post-proceso 3D se fijan en mm³ con el espaciado de cada caso (26.1 mm³ por
fragmento y 173.8 por región), así no dependen del grosor de los cortes; `meta.json` los guarda en
`umbrales_postproceso`. `--umbral-voxeles` vuelve a los 30 y 200 vóxeles.

Mínima borde a borde entre el fragmento y el principal de su hueso, con `distance_transform_edt` y
el spacing del header (ajustado al estiramiento a 384 px). El volumen de IDs sale de
`postprocess_3d(fill_unlabeled=False)`: los vóxeles que no son ni principal ni fragmento quedan sin
etiqueta, así no se rellena el hueco entre fragmentos. Es una medida del fragmento entero en 3D: en
el visualizador 2 se repite en todos los cortes donde aparece.

## Resultado por caso (H8, TTA)

Generado en CPU. Dice por macro-hueso de la semántica 3D contra el ground truth y fragmentos
secundarios reales con su distancia (real → predicha). «Cruce» es el Dice entre el coxal izquierdo
predicho y el derecho real: si la orientación estuviera mal, sería alto.

| Caso | Enderezado | Dice sacro / coxal izq. / coxal der. | Cruce | Secundarios separados del principal (real → predicho) |
|---|---|---|---|---|
| 024 | no | 0.966 / 0.979 / 0.979 | 0.000 | F3 coxal izq.: 19.0 → 19.0 mm; F4 coxal izq.: 3.2 → 3.9 mm |
| 044 | sí | 0.958 / 0.973 / 0.968 | 0.000 | ninguno (su único secundario está en contacto: 0.0 → 0.0 mm) |
| 099 | sí | 0.972 / 0.977 / 0.979 | 0.000 | F2 coxal izq.: 12.8 → 18.6 mm |
| 039 (test) | no | 0.954 / 0.955 / 0.966 | 0.002 | F3 coxal der.: 33.7 mm, **no detectado** |

En 044 y 099 el sacro queda abajo y las regiones izquierda y derecha coinciden con el ground truth
después de enderezar. Son cuatro casos: sirven para comprobar que la orientación y la distancia
funcionan, no para estimar el error (eso está en la pestaña de métricas, con los 15 casos de test).
Frente a H6, el fragmento de 3.2 mm del caso 024 se mide mejor (3.9 contra 4.5 mm) y el de 12.8 mm
del 099 peor (18.6 contra 13.3 mm). En 039 el modelo sigue sin detectar el fragmento más separado
del coxal derecho.

## Decisiones

| Tema | Decisión |
|---|---|
| Framework | Streamlit. El dashboard solo lee los archivos que deja `demo_export.py`, así que abre al instante y no necesita `torch` |
| Dónde se endereza | Al exportar el caso, no en la API: así las tres vistas, el ground truth y la inferencia en vivo comparten orientación |
| Orientación | Sale de la cabecera del `.mha`, no de la lista de casos, para que sirva con volúmenes que no sean de PENGWIN |
| Modelos en la demo | Solo el entregado (H8). El selector de modelos se descartó y los casos exportados con D5 y H6 se borraron |
| Casos | Tres de validación y uno de test (039), este último porque el modelo ya estaba congelado |
| Publicación | Streamlit Community Cloud y una API en AWS Lambda, probados desde computador y celular (6 de octubre). El profesor aprobó este despliegue en lugar del túnel de Cloudflare del enunciado |

## Pendiente

- Material del pitch.
- Línea entre el fragmento y su principal en el visualizador 3 (opcional).

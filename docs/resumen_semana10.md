# Semana 10 Pipeline multitarea

## Estado actual

La Semana 10 inició sobre la arquitectura validada de Semana 9. Ya están implementados:

- backbone y CBAM compartidos;
- cabeza independiente de clasificación anatómica multietiqueta;
- cabeza de detección por grid con NMS propia;
- cabeza de segmentación con salida semántica de cuatro clases y salida de instancia
  supervisada por los IDs `0..30` de PENGWIN;
- pérdida compuesta con lambdas explícitas para detección, clasificación, semántica e
  instancia;
- distancia borde a borde en milímetros con spacing del header y
  `scipy.ndimage.distance_transform_edt`;
- comparación de distancia predicción contra ground truth por ID de fragmento.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Las doce pruebas actuales comprueban las piezas de Semana 9, formas de las tres cabezas,
targets semánticos y de presencia, gradientes hacia todas las cabezas, distancias físicas
con spacing anisotrópico y la secuencia de inferencia con NMS y filtrado de fragmentos.

Prueba real de overfit multitarea sobre dos cortes de entrenamiento:

```text
100 épocas, base_channels=4, CPU
pérdida total: 7.89094 -> 2.91041 (2.7x)
detección final: 0.05375 | semántica: 0.8172 | instancia total: 0.7794
```

La exactitud de instancia total incluye el fondo, que domina el número de píxeles; por eso
el script también guarda `instance_foreground_accuracy`, que es la métrica que debe usarse
para juzgar la separación de fragmentos. Este overfit confirma que las cabezas reciben
gradientes y aprenden, pero todavía no es un resultado de desempeño final.

Prueba real de distancia sobre el caso `001`:

```text
coxal_izquierdo, fragmento 12 respecto a 11: 0.0 mm
```

El valor cero indica contacto entre superficies discretizadas. Este caso debe conservarse
como ejemplo de la limitación señalada por el enunciado: fragmentos en contacto pueden
fusionarse si la segmentación predicha no separa correctamente sus instancias.

## Pipeline implementado

`src/inference_week10.py` ejecuta la secuencia requerida para un corte: decodificación de
la cabeza de detección, NMS propia por región, restricción de la máscara de instancia al
bounding box, validación de pertenencia anatómica mediante la salida semántica y retorno
de IDs de fragmentos listos para apilar en un volumen.

El comando de prueba del overfit es:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python src/train_week10_overfit.py \
  --epochs 100 --samples 2 --base-channels 4 \
  --output outputs/week10_multitask_overfit.pth
```

## Pendiente de Semana 10

- entrenamiento completo con los 70 casos;
- métricas de cajas, semántica e instancia sobre validación/test;
- reconstrucción volumétrica y evaluación de distancias por caso;
- baseline zero-shot de SAM usando bounding boxes predichas;
- dashboard con visualizadores 2 y 3.

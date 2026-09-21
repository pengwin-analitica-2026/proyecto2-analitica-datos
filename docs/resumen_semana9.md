# Semana 9 Backbone CBAM y deteccion

## Objetivo

Implementar una primera versión verificable de la detección corte a corte: localizar una
bounding box para cada región anatómica visible en el corte, usando un backbone propio,
CBAM y una cabeza por grid.

## Diseño implementado

- `FundidoraBackbone`: cuatro bloques convolucionales con reducción espacial 16x.
- `CBAM`: atención de canal y espacial aplicada antes de la cabeza.
- `GridDetectionHead`: por celda predice objectness, `(cx, cy, w, h)` y una clase entre
  sacro, coxal izquierdo y coxal derecho.
- Pérdida multitarea: BCE de objectness, Smooth L1 de caja y entropía cruzada de clase.
- `nms`: supresión de no máximos escrita en PyTorch, sin frameworks de detección.

## Cómo ejecutar el overfit

Después de instalar las dependencias y descargar `data/raw/`:

```bash
python src/train_overfit_detection.py --case-id 001 --epochs 150
```

El script selecciona un batch pequeño de dos cortes con máscara, redimensiona imagen y máscara a `256x256`,
construye una caja por región visible, entrena sobre el mismo batch y guarda:

```text
outputs/week9_overfit_detection.pth
```

La prueba es de correctitud del pipeline, no una métrica final. Se considera aprobada si
la pérdida final es menor que el 25% de la pérdida inicial y las predicciones decodifican
cajas con NMS.

Para una verificación cualitativa, después del overfit se ejecuta:

```bash
python src/visualize_overfit_detection.py --case-id 001
```

Genera `outputs/figures/week9_detection_overfit.png`, con cajas ground truth sólidas y
cajas predichas discontinuas, coloreadas por región anatómica.

## Ejecución verificada

Se ejecutó sobre el caso `001`, dos cortes con etiqueta visible, resolución `256x256`,
backbone de 8 canales base, `150` épocas y CPU:

| Medida | Resultado |
|---|---:|
| Pérdida inicial | 1.68749 |
| Pérdida final | 0.00223 |
| Reducción | 758.4x |
| Criterio mínimo | 4x |
| Resultado | Aprobado |

La inspección del overlay confirma que las cajas predichas después de NMS se superponen a
las cajas de referencia en los dos cortes. Este resultado prueba que la cadena máscara →
target de grid → backbone/CBAM → cabeza → pérdida → NMS funciona; no constituye todavía
una evaluación sobre validación o prueba.

## Primer entrenamiento multi paciente

Para obtener una primera predicción fuera del batch de overfit se creó un cache reproducible
con cuatro cortes distribuidos axialmente por caso. Se entrenó con los primeros 24 casos del
split fijo de train (96 cortes) y se validó con los 15 casos de validación (60 cortes, 164
cajas visibles). El cache es local y no se versiona.

```bash
python src/prepare_week9_detection_data.py
python src/train_week9_detection.py --epochs 35
python src/visualize_week9_validation.py --case-id 014
```

| Medida | Resultado |
|---|---:|
| Mejor IoU media de validación | 0.3883 |
| Caso mostrado | 014, corte 165 |
| Salida visual | `outputs/figures/week9_detection_validation_014.png` |

La imagen muestra una hipótesis de mayor confianza por región para conservar legibilidad;
la evaluación de IoU conserva todas las propuestas luego de NMS. Las cajas siguen la
anatomía general, pero todavía tienen imprecisión de escala y posición: es una evidencia
inicial cualitativa, no una métrica final ni el objetivo de mAP del proyecto.

## Entorno reproducible

La ejecución se verificó con Python 3.12, PyTorch 2.2.2 y CPU. En macOS x86_64, esa versión
de PyTorch requiere `numpy<2`; por eso `requirements.txt` fija una familia compatible de
NumPy, SciPy y scikit-image.

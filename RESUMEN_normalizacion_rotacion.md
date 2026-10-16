# Corrección de la rotación (pedido del profesor) — resumen

**Rama:** `feat/normalizacion-orientacion` · **No requiere reentrenar para medir** · Test no se ha tocado.

## Qué pidió el profesor
1. Normalizar la rotación con los **metadatos**, usando **productos punto y cruz**, como en su artículo (Ferro Sánchez et al., *MethodsX* 2024).
2. No depender de EMA.
3. Que el modelo no falle con imágenes "feas" o ruidosas en la demo.

## Qué se hizo
| Paso | Archivo | Qué hace |
|---|---|---|
| 1. Orientación con la cabecera (del artículo) | `src/normalizacion.py` | Punto: verifica que los ejes sean perpendiculares y con qué eje del paciente coincide cada uno. Cruz: dirección de los cortes y si la imagen está en espejo. Lleva todo a la misma orientación estándar. En los 100 casos da lo mismo que `to_canonical`. |
| 2. Giro del paciente en la camilla (adaptado a pelvis) | `src/normalizacion.py` | El artículo usa estructuras de la cara; nosotros, la **simetría izquierda-derecha de la pelvis** (silueta del hueso vista desde arriba). Solo corrige si el giro es claro (> 5° y mejora la simetría). |
| Giro de entrenamiento en milímetros | `src/aumento_fisico.py`, `--aug-mm` | Antes se giraba en píxeles; con píxeles no cuadrados eso deformaba la pelvis. |
| Evaluación con enderezado | `src/evaluate_volumes.py --normalizar` | Endereza cada volumen antes de H6. |
| Prueba de estrés en milímetros | `src/evaluate_volumes.py --stress rotmm=X` | Gira al paciente X° de verdad. El `rot=X` anterior giraba en píxeles y, en **10 de los 15 casos de validación** (píxeles no cuadrados), deformaba la pelvis. |

Todo con pruebas automáticas (`tests/test_normalizacion.py`, `tests/test_evaluar_normalizado.py`, `tests/test_aumento_fisico.py`).

## Resultados hasta ahora (validación, sin modelo)
Error al recuperar un giro conocido (grados), 15 casos:

| Giro aplicado | Error medio | Error máximo |
|---|---|---|
| 30° | 1.0° (0.3° sin el 044) | 12° (044) |
| 60° | 2.8° | 21° (044) |

- Casos con giro propio detectado y corregido: 014 (+6°) y 095 (−6°).
- **Dónde falla:** el caso 044 tiene un campo de visión de solo ~16 cm de profundidad; al girarlo, la pelvis se sale del recuadro y el método no ve su simetría.

## Lo que falta
1. **Correr `notebooks/07_normalizacion_evaluacion_colab.ipynb`** (Colab o Kaggle con GPU, ~20 min): H6 con giros de 30° y 60°, con y sin enderezado.
2. Reentrenar sin EMA: enderezado + `--aug-mm` + `--ema 0` (Gustavo).
3. Robustez a ruido: prueba de estrés con ruido y aviso en la demo.
4. Conectar el enderezado a la demo.

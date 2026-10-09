# Cuantización de H6

El profesor sugirió compactar el modelo (clase del miércoles) con una condición: se adopta solo si
mantiene las métricas y no baja el Dice. Este documento explica qué se probó, por qué esas
precisiones y no otras, y qué falta correr. El script es `src/cuantizar.py`.

## Qué es cuantizar

H6 se entrenó en **fp32**: cada peso es un número de punto flotante de 32 bits. Cuantizar es guardar
y calcular esos mismos pesos con menos bits. Hay dos familias:

- **Punto flotante más corto** (fp16, bf16): sigue siendo punto flotante, pero de 16 bits. fp16
  conserva más decimales y bf16 más rango (el mismo exponente que fp32). Pasar a 16 bits es casi
  una conversión directa y suele no cambiar el resultado.
- **Enteros** (int8, int4): cada tensor guarda una escala y un cero, y los valores se redondean a
  enteros (256 niveles en int8, 16 en int4). Para las activaciones hay que calibrar el rango con
  datos reales. Las operaciones con enteros son más baratas en CPU que las de punto flotante.

## Por qué fp16, bf16 e int8, y no int4

| Precisión | Bits | Se prueba | Motivo |
|---|---|---|---|
| fp32 | 32 | Sí (referencia) | Es como se entrenó |
| fp16 | 16 | Sí | Mitad de memoria. Acelera en GPU; en CPU depende de que el procesador tenga instrucciones de 16 bits |
| bf16 | 16 | Sí | Igual que fp16, pero más estable porque no se desborda con valores grandes |
| int8 | 8 | Sí | Es la cuantización estándar para inferencia en CPU, donde corre la demo (AWS Lambda). PyTorch la soporta para convoluciones |
| int4 | 4 | No | PyTorch no tiene convoluciones int4 en CPU. Es una técnica para los pesos de modelos de lenguaje con miles de millones de parámetros, donde el problema es la memoria. Aquí el modelo pesa 11 MB, así que no hay nada que ganar y sí mucha pérdida con 16 niveles |

int8 es **cuantización estática después del entrenamiento**: no se reentrena. Los pesos se
cuantizan por canal, y el rango de las activaciones se calibra con cortes de **entrenamiento**
(nunca de validación ni de test). Si int8 perdiera Dice, el siguiente paso sería entrenar con
cuantización simulada (QAT), que sí requiere reentrenar.

## Hallazgo: las convoluciones transpuestas no se pueden cuantizar

Con todo en int8 (`int8_todo`), la segmentación de los coxales desaparece: el Dice semántico cae
de 0.976 a 0.019. Para encontrar la causa se dejó en fp32 un bloque a la vez:

| Lo que queda en fp32 | Dice sacro / coxal izq. / coxal der. (acuerdo con fp32) |
|---|---|
| Nada | 0.08 / 0.00 / 0.00 |
| Backbone | 0.09 / 0.00 / 0.00 |
| CBAM | 0.08 / 0.00 / 0.00 |
| Primer bloque del decodificador | 0.06 / 0.00 / 0.00 |
| Toda la cabeza de segmentación | 0.99 / 1.00 / 1.00 |
| **Solo las ConvTranspose2d del decodificador** | **0.99 / 1.00 / 1.00** |

La causa son las 4 convoluciones transpuestas (las que suben la resolución). Ni el CBAM ni el
backbone. Por eso la variante `int8` deja esas 4 capas en fp32 y cuantiza todo lo demás.

## Prueba de humo (un caso, sin valor de reporte)

Esto es solo para comprobar que el script funciona. Se corrió en un contenedor con 2 CPU, sobre el
caso 014 de validación, sin TTA y **calibrando con el mismo caso**. No sirve para decidir: los
números válidos salen del cuaderno de Colab, con los 15 casos de validación y la calibración con
casos de entrenamiento. Archivo: `outputs/s10/cuantizacion_humo_014.json`.

| Variante | Tamaño | Latencia (2 CPU) | Dice por fragmento | Dice semántico | F1 | Error distancia |
|---|---|---|---|---|---|---|
| fp32 | 11.0 MB | 98 ms | 0.837 | 0.976 | 0.997 | 0.873 mm |
| fp16 | 5.5 MB | 131 ms | 0.837 | 0.976 | 0.997 | 0.873 mm |
| bf16 | 5.5 MB | 62 ms | 0.838 | 0.976 | 0.997 | 0.873 mm |
| int8 (transpuestas en fp32) | 5.0 MB | 36 ms | 0.838 | 0.977 | 0.997 | 0.873 mm |
| int8 + CBAM y salidas en fp32 | 5.0 MB | 35 ms | 0.836 | 0.977 | 0.998 | 0.873 mm |
| int8_todo | 2.8 MB | 25 ms | 0.026 | 0.019 | 0.997 | — |

Lectura provisional: las cinco variantes que funcionan dan el mismo Dice (diferencias de ±0.002,
dentro del ruido). fp16 es **más lenta** en esta CPU, que no tiene instrucciones de 16 bits para
fp16. bf16 baja a 62 ms. int8 conserva el Dice y corre unas 2.7 veces más rápido. El tamaño
baja a la mitad y no a un cuarto porque las transpuestas siguen en fp32. Dejar también el CBAM en
fp32 no aporta nada.

## Qué falta para decidir

1. Correr la celda 8 de `notebooks/colab/colab_pedidos_profesor.ipynb`: los 15 casos de
   validación, con TTA, calibrando con 8 casos de entrenamiento, en la CPU de Colab y sin
   entrenamientos en paralelo.
2. Se adopta la variante más rápida que cumpla la regla del protocolo: el Dice por fragmento no
   baja más de 0.003 frente a fp32 (el JSON lo marca en `cumple_regla_0_003`).
3. Solo esa variante se reporta una vez en test.
4. Usarla en la demo de Lambda es un cambio aparte. No se toca `api/` ni `deploy/` sin decidirlo
   en grupo.

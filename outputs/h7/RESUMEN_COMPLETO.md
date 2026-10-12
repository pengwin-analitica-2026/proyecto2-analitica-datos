---
title: "PENGWIN · Correcciones del profesor: rotación, EMA e imágenes feas"
subtitle: "Resumen completo de mediciones (H6 → H7)"
---

**No hay que entrenar ni ejecutar nada.** Todo ya está corrido; los notebooks tienen sus salidas guardadas.
Todo es en **validación (15 casos)**, salvo la sección 5 (test, una sola consulta con H7).

# 0. Qué hay en esta carpeta

| Carpeta | Qué es |
|---|---|
| `RESUMEN_COMPLETO.pdf` / `.md` | Este documento |
| `notebooks/07_rotacion_H6_primera_prueba.ipynb` | **Primera prueba:** H6 (con EMA) con pacientes girados 30° y 60°, con y sin enderezado |
| `notebooks/09_H7_entrenamiento_validacion_test.ipynb` | **Entrenamiento de H7** (sin EMA) en Kaggle + validación H6 vs H7 + tercera evaluación en test |
| `notebooks/10_H7_ruido_y_CBAM.ipynb` | **Imágenes feas** (ruido, desenfoque, baja resolución) H6 vs H7 + **CBAM apagado** en H7 |
| `modelos/` | `h7_..._noema_...pth` (modelo final) y `h6_..._ema_...pth` (anterior) |
| `resultados/` | Todos los JSON de métricas (validación, test, ruido, CBAM) |
| `figuras/` | Diagnóstico del giro del caso 044 y ejemplo de imágenes dañadas |

En el notebook 09, la celda de validación que dice `^C` / `KeyError` se interrumpió a propósito
(se reutilizó la validación de H6 del notebook 07); la que vale es **"nueva validacion"**.
En el notebook 10 hay celdas repetidas: lo importante son **las dos últimas tablas**.

# 1. Qué pidió el profesor

1. Normalizar las rotaciones **con los metadatos**, usando **productos punto y cruz** (su artículo: Ferro Sánchez C. A. et al., "Methodology for the correction of a CBCT volume from the skull to the natural head position", *MethodsX* 13:103073, 2024, doi:10.1016/j.mex.2024.103073).
2. **No usar EMA**: "con imágenes feas se puede destruir".
3. Un **pipeline robusto** a imágenes feas, incluida la demo.

# 2. Los dos modelos

| | H6 (antes) | **H7 (final)** |
|---|---|---|
| EMA | 0.999 | **0 (sin EMA)** |
| Giro de entrenamiento ±35° | en píxeles | **en milímetros** (`--aug-mm`) |
| Orientación | RAI (canónica) | RAI (canónica) |
| Resto | 32 canales, CBAM 1, 60 épocas, lr 0.002, coseno, semilla 42 | igual |
| Dónde se entrenó | — | Kaggle T4, 67 min, desde cero |

Curva de H7 (Dice por fragmento en validación): época 20 → 0.827 · 30 → 0.816 · 40 → 0.863 · 50 → 0.872 · **60 → 0.875**.

# 3. Rotación

## 3.1 Paso 1 · Orientación con los metadatos (punto y cruz)
De la cabecera de cada CT se leen las direcciones de filas y columnas.
**Producto punto:** con qué eje del paciente coincide cada eje de la imagen y si son perpendiculares.
**Producto cruz** (filas × columnas): hacia dónde van los cortes y si la imagen está en espejo.
Con eso todo queda en la misma orientación **RAI**; los 34 casos que venían en **LPI** se giran 180°.
En los 100 casos coincide con lo que ya hacía `to_canonical`. (`src/normalizacion.py`)

## 3.2 Paso 2 · Giro del paciente en la camilla (enderezado)
El artículo endereza el cráneo (CBCT) a la posición natural de la cabeza; para pelvis usamos la **simetría izquierda-derecha del hueso** visto desde arriba.
Solo corrige si el giro es claro (> 5° y mejora la simetría). Solo corrige el giro en el plano axial
(no el cabeceo ni la inclinación lateral: limitación).

## 3.3 Lo que primero probé (y por qué se cambió)
- La prueba de estrés vieja `rot=X` giraba la imagen **en píxeles**. En **10 de los 15** casos de validación los píxeles
  no son cuadrados, así que eso **deformaba la pelvis** en vez de girarla.
- Señal del problema: en el caso **044** (píxeles 0.84 × 0.42 mm) el enderezado estimaba solo **+2.8°** para un giro de 30°.
  Ver `figuras/diag_044.png`.
- Corrección: nueva prueba **`rotmm=X`**, que gira X grados **en milímetros reales**, y el giro de entrenamiento en mm para H7.

Precisión del enderezado (sin modelo, ángulo estimado vs aplicado):

| Giro aplicado | Error medio | Error máximo |
|---|---|---|
| 30° | 1.0° (0.3° sin el 044) | 12° (044) |
| 60° | 2.8° | 21° (044) |

El 044 tiene solo ~16 cm de profundidad de campo: al girarlo, la pelvis se sale del recuadro.
Ángulos estimados con `rotmm=60` (lo ideal ≈ 60° + el giro propio del paciente): mediana **58.3°**;
todos entre 52° y 66°, salvo el 044 (39°). Con `rotmm=30`: mediana **31.0°**.

## 3.4 Primera prueba con modelo · H6 (notebook 07)

| Prueba | F1 | AUC | mAP50 | IoU det. | Dice seg. | **Dice fragmento** |
|---|---|---|---|---|---|---|
| Sin giro | 0.992 | 0.999 | 0.971 | 0.949 | 0.971 | **0.869** |
| Girado 30° | 0.990 | 0.999 | 0.943 | 0.927 | 0.961 | 0.853 |
| Girado 30° + enderezado | 0.992 | 0.999 | 0.971 | 0.945 | 0.959 | 0.854 |
| Girado 60° | 0.959 | 0.988 | **0.311** | 0.392 | 0.715 | **0.608** |
| Girado 60° + enderezado | 0.989 | 0.999 | **0.963** | 0.928 | 0.957 | **0.851** |

**Lectura:** con 60° el modelo se cae (mAP50 0.31); el enderezado lo recupera casi todo (0.96).

## 3.5 Ahora · H6 vs H7 (notebook 09)

| Prueba | Modelo | F1 | mAP50 | IoU det. | Dice seg. | **Dice fragmento** |
|---|---|---|---|---|---|---|
| Sin giro | H6 | 0.992 | 0.971 | 0.949 | 0.971 | 0.869 |
| | **H7** | 0.992 | 0.974 | 0.948 | 0.971 | **0.873** |
| Girado 60° | H6 | 0.959 | 0.311 | 0.392 | 0.715 | 0.608 |
| | **H7** | 0.951 | 0.347 | 0.426 | 0.725 | **0.636** |
| Girado 60° + enderezado | H6 | 0.989 | 0.963 | 0.928 | 0.957 | 0.851 |
| | **H7** | 0.988 | 0.963 | 0.924 | 0.956 | **0.855** |

**Lectura:** en rotación, quitar el EMA no empeoró: H7 queda igual o un poco mejor que H6 en las tres pruebas.

# 4. H7 cumple las 9 métricas (validación)

| Métrica | H7 | Objetivo |
|---|---|---|
| F1 clasificación | 0.992 | 0.85 ✔ |
| AUC clasificación | 0.999 | 0.85 ✔ |
| IoU detección | 0.948 | 0.65 ✔ |
| mAP50 | 0.974 | 0.65 ✔ |
| mAP50-95 | 0.903 | 0.40 ✔ |
| Dice segmentación | 0.971 | 0.85 ✔ |
| IoU segmentación | 0.944 | 0.70 ✔ |
| Dice por fragmento | 0.873 | 0.85 ✔ |
| IoU por fragmento | 0.812 | 0.70 ✔ |

# 5. Tercera evaluación oficial en test (H7)

| | F1 | AUC | mAP50 | IoU det. | Dice seg. | **Dice fragmento** |
|---|---|---|---|---|---|---|
| H7 sin enderezar | 0.989 | 0.999 | 0.962 | 0.940 | 0.971 | **0.879** |
| H7 con enderezado | 0.989 | 0.999 | 0.958 | 0.938 | 0.968 | 0.874 |

Evaluaciones oficiales (Dice por fragmento): 1.ª D5 **0.845** → 2.ª H6 **0.875** → 3.ª H7 **0.879**.
Contando todas las consultas a test del proyecto (como el model card): c2 (0.848), EMA + CASR (0.847), D5 (0.845), H6 (0.875) y H7 (0.879) = **5 consultas**. En la de H7 se corrieron las dos variantes (sin y con enderezado) en la misma sesión; la principal es **sin enderezar**, que es la configuración validada (0.873 en validación).
**¿El enderezado toca casos que no lo necesitan?** (revisado después, sin volver a evaluar test)

- En validación sin giro, con enderezado: Dice fragmento **0.872** vs 0.873 sin él. Solo corrigió 2 de 15 casos (014 +6°, 095 −6°): casi gratis.
- En test (leyendo la consulta ya hecha) corrigió 6 de 15: el **012 (+23°)**, el paciente girado conocido, y cinco con giros pequeños (5°–7°: 020, 054, 061, 062, 065). Esos giros pequeños explican la baja de 0.879 a 0.874.
- Mejora propuesta (sin usar test para decidir): subir el umbral de corrección de 5° a ~10°, para enderezar solo giros claros. Con ese umbral, en validación no cambia nada (los dos casos corregidos tenían 6°) y los giros de 30° y 60° se siguen corrigiendo.
**Test no se vuelve a evaluar.**

# 6. Imágenes feas: H6 (EMA) vs H7 (sin EMA) · notebook 10

Ejemplo de cada daño: `figuras/imagenes_danadas_ejemplo.png`.

| Daño | Modelo | F1 | mAP50 | Dice seg. | **Dice fragmento** |
|---|---|---|---|---|---|
| Sin daño | H6 | 0.992 | 0.971 | 0.971 | 0.869 |
| | H7 | 0.992 | 0.974 | 0.971 | **0.873** |
| Baja resolución (2.5) | H6 | 0.991 | 0.973 | 0.961 | 0.857 |
| | H7 | 0.990 | 0.970 | 0.961 | **0.861** |
| Borrosa (1.5) | H6 | 0.984 | 0.961 | 0.937 | 0.818 |
| | H7 | 0.982 | 0.954 | 0.935 | **0.819** |
| Ruido leve (10) | H6 | 0.881 | 0.690 | 0.889 | **0.690** |
| | H7 | 0.879 | 0.730 | 0.825 | 0.654 |
| Ruido fuerte (25) | H6 | 0.691 | 0.188 | 0.249 | 0.135 |
| | H7 | 0.517 | 0.224 | 0.383 | **0.228** |

**Lectura:**

- Borrosa y baja resolución casi no afectan a ninguno de los dos.
- **El punto débil de los dos es el ruido**: con ruido leve pierden ~0.2 y con ruido fuerte se caen.
- **El EMA no era la causa; el ruido es el punto débil de ambos modelos.** Con ruido leve H6 aguanta algo mejor (0.690 vs 0.654); con ruido fuerte, H7 (0.228 vs 0.135).
- Como el desenfoque casi no daña, se probó **suavizar la imagen antes del modelo** (sección 6.1).

## 6.1 Suavizado antes del modelo (H7, sin reentrenar)

`suav=S`: filtro gaussiano de S píxeles aplicado **después** del ruido, sobre la imagen ya en 0–255 (como lo haría el pipeline de la demo).

| Prueba | F1 | mAP50 | Dice seg. | **Dice fragmento** |
|---|---|---|---|---|
| Sin daño | 0.992 | 0.974 | 0.971 | **0.873** |
| Sin daño + suavizado 1 | 0.990 | 0.969 | 0.957 | 0.859 |
| Ruido leve (10) | 0.879 | 0.730 | 0.825 | 0.654 |
| Ruido leve + suavizado 1 | 0.984 | 0.968 | 0.955 | **0.851** |
| Ruido leve + suavizado 1.5 | 0.980 | 0.950 | 0.933 | 0.816 |
| Ruido fuerte (25) | 0.517 | 0.224 | 0.383 | 0.228 |
| Ruido fuerte + suavizado 1 | 0.822 | 0.742 | 0.876 | **0.658** |

**Lectura:**

- **Con ruido leve, suavizar recupera casi todo**: 0.654 → 0.851 (mAP50 0.730 → 0.968), sin reentrenar.
- Con ruido fuerte recupera mucho (0.228 → 0.658) pero no llega al objetivo (0.85): ahí la demo debe **avisar** que el resultado no es confiable.
- En imágenes limpias el filtro **cuesta 0.014** (0.873 → 0.859): más que el umbral de 0.01, así que **no se aplica siempre**, solo cuando la imagen viene ruidosa.
- σ = 1 es mejor que 1.5.
- Regla para la demo: medir el ruido de la imagen → si es ruidosa, suavizar con σ = 1 y mostrar aviso de calidad; si es muy ruidosa, aviso de "resultado no confiable". No hace falta un H8 con aumento de ruido para el ruido leve.

# 7. Aporte de CBAM en H7 (apagado al predecir) · notebook 10

| Cabeza | Con CBAM | CBAM apagado | Caída |
|---|---|---|---|
| Clasificación (F1) | 0.992 | 0.444 | −0.55 |
| Clasificación (AUC) | 0.999 | 0.905 | −0.10 |
| Detección (mAP50) | 0.974 | 0.401 | −0.57 |
| Detección (IoU) | 0.948 | 0.281 | −0.67 |
| Segmentación (Dice) | 0.971 | 0.678 | −0.29 |
| Fragmentos (Dice) | 0.873 | 0.461 | −0.41 |

**Cuidado al presentarlo:** esto mide cuánto **depende** H7 de CBAM, no cuánto **gana** por tenerlo.
H7 se entrenó con CBAM; quitárselo de golpe lo descuadra, por eso caídas tan grandes.
En la ablación anterior (semanas 9–10), entrenando un modelo **sin** CBAM, la diferencia fue pequeña:
F1 0.988 vs 0.975 · mAP50:95 0.883 vs 0.872 · Dice fragmento 0.851 vs 0.851 (las mismas cifras del model card).
Aparte, en la semana 10, al elegir dónde poner CBAM: un bloque mAP50 0.926, tres bloques 0.907, sin CBAM 0.900.
Las cabezas que más dependen son **detección y clasificación**; la segmentación, menos.
Para el aporte real en H7 habría que entrenar H7 sin CBAM (`--cbam none`, ~70 min en Kaggle).

# 8. Conclusiones

1. **Rotación con metadatos (punto y cruz):** hecho; todo queda en RAI.
2. **Paciente girado:** el enderezado recupera casi todo (60°: Dice fragmento 0.61 → 0.85; mAP50 0.31 → 0.96).
3. **Sin EMA:** H7 cumple las 9 métricas, iguala o mejora a H6 en rotación y queda cerca en ruido (H6 mejor con ruido leve, H7 con ruido fuerte). **H7 es el modelo final**, como pidió el profesor.
4. **Imágenes feas:** resiste desenfoque y baja resolución; **el ruido es el punto débil** (de ambos modelos), y **suavizar antes del modelo lo recupera** con ruido leve (0.654 → 0.851).

# 9. Lo que falta

- Demo: detector de ruido + suavizado σ = 1 solo si es ruidosa + aviso de calidad (umbral del detector por definir en validación).
- Subir el umbral del enderezado a ~10° (decidido en validación, sin nueva consulta a test).
- Model card: actualizar el conteo de consultas a test (ahora 5).
- Conectar el enderezado a la demo.
- Actualizar paper, model card y README con H7.
- (Opcional) Entrenar H7 sin CBAM para el aporte real.
- Limitaciones a declarar: solo se endereza el giro axial; el caso 044 (campo corto) se estima mal.

# 10. Cómo abrir el modelo (sin entrenar)

```python
import torch
ck = torch.load("modelos/h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42.pth", map_location="cpu", weights_only=False)
print(ck["args"]["ema"], ck["args"]["canonical"], ck["args"]["aug_mm"], ck["args"]["base_channels"])   # 0.0 True True 32
```
Para evaluarlo: `python src/evaluate_volumes.py --checkpoint <pth> --cache-dir <caché> --split val --tta`
(rama `feat/normalizacion-orientacion`).

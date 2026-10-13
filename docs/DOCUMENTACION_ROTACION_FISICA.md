# Documentación de orientación y rotación física

## 1. Objetivo

Esta implementación busca resolver dos problemas distintos presentes en los volúmenes médicos del conjunto de datos PENGWIN:

1. Los archivos `.mha` no están almacenados todos con la misma orientación física.
2. Una rotación realizada solamente en píxeles puede deformar la anatomía cuando el tamaño físico del píxel es diferente en los ejes X e Y.

También se desarrolló un método adicional para detectar si el paciente está girado dentro del escáner, aunque los metadatos de orientación sean correctos.

La implementación toma como referencia metodológica la normalización mediante orientación física descrita por Ferro Sánchez et al. en *MethodsX* (2024):

- Artículo: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11652745/>
- DOI: `10.1016/j.mex.2024.103073`

La corrección basada en simetría fue adaptada específicamente a la anatomía de la pelvis; no reutiliza las estructuras anatómicas craneales del artículo.

## 2. Resumen de los procesos implementados

Se deben distinguir tres procesos:

| Proceso | Propósito | Estado en el modelo final |
|---|---|---|
| Orientación canónica por metadatos | Unificar la orientación física de los volúmenes | Aplicado |
| Rotación física aleatoria de entrenamiento | Aumentar la robustez sin deformar la anatomía | Aplicado, hasta ±35° |
| Enderezado automático por simetría pélvica | Corregir al paciente girado dentro del escáner | Implementado y auditado, pero no activado permanentemente en el checkpoint final |

El checkpoint final de Semana 10 registra:

```text
canonical = True
aug_mm = True
aug_rot_deg = 35
normalizar = False
```

Por tanto, el modelo final sí utiliza orientación canónica y fue entrenado con rotación física aleatoria. El enderezado automático por simetría se dejó como una auditoría controlada sobre validación.

## 3. Orientación canónica mediante metadatos

### 3.1 Problema encontrado

Los volúmenes de PENGWIN no están todos guardados con la misma matriz de dirección:

- 34 de los 100 casos tienen `TransformMatrix = -1 0 0 0 -1 0 0 0 1`.
- Los otros 66 casos tienen la matriz identidad.

Sin normalización, aproximadamente un tercio de los pacientes entra girado 180° en el plano axial respecto a los demás. Esto obliga a la red a aprender simultáneamente dos orientaciones para estructuras como el sacro y los coxales izquierdo y derecho.

### 3.2 Procedimiento aplicado

Antes de construir la entrada 2.5D se procesa la matriz `TransformMatrix` de la cabecera MHA:

1. Se leen los tres ejes físicos de la imagen.
2. Se calculan sus normas.
3. Se utilizan productos punto para verificar que los ejes sean perpendiculares.
4. Se utiliza el producto cruz de los ejes X e Y y su producto punto con Z para comprobar la dirección del sistema de coordenadas.
5. Se calcula el determinante de la matriz.
6. Se comprueba que sea una matriz ortonormal y alineada con los ejes.
7. Imagen y máscara se llevan a una orientación común con `SimpleITK.DICOMOrient(image, "LPS")`.

Las matrices inválidas se rechazan. Una matriz oblicua válida tampoco se corrige silenciosamente, porque llevarla a identidad requeriría un remuestreo 3D explícito.

Para las matrices alineadas con los ejes, la operación usa permutaciones e inversiones de ejes. Esto evita interpolar intensidades o alterar los identificadores de la máscara.

### 3.3 Archivo principal

La implementación se encuentra en:

```text
src/orientacion.py
```

Funciones principales:

```python
read_transform_matrix()
analyse_direction()
canonicalize_arrays()
```

La integración con los datos está en:

```text
src/build_case_cache.py
src/prepare_week9_detection_data.py
```

`prepare_week9_detection_data.py` aplica la orientación canónica antes de construir cada entrada `[Z-1, Z, Z+1]` y guarda el espaciado físico ajustado al tamaño de 384 × 384.

## 4. Rotación física aleatoria durante el entrenamiento

### 4.1 Por qué se cambió la rotación anterior

La rotación antigua trabajaba directamente sobre la cuadrícula de píxeles. Si `mm_x` y `mm_y` eran diferentes, una rotación matemática en píxeles no equivalía a una rotación física real y podía estirar o comprimir la pelvis.

La nueva implementación incorpora los milímetros por píxel de cada muestra en la matriz afín. De esta manera, la transformación representa el giro del paciente en el espacio físico.

### 4.2 Configuración utilizada

En Semana 9 y Semana 10 se utilizaron los mismos parámetros:

```text
--canonical
--aug-geo 1.0
--aug-rot-deg 35
--aug-mm
```

Esto produce ángulos aleatorios uniformes entre `-35°` y `+35°`.

La transformación geométrica tiene las siguientes características:

- Probabilidad de aplicación: 80% por muestra.
- El 20% restante conserva la geometría original.
- Los tres canales de la entrada 2.5D reciben exactamente la misma transformación.
- La máscara recibe la misma transformación que la imagen.
- Las imágenes se remuestrean mediante interpolación bilineal.
- Las máscaras se remuestrean mediante vecino más cercano para preservar los IDs.
- El aumento también incluye pequeñas variaciones de escala, proporción y traslación.
- Los valores externos al campo de visión se rellenan con cero.

### 4.3 Construcción física de la matriz

El código recibe `spacing_yx`, que contiene los milímetros por píxel de cada muestra. La relación física utilizada es:

```python
physical_y_over_x = (
    spacing_y * image_height
) / (
    spacing_x * image_width
)
```

Esta relación se incorpora en los términos cruzados de la matriz de rotación. Así se evita deformar la anatomía cuando el espaciado no es isotrópico.

### 4.4 Archivos implicados

Implementación compartida:

```text
src/aumento_fisico.py
```

Función principal:

```python
random_affine()
```

Integración con Semana 9:

```text
src/train_week9_detection.py
notebooks/NUEVA_PROPUESTA_01_S9_entrenar_25D.ipynb
```

Integración con Semana 10:

```text
src/train_s10.py
notebooks/NUEVA_PROPUESTA_02_S10_multitarea.ipynb
```

La coherencia entre Semana 9 y Semana 10 es obligatoria. `train_s10.py` comprueba que el checkpoint heredado de Semana 9 tenga la misma configuración `canonical` y `aug_mm`.

## 5. Enderezado automático mediante simetría pélvica

### 5.1 Problema que cubre

La cabecera MHA describe la orientación de los ejes de la imagen, pero no necesariamente indica si el paciente estaba girado dentro del escáner.

Para abordar ese problema se implementó un segundo método que analiza la anatomía de la pelvis.

### 5.2 Procedimiento

El algoritmo realiza los siguientes pasos:

1. Selecciona cortes del volumen para reducir el coste de procesamiento.
2. Obtiene una aproximación de hueso mediante un umbral de intensidad.
3. Construye una proyección axial de frecuencia de hueso.
4. Suaviza la proyección con un filtro gaussiano.
5. Prueba ángulos entre `-70°` y `+70°`, en pasos de 1°.
6. Para cada ángulo, gira la proyección usando el espaciado físico.
7. Centra la masa de la pelvis.
8. Compara la proyección con su reflejo izquierda-derecha.
9. Selecciona el ángulo que maximiza la simetría.

Se utiliza una frecuencia de aparición de hueso y no únicamente el valor máximo. Esto reduce la influencia de metal, ruido y fragmentos aislados.

### 5.3 Criterios de seguridad

La corrección no se aplica automáticamente solo porque exista un ángulo con mayor simetría. Debe cumplir todos los controles siguientes:

- Ángulo absoluto superior a 5°.
- Mejora de simetría igual o superior a `0.02`.
- Score absoluto de simetría igual o superior a `0.35`.
- Conservación mínima del 97% de la masa proyectada.
- El ángulo no debe quedar en el límite de búsqueda.
- Debe existir suficiente hueso para realizar una estimación estable.

Las posibles razones de abstención quedan registradas:

```text
insufficient_bone
angle_below_5deg
search_boundary
small_symmetry_improvement
low_absolute_symmetry
field_of_view_crop_risk
```

Si la corrección es aceptada:

- La imagen se gira con interpolación de orden 1.
- La máscara se gira con interpolación de orden 0.
- Se conserva la forma `[Z, Y, X]` del volumen.
- Todos los cortes reciben el mismo ángulo.

### 5.4 Archivo principal

El método se encuentra en:

```text
src/orientacion.py
```

Funciones relevantes:

```python
rotate_inplane_physical()
_symmetry_score()
estimate_pelvis_correction()
normalize_patient_rotation()
```

## 6. Flujo completo de la rotación

### 6.1 Flujo usado durante el entrenamiento

```text
Volumen MHA
    ↓
Lectura de TransformMatrix y spacing
    ↓
Validación con productos punto y cruz
    ↓
Orientación canónica común
    ↓
Redimensionamiento a 384 × 384 conservando spacing_yx ajustado
    ↓
Construcción de entrada 2.5D [Z-1, Z, Z+1]
    ↓
Rotación física aleatoria entre -35° y +35°
    ↓
Misma transformación para imagen y máscara
    ↓
Backbone y cabezas del modelo
```

### 6.2 Flujo usado en la auditoría

```text
Volumen de validación en orientación canónica
    ↓
Giro físico artificial de 30° o 60° mediante rotmm
    ↓
Estimación de simetría pélvica
    ↓
Comprobaciones de confianza y riesgo de recorte
    ↓
Corrección física si se acepta
    ↓
Inferencia y comparación de métricas
```

## 7. Configuración exacta del entrenamiento final

### 7.1 Semana 9

La nueva propuesta de Semana 9 entrenó desde cero el backbone 2.5D, CBAM y la cabeza de detección con:

```text
--context 1
--canonical
--aug-geo 1.0
--aug-rot-deg 35
--aug-mm
```

Entrada del modelo:

```text
Canal 1: corte Z-1
Canal 2: corte Z
Canal 3: corte Z+1
```

### 7.2 Semana 10

Semana 10 cargó desde Semana 9:

- Backbone.
- CBAM.
- Cabeza de detección.

Después agregó y entrenó las cabezas de clasificación y segmentación. Conservó la misma orientación canónica y el mismo aumento físico de ±35°.

El comando utilizado incluyó:

```text
--week9-checkpoint outputs/week9_25d/week9_25d_rotacion_fisica_desde_cero_w32_e35_s42.pth
--context 1
--canonical
--aug-geo 1.0
--aug-rot-deg 35
--aug-mm
```

No se usó `--normalizar` durante este entrenamiento. Por eso el checkpoint final registra:

```text
normalizar = False
```

## 8. Auditoría de rotación

### 8.1 Diseño experimental

La auditoría utiliza únicamente validación. Test no se modifica ni se somete a estrés.

Se evaluaron cinco condiciones:

```text
1. limpio_normalizado
2. rotmm30_sin_corregir
3. rotmm30_corregido
4. rotmm60_sin_corregir
5. rotmm60_corregido
```

Los comandos de estrés utilizan:

```text
--stress rotmm=30
--stress rotmm=60
```

`rotmm` indica que el giro se realiza en el espacio físico. El modo antiguo `rot` trabaja sobre la cuadrícula y no debe usarse para justificar robustez médica cuando existen píxeles no cuadrados.

### 8.2 Resultados

| Condición | F1 clasificación | mAP@50 | Dice semántico | Dice instancias |
|---|---:|---:|---:|---:|
| Limpio normalizado | 0.9709 | 0.9192 | 0.9635 | 0.8558 |
| 30° sin corregir | 0.9704 | 0.8403 | 0.9485 | 0.8386 |
| 30° corregido | 0.9714 | 0.9073 | 0.9524 | 0.8426 |
| 60° sin corregir | 0.9027 | 0.2628 | 0.6974 | 0.6059 |
| 60° corregido | 0.9518 | 0.7845 | 0.8626 | 0.7691 |

Aceptación de las correcciones:

- Giro de 30°: 13 de 15 pacientes corregidos.
- Giro de 60°: 11 de 15 pacientes corregidos.
- Volúmenes limpios: no se forzó ninguna corrección innecesaria.

### 8.3 Interpretación

La corrección recupera especialmente el rendimiento de detección. Con un giro de 60°, el mAP@50 pasa de `0.2628` sin corrección a `0.7845` con corrección.

También recupera la segmentación semántica: el Dice pasa de `0.6974` a `0.8626`.

Sin embargo, el método no produce invariancia total ante giros extremos. El Dice de instancias corregido a 60° queda en `0.7691`. Cuando el giro provoca pérdida real de anatomía fuera del campo de visión, ninguna normalización puede reconstruir esa información.

La conclusión defendible es:

> La normalización física mejora sustancialmente la robustez ante giros fuera de distribución, especialmente en detección, pero los giros extremos con pérdida de campo de visión continúan siendo una limitación.

No se debe afirmar que el sistema resiste cualquier rotación ni que corrige todos los casos.

## 9. Diferencia entre lo aplicado y lo auditado

### Aplicado al modelo final

- Normalización de orientación mediante `TransformMatrix`.
- Validación de ejes con productos punto y cruz.
- Orientación canónica antes de formar la entrada 2.5D.
- Rotaciones físicas aleatorias de hasta ±35° durante entrenamiento.
- Misma transformación para los tres canales y la máscara.

### Implementado y auditado, pero no activado permanentemente

- Detección automática del giro del paciente mediante simetría de la pelvis.
- Enderezado automático de cada volumen antes de la inferencia.

Esta separación evita presentar un experimento de validación como si formara parte de la configuración congelada del modelo final.

## 10. Archivos que deben compartirse

### Archivos esenciales

```text
src/orientacion.py
src/aumento_fisico.py
src/evaluate_volumes.py
notebooks/NUEVA_PROPUESTA_02_S10_multitarea.ipynb
```

### Integración con entrenamiento

```text
src/prepare_week9_detection_data.py
src/train_week9_detection.py
src/build_case_cache.py
src/train_s10.py
notebooks/NUEVA_PROPUESTA_01_S9_entrenar_25D.ipynb
```

### Pruebas automáticas

```text
tests/test_orientacion.py
tests/test_aumento_fisico.py
tests/test_evaluar_normalizado.py
```

### Resultados de la auditoría

```text
outputs/s10/auditoria_rotacion_fisica/limpio_normalizado.json
outputs/s10/auditoria_rotacion_fisica/rotmm30_sin_corregir.json
outputs/s10/auditoria_rotacion_fisica/rotmm30_corregido.json
outputs/s10/auditoria_rotacion_fisica/rotmm60_sin_corregir.json
outputs/s10/auditoria_rotacion_fisica/rotmm60_corregido.json
```

## 11. Resumen breve para exposición

> Primero unificamos la orientación de todos los volúmenes utilizando la matriz física de la cabecera MHA. Verificamos sus ejes mediante productos punto y cruz y llevamos imagen y máscara a una orientación común. Después, durante el entrenamiento, aplicamos giros aleatorios de hasta ±35° en milímetros, teniendo en cuenta el espaciado real de los píxeles para no deformar la pelvis. Adicionalmente desarrollamos un enderezado conservador basado en la simetría izquierda-derecha de la pelvis. Este último se evaluó mediante una auditoría con giros físicos de 30° y 60° sobre validación. Recuperó gran parte del rendimiento, especialmente en detección, pero se abstiene si existe riesgo de recorte o baja confianza.

## 12. Referencia principal

Ferro Sánchez et al. *MethodsX* (2024). Implementación de referencia para la normalización de orientación mediante metadatos y relaciones geométricas entre ejes.

<https://pmc.ncbi.nlm.nih.gov/articles/PMC11652745/>

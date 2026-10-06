# Plan de la Semana 11 — visualizadores y dashboard

Versión 2, 5 de octubre de 2026 (20:30, hora de Colombia). Rama revisada:
`semana10-multiples-pruebas-working` (commit `30fdad3` más los resultados sin commitear de
`outputs/s10/`). Complementa `docs/plan_semana10.md` y `docs/decisiones_semana10.md`.

Cambios frente a la versión 1: entran las decisiones del equipo (sección 8), el candidato
con EMA ya está en disco, y se incorpora el hallazgo de `decisiones_semana10.md` sobre la
distancia en mm.

## Estado al 5 de octubre (noche)

Construido en esta sesión: inferencia sobre volumen crudo, exportador por caso, los tres
visualizadores en Streamlit, pestaña de métricas y latencia, 15 pruebas y la guía
`docs/semana11_dashboard.md`. Casos exportados: 024, 099 y 044 (validación, candidato D5 + TTA).
La distancia usa un método **provisional** hasta que llegue `fill_unlabeled`; después basta
`demo_export.py --reuse`. Pendiente: latencia en Mac y GPU, túnel de Cloudflare, video de
respaldo, selector de modelos, model card y pesos.

## 0. Respuesta corta

**Sí se puede empezar ya.** El lado del modelo alcanza para los visualizadores 2 y 3: hay
checkpoint del candidato, preprocesamiento, inferencia por volumen, NMS y post-proceso 3D.
Lo que falta es todo el lado de la aplicación, que hoy no existe: no hay ninguna app, ni
render del corte con cajas y máscaras, ni mallas por fragmento.

Antes de la primera pantalla hay que resolver cuatro cosas:

1. **La distancia en mm.** Con el código que hay en el Mac, la distancia predicha sale
   0.0 mm en los 41 fragmentos secundarios de validación. La corrección (`fill_unlabeled`,
   `--distance-nofill`) está descrita en `decisiones_semana10.md` pero no está en `src/` de
   este repo. Sin ella, los visualizadores 2 y 3 mostrarían 0.0 mm en todo.
2. El caché a 384 px de los casos de la demo en el Mac (hoy solo está en Kaggle).
3. Las predicciones guardadas por caso (`infer_volume.py` solo guarda métricas en JSON).
4. Una inferencia que no dependa del ground truth (`evaluate_case` lo exige como entrada).

Lo demás que queda abierto de la Semana 10 (SAM, test, multi-celda) no bloquea. Las dos
líneas avanzan en paralelo: la GPU entrena y evalúa, el visualizador se desarrolla en CPU.

**Fechas:** avance el miércoles 7 de octubre; entrega la semana siguiente.

## 1. Qué pide el enunciado para la Semana 11

Entregable: visualizadores 2 y 3 terminados, dashboard integrado, despliegue por túnel de
Cloudflare, latencia en CPU y GPU, model card, pitch y repositorio final con pesos.

| Visualizador | Contenido obligatorio | Método que fija el enunciado |
|---|---|---|
| 1. Volumen crudo | Reconstrucción 3D inicial con vista MIP, solo hueso | Umbral en HU sobre el volumen sin procesar; sin modelo |
| 2. Corte por corte | Detección, segmentación y clasificación, con slider de cortes | Overlay 2D de cajas y máscaras coloreadas |
| 3. Reconstrucción 3D final | Fragmentos coloreados por macro-hueso, con la distancia junto a la etiqueta de cada fragmento | Máscaras apiladas + marching cubes (skimage) + Plotly Mesh3d |

Condiciones que cruzan los tres:

- La distancia de cada fragmento debe verse **en los visualizadores 2 y 3**. Una tabla sola
  no basta.
- Advertencia de uso no clínico visible en el dashboard.
- Demo en vivo por túnel de Cloudflare, accesible desde computador y celular, con video
  pregrabado de respaldo.
- La demo es corte a corte (pedido del avance del 2 de octubre).

## 2. Inventario: qué hay y qué falta

| Pieza | Estado | Dónde |
|---|---|---|
| Checkpoint del candidato (c2 + EMA) | **Hay** | `outputs/s10/d5_ema_w32_e60_s42.pth` (11 MB) y semillas 43 y 44 |
| Preprocesamiento crudo → 384 px | **Hay** | `pengwin_io.py`, `build_case_cache.py` |
| Inferencia por volumen con TTA | **Hay**, pero pide el GT | `evaluate_volumes.evaluate_case`, `EnsembleTTA` |
| Cajas, NMS propio y refinamiento | **Hay** | `week9_detection.decode_predictions`, `pengwin_metrics.refine_boxes_with_semantic` |
| Post-proceso 3D → volumen de IDs | **Hay**, medido en validación: Dice 3D principal 0.911, secundario 0.605 con el candidato | `volume_postprocess.postprocess_3d`, `outputs/s10/d5_volumen_val.json` |
| Distancia borde a borde en mm | **Hay la función, falla el uso**: el relleno del post-proceso pega los fragmentos y todo da 0.0 mm | `fragment_distances.py`; corrección pendiente de traer a `src/` |
| Marching cubes + Mesh3d | **Hay** el patrón | `volumen_3d_visualizer.py` |
| Datos crudos de los 100 casos | **Hay** en el Mac | `data/raw/` |
| Visualizador 1 | **Parcial**: HTML sueltos de 10 a 31 MB, isosuperficie en vez de MIP | `outputs/figures/volumen_3d_*.html` |
| Caché `data/cache/s10/cache_384` | **Falta** en el Mac | Solo en el dataset de Kaggle `pengwin-s10-datos` |
| Predicciones guardadas por caso | **Falta** | — |
| Inferencia sin ground truth | **Falta** | — |
| Render del corte (visualizador 2) | **Falta** | — |
| Mallas por fragmento (visualizador 3) | **Falta** | — |
| Dashboard | **Falta**; no hay framework en `requirements.txt` | — |
| `cloudflared` | Sin verificar en el Mac | — |
| Script de latencia | **Falta** | Los 133.7 ms por corte del JSON de validación incluyen TTA y cálculo de métricas |

## 3. Modelos que entran al dashboard

Selector de modelo que lee un registro de checkpoints, para cambiar el modelo final sin
tocar código. Validación sobre el volumen completo, 15 casos, TTA espejo, semilla 42:

| Modelo | Canales | Parámetros | mAP50 ref. | Dice seg. | Dice inst. | Dice secundarios | Papel |
|---|---|---|---|---|---|---|---|
| D5: c2 + EMA | 32 | 2.74 M | 0.959 | 0.968 | 0.851 | 0.703 | **Principal**: candidato confirmado con 3 semillas |
| c2 | 32 | 2.74 M | 0.962 | 0.969 | 0.849 | 0.687 | Mismo modelo sin EMA, para comparar |
| c1 | 16 | 688 mil | 0.955 | 0.969 | 0.839 | 0.649 | Intermedio |
| B1 | 8 | 173 mil | 0.935 | 0.963 | 0.835 | 0.661 | El más liviano, para el contraste de latencia en CPU |

Para el avance del miércoles basta con D5. Los otros tres se agregan re-exportando casos.

## 4. Diseño

### 4.1 Dos pasos: exportar y mostrar

El dashboard no infiere el volumen completo en vivo. Un caso tiene de 190 a 410 cortes e
`infer_volume.py` tomó entre 42 y 70 s por caso en GPU; en la CPU del Mac sería varias
veces más.

```
PASO A · exportar (lento, una vez por caso y por modelo)
  .mha crudo → preprocesar a 384 → modelo + TTA → cajas (NMS) + semántica + instancia
  → post-proceso 3D → distancias en mm → mallas
  → outputs/demo/<modelo>/<caso>.npz

PASO B · dashboard (rápido, solo lee)
  V1 crudo · V2 corte a corte · V3 reconstrucción 3D
  + botón "inferir este corte en vivo" que carga el modelo y muestra la latencia
```

El botón de inferencia en vivo demuestra que el pipeline corre de verdad y en CPU. Lo
demás sale del archivo exportado, así la sustentación no depende de la velocidad de la
máquina.

### 4.2 Contenido de cada archivo exportado

- Imagen preprocesada `(Z, 384, 384)` en uint8 y spacing ajustado al caché.
- Semántica y volumen de IDs tras el post-proceso 3D.
- Por corte: cajas crudas tras NMS, cajas refinadas, puntajes y probabilidades de clasificación.
- Tabla de fragmentos predichos: región, ID, volumen, distancia en mm al principal, medida
  sobre el volumen **sin relleno**.
- La misma tabla sobre el ground truth, para la comparación.
- Mallas por fragmento ya reducidas (vértices y caras).
- Para el visualizador 1: MIP en los tres ejes y malla del hueso crudo, desde el `.mha`.

### 4.3 Visualizador 1 — volumen crudo

Decidido: se muestran las dos.

- **MIP** (proyección de máxima intensidad): por cada línea que atraviesa el volumen se
  toma el vóxel más brillante. El resultado es una imagen 2D parecida a una radiografía
  donde resalta el hueso. Es la que nombra el enunciado; se hace en los tres ejes sobre el
  volumen umbralizado a 250 HU.
- **Isosuperficie**: la malla 3D de la superficie del hueso, que se rota con el dedo o el
  mouse. Es la que ya existe de la Semana 8; entra en versión liviana.

### 4.4 Visualizador 2 — corte por corte

- Slider de cortes; selector de caso y de modelo.
- Capas con interruptor: cajas, máscaras, ground truth, caja cruda o refinada.
- Color por macro-hueso: un tono para sacro, otro para coxal izquierdo, otro para coxal
  derecho. El fragmento principal va en el tono pleno y los secundarios en variantes del
  mismo tono.
- Junto a cada fragmento secundario visible en el corte: `ID · distancia mm`.
- Panel lateral: probabilidad de cada región en el corte y tabla de fragmentos del caso.

La distancia es una medida 3D del fragmento entero. Se muestra el mismo valor en todos los
cortes donde aparece ese fragmento y se dice así en la leyenda.

### 4.5 Visualizador 3 — reconstrucción 3D

- Marching cubes por fragmento sobre el volumen de IDs, con el spacing en mm.
- Un `Mesh3d` por fragmento, con los mismos colores del visualizador 2.
- Etiqueta 3D en el centroide de cada fragmento: `ID · distancia mm`.
- Línea entre los dos puntos más cercanos del fragmento y su principal (opcional, si alcanza
  el tiempo; `edge_distance_mm` ya calcula los índices).
- Interruptor para superponer el ground truth y ver la diferencia de distancia.
- Presupuesto: unos 150 mil triángulos por caso, para que abra en un celular.

### 4.6 Dashboard

- Decidido: **Streamlit**. Slider y Plotly nativos, un solo archivo, se adapta al celular y
  funciona detrás del túnel de Cloudflare.
- Página web propia: queda como segunda fase opcional. Como el dashboard solo lee los
  archivos exportados, una página propia leería los mismos archivos y no obliga a rehacer
  nada del modelo. Se decide después del avance, según cómo responda el slider de
  Streamlit desde un celular a través del túnel.
- Despliegue: el Mac con el túnel de Cloudflare, que es lo que pide el enunciado. AWS no
  hace falta; la latencia en GPU se mide aparte en Kaggle o Colab.
- Tres pestañas, una por visualizador, y una cuarta con métricas y latencia.
- Advertencia de uso no clínico fija en la parte superior.
- Estructura: `app/dashboard.py`, `src/predict_volume.py`, `src/demo_export.py`,
  `src/demo_render.py`, `src/measure_latency.py`, con pruebas en `tests/`.

### 4.7 Casos

- **Desarrollo, con validación.** Solo 5 de los 41 fragmentos secundarios de validación
  están separados del principal en el ground truth; el resto se tocan y miden 0 mm. Para
  que la distancia se vea, los casos de trabajo son:
  - 024: dos fragmentos del coxal izquierdo a 19.0 y 3.2 mm, con Dice 3D 0.90 y 0.91.
  - 099: un fragmento del coxal izquierdo a 12.8 mm, con Dice 3D 0.94.
  - 044: 55 MB, el más liviano, para iterar rápido.
  - 014 y 087: fragmentos separados (3.5 y 5.3 mm) que el modelo segmenta mal (Dice 3D
    0.16 y 0.14). Sirven como casos difíciles para el informe.
- **Demo final, con test:** se exportan después de congelar el modelo. Mirar casos de test
  ahora rompe la regla de medición del plan de la Semana 10 (test una sola vez). Candidatos:
  039 (9 fragmentos, 39 MB, corte no cuadrado de 154×322) y 007 (7 fragmentos).

## 5. Orden de ejecución

### Meta del avance del miércoles 7 de octubre

Se muestra en el Mac, sin túnel:

- **Mínimo:** visualizador 2 completo sobre el caso 024 con el candidato D5: slider,
  cajas, máscaras por macro-hueso y distancia junto a cada fragmento.
- **Objetivo:** además, primera versión del visualizador 3 (mallas por fragmento con su
  distancia) y la MIP del visualizador 1, sobre 024 y 099.

Queda para después del avance: selector con varios modelos, túnel, prueba en celular,
latencia y cierre de la entrega.

### Bloques

```
BLOQUE 0 · Preparación                                              [martes]
  Traer a src/ la corrección de la distancia (fill_unlabeled, --distance-nofill)
  · instalar streamlit · caché 384 de 024, 099 y 044 en el Mac
  · correr infer_volume.py con D5 sobre esos casos
  Compuerta: en 024 y 099 la distancia predicha deja de ser 0.0 mm y se compara
             con la real (19.0, 3.2 y 12.8 mm)

BLOQUE 1 · Exportación                                              [martes]
  predict_volume.py (inferencia sin GT) · demo_export.py · pruebas
  Compuerta: un .npz por caso que se recarga igual a como se guardó

BLOQUE 2 · Visualizador 2                                 [martes a miércoles]
  Compuerta: el slider recorre el caso con cajas, máscaras y distancia junto al fragmento

BLOQUE 3 · Visualizador 3                                         [miércoles]
  Compuerta: malla por fragmento con distancia; abre en un celular

                    ── avance del miércoles 7 de octubre ──

BLOQUE 4 · Dashboard integrado
  V1 (MIP + isosuperficie liviana) · tres pestañas · advertencia · selector de modelo y caso
  Compuerta: los tres visualizadores con el mismo caso y los mismos colores

BLOQUE 5 · Despliegue y latencia
  Túnel de Cloudflare · prueba en celular · video de respaldo
  · latencia por imagen en CPU (Mac) y GPU (Kaggle o Colab), sin TTA y con TTA
  Compuerta: URL pública abierta desde un celular; tabla de latencia

BLOQUE 6 · Cierre de la entrega
  Model card · README con comandos exactos · pesos en el repositorio · IA_USAGE.md · pitch
```

Los bloques 2 y 3 son independientes entre sí una vez termina el 1; se pueden repartir
entre dos personas.

## 6. Riesgos

| Riesgo | Qué se hace |
|---|---|
| La distancia predicha sale 0.0 mm en todo | Es el estado actual del código local. Bloque 0: medir sobre el volumen sin relleno y comprobar en 024 y 099 antes de construir pantallas |
| El modelo no separa el fragmento aunque esté separado en el GT | Dice 3D de secundarios 0.605. Se muestra tal cual y se usa en el informe; la demo abre con 024 y 099, donde sí los separa |
| Las mallas pesan demasiado para un celular | Presupuesto de triángulos y reducción al exportar. Los HTML actuales del V1 pesan hasta 31 MB |
| El túnel se cae en la sustentación | Video pregrabado; la URL de un túnel rápido cambia en cada arranque, se comparte al momento |
| Fragmentos en contacto dan 0.0 mm | Se muestra como "0.0 mm (en contacto)". Es la limitación que el enunciado pide declarar |
| Fragmento real que el modelo no separa | En la comparación con el GT aparece como "no detectado", sin distancia inventada |
| El candidato final cambia | El selector lee un registro; basta re-exportar los casos |
| `.gitignore` excluye `*.pth` | El enunciado pide pesos en el repositorio final: release de GitHub o Git LFS |

## 7. Pendiente de la Semana 10 que no bloquea el visualizador

Sigue en paralelo y alimenta la pestaña de métricas y el informe: evaluación en test una
sola vez, SAM con la caja predicha, asignación multi-celda, ensamble de las tres semillas
con EMA, matriz de confusión, cinco peores casos y el notebook 04. Las ablaciones de CBAM,
transfer learning y λ ya están medidas (`decisiones_semana10.md`, sección 3.6).

## 8. Decisiones

Tomadas el 5 de octubre:

| Tema | Decisión |
|---|---|
| Framework | Streamlit. Página web propia como segunda fase opcional, sobre los mismos archivos exportados |
| Fechas | Avance el miércoles 7 de octubre; entrega la semana siguiente |
| Dónde corre | El Mac, con túnel de Cloudflare. Kaggle y Colab para la latencia en GPU. AWS no entra |
| Visualizador 1 | MIP e isosuperficie, las dos |
| Modelo | D5 (c2 + EMA) como principal |

Abiertas:

1. ¿Dónde está la corrección de la distancia (`fill_unlabeled`) y cuándo llega a `src/`?
2. Día exacto de la entrega.
3. Página web propia: se decide después del avance.

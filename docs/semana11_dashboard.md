# Semana 11 — dashboard con tres visualizadores

Estado al 5 de octubre de 2026. Plan completo en `docs/plan_semana11.md`.

## Cómo correrlo

Desde la raíz del repo, con el `.venv` activo:

```bash
pip install -r requirements.txt                       # agrega streamlit
python src/demo_export.py --case 024 --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth --tta
streamlit run app/dashboard.py
```

- `demo_export.py` corre el pipeline completo una vez por caso y deja todo en
  `outputs/demo/<modelo>/<caso>/`. En CPU toma unos 4 minutos por caso con TTA. El dashboard solo
  lee esos archivos, así que abre al instante.
- `--reuse` rehace solo el post-proceso, las distancias y las mallas, sin volver a correr el modelo.
  Es lo que hay que usar cuando llegue la corrección de la distancia.
- `DEMO_ROOT=<carpeta>` cambia dónde busca el dashboard los casos exportados.
- El botón «Inferir este corte en vivo» del visualizador 2 carga el modelo real y muestra la
  latencia; necesita `torch` y el checkpoint en `outputs/s10/`.

## Qué hay

| Archivo | Para qué |
|---|---|
| `src/predict_volume.py` | Inferencia sobre un volumen crudo, sin ground truth: preprocesa como el entrenamiento, corre las tres cabezas con NMS propio y devuelve semántica, mapa primario/secundario, cajas y clasificación por corte |
| `src/demo_export.py` | Exportador por caso: volumen de IDs, distancias en mm, comparación con el ground truth, mallas por fragmento y vista cruda |
| `src/demo_render.py` | Dibujo del visualizador 2 (corte con cajas, máscaras y etiquetas), del 3 (mallas Plotly) y de la isosuperficie cruda |
| `src/raw_view.py` | Visualizador 1: MIP en tres ejes e isosuperficie del hueso, sobre el volumen sin procesar |
| `src/measure_latency.py` | Latencia por imagen en CPU y GPU (mediana y p90); escribe `outputs/demo/latency.json` |
| `app/dashboard.py` | Dashboard Streamlit con tres pestañas, métricas y latencia, y la advertencia de uso no clínico siempre visible |
| `tests/test_demo_*.py`, `tests/test_predict_volume.py` | 15 pruebas: distancias con spacing anisotrópico, fuga de vóxeles, etiquetas, mallas en mm, forma de las salidas |

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

La distancia es la mínima borde a borde entre el fragmento y el principal de su hueso, con
`distance_transform_edt` y el spacing del header (ajustado al estiramiento a 384 px). Es una
medida del fragmento entero en 3D: en el visualizador 2 se repite en todos los cortes donde aparece.

## Pendiente de la corrección de distancia

`volume_postprocess.postprocess_3d` rellena por defecto cada vóxel de la región sin etiqueta con el
fragmento principal, y eso deja todas las distancias predichas en 0.0 mm
(`docs/decisiones_semana10.md`, sección 5). La corrección (`fill_unlabeled`) la está haciendo otra
sesión y todavía no está en `src/`.

Mientras tanto `demo_export.build_id_volume` usa una versión **provisional**: conserva solo los
vóxeles que la cabeza de instancia etiquetó y reduce el principal a su mayor componente conexa. Esto
último importa: sin ello, una fuga de 7 vóxeles del principal dentro de un fragmento llevó su
distancia de 19.0 a 0.0 mm en el caso 024. El método queda escrito en `meta.json`
(`distance_method`) y el dashboard muestra un aviso mientras sea el provisional. Cuando
`postprocess_3d` acepte `fill_unlabeled`, `build_id_volume` la detecta sola; basta con
`python src/demo_export.py --case <caso> --checkpoint <pth> --reuse` para rehacer cada caso.

## Resultado en el caso 024 (validación, candidato D5, TTA)

Distancias con el método provisional, predicción contra ground truth, solo fragmentos secundarios
separados del principal:

| Fragmento real | Real | Predicho | Error |
|---|---|---|---|
| Coxal izquierdo F2 | 19.0 mm | 19.0 mm | 0.0 mm |
| Coxal izquierdo F3 | 3.2 mm | 3.9 mm | 0.8 mm |

Son dos fragmentos de un solo caso: sirven para comprobar que la medición funciona, no para
estimar su error. El Dice 3D de esos dos fue 0.90 y 0.91. Los otros cuatro secundarios reales de
este caso están en contacto con el principal (0.0 mm); el modelo no detectó uno de ellos (coxal
derecho F2) y en el sacro y el coxal derecho F1 el Dice 3D fue bajo (0.49 y 0.52).

## Pendiente

- Casos de la demo final (con test, después de congelar el modelo) y selector con varios modelos.
- Latencia en CPU del Mac y en GPU (Kaggle o Colab) con `src/measure_latency.py`.
- Túnel de Cloudflare y prueba desde un celular.
- Línea entre el fragmento y su principal en el visualizador 3 (opcional).

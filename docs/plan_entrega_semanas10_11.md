# Plan de entrega Semanas 10 y 11

## Punto de partida

Las semanas 8 y 9 están cerradas: los 100 volúmenes y etiquetas PENGWIN están validados,
los splits se fijaron por paciente, el preprocesamiento es reproducible y la detección
propia con CBAM pasó pruebas de overfit y validación inicial. Los archivos son `.mha`, no
NIfTI, pero el spacing físico se obtiene igualmente del header de SimpleITK.

## Orden de implementación

| Fase | Resultado verificable | Criterio de aceptación |
|---|---|---|
| 10A | Backbone compartido con tres cabezas | clasificación, detección y segmentación en un solo forward |
| 10B | Segmentación semántica e instancia | máscaras por región y por fragmento con Dice e IoU |
| 10C | Pérdida multitarea | lambdas configurables, registradas y justificadas |
| 10D | Entrenamiento completo | train 70, validación 15, checkpoint por mejor métrica |
| 10E | Evaluación | F1, AUC, IoU bbox, mAP50, mAP50:95, Dice e IoU |
| 10F | Distancia física | distancia borde a borde en mm para predicción y ground truth |
| 10G | Baseline SAM | prompts con bbox predicha, Dice e IoU comparables |
| 11A | Visualizador 2 | slider de cortes con cajas, máscaras, clases y distancias |
| 11B | Visualizador 3 | mallas 3D coloreadas con etiquetas y distancias |
| 11C | Dashboard | tres visualizadores y advertencia de uso no clínico |
| 11D | Rendimiento | latencia por imagen en CPU y GPU cuando haya GPU disponible |
| 11E | Documentación final | model card, README, IA usage, mini-paper y material del pitch |
| 11F | Despliegue | demo accesible por Cloudflare y video de respaldo |

## Decisiones técnicas

- Mantener resolución `256x256` y splits por paciente.
- Usar el backbone y CBAM ya verificados; agregar las otras cabezas sin frameworks de
  detección o segmentación de alto nivel.
- Representar la segmentación con una salida semántica de cuatro clases (fondo y tres
  regiones) y una salida de instancia supervisada por los IDs `0..30` de PENGWIN.
- Entrenar las cabezas desde cero. No usar SAM dentro del pipeline ni para producir labels.
- Ejecutar primero smoke tests y overfit de cada cabeza antes del entrenamiento completo.
- Guardar métricas y configuración dentro de cada checkpoint para reproducibilidad.

## Riesgos que deben quedar explícitos

- El entorno local actual es CPU; el entrenamiento completo y SAM serán más lentos sin GPU.
- La separación de fragmentos que se tocan puede fallar y fusionar instancias.
- La intensidad cambia entre pacientes y existen artefactos de metal/endurecimiento de haz.
- Las métricas iniciales de Semana 9 no son las métricas finales del sistema.
- La descarga y ejecución de SAM requiere pesos externos y suficiente memoria.


# Semillas 43 y 44 de H8: qué se mide y cómo se usa

Escrito el 16 de octubre de 2026, **antes de entrenar ni evaluar** las dos semillas nuevas.

## Qué se hace

- Se entrena H8 con las semillas 43 y 44, con la receta exacta de la semilla 42
  (`outputs/h8/h8_aumm_noise01_noema_w32_e60_s42.json`): solo cambia `--seed`.
- Cada semilla se evalúa igual que la 42: validación sin daño, con ruido σ = 10 y 25, con giro de
  60° y con giro de 60° enderezado, y test sin daño (un modelo con TTA de espejo, tamaño 384).

## Para qué sirve

- Para medir **cuánto varía H8 con la semilla**, en validación y en test, y reportar la media y la
  desviación de las tres semillas.
- **No sirve para elegir modelo.** El modelo entregado sigue siendo H8 semilla 42 salga como salga
  cada semilla nueva, igual que se hizo con las tres semillas de H6. Elegir la mejor semilla mirando
  test sería seleccionar sobre el conjunto que debe quedar intacto.
- No se ajusta ningún umbral ni se reentrena nada después de ver los resultados.

## Cómo debe citarse

La evaluación en test de las semillas 43 y 44 usa el mismo modelo y la misma configuración congelada
de la evaluación de H8 (`outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md`) y se declara como medición de
la variación, no como una nueva consulta para elegir.

## Qué se espera, dicho antes de mirar

Con H6 las tres semillas quedaron a ±0.002 en validación (0.869, 0.868 y 0.872) y prácticamente
iguales en test (0.875, 0.875 y 0.876). Lo esperable es algo parecido con H8 en imágenes limpias. Con
ruido y giro la variación puede ser mayor, porque esas pruebas dependen de lo que aprendió el aumento
de datos.

# Configuración congelada para la evaluación de H8 en test

Escrita el 13 de octubre de 2026, **antes de correr nada sobre test con H8**.

## Modelo y configuración

- **Modelo principal:** `h8_aumm_noise01_noema_w32_e60_s42.pth` (receta de H7 más `--aug-noise 0.1`;
  sin EMA, rotación en milímetros, orientación canónica, semilla 42). Se eligió solo con validación
  (`EXPERIMENTOS_EMA_ROTACION_RUIDO.md`).
- **Inferencia:** un modelo con TTA de espejo, tamaño 384. Sin ensamble, sin enderezado del giro
  axial y sin suavizado. Es la configuración con la que H8 se midió en validación (0.865) y la misma
  de las evaluaciones anteriores de D5, H6 y H7, para que la comparación sea de igual a igual.
- **Sin pruebas de estrés sobre test** (ruido y giro se miden solo en validación).
- Se corre **una sola vez**. No se ajusta ningún umbral ni se reentrena después de ver el resultado.
  Si queda por debajo de algún objetivo, se reporta así.

## Qué más se corre en la misma sesión, sin elegir entre ellos

- **HD95 y ASSD por fragmento** de H8, en validación y en test. Usan las mismas predicciones de la
  evaluación principal; no son una configuración distinta.
- **Ablación de CBAM:** un modelo con la receta de H8 entrenado sin CBAM (`--cbam none`), evaluado en
  validación y en test; y H8 con el CBAM apagado al predecir, también en las dos particiones. Sirven
  para medir el aporte de CBAM, no para elegir modelo: el principal sigue siendo H8 aunque la
  ablación salga mejor en test.

## Cómo debe citarse

Es la **sexta consulta a test** del proyecto (c2 0.848, EMA + CASR 0.847, D5 0.845, H6 0.875,
H7 0.879 y esta). Los casos de test 012 y 039 se usaron para diagnosticar fallos antes de H6, así que
ninguna cifra posterior a D5 es una estimación independiente.

## Qué se espera, dicho antes de mirar

En validación H8 da 0.865 de Dice por fragmento, 0.008 menos que H7 (0.873). H7 dio 0.879 en test.
Lo razonable es que H8 quede cerca de 0.87 en test, algo por debajo de H7.

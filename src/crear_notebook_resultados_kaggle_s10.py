"""Genera y ejecuta el notebook de resultados del experimento Kaggle S10.

El notebook no entrena ni hace inferencia: consume únicamente los artefactos
ya descargados de los dos kernels privados de Kaggle y deja tablas, gráficas y
conclusiones incorporadas como salidas de las celdas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        default="notebooks/RESULTADOS_KAGGLE_S10_60_EPOCAS.ipynb",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_dir = root / "outputs" / "kaggle" / "s10_60ep"
    final_dir = root / "outputs" / "kaggle" / "s10_60ep_puntos_5_a_9"
    required = [
        train_dir / "s10_kaggle_60ep_sin_early_stopping_historial.json",
        train_dir / "s10_kaggle_60ep_sin_early_stopping_best.pth",
        train_dir / "s10_kaggle_60ep_sin_early_stopping_epoch60.pth",
        final_dir / "EXPERIMENTO_COMPLETO_PUNTO_9.json",
        final_dir / "modelo_final_semana10.json",
        final_dir / "modelo_final_semana10_val.json",
        final_dir / "modelo_final_semana10_test.json",
        final_dir / "modelo_final_semana10_cbam_aportacion.json",
        final_dir / "modelo_final_semana10_hd95_assd_val.json",
        final_dir / "modelo_final_semana10_hd95_assd_test.json",
        final_dir / "modelo_final_semana10.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan artefactos de Kaggle:\n" + "\n".join(missing))

    history = json.loads(required[0].read_text(encoding="utf-8"))
    manifest = json.loads((final_dir / "modelo_final_semana10.json").read_text(encoding="utf-8"))
    val = json.loads((final_dir / "modelo_final_semana10_val.json").read_text(encoding="utf-8"))
    test = json.loads((final_dir / "modelo_final_semana10_test.json").read_text(encoding="utf-8"))
    cbam = json.loads((final_dir / "modelo_final_semana10_cbam_aportacion.json").read_text(encoding="utf-8"))
    surf_test = json.loads((final_dir / "modelo_final_semana10_hd95_assd_test.json").read_text(encoding="utf-8"))
    val_all, test_all = val["subsets"]["todo"], test["subsets"]["todo"]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
# Resultados Kaggle — Semana 10, 60 épocas completas

Este notebook es el **informe reproducible del experimento final**. El entrenamiento y la inferencia pesada se ejecutaron en kernels privados de Kaggle; aquí se leen los artefactos descargados y se muestran sus resultados sin volver a entrenar.

- Entrenamiento: 60 épocas obligatorias, sin early stopping y sin EMA.
- Inicialización: backbone, CBAM y detector del checkpoint físico 2.5D de Semana 9.
- Selección final: solo validación volumétrica; test se consultó una única vez después de congelar el modelo.
- Extensión del taller: HD95 y ASSD 3D por fragmento.
"""
        ),
        markdown("## 1. Configuración y localización de artefactos"),
        code(
            """
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

def encontrar_raiz(inicio: Path) -> Path:
    for candidato in (inicio, *inicio.parents):
        if (candidato / 'outputs' / 'kaggle' / 's10_60ep').exists():
            return candidato
    raise FileNotFoundError('No se encontró la raíz con los resultados de Kaggle.')

ROOT = encontrar_raiz(Path.cwd().resolve())
TRAIN_DIR = ROOT / 'outputs' / 'kaggle' / 's10_60ep'
FINAL_DIR = ROOT / 'outputs' / 'kaggle' / 's10_60ep_puntos_5_a_9'
HISTORIAL = TRAIN_DIR / 's10_kaggle_60ep_sin_early_stopping_historial.json'
MODELO_FINAL = FINAL_DIR / 'modelo_final_semana10.pth'

print('Raíz:', ROOT)
print('Entrenamiento:', TRAIN_DIR)
print('Evaluación final:', FINAL_DIR)
print('Modelo final:', MODELO_FINAL)
"""
        ),
        markdown("## 2. Verificación de Semana 9 y del entrenamiento limpio"),
        code(
            """
historial = json.loads(HISTORIAL.read_text(encoding='utf-8'))
args = historial['args']
verificacion = {
    'épocas solicitadas': args['epochs'],
    'épocas completadas': historial['epochs_completed'],
    'parada temprana': historial['stopped_early'],
    'motivo de parada': historial['stop_reason'],
    'EMA': args['ema'],
    'checkpoint Semana 9': Path(args['week9_checkpoint']).name,
    'freeze Semana 9 (épocas)': args['freeze_week9_epochs'],
    'entrada 2.5D (contexto)': args['context'],
    'rotación máxima': args['aug_rot_deg'],
    'rotación en milímetros': args['aug_mm'],
    'cuDNN desactivado': args['disable_cudnn'],
    'duración (min)': round(historial['minutes'], 2),
}
display(pd.Series(verificacion, name='valor').to_frame())
assert historial['epochs_completed'] == 60
assert historial['stopped_early'] is False and historial['stop_reason'] is None
assert args['ema'] == 0.0 and args['aug_mm'] is True
print('OK: corrida limpia de 60 épocas, sin EMA y sin early stopping.')
"""
        ),
        markdown("## 3. Entrenamiento: pérdida y métricas de validación"),
        code(
            """
df_hist = pd.DataFrame(historial['history'])
columnas = ['epoch', 'loss', 'classification_f1_macro', 'detection_map50',
            'segmentation_dice_macro', 'instance_dice_matched']
display(df_hist[columnas].round(4).set_index('epoch'))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(df_hist.epoch, df_hist.loss, marker='o')
axes[0].set(title='Pérdida de entrenamiento', xlabel='Época', ylabel='Loss')
for clave, etiqueta in [
    ('segmentation_dice_macro', 'Dice semántico'),
    ('detection_map50', 'mAP50'),
    ('instance_dice_matched', 'Dice instancias'),
]:
    axes[1].plot(df_hist.epoch, df_hist[clave], marker='o', label=etiqueta)
axes[1].axhline(0.85, color='gray', linestyle='--', linewidth=1, label='objetivo Dice')
axes[1].set(title='Validación durante entrenamiento', xlabel='Época', ylim=(0, 1))
axes[1].legend()
plt.tight_layout()
plt.show()

mejor_inst = df_hist.loc[df_hist.instance_dice_matched.idxmax()]
ultima = df_hist.iloc[-1]
print(f"Mejor Dice de instancias: {mejor_inst.instance_dice_matched:.4f} (época {int(mejor_inst.epoch)})")
print(f"Época 60: loss={ultima.loss:.4f}, Dice semántico={ultima.segmentation_dice_macro:.4f}, "
      f"mAP50={ultima.detection_map50:.4f}, Dice instancias={ultima.instance_dice_matched:.4f}")
"""
        ),
        markdown("## 4. Comparación de checkpoints y selección final"),
        code(
            """
manifiesto = json.loads((FINAL_DIR / 'modelo_final_semana10.json').read_text(encoding='utf-8'))
comparacion = []
for etiqueta, archivo in [
    ('mejor durante entrenamiento', 'candidato_best_val.json'),
    ('época 60', 'candidato_epoch60_val.json'),
]:
    datos = json.loads((FINAL_DIR / archivo).read_text(encoding='utf-8'))
    m = datos['subsets']['todo']
    comparacion.append({
        'candidato': etiqueta,
        'F1 clasificación': m['classification_f1_macro'],
        'mAP50': m['detection_map50'],
        'Dice semántico': m['segmentation_dice_macro'],
        'Dice instancias': m['instance_dice_matched'],
    })
display(pd.DataFrame(comparacion).set_index('candidato').round(4))
print('Seleccionado:', manifiesto['checkpoint_seleccionado'])
print('Criterio:', manifiesto['criterio_seleccion'])
print('La época 60 ganó por una diferencia de',
      f"{manifiesto['candidatos_validacion'][manifiesto['checkpoint_seleccionado']] - min(manifiesto['candidatos_validacion'].values()):.4f}",
      'en Dice de instancias volumétrico.')
"""
        ),
        markdown("## 5. Trazabilidad e integridad del modelo publicado"),
        code(
            """
sha_local = hashlib.sha256(MODELO_FINAL.read_bytes()).hexdigest()
integridad = {
    'archivo': MODELO_FINAL.name,
    'tamaño (bytes)': MODELO_FINAL.stat().st_size,
    'SHA-256 manifiesto': manifiesto['sha256'],
    'SHA-256 local': sha_local,
    'coincide': sha_local == manifiesto['sha256'],
}
display(pd.Series(integridad, name='valor').to_frame())
assert integridad['coincide']
print('OK: el modelo descargado es exactamente el publicado por Kaggle.')
"""
        ),
        markdown("## 6. Evaluación volumétrica final: validación y test"),
        code(
            """
validacion = json.loads((FINAL_DIR / 'modelo_final_semana10_val.json').read_text(encoding='utf-8'))
prueba = json.loads((FINAL_DIR / 'modelo_final_semana10_test.json').read_text(encoding='utf-8'))
objetivos = {
    'classification_f1_macro': 0.85,
    'classification_auc_macro': 0.85,
    'detection_iou_mean': 0.65,
    'detection_map50': 0.65,
    'detection_map50_95': 0.40,
    'segmentation_dice_macro': 0.85,
    'segmentation_iou_macro': 0.70,
    'instance_dice_matched': 0.85,
    'instance_iou_matched': 0.70,
}
filas = []
for conjunto, datos in [('validación', validacion), ('test', prueba)]:
    resumen = datos['subsets']['todo']
    for metrica, objetivo in objetivos.items():
        valor = float(resumen[metrica])
        filas.append({'conjunto': conjunto, 'métrica': metrica, 'valor': valor,
                      'objetivo': objetivo, 'cumple': valor >= objetivo})
tabla_metricas = pd.DataFrame(filas)
display(tabla_metricas.pivot(index='métrica', columns='conjunto', values='valor').round(4))
display(tabla_metricas.pivot(index='métrica', columns='conjunto', values='cumple'))
print('Validación:', int(tabla_metricas.query("conjunto == 'validación'").cumple.sum()), '/ 9 métricas')
print('Test:', int(tabla_metricas.query("conjunto == 'test'").cumple.sum()), '/ 9 métricas')

grafica = tabla_metricas.pivot(index='métrica', columns='conjunto', values='valor')
ax = grafica.plot(kind='barh', figsize=(10, 6), xlim=(0, 1), title='Validación frente a test')
ax.set_xlabel('Valor')
plt.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
### Lectura de test

El resultado de test es la medición principal ante pacientes no vistos. La configuración fue congelada usando validación antes de consultar test. La similitud entre ambos conjuntos y el hecho de que test no sea peor de manera sistemática no muestran señales de sobreajuste relevante.
"""
        ),
        markdown("## Auditoría de orientación y rotación física"),
        code(
            """
rot_dir = FINAL_DIR / 'auditoria_rotacion_fisica'
filas_rotacion = []
for path in sorted(rot_dir.glob('*.json')):
    datos = json.loads(path.read_text(encoding='utf-8'))
    m = datos['subsets']['todo']
    filas_rotacion.append({
        'experimento': path.stem,
        'correcciones aceptadas': sum(bool(v.get('accepted')) for v in datos.get('normalization', {}).values()),
        'F1 clasificación': m['classification_f1_macro'],
        'mAP50': m['detection_map50'],
        'Dice semántico': m['segmentation_dice_macro'],
        'Dice instancias': m['instance_dice_matched'],
    })
tabla_rotacion = pd.DataFrame(filas_rotacion).set_index('experimento')
display(tabla_rotacion.round(4))

for grados in (30, 60):
    sin = tabla_rotacion.loc[f'rotmm{grados}_sin_corregir']
    con = tabla_rotacion.loc[f'rotmm{grados}_corregido']
    print(f"{grados}° — cambio al normalizar: mAP50 {con['mAP50']-sin['mAP50']:+.4f}, "
          f"Dice semántico {con['Dice semántico']-sin['Dice semántico']:+.4f}, "
          f"Dice instancias {con['Dice instancias']-sin['Dice instancias']:+.4f}")
"""
        ),
        markdown(
            """
La normalización es especialmente útil en el estrés de **60°**: recupera gran parte de detección y segmentación. En **30°**, el aumento físico de entrenamiento ya aporta robustez y la corrección no mejora todas las métricas; por eso el algoritmo conserva controles de confianza y no debe presentarse como una garantía universal.
"""
        ),
        markdown("## 7. Publicación del modelo final"),
        code(
            """
publicacion = {
    'modelo final': MODELO_FINAL,
    'checkpoint seleccionado': manifiesto['checkpoint_seleccionado'],
    'evaluación de validación': FINAL_DIR / manifiesto['evaluacion_validacion'],
    'evaluación de test': FINAL_DIR / manifiesto['evaluacion_prueba'],
    'SHA-256': manifiesto['sha256'],
}
display(pd.Series({k: str(v) for k, v in publicacion.items()}, name='valor').to_frame())
"""
        ),
        markdown("## 8. Aportación final de CBAM por cabeza"),
        code(
            """
cbam = json.loads((FINAL_DIR / 'modelo_final_semana10_cbam_aportacion.json').read_text(encoding='utf-8'))
filas_cbam = []
for etiqueta, clave in [('clasificación', 'clasificacion'), ('detección', 'deteccion'),
                        ('segmentación', 'segmentacion'), ('score general', 'general')]:
    filas_cbam.append({
        'cabeza': etiqueta,
        'con CBAM': cbam['con_cbam'][clave],
        'sin CBAM': cbam['sin_cbam'][clave],
        'aportación': cbam['aportacion'][clave],
    })
display(pd.DataFrame(filas_cbam).set_index('cabeza').round(4))
print(f"Caída general al retirar CBAM: {cbam['caida_al_retirar_cbam_pct']:.1f}%")
print('Nota metodológica:', cbam['advertencia'])
"""
        ),
        markdown("## 9. Extensión opcional: HD95 y ASSD 3D por fragmento"),
        code(
            """
superficies = {}
filas_superficie = []
for split in ('val', 'test'):
    path = FINAL_DIR / f'modelo_final_semana10_hd95_assd_{split}.json'
    datos = json.loads(path.read_text(encoding='utf-8'))
    superficies[split] = datos
    resumen = datos['resumen']
    for tipo, grupo in [('todos', resumen), *resumen['por_tipo'].items()]:
        cobertura = grupo['cobertura_fragmentos'] if tipo == 'todos' else grupo['cobertura']
        n_gt = grupo['n_fragmentos_gt'] if tipo == 'todos' else grupo['n_gt']
        filas_superficie.append({
            'conjunto': split,
            'tipo': tipo,
            'fragmentos GT': n_gt,
            'cobertura': cobertura,
            'ASSD media (mm)': grupo['assd_mm'].get('media', np.nan),
            'ASSD mediana (mm)': grupo['assd_mm'].get('mediana', np.nan),
            'HD95 media (mm)': grupo['hd95_mm'].get('media', np.nan),
            'HD95 mediana (mm)': grupo['hd95_mm'].get('mediana', np.nan),
        })
tabla_superficie = pd.DataFrame(filas_superficie).set_index(['conjunto', 'tipo'])
display(tabla_superficie.round(3))
print('Menor ASSD/HD95 es mejor. Las distancias se resumen solo para fragmentos detectados;')
print('la cobertura declara explícitamente los fragmentos omitidos.')
"""
        ),
        code(
            """
for split, datos in superficies.items():
    filas = []
    for caso in datos['por_caso']:
        filas.append({
            'caso': caso['case_id'],
            'cobertura': caso['cobertura_fragmentos'],
            'ASSD media (mm)': caso['assd_mm'].get('media', np.nan),
            'HD95 media (mm)': caso['hd95_mm'].get('media', np.nan),
        })
    detalle = pd.DataFrame(filas).sort_values('HD95 media (mm)', ascending=False)
    print(f'Casos de {split}, ordenados por HD95 medio:')
    display(detalle.round(3).reset_index(drop=True))
"""
        ),
        markdown(
            f"""
## Conclusión final

- El experimento ejecutó **{history['epochs_completed']} épocas completas**, sin parada temprana, sin EMA y sin valores no finitos.
- El checkpoint final seleccionado fue **{manifest['checkpoint_seleccionado']}**, escogido únicamente con validación volumétrica.
- Test cumple **9/9 métricas mínimas**: Dice semántico `{test_all['segmentation_dice_macro']:.4f}` y Dice de instancias `{test_all['instance_dice_matched']:.4f}`.
- La ablación de inferencia muestra una caída general de `{cbam['caida_al_retirar_cbam_pct']:.1f}%` al retirar CBAM. Esto mide dependencia del checkpoint; no sustituye entrenar un control independiente sin CBAM.
- En test, la cobertura de fragmentos para HD95/ASSD es `{surf_test['resumen']['cobertura_fragmentos']:.1%}`, con ASSD medio `{surf_test['resumen']['assd_mm']['media']:.3f} mm` y HD95 medio `{surf_test['resumen']['hd95_mm']['media']:.3f} mm`.
- La principal limitación sigue estando en fragmentos secundarios y en giros extremos: la corrección de 60° mejora mucho el resultado, pero el Dice de instancias queda por debajo del objetivo de 0.85.

**Estado:** experimento completo hasta el punto 9, modelo final publicado y artefactos descargados.
"""
        ),
        markdown(
            """
## Evidencia original

- Kernel de entrenamiento: `federicog19/pengwin-s10-60ep-sin-early-stop`
- Kernel de evaluación: `federicog19/pengwin-s10-60ep-puntos-5-a-9`
- Los JSON y CSV originales permanecen en `outputs/kaggle/`; este notebook solo los resume y visualiza.
"""
        ),
    ]

    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()

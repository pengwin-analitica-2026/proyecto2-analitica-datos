const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak, TableOfContents } = require('docx');

const NAVY = '1F3A5F', TEAL = '2A9D8F', GRAY = '5C677D', LIGHT = 'EEF3F8';
const W = 9360; // ancho útil (Letter, márgenes 1")
const fig = (f) => fs.readFileSync('./fig/' + f);

const P = (text, o = {}) => new Paragraph({ spacing: { after: 120, line: 288 }, alignment: o.align || AlignmentType.JUSTIFIED,
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === 'string' ? new TextRun(t) : t) });
const B = (t) => new TextRun({ text: t, bold: true });
const I = (t) => new TextRun({ text: t, italics: true });
const PEND = (t) => new TextRun({ text: t, highlight: 'yellow', italics: true });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (parts) => new Paragraph({ numbering: { reference: 'vi', level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(parts) ? parts : [parts]).map(t => typeof t === 'string' ? new TextRun(t) : t) });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, italics: true, size: 18, color: GRAY })] });
const image = (f, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
  children: [new ImageRun({ type: 'png', data: fig(f), transformation: { width: w, height: h },
    altText: { title: f, description: f, name: f } })] });

const border = { style: BorderStyle.SINGLE, size: 4, color: 'C9D3DF' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, header, rows, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, isHead, rowIdx) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { fill: NAVY, type: ShadingType.CLEAR, color: 'auto' }
      : (o.highlightRow === rowIdx ? { fill: 'DFF2EF', type: ShadingType.CLEAR, color: 'auto' }
        : (rowIdx % 2 ? { fill: 'F7F9FC', type: ShadingType.CLEAR, color: 'auto' } : undefined)),
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: (i === 0 || (o.leftAll && !isHead)) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: (Array.isArray(txt) ? txt : [txt]).map(t => typeof t === 'string'
        ? new TextRun({ text: t, bold: isHead || (o.boldFirst && i === 0), color: isHead ? 'FFFFFF' : '1B1B1B', size: 18 })
        : t) })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, -1)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, ri)) }))] });
}
const sp = () => new Paragraph({ spacing: { after: 80 }, children: [] });
const ok = (t) => new TextRun({ text: t, color: '1E7F4F', bold: true, size: 18 });

const children = [
  // ---------- Portada ----------
  new Paragraph({ spacing: { before: 2200, after: 200 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: 'PROYECTO INTEGRADOR · CORTE 2', color: TEAL, bold: true, size: 22 })] }),
  new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: 'Detección, segmentación y clasificación de fragmentos de fractura pélvica en CT', bold: true, size: 48, color: NAVY })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: 'Un modelo multitarea 2D propio con atención CBAM, entrenado sobre el dataset PENGWIN 2024', size: 26, color: GRAY })] }),
  new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 8 } }, spacing: { after: 120 },
    children: [new TextRun({ text: 'Analítica de Datos · Universidad Autónoma de Occidente · 2026-2', size: 22 })] }),
  P([B('Profesor: '), 'Carlos Ferro'], { align: AlignmentType.LEFT }),
  new Paragraph({ spacing: { after: 60 }, children: [B('Equipo')] }),
  table([5600, 2000], ['Integrante', 'Código'], [
    ['Gustavo Adolfo Chipantiza', '2222022'], ['Federico Gonzales Cárdenas', '2225657'],
    ['Juan Jacobo Delgado Melo', '2230110'], ['María de los Ángeles Valencia Hurtado', '2235112'],
  ]),
  sp(),
  P([B('Repositorio: '), 'github.com/pengwin-analitica-2026/proyecto2-analitica-datos'], { align: AlignmentType.LEFT }),
  new Paragraph({ spacing: { before: 900 }, shading: { fill: 'FDF1EC', type: ShadingType.CLEAR, color: 'auto' },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'E76F51', space: 8 } },
    children: [new TextRun({ text: 'Uso exclusivamente académico. ', bold: true }), new TextRun('Este sistema no es un dispositivo médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.')] }),
  new Paragraph({ children: [new PageBreak()] }),

  // ---------- Resumen ----------
  H1('Resumen'),
  P(['Presentamos un sistema que, en cada corte de una tomografía pélvica, clasifica la presencia del sacro y de los dos coxales, los localiza con cajas, segmenta sus fragmentos y mide su separación en milímetros. Usa una red 2D con un backbone convolucional propio de 32 canales, atención CBAM y tres cabezas entrenadas desde cero, sin frameworks de detección ni de segmentación. ',
    'Tras una investigación sistemática de la métrica más exigente, el Dice por fragmento, encontramos que un tercio de los volúmenes estaba almacenado girado 180°. Corregir la orientación desde la cabecera del archivo y entrenar con aumento geométrico llevó el modelo final (H6) a ',
    B('cumplir las 9 métricas del enunciado en test'), ', con un Dice por fragmento de ', B('0.875 ± 0.001'), ' en tres semillas (objetivo 0.85). Esta cifra es una segunda evaluación en test, posterior al análisis de fallos; la primera evaluación oficial, con el modelo anterior, dio 0.845. Ambas se reportan.']),


  new Paragraph({ children: [new PageBreak()] }),

  // ---------- 1 ----------
  H1('1. Introducción'),
  P(['Las fracturas pélvicas de alta energía tienen una discapacidad superior al 50 % y una mortalidad superior al 13 %. Para planear su reducción quirúrgica hay que saber cuántos fragmentos hay, a qué hueso pertenece cada uno y cuánto se separó del fragmento principal. El proyecto automatiza esa lectura con cuatro restricciones del curso: ',
    B('arquitectura 2D'), ' corte a corte, ', B('backbone propio con CBAM'), ', ', B('tres cabezas entrenadas desde cero'), ' y ', B('ningún framework de alto nivel'), ' (SAM solo como línea base).']),

  // ---------- 2 ----------
  H1('2. Datos y preprocesamiento'),
  P('Se usaron los 100 volúmenes de entrenamiento de PENGWIN 2024 (Task 1, CT), en formato .mha y con máscaras validadas por expertos. La partición es por paciente, nunca por corte, y está estratificada por número de fragmentos (semilla 42).'),
  table([3400, 5960], ['Aspecto', 'Valor'], [
    ['Partición', '70 entrenamiento · 15 validación · 15 test'],
    ['Etiquetas', '0 fondo · 1–10 sacro · 11–20 coxal izquierdo · 21–30 coxal derecho'],
    ['Fragmentos por caso', '3 a 9 (media 5.75); el ID 1/11/21 es el más grande en 300 de 300 regiones'],
    ['Spacing', '0.66–1.22 mm en el plano · 0.625–1.25 mm entre cortes'],
    ['Orientación', 'Un tercio de los volúmenes viene almacenado girado 180° según su cabecera'],
  ], { boldFirst: true }),
  sp(),
  image('nb_fragmentos.png', 600, 155),
  caption('Figura 1. Fragmentos por caso y por región (1 = sin fractura) y volumen de hueso por caso.'),
  H2('Cadena de preprocesamiento'),
  P(['Cada volumen pasa por cuatro pasos fijos, idénticos en entrenamiento y evaluación: ', B('(1) orientación canónica'), ' según la cabecera, ', B('(2) recorte de HU'), ' a [−1024, 3000], ', B('(3) ventana ósea C400/W1800'), ' y ', B('(4) corrección gamma 1.5'), ', seguida del reescalado a 384 × 384. La orientación canónica no usa etiquetas: los 34 volúmenes guardados en orientación LPI se giran 180° en el plano, igual en la imagen y en la máscara y en todos los splits. En la demo todavía falta aplicarla (sección 9).']),
  image('nb_preproc_ejemplos.png', 430, 321),
  caption('Figura 2. HU crudo, ventana ósea y entrada final al modelo (ventana + gamma 1.5) en tres casos: con más hueso, con metal y con relleno del escáner.'),
  P(['Gamma 1.5 reemplazó a CLAHE tras compararlos con dos criterios independientes (Figura 3): el contraste-ruido entre hueso y tejido blando y la calidad de una máscara de Otsu frente a la etiqueta. CLAHE triplicaba el ruido del tejido blando; gamma no pierde información y separa mejor el hueso.']),
  image('fig_preproc.png', 600, 204),
  caption('Figura 3. Comparación de preprocesamientos en 5 casos de referencia.'),

  // ---------- 3 ----------
  H1('3. Arquitectura y decisiones de diseño'),
  P('Un backbone convolucional propio alimenta tres cabezas en un solo paso hacia adelante (Figura 4). La segmentación tiene dos salidas: una semántica, que define cada hueso, y una de instancia, que separa el fragmento principal de los secundarios dentro de cada hueso.'),
  image('fig_arquitectura.png', 620, 260),
  caption('Figura 4. Arquitectura del modelo final (2.74 M de parámetros).'),
  table([2600, 3800, 2960], ['Decisión', 'Motivo', 'Evidencia'], [
    ['32 canales base', 'Más capacidad para separar fragmentos', 'Dice por fragmento: 8 → 16 → 32 canales = 0.829 → 0.839 → 0.849; 48 canales no mejora'],
    ['CBAM en 1 etapa', 'Atención antes de las cabezas', 'Mejor detección que sin CBAM y que en 3 etapas (sección 6)'],
    ['Instancia primario / secundario', 'Tarea acotada por hueso', 'Con 31 IDs se perdía el 45 % de los secundarios (0.787)'],
    ['Grid y NMS propios', 'Restricción del curso', 'Overfit intencional: pérdida de 1.687 a 0.002 (758×)'],
    ['EMA de pesos (0.999)', 'Estabiliza el entrenamiento', '+0.004 de media en 3 semillas'],
    ['Orientación canónica + aumento geométrico', 'Robustez a la posición del paciente', '+0.016 en validación y +0.028 en test (sección 7)'],
  ], { boldFirst: true }),

  // ---------- 4 ----------
  H1('4. Entrenamiento'),
  table([3400, 5960], ['Parámetro', 'Valor (modelo H6)'], [
    ['Optimizador', 'AdamW, lr 2×10⁻³ con decaimiento coseno, batch 16, precisión mixta'],
    ['Duración', '60 épocas × 2 500 cortes (≈ 65 min por modelo en una GPU T4)'],
    ['Aumento de datos', 'Espejo sagital con intercambio de coxales; rotación hasta ±35°, escala, estiramiento y desplazamiento'],
    ['Pérdida', 'Detección + clasificación (0.5) + semántica CE/Dice (2.0 / 1.0) + instancia CE (secundarios ×3) / Dice + borde'],
    ['Promedio de pesos', 'EMA con decaimiento 0.999'],
    ['Semillas', '42, 43 y 44 (todas reportadas)'],
  ], { boldFirst: true }),

  // ---------- 5 ----------
  H1('5. Resultados'),
  P('Todas las métricas se calculan sobre todos los cortes del volumen de cada caso (no sobre una muestra), con TTA de espejo. El modelo final cumple los nueve objetivos del enunciado con margen (Figura 5).'),
  image('fig_metricas.png', 560, 299),
  caption('Figura 5. Métricas del modelo final en test frente al objetivo del enunciado.'),
  table([2900, 1400, 1400, 1400, 2260], ['Métrica (test, 15 casos)', 'H6 s42', 'H6 s43', 'H6 s44', 'Objetivo'], [
    ['F1 / AUC clasificación', '0.990 / 0.999', '0.986 / 0.998', '0.989 / 0.999', ['≥ 0.85 ', ok('✓')]],
    ['IoU de detección', '0.943', '0.938', '0.942', ['≥ 0.65 ', ok('✓')]],
    ['mAP@0.5 / mAP@[0.5:0.95]', '0.961 / 0.902', '0.952 / 0.894', '0.960 / 0.902', ['≥ 0.65 / 0.40 ', ok('✓')]],
    ['Dice / IoU semántico', '0.971 / 0.943', '0.970 / 0.943', '0.970 / 0.942', ['≥ 0.85 / 0.70 ', ok('✓')]],
    ['IoU por fragmento', '0.816', '0.814', '0.815', ['≥ 0.70 ', ok('✓')]],
    ['Dice por fragmento', '0.875', '0.875', '0.876', ['≥ 0.85 ', ok('✓')]],
  ], { boldFirst: true }),
  sp(),
  P(['En validación, la misma medición da 0.869, 0.868 y 0.872. ', B('Modelo entregado: '), 'H6 semilla 42, la de referencia, fijada por escrito antes de evaluar test (docs/decisiones_semana10.md, sección 12.4). Las tres semillas son equivalentes en test (±0.001).',
    ' El ensamble de las tres alcanza 0.879, pero triplica el costo de inferencia, por lo que no se usa en la demo en tiempo real.']),
  P(['La detección se mide con cajas refinadas con la máscara semántica. Con la cabeza grid sola, en validación (H6 s42), el mAP@0.5 es 0.887, el mAP@[0.5:0.95] 0.506 y el IoU 0.766 (0.971, 0.904 y 0.949 refinados): la cabeza cruda también cumple los objetivos, y el refinamiento es un post-proceso.']),
  H2('Desempeño computacional'),
  P(['Latencia por imagen en GPU y CPU: ', PEND('[pendiente: medir y completar]'), '.']),

  // ---------- 6 ----------
  H1('6. Estudio de ablación'),
  P('Las ablaciones obligatorias se hicieron sobre la receta de 32 canales con EMA (validación, semilla 42 salvo que se indique), cambiando una sola cosa a la vez.'),
  table([2600, 2300, 2300, 2160], ['Ablación', 'Con', 'Sin', 'Conclusión'], [
    ['CBAM', 'F1 0.988 · mAP50:95 0.883', 'F1 0.975 · mAP50:95 0.872', 'Mejora clasificación y detección; no cambia la segmentación'],
    ['Transfer learning (backbone preentrenado en Intel Image Classification)', 'Dice frag. 0.851 / 0.856 (s42 / s43)', '0.851 / 0.855', 'Neutral: no se usa'],
    ['λ instancia ×2', 'Dice frag. 0.855', '0.851', 'Sin mejora clara; empeora la detección'],
  ], { boldFirst: true }),
  sp(),
  H2('Ablación de los componentes de H6'),
  P('H6 agrega dos cambios a esa receta (D5): la orientación canónica y el aumento geométrico. Cada combinación se entrenó por separado (validación, semilla 42):'),
  table([1500, 1500, 1700, 1300, 1200, 1160, 1000], ['Modelo', 'Orientación', 'Aumento geom.', 'Dice frag.', 'Secundario', 'Sec. perdidos', 'mAP@0.5'], [
    ['D5', 'no', 'no', '0.851', '0.703', '9.0 %', '0.959'],
    ['H1', 'no', 'sí, ±12°', '0.864', '0.743', '7.5 %', '0.969'],
    ['H4', 'sí', 'no', '0.861', '0.742', '8.8 %', '0.971'],
    ['H5', 'sí', 'sí, ±12°', '0.873', '0.769', '6.2 %', '0.976'],
    ['H6', 'sí', 'sí, ±35°', '0.869', '0.764', '6.9 %', '0.971'],
  ], { boldFirst: true, highlightRow: 4 }),
  sp(),
  P(['Los dos cambios aportan por separado (+0.010 la orientación y +0.014 el aumento) y casi se suman (+0.023 juntos), lo que indica que corrigen problemas distintos. La ganancia está en los fragmentos secundarios (0.703 → 0.769), justo donde estaba el error. ',
    'H5 y H6 empatan en validación (medias de 0.871 y 0.870 en sus semillas, dentro del ruido de ±0.004). Se eligió H6 por su robustez a pacientes girados (sección 8).']),

  // ---------- 7 ----------
  H1('7. Investigación del Dice por fragmento'),
  P('El Dice por fragmento fue la única métrica difícil. La investigación siguió una disciplina fija: una diferencia por experimento, tres semillas para medir el ruido (±0.004) y decisiones tomadas en validación. La Figura 6 resume el camino.'),
  image('fig_progresion.png', 620, 267),
  caption('Figura 6. Dice por fragmento en validación y test a lo largo de la investigación.'),
  P(['Más de 25 alternativas no superaron el ruido: 48 canales, contexto 2.5D, pérdidas focal y Tversky, sobremuestreo, ensambles, post-proceso 3D, watershed, recorte al cuerpo y peso en la frontera de fractura. Las que sí ayudaron fueron el ancho de 32 canales y EMA, pero el modelo seguía en ',
    B('~0.847 en test'), '.']),
  P(['El análisis por paciente mostró que dos casos de test (012 y 039) concentraban la pérdida: sin ellos el promedio era 0.869, y en el 012 el 30 % del coxal izquierdo salía como derecho. Al investigar la causa encontramos que ',
    B('un tercio de los volúmenes del dataset estaba almacenado girado 180°'), '. La corrección es general (se lee de la cabecera y se aplica igual a todos los casos) y, junto con el aumento geométrico, mejoró la validación en las tres semillas (+0.016), no solo test.']),
  P(['Como control, un ajuste fino de los modelos previos con rotaciones de ±10° solo subió la validación rotada de 0.839 a 0.847, sin cambiar la normal (0.862): más aumento de datos no bastaba, había que corregir la orientación de los datos.']),

  // ---------- 8 ----------
  H1('8. Análisis de error'),
  H2('Robustez a la rotación'),
  P('Antes de H6 evaluamos el modelo sobre la validación rotada en el plano. El Dice caía ~0.023 a ±10°, pero la confusión izquierda/derecha seguía por debajo del 0.3 % (Figura 7). Así descartamos que el fallo del caso 012 se debiera a giros leves y apuntamos a la orientación del volumen.'),
  image('fig_robustez.png', 620, 223),
  caption('Figura 7. Sensibilidad del modelo previo (EMA) a rotaciones de la validación.'),
  P('Con la validación girada 30° y 60°, el aumento geométrico de ±35° marca la diferencia entre H5 y H6 (Dice por fragmento):'),
  table([3000, 2100, 2100, 2160], ['Validación girada', 'D5', 'H5 (±12°)', 'H6 (±35°)'], [
    ['30°', '0.559', '0.807', '0.854'],
    ['60°', '—', '0.218', '0.734'],
  ], { boldFirst: true }),
  sp(),
  P('Con el mismo rendimiento en la validación normal, H6 resiste mucho mejor a pacientes girados en el escáner, que existen en el dataset (el caso 012 está girado unos 26°).'),
  H2('Matriz de confusión de la clasificación'),
  P([PEND('[pendiente: matriz de confusión de la cabeza de clasificación sobre test con H6]')]),
  H2('Los cinco peores casos'),
  P([PEND('[pendiente: galería de los 5 peores casos de test con H6, con un corte por caso y su hipótesis]'), ' Con el modelo previo, los peores eran 012 (paciente girado, hueso poco denso, pelvis pequeña) y 039 (154 × 322 px con el espaciado más grueso del dataset).']),
  image('nb_caso012.png', 600, 210),
  caption('Figura 8. Caso 012 (test), corte 157, con el modelo previo a la orientación canónica (D5, notebook 05): parte del coxal izquierdo recibe etiquetas del derecho (IDs 22–23). Dice del corte: 0.57.'),
  P(['El Dice corte a corte muestra que los errores no son puntuales: en 012 y 039 hay tramos largos del hueso por debajo de 0.6 (Figura 9). Al girar la imagen del caso 012 antes de pasarla al modelo, el Dice pasa de 0.69 (sin giro) a 0.83 (−45°) y cae hasta 0.03 con otros giros (Figura 10): el modelo previo dependía de la orientación con que llegaba el volumen. Ese hallazgo llevó a la orientación canónica de H6.']),
  image('nb_dice_cortes.png', 600, 192),
  caption('Figura 9. Dice por corte a lo largo del hueso en los cuatro casos más difíciles (012 y 039 de test; 087 y 078 de validación), con el modelo previo (D5, notebook 05). Línea punteada: objetivo 0.85.'),
  image('nb_giro012.png', 440, 197),
  caption('Figura 10. Caso 012: Dice de instancias según el giro aplicado a la imagen antes del modelo (modelo previo, D5).'),
  H2('Distancia de separación: predicción frente a ground truth'),
  P([PEND('[pendiente: distancia mínima borde a borde, en mm, de cada fragmento al principal, con distance_transform_edt y el spacing del header, en predicción y GT]')]),
  H2('Comparación con SAM'),
  P([PEND('[pendiente: SAM ViT-B zero-shot con la caja predicha como prompt, sobre test]')]),

  // ---------- 9 ----------
  H1('9. Limitaciones'),
  bullet([B('Fragmentos en contacto. '), 'Cuando dos fragmentos se tocan en la superficie de fractura, el esquema primario/secundario puede fusionarlos por error de segmentación; en esos casos la distancia medida no es confiable.']),
  bullet([B('Modelo 2D. '), 'Cada corte se procesa por separado; los mejores resultados del challenge vienen de redes 3D, fuera del alcance del curso.']),
  bullet([B('Datos. '), '100 casos de un solo dataset; validación sin casos tan atípicos como los peores de test.']),
  bullet([B('Orientación en la demo. '), 'Los volúmenes LPI se identifican con una lista de los 34 casos del dataset, derivada de sus cabeceras, y la demo todavía no aplica el giro. Para un paciente nuevo, la orientación debe leerse de la cabecera de cada archivo.']),
  bullet([B('Fragmentos secundarios y sobrantes. '), 'El secundario sigue siendo el punto débil (Dice 0.764 frente a 0.905 del primario, y el 6.9 % no se detecta). Además, el Dice por fragmento no penaliza los fragmentos predichos que sobran.']),
  bullet([B('Segunda evaluación en test. '), 'La cifra de H6 se obtuvo después de analizar los fallos de la primera evaluación; no es una estimación limpia (ver Transparencia).']),
  bullet([B('Uso no clínico. '), 'No validado clínicamente; solo para fines académicos.']),

  // ---------- 10 ----------
  H1('10. Conclusiones'),
  P(['Un modelo 2D propio, liviano (2.74 M de parámetros) y sin frameworks cumple las nueve métricas del enunciado en test. La mejora decisiva no vino de una red más grande, sino de ',
    B('diagnosticar con datos'), ' por qué fallaban dos pacientes y encontrar un error de orientación en el dataset. Cada decisión está respaldada por evidencia medida y por un protocolo que separa validación de test.']),

  // ---------- Transparencia ----------
  H1('Transparencia: consultas al conjunto de test'),
  P('Test se consultó más de una vez durante el proyecto. Las reportamos todas:'),
  table([3800, 2200, 3360], ['Consulta', 'Dice por fragmento', 'Motivo'], [
    ['C2, semilla 44', '0.848', 'Primera evaluación del modelo base'],
    ['Receta EMA + post-proceso (5 modelos)', '0.847 ± 0.003', 'Mejora consistente en validación; declarada'],
    ['D5 (EMA), semilla 42', '0.845', 'Cifra oficial de la Semana 10: configuración congelada antes de ver test'],
    ['H6 (3 semillas)', '0.875 ± 0.001', 'Segunda evaluación, posterior al análisis de fallos'],
  ], { boldFirst: true, highlightRow: 3 }),
  sp(),
  P(['La cifra de H6 se reporta como ', B('segunda evaluación, posterior al análisis de fallos'), ', sin reemplazar la de D5. H6 se diseñó después de estudiar por qué fallaban los casos 012 y 039 de test, pero se eligió solo con validación y pruebas de estrés. Su configuración se congeló por escrito antes de correrla, se evaluó una sola vez sin ajustar nada después, y se reportan también las semillas 43 y 44 y el ensamble, sin elegir entre ellos (docs/decisiones_semana10.md, secciones 12.3 a 12.5).']),

  // ---------- Reproducibilidad ----------
  H1('Reproducibilidad'),
  table([2400, 6960], ['Elemento', 'Dónde'], [
    ['Ramas por semana', 'continuacion_semana08 → unificado_semana09 → semana10-multiples-pruebas-working → semana11-visualizadores (cada una contiene a la anterior)'],
    ['Notebooks', '01 y 02 (semana 8), 03 (semana 9), 04 (semana 10), 05 (casos difíciles, con D5)'],
    ['Pesos', 'h6_auggeo_rot35_canon_ema_w32_e60_s42.pth (entregado); s43 y s44 para medir la variabilidad'],
    ['Comandos', 'README.md (caché, entrenamiento, evaluación, SAM, latencia y demo)'],
    ['Modelo', 'MODEL_CARD.md'],
    ['Resultados', 'outputs/s10/*_val.json y TEST2_*.json'],
    ['Uso de IA', 'IA_USAGE.md'],
  ], { boldFirst: true }),


  new Paragraph({ children: [new PageBreak()] }),
  H1('Anexo. Trabajo por semana'),
  table([1300, 4760, 3300], ['Semana', 'Qué se hizo', 'Resultado clave'], [
    ['8 · Datos', 'Curaduría de los 100 volúmenes (.mha con SimpleITK) y manifiesto; partición por paciente estratificada (70/15/15, semilla 42); preprocesamiento unificado (recorte de HU y ventana ósea); EDA de intensidades; Visualizador 1 (reconstrucción 3D del volumen crudo).',
      ['Mediana de HU del hueso entre 81 y 431 (CV 26.5 %): variación biológica, no de escáner. Metal en 41 casos, fuera de la pelvis. Máscara del visualizador: recall de 0.57 a 0.92. ', PEND('(confirmar el conteo de metal: el notebook 02 cuenta 25 casos)')]],
    ['9 · Detector', 'Backbone propio con CBAM, cabeza de detección por grid y NMS propios; overfit intencional; primer entrenamiento con varios pacientes; pedidos de la asesoría: γ aprendible por etapa, Grad-CAM, fragmentos desplazados y justificación del preprocesamiento.',
      'Overfit: pérdida de 1.687 a 0.002 (758×). F1 de 0.31 a 0.57 (32 canales y umbral 0.40). Gamma 1.5 elegido por contraste-ruido (8.1), Otsu (0.486) y detector (F1 0.558 vs 0.515, 3 semillas).'],
    ['10 · Multitarea', 'Modelo con tres cabezas y salida de instancia primario/secundario; evaluación sobre el volumen completo; barrido de ancho, EMA, pérdidas y post-procesos (más de 25 experimentos); ablaciones de CBAM, transfer learning y λ.',
      'Dice por fragmento de 0.787 (31 IDs) a 0.859 en validación. Test: 0.848, con 8 de 9 métricas cumplidas. Diagnóstico: secundarios en contacto y dos casos atípicos (012 y 039).'],
    ['11 · Modelo final y demo', 'Orientación canónica desde la cabecera y aumento geométrico (modelo H6, 3 semillas); visualizadores 2 y 3 y dashboard; despliegue en curso.',
      'Validación 0.870 y test 0.875 ± 0.001: 9 de 9 métricas cumplidas. Pendiente: demo con H6, túnel de Cloudflare y latencia.'],
  ], { boldFirst: true, leftAll: true }),
  image('nb_100casos.png', 470, 474),
  caption('Figura A1. Revisión visual de los 100 casos tras el preprocesamiento (contorno rojo: hueso; b: brillo medio del hueso).'),
  image('nb_ventaneo_gamma.png', 470, 220),
  caption('Figura A2. F1 del detector con ventaneo y con gamma 1.5 (puntos: 3 semillas; raya: media).'),
  image('nb_detector15.png', 600, 357),
  caption('Figura A3. Detector de la semana 9 (con CBAM) en 15 casos de validación: cajas reales (continuas) y predichas (discontinuas).'),
  image('nb_gradcam014.png', 560, 433),
  caption('Figura A4. Smooth HiResCAM en el caso 014 con CBAM, sin CBAM y con CBAM en 3 etapas: dónde mira el modelo para cada región.'),
  H1('Referencias'),
  bullet('Sang, Y. et al. Benchmark of Segmentation Techniques for Pelvic Fracture in CT and X-Ray: Summary of the PENGWIN 2024 Challenge. IEEE TMI, 2026. arXiv:2504.02382.'),
  bullet('PENGWIN Task 1: Pelvic Fracture Segmentation on CT. Zenodo, doi:10.5281/zenodo.10927452.'),
  bullet('Woo, S. et al. CBAM: Convolutional Block Attention Module. ECCV, 2018.'),
  bullet('Kirillov, A. et al. Segment Anything. ICCV, 2023.'),
];

const doc = new Document({
  creator: 'Equipo PENGWIN', title: 'Informe técnico PENGWIN',
  styles: {
    default: { document: { run: { font: 'Calibri', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Calibri', color: NAVY }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 25, bold: true, font: 'Calibri', color: TEAL }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'vi', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, right: 1440, bottom: 1300, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: 'PENGWIN · Proyecto Integrador Corte 2 · ', color: GRAY, size: 16 }),
        new TextRun({ children: [PageNumber.CURRENT], color: GRAY, size: 16 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync('./Informe_tecnico_PENGWIN.docx', b); console.log('ok'); });

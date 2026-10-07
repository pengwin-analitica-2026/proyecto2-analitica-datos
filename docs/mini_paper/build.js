// Informe técnico corto con estructura de mini-paper (entregable del enunciado). Genera Mini_paper_PENGWIN.docx.
// Uso, desde docs/mini_paper:  node build.js   (necesita el paquete npm "docx")
// Las cifras salen de outputs/s10/*.json y outputs/demo/latency.json.
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, Footer, PageNumber, LevelFormat } = require('docx');

const FUENTE = 'Times New Roman';
const W = 9360;                       // ancho útil en DXA (Letter, márgenes de 1")
const aqui = (f) => fs.readFileSync(path.join(__dirname, f));

const run = (t, o = {}) => new TextRun({ text: t, font: FUENTE, size: 21, ...o });
const B = (t) => run(t, { bold: true });
const I = (t) => run(t, { italics: true });
const partes = (x) => (Array.isArray(x) ? x : [x]).map((t) => (typeof t === 'string' ? run(t) : t));
const P = (x, o = {}) => new Paragraph({ alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: 95, line: 250 }, indent: o.indent, children: partes(x) });
const H = (t) => new Paragraph({ spacing: { before: 200, after: 90 }, keepNext: true,
  children: [run(t, { bold: true, size: 23 })] });
const H2 = (t) => new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
  children: [run(t, { bold: true, italics: true })] });
const pie = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 170 },
  children: [run(t, { size: 18 })] });
const cabezaTabla = (t) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 120, after: 60 },
  keepNext: true, children: [run(t, { size: 18 })] });
const figura = (f, ancho, alto) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
  spacing: { before: 120, after: 0 },
  children: [new ImageRun({ type: 'png', data: aqui(f), transformation: { width: ancho, height: alto },
    altText: { title: f, description: f, name: f } })] });
const ref = (n, t) => new Paragraph({ spacing: { after: 50, line: 240 }, indent: { left: 400, hanging: 400 },
  children: [run(`[${n}]  `, { size: 19 }), ...(Array.isArray(t) ? t : [run(t, { size: 19 })])] });
const item = (x) => new Paragraph({ numbering: { reference: 'lista', level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 40, line: 250 }, children: partes(x) });

// Tabla con reglas horizontales (arriba, bajo la cabecera y abajo), sin líneas verticales.
const NADA = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const FINA = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const GRUESA = { style: BorderStyle.SINGLE, size: 10, color: '000000' };
function tabla(anchos, cabecera, filas, o = {}) {
  const total = anchos.reduce((a, b) => a + b, 0);
  const celda = (texto, i, tipo, negrita) => new TableCell({
    width: { size: anchos[i], type: WidthType.DXA },
    margins: { top: 35, bottom: 35, left: 70, right: 70 },
    borders: { left: NADA, right: NADA,
      top: tipo === 'cabeza' ? GRUESA : NADA,
      bottom: tipo === 'cabeza' ? FINA : (tipo === 'ultima' ? GRUESA : NADA) },
    children: [new Paragraph({ alignment: (i === 0 || (o.izquierda || []).includes(i)) ? AlignmentType.LEFT : AlignmentType.CENTER,
      keepNext: !o.partir && tipo !== 'ultima',     // la tabla no se parte entre páginas, salvo las largas
      children: [run(texto, { size: 18, bold: tipo === 'cabeza' || negrita })] })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: anchos, alignment: AlignmentType.CENTER,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: cabecera.map((c, i) => celda(c, i, 'cabeza')) }),
      ...filas.map((f, n) => new TableRow({ cantSplit: true,
        children: f.map((c, i) => celda(c, i, n === filas.length - 1 ? 'ultima' : 'fila', o.negrita === n)) }))] });
}
// Dos figuras lado a lado, cada una con su pie (tabla sin bordes de dos columnas).
function dosFiguras(izquierda, derecha) {
  const celda = ([f, ancho, alto, texto]) => new TableCell({ width: { size: W / 2, type: WidthType.DXA },
    borders: { top: NADA, bottom: NADA, left: NADA, right: NADA }, margins: { left: 60, right: 60 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: aqui(f),
      transformation: { width: ancho, height: alto }, altText: { title: f, description: texto, name: f } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 120 }, children: [run(texto, { size: 18 })] })] });
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W / 2, W / 2],
    rows: [new TableRow({ cantSplit: true, children: [celda(izquierda), celda(derecha)] })] });
}
// Galería: una fila de dos paneles (el segundo puede faltar).
function filaGaleria(izquierda, derecha) {
  const celda = (panel) => new TableCell({ width: { size: W / 2, type: WidthType.DXA },
    borders: { top: NADA, bottom: NADA, left: NADA, right: NADA }, margins: { left: 60, right: 60 },
    children: panel ? [new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: aqui(panel[0]),
        transformation: { width: 296, height: 201 }, altText: { title: panel[0], description: panel[1], name: panel[0] } })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 30, after: 90 }, children: [run(panel[1], { size: 17 })] })]
      : [new Paragraph({ children: [] })] });
  return new TableRow({ cantSplit: true, children: [celda(izquierda), celda(derecha)] });
}
const galeria = (paneles) => new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W / 2, W / 2],
  rows: [0, 2, 4].filter((i) => i < paneles.length).map((i) => filaGaleria(paneles[i], paneles[i + 1])) });
const aire = () => new Paragraph({ spacing: { after: 90 }, children: [] });

const contenido = [
  // ---------- Título ----------
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [run('INFORME TÉCNICO', { size: 19, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [run('Detección, segmentación por fragmento y medición de fracturas pélvicas en CT con una red multitarea 2D propia', { bold: true, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 50 },
    children: [run('Gustavo Adolfo Chipantiza, Federico Gonzales Cárdenas, Juan Jacobo Delgado Melo y María de los Ángeles Valencia Hurtado', { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 50 },
    children: [run('Analítica de Datos, Universidad Autónoma de Occidente, 2026-2. Profesor: Carlos Ferro', { size: 19, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
    children: [run('Proyecto integrador, corte 2 · octubre de 2026 · github.com/pengwin-analitica-2026/proyecto2-analitica-datos', { size: 19 })] }),

  // ---------- Resumen ----------
  P([B('Resumen. '),
    'Presentamos un sistema que, en cada corte axial de una tomografía de pelvis, indica qué regiones óseas aparecen (sacro, coxal izquierdo y coxal derecho), las localiza con cajas, segmenta sus fragmentos y mide en milímetros cuánto se separó cada fragmento del fragmento principal de su hueso. Es una sola red 2D de 2.74 millones de parámetros, con backbone convolucional propio, atención CBAM y tres cabezas entrenadas desde cero, sin frameworks de detección ni de segmentación. Se entrenó y se evaluó con los 100 volúmenes públicos de PENGWIN, divididos por paciente en 70, 15 y 15. Al analizar los fallos encontramos que 34 de los 100 volúmenes están guardados girados 180° en el plano. Enderezarlos a partir de la cabecera del archivo y entrenar con rotaciones de ±35° subió el Dice por fragmento en validación de 0.851 a 0.869. En test, el modelo final cumple los nueve objetivos del curso, con un Dice por fragmento de 0.875 (0.875, 0.875 y 0.876 con tres semillas). Esa cifra es una segunda evaluación en test, hecha después de analizar los fallos; la primera, con la configuración congelada antes de ver test, dio 0.845. La distancia de separación tiene un error medio de 1.28 mm. La parte más débil son los fragmentos secundarios, con Dice de 0.750.'],
    { indent: { left: 500, right: 500 } }),
  P([B('Palabras clave: '), 'fractura pélvica, tomografía computarizada, segmentación de instancias, aprendizaje multitarea, CBAM.'],
    { indent: { left: 500, right: 500 }, align: AlignmentType.LEFT }),

  // ---------- 1 ----------
  H('1. Introducción'),
  P('Para planear la reducción quirúrgica de una fractura de pelvis hay que saber cuántos fragmentos hay, a qué hueso pertenece cada uno y cuánto se desplazó. Hacerlo a mano sobre una tomografía es lento. El reto PENGWIN 2024 [1] publicó tomografías con cada fragmento anotado y validado por expertos [2, 3]; el mejor método del reto alcanzó un IoU por fragmento de 0.930 en CT [1].'),
  P('Este trabajo aborda el mismo problema con las restricciones del curso: arquitectura 2D corte a corte, backbone propio con CBAM [4] antes de bifurcar hacia las cabezas, tres cabezas entrenadas desde cero, detección con grid [5] y supresión de no máximos propios, y ningún framework de detección o segmentación de alto nivel. SAM [6] solo se usa como comparación. Nuestras cifras no son comparables con las del reto: usamos una partición propia de los 100 casos públicos y medimos corte a corte.'),
  P('Los aportes son tres. Primero, una red ligera que resuelve clasificación, detección, segmentación por fragmento y medición en un solo paso. Segundo, el hallazgo de que un tercio del dataset está guardado en otra orientación, y la medida de cuánto afecta al modelo. Tercero, un protocolo de experimentos con el ruido medido entre semillas y con todas las consultas a test declaradas.'),

  // ---------- 2 ----------
  H('2. Datos'),
  P('Usamos los 100 volúmenes de CT con máscara de PENGWIN Task 1 [3], en formato .mha y con licencia CC BY 4.0. Cada vóxel de la máscara lleva el número de su fragmento: de 1 a 10 para el sacro, de 11 a 20 para el coxal izquierdo y de 21 a 30 para el coxal derecho. El primer número de cada región (1, 11 y 21) es el fragmento principal y los demás son fragmentos secundarios. Hay entre 3 y 9 fragmentos por caso (media 5.75). El espaciado va de 0.66 a 1.22 mm en el plano y de 0.625 a 1.25 mm entre cortes. La partición es 70 / 15 / 15 por paciente, con semilla 42, y no se generaron etiquetas nuevas.'),
  P('Cada corte se recorta a [−1024, 3000] HU, pasa por una ventana ósea (centro 400, ancho 1800) y una corrección gamma de 1.5, y se lleva a 384 × 384 píxeles.'),
  P([B('Orientación. '), 'La cabecera de 34 de los 100 archivos indica que el volumen está guardado girado 180° en el plano axial (22 en entrenamiento, 10 en validación y 2 en test). En esos casos el sacro queda arriba y los coxales intercambian su lado en la imagen. El paso de orientación canónica lee la cabecera y endereza el volumen, igual en la imagen y en la máscara, en todos los conjuntos y sin usar las etiquetas.']),

  // ---------- 3 ----------
  H('3. Método'),
  H2('3.1 Arquitectura'),
  P('Un backbone convolucional propio, con 32 canales base y un módulo CBAM en una etapa, alimenta tres cabezas en un solo paso hacia adelante (Figura 1). La cabeza de clasificación dice cuáles de las tres regiones aparecen en el corte. La de detección predice, en un grid de 24 × 24 celdas, la caja de la región cuyo centro cae en cada celda, y una supresión de no máximos propia deja una caja por región. La de segmentación tiene dos salidas: una semántica (fondo y las tres regiones) y una de instancia que, dentro de cada región, separa el fragmento principal de los secundarios. Los fragmentos secundarios individuales salen de las componentes conexas de esa salida, con un tamaño mínimo de 12 píxeles.'),
  figura('../informe/fig/fig_arquitectura.png', 440, 185),
  pie('Figura 1. Arquitectura del modelo: un backbone compartido y tres cabezas entrenadas desde cero.'),
  P('Para la cabeza de detección se consultó el principio del grid de YOLO [5]: cada celda predice la caja del objeto cuyo centro cae en ella. La cabeza, la asignación de celdas y la supresión de no máximos están escritas por el equipo, sin código ni pesos de YOLO, Detectron2 o Mask R-CNN. Cada decisión de diseño se tomó con una medida en validación (Tabla 1).'),
  cabezaTabla('Tabla 1. Decisiones de arquitectura, su motivo y la evidencia en validación (Dice por fragmento salvo que se diga otra cosa).'),
  tabla([2300, 3000, 4060], ['Decisión', 'Motivo', 'Evidencia'], [
    ['32 canales base', 'El ancho del backbone era la limitación principal', 'Con 8, 16 y 32 canales: 0.829, 0.839 y 0.850. Con 48: 0.844'],
    ['CBAM en una etapa, antes de las cabezas', 'Lo pide el curso; se probó en una y en tres etapas', 'mAP50 de 0.926 con una etapa, 0.907 con tres y 0.900 sin CBAM (40 épocas)'],
    ['Instancia como principal o secundario por región', 'Lógica de dos etapas: primero la región y luego sus fragmentos. Es una tarea más acotada que predecir 31 identificadores', 'Con 31 identificadores: 0.787, y secundarios en 0.460. Con principal y secundario: 0.829 y 0.645'],
    ['384 px por corte (el curso recomienda 224 a 256)', 'Hay fragmentos pequeños que a 256 px ocupan muy pocos píxeles', 'No se midió por separado en la serie final. Un lote de 16 con precisión mixta cabe en una GPU T4'],
    ['Promedio exponencial de los pesos (EMA)', 'Estabiliza el final del entrenamiento', 'Secundarios +0.016, +0.048 y +0.009 en tres semillas'],
    ['Sin preentrenamiento', 'El preentrenamiento del backbone no dio ganancia consistente', 'Sección 5.3'],
  ], { izquierda: [1, 2], partir: true }),
  aire(),
  H2('3.2 Pérdida y entrenamiento'),
  P('La pérdida es una suma ponderada de los términos de las tres cabezas. Los pesos no se copiaron de otra implementación: cada uno responde a algo medido en los datos o en el entrenamiento (Tabla 2).'),
  cabezaTabla('Tabla 2. Pesos de la pérdida y su justificación.'),
  tabla([2500, 900, 5960], ['Término', 'Peso', 'Por qué'], [
    ['Detección', '1', 'Peso de referencia'],
    ['Clasificación', '0.5', 'Es la tarea que converge primero: en la primera época el F1 ya es 0.907, cuando el mAP50 va en 0.287 y el Dice por fragmento en 0.529. Con más peso competiría con tareas que aún no aprenden'],
    ['Semántica (entropía cruzada y Dice)', '2 y 1', 'Es la primera de las dos etapas: un error de región arrastra a todos sus fragmentos'],
    ['Instancia (entropía cruzada y Dice)', '1 y 1', 'En la entropía cruzada los secundarios pesan el triple: son cerca de la mitad de los fragmentos y cerca de una décima parte del hueso. Con peso ×6 no superó el ruido (0.832 frente a 0.829) y con sobremuestreo ×3 empeoró (0.837 frente a 0.850)'],
    ['Borde', '1', 'Término auxiliar para el contorno de los fragmentos'],
    ['Instancia y borde ×2, detección ×0.5', '—', 'Alternativa probada: 0.855, dentro del ruido, y el mAP50 de la cabeza de grid baja de 0.868 a 0.827. Se dejan los pesos de arriba'],
  ], { izquierda: [2], partir: true }),
  aire(),
  P('El modelo se entrena desde cero con AdamW (tasa 2 × 10⁻³ con decaimiento coseno, lote de 16, precisión mixta), 60 épocas de 2 500 cortes y un promedio exponencial de los pesos (EMA, 0.999). El aumento de datos combina espejo sagital con intercambio de las etiquetas de los coxales, cambios de intensidad y, en el modelo final, rotación de hasta ±35°, escala, proporción y desplazamiento.'),
  H2('3.3 Volumen y distancia'),
  P('Las predicciones de todos los cortes se apilan y un post-proceso 3D por región une cada fragmento a lo largo del volumen. La separación de un fragmento secundario es la distancia mínima borde a borde hasta el fragmento principal de su hueso, calculada con la transformada de distancia euclídea y el espaciado en milímetros de la cabecera. En inferencia se promedia la predicción del corte con la de su espejo (TTA).'),

  // ---------- 4 ----------
  H('4. Protocolo de evaluación'),
  P('Todas las métricas se calculan sobre todos los cortes de cada volumen: 5 047 en validación y 4 664 en test. El Dice por fragmento compara cada fragmento real de un corte con el fragmento predicho que más se le superpone, y es la métrica más exigente de las nueve.'),
  P('Cada experimento cambió una sola cosa. El ruido entre corridas se midió con tres semillas (±0.004 en el Dice por fragmento y ±0.012 en el de los secundarios), y un cambio solo se aceptó si mejoraba la validación por encima de ese ruido. Los modelos se eligieron únicamente con validación y con pruebas de estrés, que repiten la validación con los volúmenes girados.'),
  P('Test se evaluó dos veces con modelos distintos. La primera, con el modelo D5 y la configuración congelada antes de ver test, es la cifra limpia. La segunda, con el modelo final H6, se hizo después de estudiar por qué fallaban dos casos de test (012 y 039). H6 se eligió solo con validación y su configuración se fijó por escrito antes de evaluarlo, pero la decisión de investigar la orientación nació de mirar test, y por eso reportamos las dos cifras. Las demás consultas a test (otras semillas y variantes que no se adoptaron) están en la bitácora del repositorio.'),

  // ---------- 5 ----------
  H('5. Resultados'),
  H2('5.1 Métricas frente a los objetivos'),
  P('El modelo final cumple los nueve objetivos en test (Tabla 3). La primera evaluación cumplía ocho: el Dice por fragmento quedaba en 0.845. Las tres semillas de H6 dan 0.875, 0.875 y 0.876; promediar los tres modelos da 0.879, pero triplica el costo y no se usa. La detección se reporta con la caja refinada con la máscara semántica; la cabeza de grid sola obtiene en test un IoU medio de 0.783, mAP50 de 0.912 y mAP50:95 de 0.556, que también cumplen.'),
  cabezaTabla('Tabla 3. Las nueve métricas del curso. Test: 15 casos; un modelo con TTA.'),
  tabla([3300, 1100, 1650, 1650, 1660],
    ['Métrica', 'Objetivo', 'Test, 1.ª eval. (D5)', 'Test, 2.ª eval. (H6)', 'Validación (H6)'], [
      ['F1 de clasificación', '0.85', '0.984', '0.990', '0.992'],
      ['AUC de clasificación', '0.85', '0.998', '0.999', '0.999'],
      ['IoU medio de detección', '0.65', '0.898', '0.943', '0.949'],
      ['mAP50 de detección', '0.65', '0.920', '0.961', '0.971'],
      ['mAP50:95 de detección', '0.40', '0.853', '0.902', '0.904'],
      ['Dice de segmentación semántica', '0.85', '0.949', '0.971', '0.971'],
      ['IoU de segmentación semántica', '0.70', '0.904', '0.943', '0.944'],
      ['Dice por fragmento', '0.85', '0.845', '0.875', '0.869'],
      ['IoU por fragmento', '0.70', '0.781', '0.816', '0.808'],
    ], { negrita: 7 }),
  aire(),
  P('La mejora no viene de un solo paciente: H6 sube en 13 de los 15 casos de test (Figura 2). Los dos que más cambian son los que motivaron el análisis, 012 (de 0.692 a 0.865) y 039 (de 0.710 a 0.784).'),

  H2('5.2 Qué aporta cada cambio'),
  P('H6 añade dos cosas a D5: la orientación canónica y el aumento geométrico. Cada combinación se entrenó por separado (Tabla 4). Cada cambio aporta por su cuenta (+0.010 la orientación y +0.014 el aumento con ±12°) y juntos casi se suman (+0.023). La ganancia está en los fragmentos secundarios, que pasan de 0.703 a 0.764.'),
  P('H5 y H6 empatan en la validación normal, dentro del ruido. Los separa la prueba de estrés (Figura 3): con la validación girada 30°, H6 conserva 0.854 y D5 cae a 0.559; con 60°, H6 baja a 0.734 y los demás quedan por debajo de 0.22. Elegimos H6 porque en el dataset hay pacientes girados dentro del escáner, como el caso 012.'),
  cabezaTabla('Tabla 4. Componentes de H6. Dice por fragmento en validación, semilla 42; las dos últimas columnas repiten la validación con los volúmenes girados.'),
  tabla([900, 1350, 1350, 1250, 1250, 1260, 1000, 1000],
    ['Modelo', 'Orientación canónica', 'Rotación al entrenar', 'Dice por fragmento', 'Dice secundarios', 'Secundarios sin pareja', 'Girada 30°', 'Girada 60°'], [
      ['D5', 'no', 'no', '0.851', '0.703', '9.0 %', '0.559', '0.127'],
      ['H1', 'no', '±12°', '0.864', '0.743', '7.5 %', '0.771', '0.212'],
      ['H4', 'sí', 'no', '0.861', '0.742', '8.8 %', '—', '—'],
      ['H5', 'sí', '±12°', '0.873', '0.769', '6.2 %', '0.807', '0.218'],
      ['H6', 'sí', '±35°', '0.869', '0.764', '6.9 %', '0.854', '0.734'],
    ], { negrita: 4 }),
  aire(),
  dosFiguras(['fig/por_caso_test.png', 292, 254, 'Figura 2. Dice por fragmento de cada caso de test en la primera evaluación (D5) y en la segunda (H6).'],
    ['fig/estres_giro.png', 292, 254, 'Figura 3. Dice por fragmento en validación según el giro aplicado a los volúmenes.']),

  H2('5.3 Estudio de ablación'),
  P('Las ablaciones se midieron en validación sobre la receta de D5, que comparte arquitectura y pérdida con H6, cambiando una sola cosa (Tabla 5). CBAM mejora la clasificación y la detección y no cambia la segmentación. El backbone preentrenado en Intel Image Classification no da una ganancia consistente: los secundarios suben 0.011 con una semilla y bajan 0.008 con la otra, así que el modelo final no usa preentrenamiento.'),
  cabezaTabla('Tabla 5. Ablaciones con y sin CBAM y con y sin transfer learning en el backbone (validación).'),
  tabla([2700, 3330, 3330], ['Ablación', 'Con', 'Sin'], [
    ['CBAM (semilla 42)', 'F1 0.988 · mAP50:95 0.883 · Dice por fragmento 0.851', 'F1 0.975 · mAP50:95 0.872 · Dice por fragmento 0.851'],
    ['Transfer learning en el backbone (semillas 42 y 43)', 'Dice por fragmento 0.851 y 0.856 · secundarios 0.714 y 0.717', 'Dice por fragmento 0.851 y 0.855 · secundarios 0.703 y 0.724'],
  ]),
  aire(),

  H2('5.4 Comparación con SAM'),
  P('SAM ViT-B [6] se ejecutó en modo zero-shot sobre test, con la caja de nuestro detector como indicación, y se comparó con nuestra máscara semántica dentro de la misma caja (1 181 cortes, uno de cada tres; 3 002 pares corte-región). El modelo propio gana en las tres regiones y la mayor diferencia está en el sacro (Tabla 6). SAM, además, no separa ni clasifica fragmentos y no forma parte del sistema.'),
  cabezaTabla('Tabla 6. SAM zero-shot frente al modelo propio en test.'),
  tabla([2760, 1650, 1650, 1650, 1650], ['Región', 'Dice SAM', 'Dice propio', 'IoU SAM', 'IoU propio'], [
    ['Sacro', '0.792', '0.884', '0.690', '0.820'],
    ['Coxal izquierdo', '0.915', '0.947', '0.859', '0.905'],
    ['Coxal derecho', '0.930', '0.957', '0.879', '0.921'],
    ['Las tres regiones', '0.889', '0.935', '0.823', '0.889'],
  ], { negrita: 3 }),
  aire(),

  H2('5.5 Distancia de separación: predicción frente a ground truth'),
  P('La distancia de cada fragmento secundario a su principal se calcula igual sobre la máscara predicha y sobre el ground truth del mismo caso, y se comparan (Tabla 7). En test hay 41 fragmentos secundarios y el modelo entrega distancia para 36. Con H6 el error medio es 1.28 mm y la correlación con la distancia real 0.82. Un solo fragmento del caso 007, que está a 9.4 mm y el modelo sitúa a 45.8 mm, concentra el 79 % del error; sin él la media es 0.28 mm. En validación solo 5 de los 41 secundarios están separados en el ground truth, así que esa columna es indicativa.'),
  cabezaTabla('Tabla 7. Distancia predicha frente a la real y métricas de superficie por fragmento (41 secundarios en cada conjunto).'),
  tabla([3900, 1820, 1820, 1820], ['Medida', 'Validación, H6', 'Test, D5', 'Test, H6'], [
    ['Error medio de la distancia (mm)', '0.185', '2.956', '1.277'],
    ['Error del percentil 90 (mm)', '0.000', '3.453', '0.556'],
    ['Correlación con la distancia real', '0.972', '0.543', '0.818'],
    ['Secundarios con distancia predicha', '38', '36', '36'],
    ['Dice 3D, principal y secundarios', '0.921 y 0.701', '0.909 y 0.682', '0.931 y 0.756'],
    ['ASSD medio (mm)', '4.08', '3.02', '2.06'],
    ['HD95 medio (mm)', '17.99', '16.33', '11.55'],
  ]),
  aire(),

  H2('5.6 Latencia por imagen'),
  P('La Tabla 8 da el tiempo de la red por corte de 384 × 384, con lote de 1. La configuración entregada es un modelo con TTA.'),
  cabezaTabla('Tabla 8. Latencia por corte (mediana, en ms).'),
  tabla([3300, 2020, 2020, 2020], ['Configuración', 'GPU L4', 'CPU, 12 hilos', 'CPU portátil, 8 hilos'], [
    ['Un modelo', '63.7', '161.7', '336.1'],
    ['Un modelo con TTA (entregado)', '71.6', '249.6', '695.5'],
    ['Ensamble de tres semillas con TTA', '87.4', '518.0', '—'],
  ]),
  aire(),

  // ---------- 6 ----------
  H('6. Análisis de error y limitaciones'),
  H2('6.1 Matriz de confusión de la clasificación'),
  P('La cabeza de clasificación decide en cada corte si cada región está presente, así que hay una matriz de dos por dos por región (Tabla 9, sobre los 4 664 cortes de test). El error más frecuente es no detectar el sacro: pasa en 56 de los 2 294 cortes en que está.'),
  cabezaTabla('Tabla 9. Matriz de confusión de la clasificación por región en test (H6, semilla 42).'),
  tabla([1700, 1200, 1200, 1300, 1250, 900, 900, 910],
    ['Región', 'Presente y predicha', 'Ausente y predicha', 'Presente y no predicha', 'Ausente y no predicha', 'Precisión', 'Recall', 'F1'], [
      ['Sacro', '2 238', '37', '56', '2 333', '0.984', '0.976', '0.980'],
      ['Coxal izquierdo', '3 372', '11', '12', '1 269', '0.997', '0.996', '0.997'],
      ['Coxal derecho', '3 316', '28', '4', '1 316', '0.992', '0.999', '0.995'],
    ]),
  aire(),
  P('A nivel de píxel (Tabla 10), casi todo el error de la máscara semántica es hueso que se predice como fondo, sobre todo en el sacro. La confusión entre coxal izquierdo y derecho, que con D5 llegaba al 13.7 % del hueso en el caso 012, queda por debajo del 0.1 %.'),
  cabezaTabla('Tabla 10. Matriz de confusión de la segmentación semántica en test, en porcentaje de cada clase real (filas).'),
  tabla([2760, 1650, 1650, 1650, 1650], ['Real \\ predicho', 'Fondo', 'Sacro', 'Coxal izquierdo', 'Coxal derecho'], [
    ['Fondo', '99.93', '0.02', '0.02', '0.02'],
    ['Sacro', '3.28', '96.65', '0.04', '0.04'],
    ['Coxal izquierdo', '1.52', '0.01', '98.47', '0.00'],
    ['Coxal derecho', '1.45', '0.02', '0.06', '98.47'],
  ]),
  aire(),

  H2('6.2 Los cinco peores casos'),
  P('La Tabla 11 y la Figura 4 muestran los cinco casos de test con menor Dice por fragmento. En los cinco el fragmento principal se segmenta bien (entre 0.87 y 0.91) y el error está en los secundarios: lo que falta resolver ya no es a qué hueso pertenece cada píxel, sino dónde se parte el hueso en fragmentos.'),
  cabezaTabla('Tabla 11. Los cinco peores casos de test con H6 y la causa más probable de cada uno.'),
  tabla([650, 750, 900, 1150, 1000, 4910], ['Caso', 'Dice', 'Principal', 'Secundarios', 'Sin pareja', 'Hipótesis'], [
    ['039', '0.784', '0.890', '0.635', '7.7 %', 'Es el volumen de menor resolución en el plano (154 × 322 px, 1.22 mm). Al llevarlo a 384 px el sacro sale dividido en más fragmentos de los reales'],
    ['061', '0.822', '0.891', '0.635', '17.4 %', 'El coxal derecho tiene dos fragmentos grandes. El modelo ve un corte a la vez y en parte de los cortes intercambia cuál es el principal'],
    ['007', '0.831', '0.913', '0.545', '24.5 %', 'Fragmentos secundarios pequeños en las ramas púbicas, y un coxal que el modelo parte donde el hueso se adelgaza. Uno de cada cuatro secundarios queda sin pareja'],
    ['054', '0.834', '0.899', '0.663', '16.2 %', 'Fragmentos en contacto: el modelo predice fragmentos de más y confunde principal y secundario donde se tocan'],
    ['047', '0.854', '0.870', '0.753', '17.1 %', 'Hueso fino y poco denso en los extremos del volumen (0.745 en los extremos y 0.888 en el centro). El 5.4 % del hueso se predice como fondo'],
  ], { izquierda: [5], partir: true }),
  aire(),
  galeria([
    ['../../outputs/s10/informe_h6_test/peor_039.png', 'a) Caso 039'],
    ['../../outputs/s10/informe_h6_test/peor_061.png', 'b) Caso 061'],
    ['../../outputs/s10/informe_h6_test/peor_007.png', 'c) Caso 007'],
    ['../../outputs/s10/informe_h6_test/peor_054.png', 'd) Caso 054'],
    ['../../outputs/s10/informe_h6_test/peor_047.png', 'e) Caso 047'],
  ]),
  pie('Figura 4. Galería de los cinco peores casos de test. En cada panel: imagen, fragmentos reales y fragmentos predichos, en el corte del décimo percentil (arriba) y en el corte mediano (abajo). Azul: sacro; naranja: coxal izquierdo; verde: coxal derecho; tono claro: fragmento secundario.'),

  H2('6.3 Limitaciones'),
  item([B('Fragmentos secundarios. '), 'Su Dice es 0.750 en test y el 7.8 % no tiene pareja en la predicción.']),
  item([B('Fragmentos en contacto. '), 'Cuando dos fragmentos se tocan, el modelo puede fusionarlos: el secundario se pierde y su distancia sale 0 mm o no existe.']),
  item([B('Fragmentos de más. '), 'El Dice por fragmento empareja cada fragmento real con uno predicho, pero no penaliza los fragmentos predichos que sobran.']),
  item([B('Modelo 2D. '), 'Procesa cada corte por separado; la coherencia entre cortes depende del post-proceso.']),
  item([B('Orientación. '), 'El modelo depende de la cabecera del archivo. Un volumen sin esa información, o un paciente girado más de lo que cubre el aumento, entra mal orientado.']),
  item([B('Muestra. '), 'Test tiene 15 pacientes de un solo dataset, y la cifra de H6 es una segunda evaluación. No se probó con otros centros ni escáneres.']),
  item([B('Uso. '), 'Es un trabajo académico. No es un dispositivo médico y no debe usarse para decisiones clínicas.']),

  // ---------- 7 ----------
  H('7. Conclusiones'),
  P('Una red 2D propia de 2.74 millones de parámetros, sin frameworks de detección ni de segmentación, cumple en test los nueve objetivos del curso. La mejora que cerró la brecha no fue una red más grande: casi todas las variantes de arquitectura, pérdida y post-proceso que probamos quedaron dentro del ruido. Lo que funcionó fue revisar caso por caso por qué fallaba el modelo, encontrar que parte de los datos estaba en otra orientación y corregirlo con una regla fija. El trabajo pendiente está en los fragmentos secundarios en contacto, donde un modelo con contexto 3D debería ayudar.'),

  P([B('Disponibilidad. '), 'El código, los pesos, los resultados de cada corrida y la bitácora de decisiones están en el repositorio. El dashboard con los tres visualizadores está publicado en Streamlit Community Cloud y la inferencia en vivo corre en una API en AWS Lambda. El uso de herramientas de IA durante el proyecto está documentado en IA_USAGE.md.'], { align: AlignmentType.LEFT }),

  // ---------- Referencias ----------
  H('Referencias'),
  ref(1, 'Y. Sang, Y. Liu, S. Yibulayimu et al., «Benchmark of Segmentation Techniques for Pelvic Fracture in CT and X-Ray: Summary of the PENGWIN 2024 Challenge», IEEE Transactions on Medical Imaging, vol. 45, n.º 5, pp. 2212-2228, 2026. doi:10.1109/TMI.2025.3650126.'),
  ref(2, 'Y. Liu, S. Yibulayimu, G. Zhu et al., «Automatic pelvic fracture segmentation: a deep learning approach and benchmark dataset», Frontiers in Medicine, vol. 12, 1511487, 2025.'),
  ref(3, 'Y. Sang, Y. Liu, S. Yibulayimu et al., «PENGWIN Task 1: Pelvic Fracture Segmentation on CT» (conjunto de datos, v1), Zenodo, 2024. doi:10.5281/zenodo.10927452. Licencia CC BY 4.0.'),
  ref(4, 'S. Woo, J. Park, J.-Y. Lee e I. S. Kweon, «CBAM: Convolutional Block Attention Module», ECCV, 2018.'),
  ref(5, 'J. Redmon, S. Divvala, R. Girshick y A. Farhadi, «You Only Look Once: Unified, Real-Time Object Detection», CVPR, 2016. Se consultó solo el principio del grid; la implementación es propia.'),
  ref(6, 'A. Kirillov, E. Mintun, N. Ravi et al., «Segment Anything», ICCV, 2023.'),
];

const doc = new Document({
  creator: 'Equipo PENGWIN', title: 'Informe técnico PENGWIN (mini-paper)',
  styles: { default: { document: { run: { font: FUENTE, size: 21 } } } },
  numbering: { config: [{ reference: 'lista', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•',
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1150, bottom: 1150, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FUENTE, size: 18 })] })] }) },
    children: contenido,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(__dirname, 'Mini_paper_PENGWIN.docx'), buffer);
  console.log('Mini_paper_PENGWIN.docx');
});

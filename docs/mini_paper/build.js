// Mini-paper del proyecto PENGWIN. Genera Mini_paper_PENGWIN.docx.
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
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      keepNext: tipo !== 'ultima',     // la tabla no se parte entre páginas
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
const aire = () => new Paragraph({ spacing: { after: 90 }, children: [] });

const contenido = [
  // ---------- Título ----------
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
  H2('3.2 Pérdida y entrenamiento'),
  P('La pérdida es una suma ponderada: detección (peso 1), clasificación (0.5), segmentación semántica (2 para la entropía cruzada y 1 para el Dice), instancia (1 y 1) y borde (1). En la entropía cruzada de instancia los píxeles de fragmentos secundarios pesan el triple, porque son cerca de la mitad de los fragmentos pero cerca de una décima parte del hueso. La clasificación pesa menos porque es la tarea que converge primero; la semántica pesa más porque un error de región arrastra a todos sus fragmentos. Duplicar el peso de instancia y borde no mejoró los fragmentos y empeoró la detección (sección 5.3).'),
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
  P('El modelo final cumple los nueve objetivos en test (Tabla 1). La primera evaluación cumplía ocho: el Dice por fragmento quedaba en 0.845. Las tres semillas de H6 dan 0.875, 0.875 y 0.876; promediar los tres modelos da 0.879, pero triplica el costo y no se usa. La detección se reporta con la caja refinada con la máscara semántica; la cabeza de grid sola obtiene en test un IoU medio de 0.783, mAP50 de 0.912 y mAP50:95 de 0.556, que también cumplen.'),
  cabezaTabla('Tabla 1. Las nueve métricas del curso. Test: 15 casos; un modelo con TTA.'),
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
  P('H6 añade dos cosas a D5: la orientación canónica y el aumento geométrico. Cada combinación se entrenó por separado (Tabla 2). Cada cambio aporta por su cuenta (+0.010 la orientación y +0.014 el aumento con ±12°) y juntos casi se suman (+0.023). La ganancia está en los fragmentos secundarios, que pasan de 0.703 a 0.764.'),
  P('H5 y H6 empatan en la validación normal, dentro del ruido. Los separa la prueba de estrés (Figura 3): con la validación girada 30°, H6 conserva 0.854 y D5 cae a 0.559; con 60°, H6 baja a 0.734 y los demás quedan por debajo de 0.22. Elegimos H6 porque en el dataset hay pacientes girados dentro del escáner, como el caso 012.'),
  cabezaTabla('Tabla 2. Componentes de H6. Dice por fragmento en validación, semilla 42; las dos últimas columnas repiten la validación con los volúmenes girados.'),
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

  H2('5.3 Ablaciones pedidas'),
  P('Las tres se midieron en validación sobre la receta de D5, que comparte arquitectura y pérdida con H6. Sin CBAM, el F1 de clasificación baja de 0.988 a 0.975 y el mAP50:95 de 0.883 a 0.872; el Dice por fragmento no cambia (0.851). Con el backbone preentrenado en Intel Image Classification el Dice por fragmento es 0.851 y 0.856 en dos semillas, frente a 0.851 y 0.855 sin preentrenar, y los secundarios se mueven +0.011 y −0.008: no hay ganancia consistente, así que el modelo final no usa preentrenamiento. Duplicar los pesos de instancia y borde da 0.855, dentro del ruido, y baja el mAP50 de la cabeza de grid de 0.868 a 0.827.'),

  H2('5.4 Comparación con SAM, distancia y latencia'),
  P('SAM ViT-B en modo zero-shot, con la caja de nuestro detector como indicación, obtiene en test un Dice por región de 0.89 frente a 0.93 de nuestra máscara semántica dentro de la misma caja (1 181 cortes, uno de cada tres; 3 002 pares corte-región). La mayor diferencia está en el sacro, 0.79 frente a 0.88. SAM no separa ni clasifica fragmentos.'),
  P('En test hay 41 fragmentos secundarios reales y el modelo entrega distancia para 36. El error medio es 1.28 mm, la mediana 0.0 mm y la correlación con la distancia real 0.82. Un solo fragmento del caso 007, que está a 9.4 mm y el modelo sitúa a 45.8 mm, concentra el 79 % del error; sin él la media es 0.28 mm. En el volumen, el Dice 3D es 0.931 para el fragmento principal y 0.756 para los secundarios.'),
  P('Un corte de 384 × 384 tarda 64 ms en una GPU L4 y 162 ms en una CPU de 12 hilos con un modelo, y 72 y 250 ms con TTA. En un portátil con Intel i9-9980HK y 8 hilos tarda 336 ms, y 696 ms con TTA.'),

  // ---------- 6 ----------
  H('6. Análisis de error y limitaciones'),
  P('La clasificación casi no confunde regiones: el F1 es 0.980 en el sacro, 0.997 en el coxal izquierdo y 0.995 en el derecho. En la máscara semántica, el 1.9 % del hueso se predice como fondo (3.3 % en el sacro) y la confusión entre coxal izquierdo y derecho queda por debajo del 0.1 %.'),
  P('Los cinco peores casos de test son 039 (0.784), 061 (0.822), 007 (0.831), 054 (0.834) y 047 (0.854). En todos el fragmento principal se segmenta bien y el error está en los secundarios. El caso 039 (Figura 4) es un volumen de baja resolución cuyo sacro sale dividido en más fragmentos de los reales; en 061 y 054 el modelo intercambia principal y secundario entre dos fragmentos de tamaño parecido.'),
  figura('../../outputs/s10/informe_h6_test/peor_039.png', 360, 244),
  pie('Figura 4. Caso 039, el peor de test: imagen, fragmentos reales y fragmentos predichos en dos cortes.'),
  P('Limitaciones que hay que declarar:'),
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

  P([B('Disponibilidad. '), 'El código, los pesos, los resultados de cada corrida y la bitácora de decisiones están en el repositorio. El uso de herramientas de IA durante el proyecto está documentado en IA_USAGE.md.'], { align: AlignmentType.LEFT }),

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
  creator: 'Equipo PENGWIN', title: 'Mini-paper PENGWIN',
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

import { useRef, useState } from "react";
import { api, subirArchivo } from "../lib/api";
import { Segmentado } from "./Pestanas";

const VOLUMEN = [".mha", ".mhd", ".raw", ".zraw", ".nii", ".nii.gz", ".gz", ".nrrd", ".nhdr"];
const IMAGEN = [".png", ".jpg", ".jpeg", ".bmp", ".gif"];

interface Archivo {
  archivo: File;
  nombre: string; // ruta relativa en la carpeta, si viene de una
}

const tamano = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`);

// Recorre lo que se suelta en la zona: archivos sueltos o una carpeta completa (DICOM).
async function leerSoltados(items: DataTransferItemList): Promise<Archivo[]> {
  const salida: Archivo[] = [];
  const recorrer = async (entrada: FileSystemEntry, ruta: string): Promise<void> => {
    if (entrada.isFile) {
      const archivo = await new Promise<File>((res, rej) => (entrada as FileSystemFileEntry).file(res, rej));
      salida.push({ archivo, nombre: ruta + archivo.name });
    } else if (entrada.isDirectory) {
      const lector = (entrada as FileSystemDirectoryEntry).createReader();
      let lote: FileSystemEntry[];
      do {
        lote = await new Promise((res, rej) => lector.readEntries(res, rej));
        for (const hijo of lote) await recorrer(hijo, ruta + entrada.name + "/");
      } while (lote.length);
    }
  };
  const entradas = Array.from(items).map((i) => i.webkitGetAsEntry()).filter(Boolean) as FileSystemEntry[];
  for (const e of entradas) await recorrer(e, "");
  return salida;
}

export default function Subida({ destino, nombreDestino, maxMb, trabajo }: {
  destino: string; nombreDestino: string; maxMb: number; trabajo: (id: string, titulo: string) => void;
}) {
  const [modo, setModo] = useState<"volumen" | "dicom">("volumen");
  const [archivos, setArchivos] = useState<Archivo[]>([]);
  const [tta, setTta] = useState(false);
  const [enderezar, setEnderezar] = useState(true);
  const [grosor, setGrosor] = useState("");
  const [arrastrando, setArrastrando] = useState(false);
  const [subido, setSubido] = useState<{ bytes: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const entradaArchivo = useRef<HTMLInputElement>(null);
  const entradaCarpeta = useRef<HTMLInputElement>(null);

  const total = archivos.reduce((s, a) => s + a.archivo.size, 0);
  const visibles = archivos.filter((a) => !a.nombre.split("/").pop()!.startsWith("."));
  const volumenes = visibles.filter((a) => VOLUMEN.some((e) => a.nombre.toLowerCase().endsWith(e)));
  const imagenes = visibles.filter((a) => IMAGEN.some((e) => a.nombre.toLowerCase().endsWith(e)));
  const aviso =
    imagenes.length && !volumenes.length
      ? "Son imágenes PNG/JPG: no guardan las unidades Hounsfield ni el tamaño del vóxel en mm, que el modelo y la distancia necesitan. Sube el volumen (.mha, .nii, .nrrd) o la serie DICOM original."
      : total > maxMb * 1024 * 1024 ? `La subida pesa ${tamano(total)} y el máximo es ${maxMb} MB.` : null;

  const elegir = (lista: FileList | null) => {
    if (!lista) return;
    setError(null);
    setArchivos(Array.from(lista).map((archivo) => ({ archivo, nombre: archivo.webkitRelativePath || archivo.name })));
  };

  const procesar = async () => {
    setError(null);
    try {
      const enviar = visibles;
      const { id, formularios } = await api.nuevaSubida(enviar.map((a) => ({ nombre: a.nombre, tamano: a.archivo.size })));
      const avance = new Array(enviar.length).fill(0);
      const totalEnviar = enviar.reduce((s, a) => s + a.archivo.size, 0);
      setSubido({ bytes: 0, total: totalEnviar });
      let siguiente = 0;
      const obrero = async () => {
        while (siguiente < enviar.length) {
          const i = siguiente++;
          await subirArchivo(formularios[i], enviar[i].archivo, (b) => {
            avance[i] = b;
            setSubido({ bytes: avance.reduce((s, v) => s + v, 0), total: totalEnviar });
          });
          avance[i] = enviar[i].archivo.size;
        }
      };
      await Promise.all(Array.from({ length: Math.min(6, enviar.length) }, obrero));
      const grosorMm = modo === "dicom" && parseFloat(grosor.replace(",", ".")) > 0 ? parseFloat(grosor.replace(",", ".")) : null;
      const { trabajo: idTrabajo } = await api.procesarSubida(id, { destino, tta, enderezar, grosor_mm: grosorMm });
      const nombre = modo === "dicom" ? enviar[0].nombre.split("/")[0] || "carpeta de cortes" : enviar[0].nombre;
      trabajo(idTrabajo, `CT subido · ${nombre}`);
    } catch (e) {
      setError((e as Error).message);
      setSubido(null);
    }
  };

  const subiendo = subido !== null;
  return (
    <div className="tarjeta subida">
      <div className="encabezado-seccion">
        <span className="sobretitulo">Subir un CT propio</span>
        <h2>Procesa tu volumen con el mismo pipeline</h2>
        <p className="tenue">
          Un volumen en un solo archivo (.mha, .nii, .nii.gz, .nrrd; un .mhd con su .raw) o una carpeta de cortes: los
          DICOM de una serie o un .mha por corte. Se orienta con los metadatos, se endereza el giro del paciente si es claro, y pasa por el
          mismo pipeline que los casos de ejemplo. No hay ground truth, así que no se calculan métricas.
        </p>
      </div>
      <Segmentado valor={modo} cambiar={(v) => { setModo(v); setArchivos([]); }}
        opciones={[{ valor: "volumen", texto: "Archivo de volumen" }, { valor: "dicom", texto: "Carpeta de cortes (DICOM o .mha)" }]} />
      <div
        className={`zona-soltar ${arrastrando ? "encima" : ""} ${archivos.length ? "con-archivos" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setArrastrando(true); }}
        onDragLeave={() => setArrastrando(false)}
        onDrop={async (e) => {
          e.preventDefault();
          setArrastrando(false);
          const soltados = await leerSoltados(e.dataTransfer.items);
          if (soltados.length) {
            setArchivos(soltados);
            if (soltados.some((a) => a.nombre.includes("/"))) setModo("dicom");
          }
        }}
        onClick={() => !subiendo && (modo === "dicom" ? entradaCarpeta : entradaArchivo).current?.click()}
        role="button" tabIndex={0}
      >
        <input ref={entradaArchivo} type="file" multiple hidden accept={VOLUMEN.join(",")} onChange={(e) => elegir(e.target.files)} />
        <input ref={entradaCarpeta} type="file" hidden onChange={(e) => elegir(e.target.files)}
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)} />
        <div className="zona-icono">{modo === "dicom" ? "🗂️" : "🩻"}</div>
        {archivos.length ? (
          <>
            <b>{modo === "dicom" ? `${visibles.length} archivos · ${visibles[0]?.nombre.split("/")[0] ?? ""}` : visibles.map((a) => a.nombre).join(" + ")}</b>
            <span className="tenue">{tamano(total)} · haz clic para cambiar</span>
          </>
        ) : (
          <>
            <b>{modo === "dicom" ? "Arrastra aquí la carpeta con los cortes" : "Arrastra aquí el archivo del volumen"}</b>
            <span className="tenue">o haz clic para elegir{modo === "volumen" ? " (un .mhd o .nhdr va junto con su archivo de datos)" : " la carpeta"} · hasta {maxMb} MB</span>
          </>
        )}
      </div>
      {aviso && <div className="alerta aviso">{aviso}</div>}
      {modo === "dicom" && (
        <label className="campo campo-grosor">
          <span>Grosor entre cortes (mm) · solo si son cortes .mha en 2D</span>
          <input inputMode="decimal" placeholder="Déjalo vacío si es DICOM o si los .mha traen su posición" value={grosor}
            onChange={(e) => setGrosor(e.target.value)} />
          <small>Un .mha 2D no guarda su posición en el paciente, así que no se sabe la distancia entre cortes. Si hace falta, el servidor lo pide.</small>
        </label>
      )}
      <div className="opciones">
        <label className="casilla"><input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)} />
          <span><b>TTA de espejo</b><small>Predice también la imagen en espejo y promedia: un poco más preciso y el doble de tiempo. En la GPU la fija el servicio.</small></span></label>
        <label className="casilla"><input type="checkbox" checked={enderezar} onChange={(e) => setEnderezar(e.target.checked)} />
          <span><b>Enderezar el giro axial del paciente</b><small>Estima el giro con la simetría de la pelvis y lo corrige si es de 10° o más.</small></span></label>
      </div>
      {subido && (
        <div className="subiendo">
          <div className="barra-progreso"><div style={{ width: `${(100 * subido.bytes) / Math.max(1, subido.total)}%` }} /></div>
          <span className="dato">Subiendo a AWS · {tamano(subido.bytes)} de {tamano(subido.total)}</span>
        </div>
      )}
      {error && <div className="alerta error">{error}</div>}
      <div className="fila-controles">
        <button className="boton primario grande" disabled={!visibles.length || !!aviso || subiendo} onClick={procesar}>
          {subiendo ? <><span className="giro" /> Subiendo…</> : `Procesar en ${nombreDestino}`}
        </button>
        <span className="dato">El CT se procesa en un servidor de AWS con 10 GB de memoria; tu navegador solo recibe el resultado.</span>
      </div>
    </div>
  );
}

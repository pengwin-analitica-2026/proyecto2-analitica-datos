import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type CasoPengwin, type Casos, type Config, type GpuEstado } from "./lib/api";
import { cargarCaso, type Caso } from "./lib/caso";
import { BASE_RGB, NON_CLINICAL, REGION_LABELS, REGION_ORDER, hex } from "./lib/paleta";
import {
  PestanaCorte, PestanaCrudo, corteInicial, PestanaMetricas, PestanaPasos, PestanaRecon, Segmentado, type Resumen,
} from "./components/Pestanas";
import Subida from "./components/Subida";
import ProgresoTrabajo from "./components/Trabajo";

type Origen = "ejemplos" | "pengwin" | "subir";
type Ruta =
  | { tipo: "origen"; origen: Origen }
  | { tipo: "caso"; prefijo: string }
  | { tipo: "trabajo"; id: string; titulo: string };

// La dirección (#/...) guarda dónde estás: recargar o compartir el enlace abre lo mismo.
function leerRuta(): Ruta {
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  if (h.startsWith("caso/")) return { tipo: "caso", prefijo: h.slice(5) };
  if (h.startsWith("trabajo/")) {
    const [id, ...titulo] = h.slice(8).split("/");
    return { tipo: "trabajo", id, titulo: titulo.join("/") || "Procesando" };
  }
  if (h === "pengwin" || h === "subir") return { tipo: "origen", origen: h };
  return { tipo: "origen", origen: "ejemplos" };
}

function irA(ruta: Ruta) {
  const h = ruta.tipo === "caso" ? `caso/${ruta.prefijo}` : ruta.tipo === "trabajo" ? `trabajo/${ruta.id}/${ruta.titulo}` : ruta.origen;
  location.hash = "/" + h;
}

const PESTANAS = [
  { id: "pasos", texto: "Preprocesamiento" },
  { id: "crudo", texto: "1 · Volumen crudo" },
  { id: "corte", texto: "2 · Corte por corte" },
  { id: "recon", texto: "3 · Reconstrucción 3D" },
  { id: "metricas", texto: "Métricas y latencia" },
] as const;
type Pestana = (typeof PESTANAS)[number]["id"];

const ORIGENES: { id: Origen; titulo: string; texto: string; icono: string }[] = [
  { id: "ejemplos", titulo: "Casos de ejemplo", texto: "Listos al instante", icono: "★" },
  { id: "pengwin", titulo: "Casos PENGWIN (100)", texto: "Del bucket de S3", icono: "▦" },
  { id: "subir", titulo: "Subir un CT", texto: ".mha, NIfTI o DICOM", icono: "⇪" },
];

function origenDe(prefijo: string): Origen {
  return prefijo.includes("/subidas/") ? "subir" : prefijo.includes("/ejemplos/") ? "ejemplos" : "pengwin";
}

export default function App() {
  const [ruta, setRuta] = useState<Ruta>(leerRuta);
  const [config, setConfig] = useState<Config | null>(null);
  const [casos, setCasos] = useState<Casos | null>(null);
  const [errorInicio, setErrorInicio] = useState<string | null>(null);
  const [destino, setDestino] = useState<string>("cpu");
  const [resumen, setResumen] = useState<Resumen | null>(null);
  const [origen, setOrigen] = useState<Origen>(() => (ruta.tipo === "origen" ? ruta.origen : ruta.tipo === "caso" ? origenDe(ruta.prefijo) : "subir"));
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    const cambio = () => setRuta(leerRuta());
    window.addEventListener("hashchange", cambio);
    return () => window.removeEventListener("hashchange", cambio);
  }, []);
  useEffect(() => {
    if (ruta.tipo === "origen") setOrigen(ruta.origen);
    if (ruta.tipo === "caso") setOrigen(origenDe(ruta.prefijo));
    setMenu(false);
  }, [ruta]);

  const recargarConfig = useCallback(() => api.config().then(setConfig, (e) => setErrorInicio(e.message)), []);
  useEffect(() => {
    recargarConfig();
    api.casos().then(setCasos, (e) => setErrorInicio(e.message));
    fetch("/datos/resumen.json").then((r) => (r.ok ? r.json() : null)).then(setResumen, () => undefined);
  }, [recargarConfig]);
  useEffect(() => {
    if (config && !config.destinos.some((d) => d.id === destino)) setDestino("cpu");
  }, [config, destino]);

  const nombreDestino = config?.destinos.find((d) => d.id === destino)?.nombre ?? "AWS · CPU (Lambda)";
  const abrirTrabajo = useCallback((id: string, titulo: string) => irA({ tipo: "trabajo", id, titulo }), []);
  const trabajoListo = useCallback((prefijo: string) => {
    history.replaceState(null, "", `#/caso/${prefijo}`);
    setRuta({ tipo: "caso", prefijo });
  }, []);

  return (
    <div className="app">
      <header className="cabecera">
        <button className="boton-menu" onClick={() => setMenu(!menu)} aria-label="Abrir el menú">☰</button>
        <a className="marca" href="#/">
          <img src="/favicon.svg" alt="" />
          <div>
            <h1>Fracturas pélvicas en CT</h1>
            <span>Detección, segmentación por fragmentos y distancia de separación en mm · dataset PENGWIN</span>
          </div>
        </a>
        <div className="estado-servidor">
          <span className={`led ${config ? "ok" : errorInicio ? "mal" : ""}`} />
          {config ? `Modelo H8 · ${nombreDestino}` : errorInicio ? "Sin conexión con AWS" : "Conectando…"}
        </div>
      </header>
      <div className="aviso-clinico" role="note">⚠️ {NON_CLINICAL}</div>

      <div className="cuerpo">
        <aside className={`lateral ${menu ? "abierto" : ""}`}>
          <section>
            <h4>Caso</h4>
            <nav className="origenes">
              {ORIGENES.map((o) => (
                <button key={o.id} className={origen === o.id ? "activo" : ""} onClick={() => irA({ tipo: "origen", origen: o.id })}>
                  <span className="origen-icono">{o.icono}</span>
                  <span><b>{o.titulo}</b><small>{o.texto}</small></span>
                </button>
              ))}
            </nav>
          </section>
          <section>
            <h4>Procesamiento</h4>
            <label className="campo">
              <span>Dónde corre el modelo</span>
              <select value={destino} onChange={(e) => setDestino(e.target.value)}>
                {(config?.destinos ?? [{ id: "cpu", nombre: "AWS · CPU (Lambda)", detalle: "" }]).map((d) => (
                  <option key={d.id} value={d.id}>{d.nombre}</option>
                ))}
              </select>
            </label>
            <p className="dato">
              {config?.destinos.find((d) => d.id === destino)?.detalle} Para la inferencia en vivo, «Procesar ahora» y los
              CT subidos; los casos precalculados no se vuelven a procesar.
            </p>
            {config?.gpu && <PanelGpu inicial={config.gpu} cambio={recargarConfig} />}
          </section>
        </aside>
        {menu && <div className="velo" onClick={() => setMenu(false)} />}

        <main className="principal">
          {errorInicio && <div className="alerta error">No pude hablar con el servidor de AWS: {errorInicio}</div>}
          {ruta.tipo === "trabajo" ? (
            <ProgresoTrabajo id={ruta.id} titulo={ruta.titulo} listo={trabajoListo}
              volver={() => irA({ tipo: "origen", origen })} />
          ) : ruta.tipo === "caso" ? (
            <VistaCaso key={ruta.prefijo} prefijo={ruta.prefijo} destino={destino} nombreDestino={nombreDestino}
              resumen={resumen} volver={() => irA({ tipo: "origen", origen: origenDe(ruta.prefijo) })} />
          ) : origen === "ejemplos" ? (
            <Ejemplos casos={casos} />
          ) : origen === "pengwin" ? (
            <Pengwin casos={casos} destino={destino} nombreDestino={nombreDestino} trabajo={abrirTrabajo} />
          ) : (
            <Subida destino={destino} nombreDestino={nombreDestino} maxMb={config?.max_subida_mb ?? 1024} trabajo={abrirTrabajo} />
          )}
        </main>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- GPU bajo demanda
const GPU_TEXTO: Record<string, [string, string]> = {
  InService: ["encendida", "ok"], Creating: ["encendiendo (~8 min)", "medio"], Updating: ["actualizando", "medio"],
  Deleting: ["apagándose", "medio"], Failed: ["falló al encender", "mal"],
};

function falta(iso: string | null): string {
  if (!iso) return "";
  const min = Math.max(0, Math.floor((new Date(iso).getTime() - Date.now()) / 60000));
  return `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
}

function PanelGpu({ inicial, cambio }: { inicial: GpuEstado; cambio: () => void }) {
  const [gpu, setGpu] = useState(inicial);
  const [clave, setClave] = useState("");
  const [ocupado, setOcupado] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [texto, nivel] = gpu.estado ? GPU_TEXTO[gpu.estado] ?? [gpu.estado, "medio"] : ["apagada", "apagado"];

  const actualizar = useCallback(async () => {
    try {
      setGpu(await api.gpu());
      cambio();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [cambio]);
  useEffect(() => {
    if (gpu.estado !== "Creating" && gpu.estado !== "Deleting") return;
    const t = window.setInterval(actualizar, 20000);
    return () => clearInterval(t);
  }, [gpu.estado, actualizar]);

  const accion = async (a: "encender" | "apagar") => {
    setOcupado(true);
    setError(null);
    try {
      setGpu(await api.gpuAccion(a, clave));
      cambio();
    } catch (e) {
      setError(`No se pudo ${a}: ${(e as Error).message}`);
    } finally {
      setOcupado(false);
    }
  };
  return (
    <details className="panel-gpu">
      <summary><span className={`led ${nivel}`} /> GPU (SageMaker) · {texto}</summary>
      <p className="dato">NVIDIA T4 (ml.g4dn.xlarge, ~0.74 USD/h mientras esté encendida). Encenderla tarda unos 8 minutos; se apaga sola a las {gpu.horas_apagado ?? 4} h.</p>
      {gpu.apagado_programado && <p className="dato">Se apaga sola en <b>{falta(gpu.apagado_programado)}</b>.</p>}
      <label className="campo">
        <span>Clave de administración</span>
        <input type="password" value={clave} onChange={(e) => setClave(e.target.value)} placeholder="Solo el equipo" autoComplete="off" />
      </label>
      <div className="dos-botones">
        <button className="boton" disabled={ocupado || !clave || gpu.estado === "Deleting"} onClick={() => accion("encender")}>
          {gpu.estado ? "Extender a 4 h" : "Encender (4 h)"}
        </button>
        <button className="boton" disabled={ocupado || !clave || !gpu.estado || gpu.estado === "Deleting"} onClick={() => accion("apagar")}>Apagar</button>
      </div>
      <button className="boton-texto" onClick={actualizar}>Actualizar estado</button>
      {error && <div className="alerta error pequena">{error}</div>}
    </details>
  );
}

// --------------------------------------------------------------------------- elegir caso
function Ejemplos({ casos }: { casos: Casos | null }) {
  if (!casos) return <Cargando texto="Buscando los casos…" />;
  return (
    <div>
      <div className="encabezado-seccion">
        <span className="sobretitulo">Casos de ejemplo</span>
        <h2>Elige un caso para explorar</h2>
        <p className="tenue">Exportados con H8 y TTA, con su ground truth. Se abren al instante.</p>
      </div>
      {casos.ejemplos.length ? (
        <div className="rejilla-casos">
          {casos.ejemplos.map((c) => (
            <a key={c.caso} className="tarjeta-caso" href={`#/caso/${c.prefijo}`}>
              <div className="miniatura"><img src={`/${c.prefijo}/mip_coronal.png`} alt="" loading="lazy" /></div>
              <div className="tarjeta-caso-pie"><b>Caso {c.caso}</b><span className="etiqueta">con ground truth</span></div>
            </a>
          ))}
        </div>
      ) : <div className="vacio">Todavía no hay casos de ejemplo publicados.</div>}
    </div>
  );
}

function Pengwin({ casos, destino, nombreDestino, trabajo }: {
  casos: Casos | null; destino: string; nombreDestino: string; trabajo: (id: string, titulo: string) => void;
}) {
  const [filtro, setFiltro] = useState<"todos" | "train" | "val" | "test">("todos");
  const [buscar, setBuscar] = useState("");
  const [elegido, setElegido] = useState<CasoPengwin | null>(null);
  const [modo, setModo] = useState<"precalculado" | "vivo">("precalculado");
  const [tta, setTta] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [abriendo, setAbriendo] = useState(false);

  const lista = useMemo(() => (casos?.pengwin ?? []).filter((c) =>
    (filtro === "todos" || c.split === filtro) && c.caso.includes(buscar.trim())), [casos, filtro, buscar]);
  if (!casos) return <Cargando texto="Consultando los casos en S3…" />;
  if (casos.error) return <div className="alerta error">{casos.error}</div>;
  const listos = casos.pengwin.filter((c) => c.precalculado).length;

  const elegir = (c: CasoPengwin) => {
    setElegido(c);
    setModo(c.precalculado ? "precalculado" : "vivo");
    setError(null);
  };
  const abrir = async () => {
    if (!elegido) return;
    setAbriendo(true);
    setError(null);
    try {
      const r = await api.abrir({ caso: elegido.caso, modo, destino, tta });
      if (r.resultado) irA({ tipo: "caso", prefijo: r.resultado });
      else if (r.trabajo) trabajo(r.trabajo, `Caso ${elegido.caso} de PENGWIN · ${modo === "vivo" ? "procesar ahora" : "precalculado"}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setAbriendo(false);
    }
  };

  return (
    <div>
      <div className="encabezado-seccion">
        <span className="sobretitulo">Casos PENGWIN</span>
        <h2>{casos.pengwin.length} casos · {listos} precalculados</h2>
        <p className="tenue">Precalculado: se abre en segundos. Procesar ahora: descarga el .mha original y corre el pipeline completo en AWS (unos minutos).</p>
      </div>
      <div className="barra-filtros">
        <Segmentado valor={filtro} cambiar={setFiltro} opciones={[
          { valor: "todos", texto: "Todos" }, { valor: "train", texto: "Train" }, { valor: "val", texto: "Validación" }, { valor: "test", texto: "Test" }]} />
        <input className="buscador" placeholder="Buscar caso (p. ej. 044)" value={buscar} onChange={(e) => setBuscar(e.target.value)} />
      </div>
      <div className="pengwin-contenido">
        <div className="rejilla-pengwin">
          {lista.map((c) => (
            <button key={c.caso} className={`ficha ${elegido?.caso === c.caso ? "activa" : ""}`} onClick={() => elegir(c)}>
              <b>{c.caso}</b>
              <span className={`split ${c.split}`}>{c.split}</span>
              <span className="marcas">
                {c.precalculado && <i title="precalculado" className="m-pre" />}
                {c.girado && <i title="guardado girado 180°" className="m-giro" />}
              </span>
            </button>
          ))}
          {!lista.length && <div className="vacio pequeno">Ningún caso coincide.</div>}
        </div>
        <div className={`tarjeta panel-elegido ${elegido ? "" : "inactivo"}`}>
          {elegido ? (
            <>
              <span className="sobretitulo">Caso seleccionado</span>
              <h3>Caso {elegido.caso}</h3>
              <p className="tenue">
                Split {elegido.split}{elegido.girado ? " · la cabecera dice que está guardado girado 180°" : ""}
                {elegido.precalculado ? " · precalculado" : ""}
              </p>
              <Segmentado valor={modo} cambiar={setModo} opciones={[
                ...(elegido.precalculado ? [{ valor: "precalculado" as const, texto: "Precalculado" }] : []),
                ...(elegido.crudo ? [{ valor: "vivo" as const, texto: "Procesar ahora" }] : []),
              ]} />
              {modo === "vivo" && (
                <>
                  <p className="dato">Se descarga el .mha original de S3 (con su ground truth) y pasa por todo el pipeline: orientación con los metadatos, giro axial, modelo, post-proceso 3D, distancias y mallas. Corre en {nombreDestino}.</p>
                  <label className="casilla"><input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)} />
                    <span><b>TTA de espejo</b><small>Más preciso y el doble de tiempo.</small></span></label>
                </>
              )}
              {error && <div className="alerta error">{error}</div>}
              <button className="boton primario grande ancho" disabled={abriendo} onClick={abrir}>
                {abriendo ? <><span className="giro" /> Abriendo…</> : modo === "vivo" ? `Procesar el caso ${elegido.caso}` : `Abrir el caso ${elegido.caso}`}
              </button>
            </>
          ) : (
            <div className="tenue centro">Elige un caso de la lista.<br /><small><i className="m-pre" /> precalculado · <i className="m-giro" /> guardado girado 180°</small></div>
          )}
        </div>
      </div>
    </div>
  );
}

const Cargando = ({ texto }: { texto: string }) => <div className="cargando-bloque"><span className="giro" /> {texto}</div>;

// --------------------------------------------------------------------------- caso abierto
function VistaCaso({ prefijo, destino, nombreDestino, resumen, volver }: {
  prefijo: string; destino: string; nombreDestino: string; resumen: Resumen | null; volver: () => void;
}) {
  const [caso, setCaso] = useState<Caso | null>(null);
  const [cargando, setCargando] = useState("Leyendo el caso…");
  const [error, setError] = useState<string | null>(null);
  const [pestana, setPestana] = useState<Pestana>("corte");
  const [minMm3, setMinMm3] = useState(1000);
  const [z, setZ] = useState(0);

  useEffect(() => {
    cargarCaso(prefijo, setCargando).then((c) => { setZ(corteInicial(c)); setCaso(c); }, (e) => setError((e as Error).message));
  }, [prefijo]);

  const fragmentos = useMemo(() => caso?.meta.fragments_pred.filter((f) => f.is_main || f.volume_mm3 >= minMm3) ?? [], [caso, minMm3]);
  const visibles = useMemo(() => {
    const v = new Uint8Array(256);
    fragmentos.forEach((f) => (v[f.id] = 1));
    return v;
  }, [fragmentos]);

  if (error) return (
    <div className="tarjeta"><div className="alerta error">{error}</div><button className="boton" onClick={volver}>Volver</button></div>
  );
  if (!caso) return <Cargando texto={cargando} />;
  const m = caso.meta;
  const ocultos = m.fragments_pred.length - fragmentos.length;
  const presentes = REGION_ORDER.filter((r) => fragmentos.some((f) => f.region === r));
  const titulo = m.subido ? `CT subido · ${m.case_id}` : `Caso ${m.case_id}`;

  return (
    <div className="vista-caso">
      <div className="cabeza-caso">
        <button className="boton-texto" onClick={volver}>← Cambiar de caso</button>
        <div className="titulo-caso">
          <h2>{titulo}</h2>
          <div className="chips">
            <span className="chip">{m.n_slices} cortes</span>
            <span className="chip">vóxel {m.spacing_native_xyz.map((v) => v.toFixed(2)).join("×")} mm</span>
            <span className="chip">{m.fragments_pred.length} fragmentos</span>
            {m.fragments_gt && <span className="chip gt">GT: {m.fragments_gt.length} fragmentos</span>}
            {m.en_vivo && <span className="chip vivo">procesado ahora</span>}
          </div>
        </div>
      </div>

      <div className="barra-caso">
        <div className="leyenda">
          {presentes.map((r) => (
            <span key={r}><i style={{ background: hex(BASE_RGB[r]) }} />{REGION_LABELS[r]}</span>
          ))}
          <span className="tenue">· los secundarios son tintes más claros con contorno blanco</span>
        </div>
        <label className="deslizador-en-linea" title="Oculta los fragmentos secundarios más chicos que esto: casi siempre son ruido de la segmentación. El fragmento principal nunca se oculta.">
          Volumen mínimo del fragmento <b>{minMm3.toLocaleString("es-CO")} mm³</b>
          <input type="range" min={0} max={3000} step={100} value={minMm3} onChange={(e) => setMinMm3(+e.target.value)} />
          <small>{fragmentos.length} visibles{ocultos ? ` · ${ocultos} ocultos` : ""}</small>
        </label>
      </div>

      <nav className="pestanas" role="tablist">
        {PESTANAS.map((p) => (
          <button key={p.id} role="tab" aria-selected={pestana === p.id} className={pestana === p.id ? "activa" : ""} onClick={() => setPestana(p.id)}>
            {p.texto}
          </button>
        ))}
      </nav>
      <div className="contenido-pestana">
        {pestana === "pasos" && <PestanaPasos caso={caso} />}
        {pestana === "crudo" && <PestanaCrudo caso={caso} />}
        {pestana === "corte" && <PestanaCorte caso={caso} fragmentos={fragmentos} visibles={visibles} destino={destino} nombreDestino={nombreDestino} z={z} setZ={setZ} />}
        {pestana === "recon" && <PestanaRecon caso={caso} fragmentos={fragmentos} />}
        {pestana === "metricas" && <PestanaMetricas caso={caso} resumen={resumen} />}
      </div>

      <details className="desplegable detalles-caso">
        <summary>Detalles técnicos del caso</summary>
        <div className="desplegable-cuerpo">
          <ul className="lista-detalles">
            {m.subido && <li>CT subido: {m.subido.fuente} · orientación con los metadatos: {m.subido.orientacion.accion} · inferencia: {m.subido.donde}</li>}
            {m.en_vivo && <li>Procesado en vivo en {m.en_vivo.donde}</li>}
            <li>{m.n_slices} cortes · {m.size}×{m.size} px · spacing del header {m.spacing_native_xyz.map((v) => v.toFixed(2)).join("×")} mm</li>
            <li>Modelo {m.model.checkpoint} · {(m.model.parameters / 1e6).toFixed(2)} M parámetros · precisión {m.model.precision ?? "fp32"} · TTA {m.tta ? "sí" : "no"}</li>
            {m.canonical && !m.subido && (
              <li>Orientación canónica (sacro abajo): {m.flipped
                ? "este volumen venía guardado girado 180° (cabecera LPI) y se enderezó antes de predecir; las tres vistas y el ground truth están enderezados."
                : "este volumen ya venía en esa orientación."}</li>
            )}
            {m.enderezado && (
              <li>Giro axial del paciente: {m.enderezado.angulo >= 0 ? "+" : ""}{m.enderezado.angulo.toFixed(1)}° estimado; {m.enderezado.corregido
                ? "se enderezó antes de predecir (las tres vistas y el ground truth están enderezados)."
                : `no se corrigió (solo se corrigen giros claros de ${m.enderezado.giro_minimo.toFixed(0)}° o más).`}</li>
            )}
          </ul>
        </div>
      </details>
    </div>
  );
}

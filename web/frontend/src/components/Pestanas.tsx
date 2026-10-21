import { useEffect, useMemo, useRef, useState } from "react";
import { api, type CorteVivo } from "../lib/api";
import { cargarHueso, grisDelCorte, type Caso, type Fragmento, type Malla } from "../lib/caso";
import { dibujarCorte } from "../lib/corte";
import {
  GT_RGB, REGION_LABELS, REGION_ORDER, REGION_STARTS, distanceText, fragmentRgb, hex, localName, shortDistance,
} from "../lib/paleta";
import Visor3D, { type MallaVisible, type Rotulo3D } from "./Visor3D";

export const Nota = ({ children }: { children: React.ReactNode }) => <p className="nota">{children}</p>;

function Interruptor({ activo, cambiar, children, deshabilitado }: {
  activo: boolean; cambiar: (v: boolean) => void; children: React.ReactNode; deshabilitado?: boolean;
}) {
  return (
    <label className={`interruptor ${deshabilitado ? "apagado" : ""}`}>
      <input type="checkbox" checked={activo} disabled={deshabilitado} onChange={(e) => cambiar(e.target.checked)} />
      <span className="pista" />
      <span>{children}</span>
    </label>
  );
}

export function Segmentado<T extends string>({ opciones, valor, cambiar }: {
  opciones: { valor: T; texto: string }[]; valor: T; cambiar: (v: T) => void;
}) {
  return (
    <div className="segmentado" role="radiogroup">
      {opciones.map((o) => (
        <button key={o.valor} role="radio" aria-checked={valor === o.valor}
          className={valor === o.valor ? "activo" : ""} onClick={() => cambiar(o.valor)}>
          {o.texto}
        </button>
      ))}
    </div>
  );
}

// --------------------------------------------------------------------------- preprocesamiento
export function PestanaPasos({ caso }: { caso: Caso }) {
  const pasos = caso.meta.pasos;
  if (!pasos?.length || !Object.keys(caso.meta.paquete.pasos).length)
    return (
      <div className="vacio">
        Este caso se exportó antes de que existiera esta vista. Elige un caso de PENGWIN o sube un CT para verla.
      </div>
    );
  return (
    <div>
      <Nota>
        Qué le pasa al volumen antes de entrar a la red. A la izquierda, el corte axial con más hueso (el mismo en
        todas las etapas); a la derecha, la proyección de máxima intensidad (MIP) axial. Las etapas 1 a 3 se ven con
        la ventana ósea y la proporción real del vóxel; la 4, tal cual entra al modelo.
      </Nota>
      <ol className="linea-tiempo">
        {pasos.map((paso, i) => (
          <li key={paso.clave} className="paso">
            <div className="paso-num">{i + 1}</div>
            <div className="paso-cuerpo">
              <div className="paso-texto">
                <h3>{paso.titulo}</h3>
                <p>{paso.texto}</p>
                {paso.forma && (
                  <span className="dato">
                    {paso.forma[0]} cortes de {paso.forma[1]}×{paso.forma[2]} · corte mostrado: {(paso.z ?? 0) + 1}
                  </span>
                )}
              </div>
              {(["corte", "mip"] as const).map((v) => {
                const archivo = caso.meta.paquete.pasos[`${paso.clave}_${v}`];
                return archivo ? (
                  <figure key={v} className="figura">
                    <img src={caso.url(archivo)} alt="" loading="lazy" />
                    <figcaption>{v === "corte" ? "Corte axial" : "MIP axial"}</figcaption>
                  </figure>
                ) : null;
              })}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

// --------------------------------------------------------------------------- visualizador 1
export function PestanaCrudo({ caso }: { caso: Caso }) {
  const [ver3d, setVer3d] = useState(false);
  const [version, setVersion] = useState<"pulido" | "crudo">("pulido");
  const [hueso, setHueso] = useState<Partial<Record<"crudo" | "pulido", Malla>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mips = caso.meta.paquete.mips;

  useEffect(() => {
    setHueso(null);
    setVer3d(false);
  }, [caso]);
  useEffect(() => {
    if (ver3d && !hueso) cargarHueso(caso).then(setHueso, (e) => setError(String(e.message ?? e)));
  }, [ver3d, hueso, caso]);

  const malla = hueso?.[version] ?? hueso?.crudo;
  const mallas = useMemo<MallaVisible[]>(
    () => (malla ? [{ clave: "hueso", malla, color: "#f2ead3", opacidad: 1, nombre: "Hueso (umbral 250 HU)" }] : []),
    [malla],
  );
  if (!mips) return <div className="vacio">Este caso se exportó sin la vista cruda.</div>;
  return (
    <div>
      <Nota>
        Sin modelo: umbral de 250 HU sobre el volumen sin procesar, limpieza de camilla y cables, y proyección de
        máxima intensidad (MIP) en los tres ejes.
      </Nota>
      <div className="rejilla-mips">
        {(["axial", "coronal", "sagital"] as const).map((eje) => (
          <figure key={eje} className="figura tarjeta-img">
            <img src={caso.url(mips[eje])} alt={`MIP ${eje}`} />
            <figcaption>MIP {eje[0].toUpperCase() + eje.slice(1)}</figcaption>
          </figure>
        ))}
      </div>
      <div className="fila-controles">
        <Interruptor activo={ver3d} cambiar={setVer3d}>Mostrar isosuperficie 3D del hueso</Interruptor>
        {ver3d && hueso?.pulido && (
          <Segmentado valor={version} cambiar={setVersion}
            opciones={[{ valor: "pulido", texto: "Pulido" }, { valor: "crudo", texto: "Crudo" }]} />
        )}
      </div>
      {ver3d && version === "pulido" && hueso?.pulido && (
        <Nota>Pulido: además del umbral de 250 HU recupera el hueso tenue (150-250 HU conectado al esqueleto) y rellena las cavidades. Crudo: solo el umbral y la limpieza, como la MIP.</Nota>
      )}
      {ver3d && (error ? <div className="alerta error">{error}</div> : malla ? (
        <Visor3D mallas={mallas} vista={caso.prefijo + version} alto={560} />
      ) : <div className="cargando-bloque"><span className="giro" /> Descargando la malla del hueso…</div>)}
      <Nota>El pulido 3D es solo visual; el modelo, las métricas y las distancias usan los datos sin modificar.</Nota>
    </div>
  );
}

// --------------------------------------------------------------------------- visualizador 2
export function TablaFragmentos({ fragmentos }: { fragmentos: Fragmento[] }) {
  return (
    <div className="tabla-envoltura">
      <table className="tabla">
        <thead>
          <tr><th>Hueso</th><th>Fragmento</th><th>Distancia al principal</th><th className="num">Volumen (mm³)</th><th>Cortes</th></tr>
        </thead>
        <tbody>
          {fragmentos.map((f) => (
            <tr key={f.id}>
              <td><span className="punto" style={{ background: hex(fragmentRgb(f.id)) }} />{REGION_LABELS[f.region]}</td>
              <td>{localName(f.id)}</td>
              <td>{f.is_main ? "—" : distanceText(f.distance_mm)}{f.metal && <span className="marca-metal" title="En cortes con metal: puede ser un artefacto">⚠ metal</span>}</td>
              <td className="num">{Math.round(f.volume_mm3).toLocaleString("es-CO")}</td>
              <td>{f.z_range[0]}–{f.z_range[1]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function corteInicial(caso: Caso): number {
  const [primero, ultimo] = caso.meta.bone_z_range ?? [0, caso.meta.n_slices - 1];
  return Math.floor((primero + ultimo) / 2);
}

export function PestanaCorte({ caso, fragmentos, visibles, destino, nombreDestino, z, setZ }: {
  caso: Caso; fragmentos: Fragmento[]; visibles: Uint8Array; destino: string; nombreDestino: string;
  z: number; setZ: React.Dispatch<React.SetStateAction<number>>;
}) {
  const m = caso.meta;
  const S = m.size;
  const [primero, ultimo] = m.bone_z_range ?? [0, m.n_slices - 1];
  const [mascaras, setMascaras] = useState(true);
  const [cajas, setCajas] = useState(true);
  const [tipoCaja, setTipoCaja] = useState<"refined" | "raw">("refined");
  const [verGt, setVerGt] = useState(false);
  const [gris, setGris] = useState<{ z: number; datos: Uint8Array } | null>(null);
  const lienzo = useRef<HTMLCanvasElement>(null);

  // Corte en gris (PNG) y los vecinos, para que el deslizador vaya fluido.
  useEffect(() => {
    let vigente = true;
    grisDelCorte(caso, z).then((datos) => vigente && setGris({ z, datos }), () => undefined);
    for (const d of [1, -1, 2, -2, 3, 4]) if (z + d >= 0 && z + d < m.n_slices) grisDelCorte(caso, z + d).catch(() => undefined);
    return () => { vigente = false; };
  }, [caso, z, m.n_slices]);

  const idsCorte = useMemo(() => {
    const crudo = caso.ids.subarray(z * S * S, (z + 1) * S * S);
    const salida = new Uint8Array(S * S);
    for (let i = 0; i < salida.length; i++) salida[i] = visibles[crudo[i]] ? crudo[i] : 0;
    return salida;
  }, [caso, z, S, visibles]);

  useEffect(() => {
    if (!gris || !lienzo.current) return;
    dibujarCorte(lienzo.current, {
      gris: gris.datos, ids: idsCorte, gt: caso.gt ? caso.gt.subarray(gris.z * S * S, (gris.z + 1) * S * S) : null,
      fragmentos, cajas: tipoCaja === "refined" ? m.slices[gris.z].boxes_refined : m.slices[gris.z].boxes_raw,
      mostrarMascaras: mascaras, mostrarCajas: cajas, mostrarGt: verGt && !!caso.gt,
      titulo: `corte ${gris.z + 1}/${m.n_slices}`,
    }, S);
  }, [gris, idsCorte, fragmentos, mascaras, cajas, tipoCaja, verGt, caso, m, S]);

  const mover = (d: number) => setZ((v) => Math.min(m.n_slices - 1, Math.max(0, v + d)));
  const rueda = (e: React.WheelEvent) => { e.preventDefault(); mover(e.deltaY > 0 ? 1 : -1); };
  useEffect(() => {
    const el = lienzo.current;
    if (!el) return;
    const bloquear = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", bloquear, { passive: false });
    return () => el.removeEventListener("wheel", bloquear);
  }, []);

  const corteMeta = m.slices[z];
  return (
    <div>
      <div className="visor-corte">
        <div className="marco-corte" tabIndex={0}
          onKeyDown={(e) => { if (e.key === "ArrowUp" || e.key === "ArrowRight") mover(1); if (e.key === "ArrowDown" || e.key === "ArrowLeft") mover(-1); }}>
          <canvas ref={lienzo} className="lienzo-corte" onWheel={rueda} />
          {(!gris || gris.z !== z) && <span className="giro esquina" />}
        </div>
        <aside className="panel-corte">
          <div className="control">
            <div className="control-titulo">
              <span>Corte</span>
              <span className="valor-grande">{z + 1}<small> / {m.n_slices}</small></span>
            </div>
            <input type="range" min={0} max={m.n_slices - 1} value={z} onChange={(e) => setZ(+e.target.value)} />
            <div className="pista-hueso">
              <div style={{ left: `${(primero / m.n_slices) * 100}%`, width: `${((ultimo - primero + 1) / m.n_slices) * 100}%` }} />
            </div>
            <span className="dato">El hueso aparece entre los cortes {primero + 1} y {ultimo + 1}. Rueda del mouse o flechas para moverte.</span>
          </div>
          <Interruptor activo={mascaras} cambiar={setMascaras}>Máscaras por macro-hueso</Interruptor>
          <Interruptor activo={cajas} cambiar={setCajas}>Cajas de detección</Interruptor>
          {cajas && (
            <Segmentado valor={tipoCaja} cambiar={setTipoCaja}
              opciones={[{ valor: "refined", texto: "Refinada" }, { valor: "raw", texto: "Cabeza grid cruda" }]} />
          )}
          <Interruptor activo={verGt} cambiar={setVerGt} deshabilitado={!caso.gt}>
            Contorno del ground truth {!caso.gt && <small>(sin GT)</small>}
          </Interruptor>
          <div className="control">
            <div className="control-titulo"><span>Presencia por región en este corte</span></div>
            {REGION_ORDER.map((r, i) => {
              const p = corteMeta.classification[i];
              return (
                <div key={r} className="barra-presencia">
                  <div className="barra-etiqueta"><span>{REGION_LABELS[r]}</span><b>{p.toFixed(2)}</b></div>
                  <div className="barra"><div style={{ width: `${p * 100}%`, background: hex(fragmentRgb(REGION_STARTS[r])) }} /></div>
                </div>
              );
            })}
          </div>
        </aside>
      </div>
      <Nota>
        La distancia junto a cada fragmento es la mínima borde a borde, en mm, entre ese fragmento y el principal de su
        hueso, medida sobre el volumen 3D completo (spacing del header): es la misma en todos los cortes donde el
        fragmento aparece.
      </Nota>
      <TablaFragmentos fragmentos={fragmentos} />
      <InferenciaEnVivo caso={caso} z={z} destino={destino} nombreDestino={nombreDestino} />
    </div>
  );
}

function InferenciaEnVivo({ caso, z, destino, nombreDestino }: { caso: Caso; z: number; destino: string; nombreDestino: string }) {
  const [abierto, setAbierto] = useState(false);
  const [estado, setEstado] = useState<"quieto" | "corriendo">("quieto");
  const [resultado, setResultado] = useState<{ z: number; datos: CorteVivo; ida: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lienzo = useRef<HTMLCanvasElement>(null);
  const S = caso.meta.size;

  useEffect(() => { setResultado(null); setError(null); }, [caso]);

  const inferir = async () => {
    setEstado("corriendo");
    setError(null);
    try {
      const gris = await grisDelCorte(caso, z);
      const inicio = performance.now();
      const datos = await api.corte(gris, destino);
      setResultado({ z, datos, ida: performance.now() - inicio });
    } catch (e) {
      setError(`No se pudo inferir en vivo: ${(e as Error).message}`);
    } finally {
      setEstado("quieto");
    }
  };

  useEffect(() => {
    if (!resultado || !lienzo.current) return;
    const crudo = Uint8Array.from(atob(resultado.datos.semantica_b64), (c) => c.charCodeAt(0));
    const inicios = [0, REGION_STARTS.sacro, REGION_STARTS.coxal_izquierdo, REGION_STARTS.coxal_derecho];
    const ids = crudo.map((v) => inicios[v] ?? 0);
    grisDelCorte(caso, resultado.z).then((gris) => {
      if (!lienzo.current) return;
      dibujarCorte(lienzo.current, {
        gris, ids, fragmentos: [], cajas: resultado.datos.boxes_refined,
        mostrarMascaras: true, mostrarCajas: true, mostrarGt: false, titulo: `EN VIVO · corte ${resultado.z + 1}`,
      }, S);
    });
  }, [resultado, caso, S]);

  const modeloDistinto = resultado?.datos.model && resultado.datos.model !== caso.meta.model.checkpoint;
  return (
    <details className="desplegable" open={abierto} onToggle={(e) => setAbierto((e.target as HTMLDetailsElement).open)}>
      <summary>
        <span className="icono-vivo" /> Inferir este corte en vivo (modelo real, con latencia)
      </summary>
      <div className="desplegable-cuerpo">
        <div className="fila-controles">
          <button className="boton primario" disabled={estado === "corriendo"} onClick={inferir}>
            {estado === "corriendo" ? <><span className="giro" /> Infiriendo…</> : `Inferir el corte ${z + 1} · ${nombreDestino}`}
          </button>
          <span className="dato">La primera vez el servidor carga el modelo y tarda más (arranque en frío).</span>
        </div>
        {error && <div className="alerta error">{error}</div>}
        {modeloDistinto && (
          <div className="alerta aviso">El servicio usa {resultado!.datos.model} y este caso se exportó con {caso.meta.model.checkpoint}.</div>
        )}
        {resultado && (
          <div className="resultado-vivo">
            <canvas ref={lienzo} className="lienzo-corte" />
            <div>
              <div className="kpi">
                <span className="kpi-titulo">Latencia (modelo + NMS)</span>
                <span className="kpi-valor">{resultado.datos.latency_ms.toFixed(0)} ms</span>
              </div>
              <div className="kpi">
                <span className="kpi-titulo">Ida y vuelta desde el navegador</span>
                <span className="kpi-valor chico">{resultado.ida.toFixed(0)} ms</span>
              </div>
              <Nota>
                Dispositivo: {resultado.datos.device} en {resultado.datos.donde} · precisión {resultado.datos.precision} ·
                sin TTA · un corte {S}×{S}. En vivo se muestra la región anatómica y las cajas; la separación de
                fragmentos y la distancia en mm usan el volumen completo y vienen de la exportación.
              </Nota>
            </div>
          </div>
        )}
      </div>
    </details>
  );
}

// --------------------------------------------------------------------------- visualizador 3
export function PestanaRecon({ caso, fragmentos }: { caso: Caso; fragmentos: Fragmento[] }) {
  const [principal, setPrincipal] = useState(true);
  const [opacidad, setOpacidad] = useState(0.4);
  const [gt, setGt] = useState(false);
  const [contactos, setContactos] = useState(true);
  const m = caso.meta;
  const tieneGt = !!m.fragments_gt?.some((f) => caso.mallas[`g_${f.id}`]);
  const [sx, sy, sz] = m.spacing_cache_xyz;

  const mallas = useMemo<MallaVisible[]>(() => {
    const salida: MallaVisible[] = [];
    for (const f of fragmentos) {
      if (f.is_main && !principal) continue;
      const malla = caso.mallas[`p_${f.id}`];
      if (!malla) continue;
      const dist = f.is_main ? "" : ` · ${shortDistance(f.distance_mm)}`;
      salida.push({ clave: `p_${f.id}`, malla, color: hex(fragmentRgb(f.id)), opacidad: f.is_main ? opacidad : 1,
        nombre: `${REGION_LABELS[f.region]} · ${localName(f.id)}${dist}` });
    }
    if (gt && m.fragments_gt)
      for (const f of m.fragments_gt) {
        const malla = caso.mallas[`g_${f.id}`];
        if (malla) salida.push({ clave: `g_${f.id}`, malla, color: hex(GT_RGB), opacidad: 0.18,
          nombre: `Real · ${REGION_LABELS[f.region]} · ${localName(f.id)}` });
      }
    return salida;
  }, [caso, fragmentos, principal, opacidad, gt, m.fragments_gt]);

  const secundarios = fragmentos.filter((f) => !f.is_main);
  const centro = (f: Fragmento): [number, number, number] =>
    [f.centroid_zyx_vox[2] * sx, f.centroid_zyx_vox[1] * sy, f.centroid_zyx_vox[0] * sz];
  const marcadores = useMemo(() => secundarios.map((f) => ({ pos: centro(f),
    texto: `${REGION_LABELS[f.region]} · ${localName(f.id)} · ${distanceText(f.distance_mm)}${f.metal ? " · ⚠ en cortes con metal" : ""}` })), [fragmentos]); // eslint-disable-line
  const rotulos = useMemo<Rotulo3D[]>(() => secundarios
    .filter((f) => contactos || (f.distance_mm ?? 0) >= 0.05)
    .map((f, indice) => ({ pos: centro(f), indice, color: hex(fragmentRgb(f.id)),
      texto: `<b>${localName(f.id)} · ${shortDistance(f.distance_mm)}${f.metal ? " ⚠" : ""}</b>` })), [fragmentos, contactos]); // eslint-disable-line

  return (
    <div>
      <Nota>
        Máscaras predichas apiladas, marching cubes por fragmento y malla interactiva. Cada fragmento conserva el color
        de su macro-hueso y lleva su ID y su distancia al principal.
      </Nota>
      <div className="fila-controles envolver">
        <Interruptor activo={principal} cambiar={setPrincipal}>Fragmento principal</Interruptor>
        <label className={`deslizador-en-linea ${principal ? "" : "apagado"}`}>
          Opacidad del principal <b>{opacidad.toFixed(2)}</b>
          <input type="range" min={0.1} max={1} step={0.05} value={opacidad} disabled={!principal}
            onChange={(e) => setOpacidad(+e.target.value)} />
        </label>
        <Interruptor activo={gt} cambiar={setGt} deshabilitado={!tieneGt}>Superponer ground truth</Interruptor>
        <Interruptor activo={contactos} cambiar={setContactos}>Rotular los fragmentos en contacto (0.0 mm)</Interruptor>
      </div>
      <Visor3D mallas={mallas} rotulos={rotulos} marcadores={marcadores} vista={caso.prefijo} />
      <Nota>El pulido 3D es solo visual; el modelo, las métricas y las distancias usan los datos sin modificar.</Nota>
    </div>
  );
}

// --------------------------------------------------------------------------- métricas
export interface Resumen {
  test: { n_casos: number; n_cortes: number; columnas: string[];
    filas: { metrica: string; objetivo: number; valores: Record<string, number>; cumple: boolean }[] } | null;
  indicadores: { titulo: string; valor: string; ayuda: string }[];
  latencia: { dispositivo: string; configuracion: string; mediana_ms: number; alto: string; fuente: string }[];
}

const mm = (v: number | null) => (v === null || v === undefined ? "—" : v.toFixed(1));

export function PestanaMetricas({ caso, resumen }: { caso: Caso; resumen: Resumen | null }) {
  const m = caso.meta;
  const comp = m.comparison;
  const pareados = comp?.filter((r) => r.abs_error_mm !== null) ?? [];
  const separados = pareados.filter((r) => r.gt_mm >= 0.05);
  const media = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return (
    <div className="metricas">
      <section>
        <h2>Distancia de separación: predicción contra ground truth</h2>
        <Nota>Método de la distancia: <code>{m.distance_method ?? "desconocido"}</code></Nota>
        {comp ? (
          <>
            <div className="rejilla-kpi">
              <div className="kpi"><span className="kpi-titulo">Fragmentos secundarios reales</span><span className="kpi-valor">{comp.length}</span>
                <span className="kpi-ayuda">Los que no tienen pareja predicha aparecen como «no detectado».</span></div>
              <div className="kpi"><span className="kpi-titulo">Error medio, todos los emparejados</span>
                <span className="kpi-valor">{pareados.length ? `${media(pareados.map((r) => r.abs_error_mm!)).toFixed(1)} mm` : "—"}</span>
                <span className="kpi-ayuda">{pareados.length} fragmentos; incluye los que están en contacto (0.0 mm).</span></div>
              <div className="kpi destacado"><span className="kpi-titulo">Error medio, solo separados (&gt; 0 mm)</span>
                <span className="kpi-valor">{separados.length ? `${media(separados.map((r) => r.abs_error_mm!)).toFixed(1)} mm` : "—"}</span>
                <span className="kpi-ayuda">{separados.length} fragmentos: es la medida que importa para la separación.</span></div>
            </div>
            <div className="tabla-envoltura">
              <table className="tabla">
                <thead><tr><th>Hueso</th><th>Fragmento real</th><th className="num">Distancia real (mm)</th><th>Fragmento predicho</th>
                  <th className="num">Distancia predicha (mm)</th><th className="num">Error absoluto (mm)</th><th className="num">Dice 3D</th></tr></thead>
                <tbody>
                  {comp.map((r) => (
                    <tr key={r.gt_id}>
                      <td><span className="punto" style={{ background: hex(fragmentRgb(r.gt_id)) }} />{REGION_LABELS[r.region]}</td>
                      <td>{localName(r.gt_id)}</td>
                      <td className="num">{mm(r.gt_mm)}</td>
                      <td>{r.pred_id === null ? <span className="etiqueta mala">no detectado</span> : localName(r.pred_id)}</td>
                      <td className="num">{mm(r.pred_mm)}</td>
                      <td className="num">{mm(r.abs_error_mm)}</td>
                      <td className="num">{r.dice3d.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : <div className="vacio pequeno">Este caso no tiene ground truth.</div>}
      </section>

      {resumen?.test && (
        <section>
          <h2>Resultados en test ({resumen.test.n_casos} casos, {resumen.test.n_cortes.toLocaleString("es-CO")} cortes)</h2>
          <div className="tabla-envoltura">
            <table className="tabla">
              <thead><tr><th>Métrica</th><th className="num">Objetivo</th>
                {resumen.test.columnas.map((c) => <th key={c} className={`num ${c.includes("H8") ? "col-final" : ""}`}>{c}</th>)}
                <th>Cumple (H8)</th></tr></thead>
              <tbody>
                {resumen.test.filas.map((f) => (
                  <tr key={f.metrica}>
                    <td>{f.metrica}</td>
                    <td className="num">{f.objetivo.toFixed(2)}</td>
                    {resumen.test!.columnas.map((c) => <td key={c} className={`num ${c.includes("H8") ? "col-final" : ""}`}>{f.valores[c]?.toFixed(3) ?? "—"}</td>)}
                    <td><span className={`etiqueta ${f.cumple ? "buena" : "mala"}`}>{f.cumple ? "sí" : "no"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Nota>
            <b>Cómo leer las columnas.</b> H8 es el modelo de esta demo: sin EMA, con el giro del aumento en milímetros y
            con aumento de ruido gaussiano; se eligió solo con validación y se evaluó en test una vez, con la configuración
            escrita de antemano. La primera evaluación (D5) se hizo con la configuración congelada antes de ver test y es la
            única cifra limpia: no cumplía el Dice por fragmento (0.845). Las siguientes se hicieron después de analizar los
            fallos de esa primera evaluación, que miró dos casos de test (012 y 039).
          </Nota>
          <div className="rejilla-kpi">
            {resumen.indicadores.map((k) => (
              <div key={k.titulo} className="kpi"><span className="kpi-titulo">{k.titulo}</span>
                <span className="kpi-valor">{k.valor}</span><span className="kpi-ayuda">{k.ayuda}</span></div>
            ))}
          </div>
        </section>
      )}

      <section>
        <h2>Latencia por imagen (384×384, lote 1)</h2>
        {resumen?.latencia.length ? (
          <div className="tabla-envoltura">
            <table className="tabla">
              <thead><tr><th>Dispositivo</th><th>Configuración</th><th className="num">Mediana (ms)</th><th>Percentil alto (ms)</th><th>Fuente</th></tr></thead>
              <tbody>
                {resumen.latencia.map((f, i) => (
                  <tr key={i}><td>{f.dispositivo}</td><td>{f.configuracion}</td><td className="num">{f.mediana_ms}</td><td>{f.alto}</td><td className="tenue">{f.fuente}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="vacio pequeno">Aún no hay mediciones de latencia.</div>}
        <Nota>
          Al exportar este caso: {m.ms_per_slice.toFixed(0)} ms por corte ({m.device}, TTA {m.tta ? "sí" : "no"}, lote de 8).
          La inferencia en vivo de la pestaña 2 mide un corte suelto en el servidor de AWS.
        </Nota>
      </section>
    </div>
  );
}

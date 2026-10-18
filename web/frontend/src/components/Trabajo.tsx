import { useEffect, useState } from "react";
import { api, type Trabajo } from "../lib/api";

// Etapas del pipeline según la bitácora de demo_export, para una barra de progreso honesta.
const ETAPAS: { patron: RegExp; avance: number; nombre: string }[] = [
  { patron: /Descargando|Leyendo \d+ archivo/, avance: 4, nombre: "Descargando" },
  { patron: /Modelo:/, avance: 8, nombre: "Cargando el modelo" },
  { patron: /Leído:|cabecera|girado 180/, avance: 12, nombre: "Leyendo y orientando" },
  { patron: /giro axial/, avance: 16, nombre: "Enderezando el giro axial" },
  { patron: /volumen \(/, avance: 18, nombre: "Preprocesando" },
  { patron: /inferencia:/, avance: 70, nombre: "Modelo listo" },
  { patron: /distancias predichas/, avance: 76, nombre: "Distancias en mm" },
  { patron: /ground truth:/, avance: 79, nombre: "Comparando con el ground truth" },
  { patron: /mallas:/, avance: 84, nombre: "Mallas 3D" },
  { patron: /visualizador 1|vista cruda/, avance: 90, nombre: "Vista cruda" },
  { patron: /Preparando el caso/, avance: 94, nombre: "Preparando la página" },
];

function progreso(t: Trabajo): { pct: number; etapa: string } {
  let pct = t.estado === "en_cola" ? 1 : 2;
  let etapa = t.estado === "en_cola" ? "En cola (arrancando el servidor)" : "Procesando";
  for (const { m } of t.log) {
    const cortes = m.match(/cortes (\d+)\/(\d+)/);
    if (cortes) {
      pct = Math.max(pct, 18 + (52 * +cortes[1]) / +cortes[2]);
      etapa = `Modelo: corte ${cortes[1]} de ${cortes[2]}`;
      continue;
    }
    for (const e of ETAPAS)
      if (e.patron.test(m) && e.avance >= pct) {
        pct = e.avance;
        etapa = e.nombre;
      }
  }
  if (t.estado === "listo") return { pct: 100, etapa: "Listo" };
  return { pct, etapa };
}

const reloj = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function ProgresoTrabajo({ id, titulo, listo, volver }: {
  id: string; titulo: string; listo: (prefijo: string) => void; volver: () => void;
}) {
  const [trabajo, setTrabajo] = useState<Trabajo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ahora, setAhora] = useState(Date.now());

  useEffect(() => {
    let vivo = true;
    let espera: number;
    const consultar = async () => {
      try {
        const t = await api.trabajo(id);
        if (!vivo) return;
        setTrabajo(t);
        setError(null);
        if (t.estado === "listo" && t.resultado) {
          listo(t.resultado);
          return;
        }
        if (t.estado === "error") return;
      } catch (e) {
        if (vivo) setError((e as Error).message);
      }
      espera = window.setTimeout(consultar, 1500);
    };
    consultar();
    const tic = window.setInterval(() => setAhora(Date.now()), 1000);
    return () => { vivo = false; clearTimeout(espera); clearInterval(tic); };
  }, [id, listo]);

  const { pct, etapa } = trabajo ? progreso(trabajo) : { pct: 1, etapa: "Conectando…" };
  const segundos = trabajo ? (trabajo.estado === "error" ? trabajo.actualizado - trabajo.creado : ahora / 1000 - trabajo.creado) : 0;
  const fallo = trabajo?.estado === "error";
  return (
    <div className="tarjeta progreso">
      <div className="progreso-cabeza">
        <div>
          <span className="sobretitulo">{fallo ? "No se pudo procesar" : "Procesando en AWS"}</span>
          <h2>{titulo}</h2>
        </div>
        <span className="reloj">{reloj(Math.max(0, segundos))}</span>
      </div>
      {!fallo && (
        <>
          <div className="barra-progreso"><div style={{ width: `${pct}%` }} /></div>
          <div className="progreso-etapa"><span className="giro" /> {etapa} <b>{Math.round(pct)}%</b></div>
        </>
      )}
      {fallo && <div className="alerta error">{trabajo!.error}</div>}
      {error && !fallo && <div className="alerta aviso">Reintentando la consulta del estado: {error}</div>}
      <ol className="bitacora">
        {(trabajo?.log ?? []).slice(-14).map((l, i) => (
          <li key={i}><span className="t">{reloj(l.t)}</span>{l.m.replace(/^\[[^\]]+\] /, "")}</li>
        ))}
      </ol>
      <div className="fila-controles">
        <button className="boton" onClick={volver}>{fallo ? "Volver" : "Cancelar y volver"}</button>
        {!fallo && <span className="dato">Puedes dejar esta pestaña abierta: el servidor sigue aunque cierres la página, y este enlace muestra el resultado al terminar.</span>}
      </div>
    </div>
  );
}

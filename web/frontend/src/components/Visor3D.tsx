import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import type { Malla } from "../lib/caso";

export interface MallaVisible {
  clave: string;
  malla: Malla;
  color: string;
  opacidad: number;
  nombre: string;
}

export interface Rotulo3D {
  pos: [number, number, number]; // x, y, z en mm
  texto: string;
  color: string;
  indice: number;
}

interface Props {
  mallas: MallaVisible[];
  rotulos?: Rotulo3D[];
  marcadores?: { pos: [number, number, number]; texto: string }[];
  vista: string; // al cambiar, la cámara vuelve a la vista anterior (cabeza arriba)
  alto?: number;
}

// Geometrías por malla: se arman una vez (los vértices vienen z,y,x y se pasan a x,y,z).
const geometrias = new WeakMap<Malla, THREE.BufferGeometry>();

function geometria(malla: Malla): THREE.BufferGeometry {
  let g = geometrias.get(malla);
  if (!g) {
    const n = malla.verts.length / 3;
    const pos = new Float32Array(malla.verts.length);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = malla.verts[i * 3 + 2];
      pos[i * 3 + 1] = malla.verts[i * 3 + 1];
      pos[i * 3 + 2] = malla.verts[i * 3];
    }
    g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(malla.faces, 1));
    g.computeVertexNormals();
    g.computeBoundingBox();
    geometrias.set(malla, g);
  }
  return g;
}

export default function Visor3D({ mallas, rotulos = [], marcadores = [], vista, alto = 620 }: Props) {
  const caja = useRef<HTMLDivElement>(null);
  const estado = useRef<{
    renderer: THREE.WebGLRenderer;
    etiquetas: CSS2DRenderer;
    escena: THREE.Scene;
    camara: THREE.PerspectiveCamera;
    controles: OrbitControls;
    grupo: THREE.Group;
    vista: string | null;
    pedir: () => void;
  } | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; texto: string } | null>(null);

  // Escena, cámara y renderizadores: una vez por montaje.
  useEffect(() => {
    const div = caja.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    div.appendChild(renderer.domElement);
    const etiquetas = new CSS2DRenderer();
    etiquetas.domElement.className = "capa-rotulos";
    div.appendChild(etiquetas.domElement);

    const escena = new THREE.Scene();
    const camara = new THREE.PerspectiveCamera(35, 1, 1, 10000);
    camara.up.set(0, 0, 1);
    escena.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.4));
    const luz = new THREE.DirectionalLight(0xffffff, 1.9);
    luz.position.set(0.4, -1, 0.8);
    camara.add(luz);
    escena.add(camara);
    const grupo = new THREE.Group();
    escena.add(grupo);
    const controles = new OrbitControls(camara, renderer.domElement);
    controles.enableDamping = true;
    controles.dampingFactor = 0.12;

    let cuadro = 0;
    const dibujar = () => {
      cuadro = 0;
      const moviendo = controles.update();
      renderer.render(escena, camara);
      etiquetas.render(escena, camara);
      if (moviendo) pedir();
    };
    const pedir = () => {
      if (!cuadro) cuadro = requestAnimationFrame(dibujar);
    };
    controles.addEventListener("change", pedir);

    const ajustar = () => {
      const w = div.clientWidth;
      const h = div.clientHeight;
      renderer.setSize(w, h);
      etiquetas.setSize(w, h);
      camara.aspect = w / Math.max(h, 1);
      camara.updateProjectionMatrix();
      pedir();
    };
    const observador = new ResizeObserver(ajustar);
    observador.observe(div);
    estado.current = { renderer, etiquetas, escena, camara, controles, grupo, vista: null, pedir };
    ajustar();

    // Nombre del fragmento bajo el puntero.
    const rayo = new THREE.Raycaster();
    const puntero = new THREE.Vector2();
    let ultimo = 0;
    const mover = (e: PointerEvent) => {
      if (e.buttons || performance.now() - ultimo < 50) return;
      ultimo = performance.now();
      const r = renderer.domElement.getBoundingClientRect();
      puntero.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      rayo.setFromCamera(puntero, camara);
      const golpe = rayo.intersectObjects(grupo.children, false).find((g) => g.object.userData.nombre);
      setHover(golpe ? { x: e.clientX - r.left, y: e.clientY - r.top, texto: golpe.object.userData.nombre } : null);
    };
    const salir = () => setHover(null);
    renderer.domElement.addEventListener("pointermove", mover);
    renderer.domElement.addEventListener("pointerleave", salir);

    return () => {
      cancelAnimationFrame(cuadro);
      observador.disconnect();
      controles.dispose();
      renderer.domElement.removeEventListener("pointermove", mover);
      renderer.domElement.removeEventListener("pointerleave", salir);
      renderer.dispose();
      div.innerHTML = "";
      estado.current = null;
    };
  }, []);

  // Contenido: mallas, marcadores y rótulos.
  useEffect(() => {
    const s = estado.current;
    if (!s) return;
    s.grupo.children.slice().forEach((hijo) => {
      s.grupo.remove(hijo);
      if (hijo instanceof THREE.Mesh) (hijo.material as THREE.Material).dispose();
      if (hijo instanceof CSS2DObject) hijo.element.remove();
    });
    const limites = new THREE.Box3();
    for (const m of mallas) {
      const g = geometria(m.malla);
      const transparente = m.opacidad < 0.999;
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(m.color),
        roughness: 0.55,
        metalness: 0.05,
        flatShading: true,
        transparent: transparente,
        opacity: m.opacidad,
        depthWrite: !transparente,
        side: transparente ? THREE.FrontSide : THREE.DoubleSide,
      });
      const malla = new THREE.Mesh(g, material);
      malla.userData.nombre = m.nombre;
      malla.renderOrder = transparente ? 2 : 1;
      s.grupo.add(malla);
      limites.union(g.boundingBox!);
    }
    const esfera = new THREE.SphereGeometry(1.6, 12, 8);
    for (const mk of marcadores) {
      const punto = new THREE.Mesh(esfera, new THREE.MeshBasicMaterial({ color: 0x111111 }));
      punto.position.set(...mk.pos);
      punto.userData.nombre = mk.texto;
      punto.renderOrder = 3;
      s.grupo.add(punto);
    }
    for (const r of rotulos) {
      // Desplazamiento en pantalla como las anotaciones de Plotly del dashboard, para no apilarlos.
      const ax = -46 + 28 * (r.indice % 3);
      const ay = -48 - 16 * (r.indice % 4);
      const div = document.createElement("div");
      div.className = "rotulo3d";
      div.innerHTML = `<svg width="1" height="1"><line x1="0" y1="0" x2="${ax}" y2="${ay}" /></svg>
        <span style="transform: translate(calc(-50% + ${ax}px), calc(-100% + ${ay}px)); border-color:${r.color}">${r.texto}</span>`;
      const objeto = new CSS2DObject(div);
      objeto.position.set(...r.pos);
      objeto.center.set(0, 0);
      s.grupo.add(objeto);
    }
    if (s.vista !== vista && !limites.isEmpty()) {
      s.vista = vista;
      encuadrar(s.camara, s.controles, limites);
    }
    s.pedir();
  }, [mallas, rotulos, marcadores, vista]);

  const reencuadrar = () => {
    const s = estado.current;
    if (!s) return;
    const limites = new THREE.Box3();
    s.grupo.children.forEach((h) => h instanceof THREE.Mesh && h.geometry.boundingBox && limites.union(h.geometry.boundingBox));
    if (!limites.isEmpty()) encuadrar(s.camara, s.controles, limites);
    s.pedir();
  };

  return (
    <div className="visor3d" style={{ height: alto }}>
      <div ref={caja} className="visor3d-lienzo" />
      <button className="boton-flotante" onClick={reencuadrar} title="Volver a la vista anterior">
        ⟲ Vista inicial
      </button>
      <div className="ayuda-flotante">Arrastra para rotar · rueda para acercar · clic derecho para mover</div>
      {hover && (
        <div className="tooltip3d" style={{ left: hover.x + 14, top: hover.y + 10 }}>
          {hover.texto}
        </div>
      )}
    </div>
  );
}

/** Vista anterior con la cabeza arriba, como la cámara de mesh_figure (eye 0, -2.1, 0.55). */
function encuadrar(camara: THREE.PerspectiveCamera, controles: OrbitControls, limites: THREE.Box3) {
  const centro = limites.getCenter(new THREE.Vector3());
  const radio = limites.getSize(new THREE.Vector3()).length() / 2;
  const distancia = radio / Math.sin(THREE.MathUtils.degToRad(camara.fov / 2)) * 0.95;
  const dir = new THREE.Vector3(0, -2.1, 0.55).normalize();
  camara.position.copy(centro).addScaledVector(dir, distancia);
  camara.near = distancia / 100;
  camara.far = distancia * 10;
  camara.updateProjectionMatrix();
  controles.target.copy(centro);
  controles.update();
}

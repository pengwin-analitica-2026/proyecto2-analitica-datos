# Página web en AWS (alternativa al dashboard de Streamlit)

El dashboard de Streamlit Community Cloud procesa los CT dentro de su propio proceso, con ~2.7 GB de
memoria: un `.mha` grande subido en vivo (como el del caso 044) lo puede tumbar. Esta página hace lo mismo
que el dashboard, pero el trabajo pesado corre en una Lambda propia con **10 GB y 15 minutos**, y el
navegador solo dibuja. **El dashboard de Streamlit y la Lambda `pelvis-api` no cambian**: las dos versiones
conviven.

```
Navegador (celular o computador)
   │  página, cortes PNG, mallas e IDs (estáticos)          │  API: casos, trabajos, corte en vivo, GPU
   ▼                                                         ▼
CloudFront ──► S3 pengwin-web-<cuenta>  ◄──── escribe ──── Lambda pelvis-web (10 GB, 15 min, web/servidor.py)
                 index.html, assets/                          │  lee crudos/ y exportados/
                 resultados/<modelo>/...                       ├──► S3 pengwin-ct-<cuenta> (los 100 casos)
                 subidas/ (CT subidos, 1 día)                  └──► Lambda pelvis-api ──► GPU SageMaker (si está encendida)
                 trabajos/ (estado, 7 días)
```

## Qué hace igual que el dashboard

| Dashboard de Streamlit | Página web |
|---|---|
| Origen: casos de ejemplo, casos PENGWIN (100), subir un CT | Igual. Los 100 casos, con filtro por split, precalculado o «procesar ahora» |
| Procesamiento: dónde corre el modelo (CPU / GPU), precisión | CPU de la Lambda `pelvis-web` (int8) o GPU de SageMaker si está encendida |
| Panel de la GPU con clave de administración | Igual (pasa por `pelvis-api`, que tiene los permisos) |
| Volumen mínimo del fragmento, leyenda, detalles del caso | Igual |
| Preprocesamiento paso a paso | Igual |
| 1 · Volumen crudo: MIP e isosuperficie (pulida / cruda) | Igual, con three.js |
| 2 · Corte por corte: máscaras, cajas refinadas o crudas, GT, presencia por región, tabla, inferencia en vivo | Igual. El corte se dibuja en el navegador (`lib/corte.ts` reproduce `render_slice`); además rueda del mouse y flechas |
| 3 · Reconstrucción 3D con rótulos `F2 · 19.0 mm` | Igual, con three.js y el nombre del fragmento al pasar el mouse |
| Métricas y latencia | Igual (`datos/resumen.json`, que arma `paquete_web.resumen_proyecto`) |

Además, la dirección guarda dónde estás (`#/caso/...`): recargar o compartir el enlace abre el mismo caso.

## Cómo funciona

1. **Procesar** (un caso «procesar ahora» o un CT subido) crea un **trabajo**: la Lambda se invoca a sí
   misma de forma asíncrona y el Lambda Web Adapter entrega el evento en `POST /events`. Ahí corre
   `demo_export` completo (orientación, giro axial, modelo, post-proceso 3D, distancias, mallas, vista
   cruda). El estado y la bitácora quedan en `trabajos/<id>.json`; la página los consulta cada 1.5 s y
   muestra la barra de progreso.
2. **El CT se sube directo a S3** con un formulario firmado (hasta 1 GB, `content-length-range`), sin pasar
   por la Lambda, que acepta como mucho 6 MB por pedido.
3. `src/paquete_web.py` convierte la carpeta de `demo_export` en un **paquete web** (`caso.json`, un PNG por
   corte, IDs y mallas en binario comprimido) en `resultados/<modelo>/...`. `caso.json` se escribe al final:
   si existe, el paquete está completo. Un caso ya procesado con la misma configuración no se repite.
4. El navegador baja el paquete de CloudFront: los cortes se piden a medida que se ven, así el caso abre
   rápido también en un celular.

## Desplegar

Requisitos: AWS CLI con el perfil `almia`, Docker, Node 20+ y Python con las dependencias del repo.

```bash
AWS_PROFILE=almia bash deploy/aws/web.sh todo
```

`preparar` (una vez) crea el bucket `pengwin-web-<cuenta>` (privado, solo CloudFront lo lee, y solo
`index.html`, `assets/`, `resultados/` y `datos/`), el repositorio ECR `pelvis-web`, el rol
`pelvis-web-lambda-exec` (`deploy/aws/web-lambda-policy.json`: escribir su bucket, leer `crudos/` y
`exportados/` del bucket de PENGWIN e invocarse a sí misma), la función `pelvis-web`, su Function URL (CORS
solo para la página) y la distribución de CloudFront. Toma `INFERENCE_URL` e `INFERENCE_KEY` de
`pelvis-api` para la GPU: la clave queda en la Lambda, nunca en el navegador.

Después, cuando cambie el código:

```bash
AWS_PROFILE=almia bash deploy/aws/web.sh imagen   # backend (web/servidor.py, src/)
AWS_PROFILE=almia bash deploy/aws/web.sh pagina   # frontend (web/frontend) y casos de ejemplo
AWS_PROFILE=almia bash deploy/aws/web.sh estado   # direcciones
```

## Desarrollo en el Mac

```bash
cd web/frontend && npm install && npm run build && cd ../..
AWS_PROFILE=almia WEB_DATA_DIR=/tmp/pengwin_web uvicorn web.servidor:app --port 8080
python src/paquete_web.py outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/* \
    --destino /tmp/pengwin_web/resultados/h8_aumm_noise01_noema_w32_e60_s42/ejemplos \
    --resumen /tmp/pengwin_web/datos/resumen.json
```

Sin `WEB_BUCKET` todo queda en `WEB_DATA_DIR`, los trabajos corren en un hilo y la misma API sirve la página
en `http://localhost:8080`. Para iterar en el frontend, `npm run dev` (Vite en el 5173, con proxy a la API).

## Costos y límites

- Lambda cobra por uso: un CT de ~400 cortes tarda unos 2-3 minutos con 10 GB (~0.02-0.03 USD). Sin
  visitas no cuesta nada. El arranque en frío (cargar torch) suma unos segundos al primer pedido.
- S3 y CloudFront: centavos al mes con el uso de una demo. Los CT subidos se borran al día; sus resultados,
  a la semana.
- Un trabajo que pasa de 15 minutos o de 10 GB se corta; la página lo informa como error.

## Al terminar el curso

```bash
AWS_PROFILE=almia bash deploy/aws/web.sh borrar
```

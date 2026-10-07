# CI/CD con GitHub Actions

Estado al 6 de octubre de 2026: los dos flujos ya corrieron en GitHub y el despliegue a AWS funciona
desde `main` (primer despliegue real tras integrar el PR #6).

```
PR a main     ──► ci.yml ──► pruebas
push a main   ──► ci.yml ──► pruebas
              └─► deploy-api.yml ──► pruebas ──► imagen Docker ──► ECR ──► Lambda pelvis-api ──► /health?deep=1
push a main   ──► Streamlit Community Cloud vuelve a desplegar app/dashboard.py (lo hace Streamlit, no Actions)
```

| Archivo | Cuándo corre | Qué hace |
|---|---|---|
| `ci.yml` | En cada PR hacia `main`, en cada push a `main` y a mano | Instala dependencias (torch para CPU, `fastapi`, `httpx`, `pytest`) y corre toda la suite de pruebas |
| `deploy-api.yml` | En cada push a `main` y a mano desde Actions | Corre las pruebas, construye `api/Dockerfile`, sube la imagen a ECR con el SHA del commit como etiqueta, actualiza la función Lambda y hace una prueba de humo a `/health?deep=1` (carga el modelo) |

`deploy/workflows/` guarda una copia de los dos archivos para quien quiera llevarlos a otro repositorio.

## Autenticación: OIDC, sin claves de AWS en GitHub

El flujo no usa `AWS_ACCESS_KEY_ID` ni `AWS_SECRET_ACCESS_KEY`. GitHub emite un token OIDC por
ejecución y AWS lo cambia por credenciales temporales del rol `github-actions-pelvis-api`
(`permissions: id-token: write` en el workflow).

- Cuenta de AWS: la de Almia (`050752608083`), región `us-east-2`. El proveedor OIDC
  `token.actions.githubusercontent.com` ya existía en la cuenta.
- Política de confianza (`deploy/aws/trust-policy.json`): solo acepta tokens con
  `sub = repo:pengwin-analitica-2026@338435702/proyecto2-analitica-datos@1373596198:ref:refs/heads/main`.
  Este repositorio usa el formato **inmutable** del `sub` (con los IDs numéricos de la organización
  y del repo). Con el formato por nombre (`repo:pengwin-analitica-2026/...`) el primer despliegue
  falló con `Not authorized to perform sts:AssumeRoleWithWebIdentity`. Se ve con
  `gh api repos/pengwin-analitica-2026/proyecto2-analitica-datos/actions/oidc/customization/sub`.
- Permisos del rol (`deploy/aws/deploy-policy.json`): subir imágenes solo al repositorio ECR
  `pelvis-api` y actualizar solo la función `pelvis-api`. Nada más.
- Una rama distinta de `main` o un fork no pueden asumir el rol.

**Variables del repositorio** (Settings → Secrets and variables → Actions → Variables; no son secretas):

| Nombre | Valor |
|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::050752608083:role/github-actions-pelvis-api` |
| `AWS_REGION` | `us-east-2` |
| `ECR_REPOSITORY` | `pelvis-api` |
| `LAMBDA_FUNCTION` | `pelvis-api` |

Lo que NO va en GitHub:

- `API_KEY` vive en la configuración de la función Lambda (variable de entorno).
- `INFERENCE_URL` e `INFERENCE_KEY` viven en los Secrets de Streamlit Cloud.

## El checkpoint

La imagen copia `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth` (11 MB), así que tiene que
estar en el repositorio. `.gitignore` ignora `*.pth` y tiene excepciones solo para las tres semillas
de H6 y para D5 (la primera evaluación en test).

## Problemas que aparecieron al desplegar de verdad

| Síntoma | Causa | Arreglo |
|---|---|---|
| `/health` respondía 502 y el log decía `PermissionError: '/app/api/server.py'` | Los archivos del Mac tienen permisos `600`; Lambda ejecuta el contenedor con un usuario sin privilegios | `RUN chmod -R a+rX /app` en `api/Dockerfile` |
| El arranque en frío se cortaba a los 10 s | El adaptador de Lambda espera a que `/health` responda, y cargar torch y el modelo tarda más | `/health` responde al instante; `/health?deep=1` carga el modelo (lo usa la prueba de humo) |
| `ci.yml` fallaba con `No module named 'pytest'` | `tests/test_orientacion.py` usa `pytest` | Se instala `pytest` en los dos workflows |
| `Not authorized to perform sts:AssumeRoleWithWebIdentity` | Formato inmutable del `sub` (ver arriba) | Política de confianza con el `sub` inmutable |

## Cuidados

- **Todo push a `main` despliega.** No hay filtro por rutas: un cambio solo de documentación también
  reconstruye la imagen (unos 8 minutos). Es intencional: `main` y la Lambda nunca quedan desfasadas.
- **`main` protegida.** El enunciado pide merges con revisión por pares. Pendiente de activar en
  Settings → Branches: requerir PR con una aprobación antes de integrar.
- **Al terminar el curso:** borrar la función Lambda, el repositorio ECR y los dos roles
  (`github-actions-pelvis-api`, `pelvis-api-lambda-exec`), o al menos rotar `API_KEY`.

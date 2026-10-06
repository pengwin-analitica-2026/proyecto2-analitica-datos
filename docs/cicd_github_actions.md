# CI/CD con GitHub Actions

Dos flujos en `.github/workflows/`:

| Archivo | Cuándo corre | Qué hace |
|---|---|---|
| `ci.yml` | En cada push y pull request | Instala dependencias (torch para CPU) y corre toda la suite de pruebas (83 pruebas, ~11 s en local) |
| `deploy-api.yml` | Al integrar en `main` cambios de `api/`, `src/` o del checkpoint, o a mano desde Actions | Corre las pruebas, construye la imagen de `api/Dockerfile`, la sube a ECR, actualiza la función Lambda y hace una prueba de humo a `/health` |

El dashboard no necesita pipeline propio: Streamlit Community Cloud se vuelve a desplegar solo cada
vez que se hace push a la rama que tiene conectada.

## Dónde van los secretos

En GitHub: repositorio → Settings → Secrets and variables → Actions.

**Secrets** (pestaña "Secrets", New repository secret):

| Nombre | Valor |
|---|---|
| `AWS_ACCESS_KEY_ID` | Clave de acceso de un usuario IAM creado solo para el despliegue |
| `AWS_SECRET_ACCESS_KEY` | Su clave secreta |

**Variables** (pestaña "Variables"; no son secretas):

| Nombre | Ejemplo |
|---|---|
| `AWS_REGION` | `us-east-1` |
| `ECR_REPOSITORY` | `pelvis-api` |
| `LAMBDA_FUNCTION` | `pelvis-api` |

Otras cosas que NO van en GitHub:

- `API_KEY` vive en la configuración de la función Lambda (variable de entorno).
- `INFERENCE_URL` e `INFERENCE_KEY` viven en los Secrets de Streamlit Cloud.

Además, Settings → Environments → New environment → `aws`. El workflow usa ese entorno, así que
ahí puedes exigir aprobación manual antes de cada despliegue y limitarlo a la rama `main`.

## Permisos mínimos del usuario IAM

Crear un usuario IAM solo para esto (sin acceso a la consola) con esta política. Reemplaza
`REGION` y `CUENTA`.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "arn:aws:ecr:REGION:CUENTA:repository/pelvis-api"
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:UpdateFunctionCode", "lambda:GetFunction", "lambda:GetFunctionConfiguration",
        "lambda:GetFunctionUrlConfig"
      ],
      "Resource": "arn:aws:lambda:REGION:CUENTA:function:pelvis-api"
    }
  ]
}
```

## Cuidados

- **Repositorio compartido.** El remoto actual es `mariavalencia30/proyecto2-analitica-datos`.
  Solo el dueño o un administrador puede crear secretos allí. Y cualquier colaborador con permiso de
  escritura puede escribir un workflow que lea esos secretos. Si no todos los del equipo son de
  confianza para tener acceso a AWS, hay dos salidas: usar el entorno `aws` con revisores
  obligatorios, o poner el despliegue en un repositorio propio.
- **Claves de larga vida.** Las claves de acceso son la opción simple. La alternativa más segura es
  OIDC (GitHub asume un rol de IAM sin guardar claves); se puede migrar después cambiando solo el
  paso `configure-aws-credentials`.
- **Rota o borra** las claves cuando termine el curso.
- **El checkpoint** `outputs/s10/d5_ema_w32_e60_s42.pth` (11 MB) tiene que estar en el repositorio,
  porque la imagen lo copia. `.gitignore` ignora `*.pth`; por eso se agregó una excepción solo para
  ese archivo.

## Primera vez (manual)

El workflow solo actualiza una función que ya existe. La primera imagen y la función se crean a
mano siguiendo `docs/despliegue_aws_streamlit.md`, sección 1. Después, cada push a `main` que toque
la API la actualiza sola.

## Sin verificar

Los workflows no se han ejecutado en GitHub. Las pruebas que corren ahí sí las ejecuté en un Python
3.11 limpio con `requirements.txt` (83 pruebas, todas pasan), pero la instalación de torch para CPU
en los runners de GitHub, la construcción de la imagen y el despliegue a Lambda quedan por probar.

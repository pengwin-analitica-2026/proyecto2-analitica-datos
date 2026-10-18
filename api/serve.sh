#!/bin/sh
# Punto de entrada de la imagen GPU. SageMaker arranca el contenedor con el argumento «serve»; se
# ignora y se levanta el mismo servidor de siempre en el puerto 8080 (/ping y /invocations incluidos).
exec uvicorn api.server:app --host 0.0.0.0 --port 8080

# Uso de Docker

Este documento define el contexto de ejecución con Docker.

## Requisitos

- Docker Desktop instalado (modo Linux containers)
- Dataset descargado desde Kaggle
- Dataset completo disponible en `data/training_set/` (ruta por defecto del proyecto)

Si tu dataset está en otra ubicación, actualiza la variable de ruta en el script de ejecución.

## Estructura de trabajo

Entradas:

- `data/training_set/` (dataset completo)
- `data/training_smoke/` (dataset reducido para modo desarrollo (smoke))

Salidas:

- `model/` y `outputs/` (flujo completo)
- `model_smoke/` y `outputs_smoke/` (flujo smoke/desarrollo)

## Orden recomendado de ejecución

1. Construir imagen Docker (`build`)
2. Preparar dataset smoke (`smoke`)
3. Iterar en modo desarrollo (smoke) (`train-dev` / `run-dev`)
4. Ejecutar validación completa (`train` / `run`)
5. Limpiar artefactos cuando corresponda (`clean`)

La guía paso a paso está en `docs/04_run_script.md`.

## Compatibilidad de scripts

El flujo principal del equipo está documentado con `run.sh` (Git Bash).
También existen equivalentes en PowerShell: `run.ps1` y `scripts/create_smoke.ps1`.

## Resultado esperado

Tras ejecutar la generación de predicciones (inferencia) completa, en `outputs/` se genera un `demographics.csv` con:

- Columnas originales
- `Cognitive_Impairment`
- `Cognitive_Impairment_Probability`
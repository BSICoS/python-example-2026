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

Salidas:

- `model/` (modelo entrenado)
- `outputs_supplementary/` (comprobación suplementaria opcional)

## Orden recomendado de ejecución

1. Construir imagen Docker (`build`)
2. Iterar en modo desarrollo con datasets completos (`train-dev`)
3. Entrenar el modelo (`train`)
4. Limpiar artefactos cuando corresponda (`clean`)

La guía paso a paso está en `docs/04_run_script.md`.

## Compatibilidad de scripts

El flujo principal del equipo está documentado con `run.sh` (Git Bash).
También existe un equivalente en PowerShell: `run.ps1`.

## Resultado esperado

Tras ejecutar la comprobación suplementaria, en `outputs_supplementary/` se genera un `demographics.csv` con:

- Columnas originales
- `Cognitive_Impairment`
- `Cognitive_Impairment_Probability`

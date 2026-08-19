# Script unificado de ejecución (`run.ps1`)

Este documento es la guía operativa única para ejecutar el proyecto.
Aquí se define el orden recomendado y los comandos asociados.

---

# Requisitos

- Docker Desktop instalado  
- Dataset descargado en:

```
data/training_set/
data/supplementary_set/
```

⚠️ Si el dataset está en otra ubicación, modificar las rutas correspondientes en `run.ps1`.

⚠️ Ejecutar los comandos desde PowerShell.

ℹ️ Para contexto general y definición de artefactos, ver `docs/02_docker.md`.
---

# Orden de ejecución recomendado

Desde la raíz del repositorio.

## 1) Preparar entorno

### Construir imagen Docker

```powershell
.\run.ps1 build
```

Ejecutar la primera vez y cada vez que cambien `requirements.txt` o `Dockerfile`.

## 2) Entrenamiento completo

### Entrenar sin reconstruir la imagen

```powershell
.\run.ps1 train-dev
```

Usa `data/training_set/` y guarda el modelo en `model/`.

### Comprobación opcional de compatibilidad suplementaria

```powershell
.\run.ps1 supplementary
```

Ejecuta la ruta oficial de inferencia de la imagen Docker sobre `data/supplementary_set/` y escribe predicciones en `outputs_supplementary/`. El set suplementario no es un conjunto de validación: sus predicciones no se evalúan ni se usan para selección de modelo.

## 3) Limpieza de artefactos

```powershell
.\run.ps1 clean
```

Elimina `model/` y `outputs_supplementary/`.
No elimina datasets.

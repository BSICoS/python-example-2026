# Script unificado de ejecución (`run.sh`)

Este documento es la guía operativa única para ejecutar el proyecto.
Aquí se define el orden recomendado y los comandos asociados.

---

# Requisitos

- Docker Desktop instalado  
- Dataset descargado en:

```
data/training_set/
```

⚠️ Si el dataset está en otra ubicación, modificar la variable `$FULL_DATA_REL`
dentro de `run.sh`.

⚠️ Ejecutar los comandos desde Git Bash.

ℹ️ Existen scripts equivalentes en PowerShell (`run.ps1` y `scripts/create_smoke.ps1`) para quienes prefieran ese entorno.

ℹ️ Para contexto general y definición de artefactos, ver `docs/02_docker.md` y `docs/03_smoke_dataset.md`.
---

# Orden de ejecución recomendado

Desde la raíz del repositorio.

## 1) Preparar entorno

### Construir imagen Docker

```bash
./run.sh build
```

Ejecutar la primera vez y cada vez que cambien `requirements.txt` o `Dockerfile`.

### Crear dataset smoke (5 sujetos)

```bash
./run.sh smoke
```

Genera `data/training_smoke/`.

## 2) Ciclo en modo desarrollo (smoke)

### Entrenar en modo desarrollo (smoke)

```bash
./run.sh train-dev
```

Usa `data/training_smoke/` y guarda modelo en `model_smoke/`.

### Generar predicciones (inferencia) en modo desarrollo (smoke)

```bash
./run.sh run-dev
```

Genera resultados en `outputs_smoke/`.

### Secuencia típica en modo desarrollo (smoke)

```bash
./run.sh build        # solo la primera vez
./run.sh smoke        # solo si no existe
./run.sh train-dev
./run.sh run-dev
```

## 3) Validación completa

### Entrenar con dataset completo

```bash
./run.sh train
```

Guarda el modelo en `model/`.

### Generar predicciones (inferencia) completas

```bash
./run.sh run
```

Genera resultados en `outputs/`.

## 4) Limpieza de artefactos

```bash
./run.sh clean
```

Elimina `model/`, `model_smoke/`, `outputs/` y `outputs_smoke/`.
No elimina datasets.

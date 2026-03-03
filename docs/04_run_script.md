# Script unificado de ejecución (`run.ps1`)

Para simplificar el trabajo del equipo hemos creado un único script que encapsula todos los comandos necesarios para:

- Construir la imagen Docker
- Crear el dataset smoke
- Entrenar (completo o smoke)
- Generar predicciones
- Limpiar artefactos

---

## Requisitos

- Docker Desktop instalado
- Dataset descargado en: `data/training_set/`


⚠️ Si el dataset está en otra ubicación, modificar la variable `$FULL_DATA` dentro de `run.ps1`.

---

# Comandos disponibles

Nota: Si PowerShell bloquea la ejecución, ejecutar primero:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Desde la raíz del repositorio:

## 1️⃣ Construir la imagen Docker

```powershell
.\scripts\run.ps1 build
```

Solo es necesario hacerlo:

- La primera vez
- Cuando cambien dependencias o el Dockerfile

## 2️⃣ Crear dataset smoke (5 sujetos)

```powershell
.\scripts\run.ps1 smoke
```

Genera: `data/training_smoke/`

Este dataset se usa para desarrollo rápido.

## 3️⃣ Entrenar modelo

### Smoke (rápido)

```powershell
.\scripts\run.ps1 train-smoke
```


### Completo

```powershell
.\scripts\run.ps1 train
```

El modelo se guarda en:

- model/          (full)
- model_smoke/    (smoke)

## 4️⃣ Generar predicciones

### Smoke

```powershell
.\scripts\run.ps1 run-smoke
```

### Completo

```powershell
.\scripts\run.ps1 run
```

Los resultados se generan en:

- outputs/
- outputs_smoke/

El archivo clave es: `demographics.csv` que contiene las predicciones añadidas.

## 5️⃣ Limpiar artefactos

```powershell
.\scripts\run.ps1 clean
```

Elimina:

- model/
- model_smoke/
- outputs/
- outputs_smoke/

No elimina datasets.

# Flujo recomendado para desarrollo

```powershell
.\scripts\run.ps1 build
.\scripts\run.ps1 smoke
.\scripts\run.ps1 train-smoke
.\scripts\run.ps1 run-smoke
```

Solo cuando el modelo esté estable:

```powershell
.\scripts\run.ps1 train
.\scripts\run.ps1 run
```
# Script unificado de ejecución (`run.ps1`)

Este script centraliza todos los comandos necesarios para trabajar en el proyecto:

- Construir la imagen Docker  
- Crear el dataset smoke  
- Entrenar (modo desarrollo o completo)  
- Generar predicciones  
- Limpiar artefactos  

---

# Requisitos

- Docker Desktop instalado  
- Dataset descargado en:

```
data/training_set/
```

⚠️ Si el dataset está en otra ubicación, modificar la variable `$FULL_DATA_REL`
dentro de `run.ps1`.

⚠️ Si PowerShell bloquea la ejecución de scripts, ejecutar (aplica solo para la sesión actual):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
---

# Comandos disponibles

Desde la raíz del repositorio:

---

## 1️⃣ Construir la imagen Docker

```powershell
.\run.ps1 build
```

Solo es necesario hacerlo:

- La primera vez  
- Cuando cambien `requirements.txt`  
- Cuando cambie el `Dockerfile`  

No es necesario al modificar `team_code.py` en modo desarrollo.

---

## 2️⃣ Crear dataset smoke (5 sujetos)

```powershell
.\run.ps1 smoke
```

Genera:

```
data/training_smoke/
```

Este dataset se utiliza exclusivamente para desarrollo rápido.

---

# 🚀 Modo desarrollo (rápido)

Estos comandos:

- Usan el dataset smoke  
- Montan el código como volumen  
- No requieren rebuild al modificar Python  

---

## 3️⃣ Entrenar en modo desarrollo

```powershell
.\run.ps1 train-dev
```

Utiliza:

- `data/training_smoke`  
- `model_smoke/`  

---

## 4️⃣ Generar predicciones en modo desarrollo

```powershell
.\run.ps1 run-dev
```

Genera resultados en:

```
outputs_smoke/
```

---

## 🔁 Flujo recomendado de desarrollo

```powershell
.\run.ps1 build        # solo la primera vez
.\run.ps1 smoke        # solo si no existe
.\run.ps1 train-dev
.\run.ps1 run-dev
```

Este flujo debe usarse para:

- Probar nuevas features  
- Ajustar el modelo  
- Depurar errores  
- Iterar rápidamente  

---

# 🧪 Entrenamiento completo

Solo cuando el modelo esté estable.

---

## 5️⃣ Entrenar con dataset completo

```powershell
.\run.ps1 train
```

Guarda el modelo en:

```
model/
```

---

## 6️⃣ Generar predicciones completas

```powershell
.\run.ps1 run
```

Genera resultados en:

```
outputs/
```

---

# 🧹 Limpiar artefactos

```powershell
.\run.ps1 clean
```

Elimina:

- `model/`  
- `model_smoke/`  
- `outputs/`  
- `outputs_smoke/`  

No elimina datasets.

# Estrategia recomendada del equipo

1. Desarrollar siempre en modo `*-dev`.  
2. Entrenar en full solo antes de:
   - Hacer merge a `main`
   - Generar submission  
3. Antes de enviar al challenge:
   - Ejecutar `build`
   - Ejecutar `train`
   - Ejecutar `run`
   - Verificar que funciona sin modo dev  

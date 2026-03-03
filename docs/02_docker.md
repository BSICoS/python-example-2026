# Uso de Docker

## Requisitos

- Docker Desktop instalado (modo Linux containers)
- Dataset descargado desde Kaggle
- Se asume que el dataset está en: data/training_set/

Cada miembro del equipo puede tener el dataset en una ubicación diferente, pero en este repositorio asumimos que está dentro de `data/`.

---

## Construir la imagen

Desde la raíz del repositorio:

```powershell
docker build -t cinc2026 .
```

## Entenar con el dataset completo

```powershell
$DATA="data/training_set"
$MODEL="$PWD/model"

docker run --rm `
  -v "${DATA}:/challenge/training_data:ro" `
  -v "${MODEL}:/challenge/model" `
  cinc2026 `
  python train_model.py -d training_data -m model -v
```

## Generar predicciones

```powershell
$OUT="$PWD/outputs"

docker run --rm `
  -v "${DATA}:/challenge/holdout_data:ro" `
  -v "${MODEL}:/challenge/model:ro" `
  -v "${OUT}:/challenge/holdout_outputs" `
  cinc2026 `
  python run_model.py -d holdout_data -m model -o holdout_outputs -v
```

## Resultado esperado

En la carpeta `outputs/` se generará un `demographics.csv` con:

- Columnas originales
- Cognitive_Impairment
- Cognitive_Impairment_Probability
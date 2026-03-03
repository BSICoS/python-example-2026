# Dataset Smoke (Desarrollo Rápido)

Entrenar con el dataset completo tarda aproximadamente 30–40 minutos con el modelo de ejemplo.

Para desarrollo utilizamos un dataset reducido (5 sujetos).

---

## Crear dataset smoke

Ejecutar:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_smoke.ps1
```

Esto generará: `data/training_smoke/`

## Entrenar con el smoke dataset

```powershell
$DATA="$PWD\data\training_smoke"
$MODEL="$PWD\model_smoke"

docker run --rm `
  -v "${DATA}:/challenge/training_data:ro" `
  -v "${MODEL}:/challenge/model" `
  cinc2026 `
  python train_model.py -d training_data -m model -v
```

## Generar predicciones con smoke dataset

```powershell
$OUT="$PWD/outputs_smoke"

docker run --rm `
  -v "${DATA}:/challenge/holdout_data:ro" `
  -v "${MODEL}:/challenge/model:ro" `
  -v "${OUT}:/challenge/holdout_outputs" `
  cinc2026 `
  python run_model.py -d holdout_data -m model -o holdout_outputs -v
```

## ¿Cuándo usar smoke?

- Desarrollo de nuevas features
- Comprobación rápida de que el código no rompe
- Validación de cambios en team_code.py

Nunca usar smoke para evaluar rendimiento final.
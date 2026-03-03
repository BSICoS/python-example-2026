# CINC 2026 – Visión General del Proyecto

Estamos participando en el Challenge 2026 de Computing in Cardiology.

El objetivo es predecir deterioro cognitivo a partir de datos de polisomnografía (PSG).

## Cómo nos evaluarán

La organización:

1. Construirá nuestra imagen Docker.
2. Ejecutará `train_model.py`.
3. Ejecutará `run_model.py`.
4. Evaluará las predicciones generadas.

Por tanto, la reproducibilidad mediante Docker es obligatoria.

Nuestro objetivo es garantizar que:
- El código se ejecuta sin intervención manual.
- El modelo se entrena correctamente.
- Las predicciones se generan en el formato requerido.

## Qué se puede modificar y qué no

❌ No modificar

- `train_model.py`
- `run_model.py`
- `helper_code.py`
- `evaluate_model.py`

✅ Modificar/Añadir

- `team_code.py` <-- Toda la lógica científica y de modelado debe implementarse ahí.
- Helpers, scripts, métodos: añadir a voluntad en `src/`
# ♻️ Clasificador de Reciclaje

Modelo de clasificación supervisada de residuos (MobileNetV2, transfer learning) desplegado con Streamlit.

**Clases:** cardboard, glass, metal, paper, plastic, trash

## Archivos
- `app.py` — aplicación Streamlit
- `modelo_reciclaje.keras` — modelo entrenado (generado desde el notebook)
- `class_names.json` — nombres de las clases
- `requirements.txt` — dependencias

## Ejecutar localmente
```
pip install -r requirements.txt
streamlit run app.py
```

## Notebook de entrenamiento
`Actividad_2__Clase_HectorMolina.ipynb` (Google Colab, MobileNetV2).

Autor: Hector Molina — UNAH

import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title=" Clasificador de Reciclaje ", page_icon="♻️")

IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Cargar modelo y clases una sola vez (cacheado)
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_reciclaje.keras")
    with open("class_names.json", "r") as f:
        clases = json.load(f)
    return modelo, clases

model, class_names = cargar_modelo()

st.title("♻️ Modelo predictivo de reciclaje clase de IA-ISC-campus Comayagua-2026-Hector Fortín")
st.write("Sube una foto de un residuo y el modelo predice su categoría.")
st.caption("Clases: " + ", ".join(class_names))

archivo = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)

    # Mismo preprocesamiento que en el entrenamiento
    img_redim = img.resize(IMG_SIZE)
    arr = np.array(img_redim, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]
    idx = int(np.argmax(pred))
    confianza = float(pred[idx]) * 100

    st.success(f"Predicción: **{class_names[idx]}**  ({confianza:.1f}%)")

    st.subheader("Probabilidades por clase")
    for nombre, prob in sorted(zip(class_names, pred), key=lambda x: -x[1]):
        st.write(f"{nombre}: {prob*100:.1f}%")
        st.progress(float(prob))

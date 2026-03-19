import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import joblib
import os

st.set_page_config(page_title="IA MNIST - Scikit-Learn", layout="centered")

# --- LÓGICA DE CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    """Carga el modelo pre-entrenado desde el archivo."""
    model_file = "mnist_mlp_model.pkl"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        return None

# --- CARGAMOS EL MODELO ---
model = load_model()

# --- INTERFAZ DE USUARIO ---
st.title("🔢 Clasificador de Dígitos (MLP)")

# Si el modelo no está cargado, mostramos un mensaje y detenemos la ejecución
if model is None:
    st.error("El archivo del modelo 'mnist_mlp_model.pkl' no fue encontrado.")
    st.warning("Por favor, ejecuta el script de entrenamiento primero:")
    st.code("python train_model.py")
    st.stop()


st.write("Dibuja un número claro y centrado:")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20, # Un poco más grueso ayuda a la IA
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# Solo procesar si hay dibujo
if canvas_result.image_data is not None:
    # Procesar la imagen dibujada
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L').resize((28, 28)) 
    
    # Preparar el vector para Scikit-Learn
    img_array = np.array(img).reshape(1, -1).astype("float32") / 255

    with col2:
        if st.button("Predecir con IA"):
            if model is not None:
                # Realizar predicción
                pred = model.predict(img_array)[0]
                probs = model.predict_proba(img_array)[0]
                
                st.header(f"Es un: {pred}")
                st.write(f"Confianza: {np.max(probs):.2%}")
                st.bar_chart(probs)
            else:
                st.error("El modelo no se cargó correctamente.")
import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import joblib
import os

st.set_page_config(page_title="IA MNIST - Scikit-Learn", layout="centered")

# --- LÓGICA DE ENTRENAMIENTO (MLP) ---
@st.cache_resource
def get_trained_model():
    model_file = "mnist_mlp_model.pkl"
    
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        with st.spinner("Entrenando IA... esto tardará solo un momento"):
            # Cargamos el dataset
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            
            # --- TRUCO DE VELOCIDAD ---
            # Usamos solo las primeras 10,000 imágenes para que sea rápido en Streamlit
            X = X[:10000] / 255.0
            y = y[:10000]
            
            mlp = MLPClassifier(
                hidden_layer_sizes=(100,), # Una sola capa es más rápida
                max_iter=15, 
                alpha=1e-4,
                solver='adam', 
                random_state=1
            )
            
            mlp.fit(X, y)
            joblib.dump(mlp, model_file)
            return mlp

# --- INTERFAZ DE USUARIO ---
st.title("🔢 Clasificador de Dígitos (MLP)")
st.write("Dibuja un número claro en el centro del recuadro:")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    # Procesar la imagen dibujada
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L') # Gris
    img = img.resize((28, 28)) # Tamaño MNIST
    
    # Convertir a vector plano (784 características) que es lo que espera sklearn
    img_array = np.array(img).reshape(1, -1).astype("float32") / 255

    with col2:
        if st.button("Predecir con IA"):
            # Predicción
            pred = model.predict(img_array)[0]
            probs = model.predict_proba(img_array)[0]
            
            st.header(f"Es un: {pred}")
            st.write(f"Confianza: {np.max(probs):.2%}")
            st.bar_chart(probs)
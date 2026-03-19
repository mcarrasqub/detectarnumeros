import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

# --- FUNCIONES DE IA ---
@st.cache_resource
def get_model():
    model_path = 'modelo_mnist.h5'
    
    # Si el modelo ya existe, lo carga. Si no, lo entrena.
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        with st.spinner("Entrenando modelo por primera vez..."):
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), _ = mnist.load_data()
            x_train = x_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
            
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5, verbose=0)
            model.save(model_path)
            return model

model = get_model()

# --- INTERFAZ ---
st.title("🔢 Clasificador de Dígitos (CNN)")
st.write("Dibuja un número en el recuadro negro:")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    # 1. Convertir imagen del canvas a formato MNIST (28x28, escala de grises)
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L') # Escala de grises
    img = img.resize((28, 28)) # Redimensionar
    
    # 2. Convertir a array y normalizar
    img_array = np.array(img).astype("float32") / 255
    img_array = img_array.reshape(1, 28, 28, 1)

    with col2:
        if st.button("Identificar Número"):
            prediction = model.predict(img_array)
            resultado = np.argmax(prediction)
            confianza = np.max(prediction)
            
            st.header(f"Es un: {resultado}")
            st.write(f"Confianza: {confianza:.2%}")
            st.bar_chart(prediction[0])
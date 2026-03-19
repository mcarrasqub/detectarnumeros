import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="IA Digit Recognizer", layout="centered")
st.title("🔢 Clasificador de Dígitos con CNN")
st.write("Dibuja un número del 0 al 9 y la IA intentará adivinarlo.")

# 1. Cargar y entrenar el modelo (Caché para no repetir el proceso)
@st.cache_resource
def load_and_train_model():
    # Usamos MNIST de Keras (28x28) para mejor precisión que el de sklearn
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalización y ajuste de dimensiones para CNN (batch, 28, 28, 1)
    x_train = x_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype("float32") / 255

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Entrenamiento rápido
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    return model

model = load_and_train_model()

# 2. Área de dibujo (Canvas)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Lienzo")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# 3. Procesamiento y Predicción
if canvas_result.image_data is not None:
    # Obtener la imagen del canvas y convertir a escala de grises
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L') # Convertir a escala de grises
    img = img.resize((28, 28)) # Redimensionar a 28x28 para la CNN
    
    # Preparar para el modelo
    img_array = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255

    if st.button("Predecir"):
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        with col2:
            st.markdown("### Resultado")
            st.metric(label="Número Identificado", value=digit)
            st.write(f"Confianza: **{confidence:.2%}**")
            st.bar_chart(prediction[0])
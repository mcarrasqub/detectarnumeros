import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate():
    """
    Esta función entrena y evalúa un modelo MLP para el dataset MNIST.
    """
    print("Cargando el dataset MNIST...")
    # Cargamos el dataset completo
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Normalizamos los datos
    X = X / 255.0
    
    # Dividimos en conjunto de entrenamiento y prueba (aunque usaremos CV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Configurando el modelo MLP...")
    # Usamos una arquitectura un poco más profunda y más iteraciones
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=100,
        alpha=1e-4,
        solver='adam', 
        random_state=1,
        learning_rate_init=0.001,
        verbose=10 # Muestra el progreso del entrenamiento
    )

    print("\nRealizando validación cruzada (puede tardar)...")
    # Realizamos validación cruzada con 3 folds para una estimación rápida y robusta
    # cv=5 es más robusto pero tarda más
    scores = cross_val_score(mlp, X_train, y_train, cv=3, scoring='accuracy')
    print(f"\nPrecisión media con validación cruzada (3-fold): {np.mean(scores):.4f}")
    print(f"Desviación estándar de la precisión: {np.std(scores):.4f}")

    print("\nEntrenando el modelo final con todos los datos de entrenamiento...")
    mlp.fit(X_train, y_train)

    print("\nEvaluando el modelo con el conjunto de prueba...")
    y_pred = mlp.predict(X_test)
    
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Matriz de Confusión ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualización de la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig('confusion_matrix.png')
    print("\nMatriz de confusión guardada en 'confusion_matrix.png'")


    print("\nGuardando el modelo entrenado en 'mnist_mlp_model.pkl'...")
    joblib.dump(mlp, "mnist_mlp_model.pkl")
    print("¡Modelo guardado con éxito!")

if __name__ == "__main__":
    train_and_evaluate()

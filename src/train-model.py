import pickle  # Librería para guardar y cargar objetos (usada para exportar el modelo)

# Importaciones de Scikit-Learn (la librería estándar para ML en Python)
from sklearn.datasets import fetch_openml          # Para descargar el dataset MNIST
from sklearn.ensemble import RandomForestClassifier # El algoritmo de clasificación
from sklearn.model_selection import train_test_split # Para dividir los datos en entrenamiento y prueba

# 1. CARGA DE DATOS
# Descarga el dataset MNIST (70,000 imágenes de dígitos escritos a mano del 0 al 9)
# X contiene los píxeles (características) y 'y' contiene las etiquetas (el número real)
X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

# 2. DIVISIÓN DEL DATASET
# Separa los datos: 80% para entrenar el modelo y 20% (test_size=0.2) para evaluar qué tan bueno es.
# Es vital no evaluar el modelo con los mismos datos que usó para aprender.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. DEFINICIÓN DEL MODELO
# Crea una instancia de Random Forest (Bosque Aleatorio).
# n_jobs=-1 es un parámetro clave: le dice a Python que use TODOS los núcleos de tu procesador 
# en el Codespace para entrenar más rápido.
clf = RandomForestClassifier(n_jobs=-1)

# 4. ENTRENAMIENTO (FIT)
# Aquí es donde ocurre la "magia". El modelo busca patrones en X_train para predecir y_train.
clf.fit(X_train, y_train)

# 5. EVALUACIÓN
# Calcula la precisión (accuracy) del modelo usando los datos que nunca ha visto (X_test).
# El resultado es un valor entre 0 y 1 (ej: 0.97 significa 97% de aciertos).
print(f"Precisión del modelo: {clf.score(X_test, y_test)}")

# 6. EXPORTACIÓN DEL MODELO
# Guarda el modelo entrenado en un archivo físico llamado 'mnist_model.pkl'.
# 'wb' significa "Write Binary" (Escritura Binaria).
# Este archivo .pkl es el que luego cargarás en tu API de FastAPI para hacer predicciones reales.
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
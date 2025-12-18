import io
import pickle

import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# 1. CARGA DEL MODELO
# Se abre el archivo serializado en modo lectura binaria ('rb').
# 'model' ahora es un objeto de Scikit-Learn listo para predecir.
with open("mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# 2. CONFIGURACIÓN DE CORS (Cross-Origin Resource Sharing)
# Esto permite que aplicaciones externas (como un frontend en React o Vue)
# puedan hacer consultas a tu API sin ser bloqueadas por el navegador.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Permite peticiones desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],      # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],      # Permite todas las cabeceras
)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    # 3. LECTURA DE LA IMAGEN
    # Se leen los bytes del archivo subido de forma asíncrona.
    image_data = await file.read()

    # 4. PROCESAMIENTO CON PILLOW
    # - io.BytesIO: Convierte los bytes en un flujo que PIL puede leer como si fuera un archivo.
    # - .convert("L"): Convierte la imagen a escala de grises (L = Luminance).
    image = PIL.Image.open(io.BytesIO(image_data)).convert("L")

    # 5. INVERSIÓN DE COLORES
    # El dataset MNIST original tiene fondo negro (0) y trazos blancos (255).
    # Si el usuario dibuja en negro sobre fondo blanco, invertimos los colores 
    # para que coincida con el formato con el que el modelo fue entrenado.
    image = PIL.ImageOps.invert(image)

    # 6. REDIMENSIONADO (RESIZE)
    # Se ajusta la imagen a 28x28 píxeles usando el filtro LANCZOS para máxima nitidez.
    image = image.resize((28, 28), PIL.Image.Resampling.LANCZOS)

    # 7. PREPARACIÓN PARA EL MODELO (NUMPY)
    # - np.array(image): Convierte los píxeles en una matriz numérica.
    # - .reshape(1, -1): Convierte la matriz de 28x28 en una sola fila de 784 columnas.
    image_array = np.array(image).reshape(1, -1)

    # 8. PREDICCIÓN
    # El modelo devuelve un array, tomamos el primer elemento [0] 
    # y lo convertimos a int para que sea compatible con el formato JSON.
    prediction = model.predict(image_array)
    return {"prediction": int(prediction[0])}
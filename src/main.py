import io
import pickle

import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open("mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = PIL.Image.open(io.BytesIO(image_data)).convert("L")
    image = PIL.ImageOps.invert(image)
    image = image.resize((28, 28), PIL.Image.Resampling.LANCZOS)
    image_array = np.array(image).reshape(1, -1)
    prediction = model.predict(image_array)
    return {"prediction": int(prediction[0])}
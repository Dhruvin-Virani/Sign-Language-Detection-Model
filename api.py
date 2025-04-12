from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("actions.h5")
app = FastAPI()

class ModelInput(BaseModel):
    landmarks: list  # e.g. 21 hand landmarks * 3 coords = 63 values

@app.post("/predict")
def predict(input: ModelInput):
    input_array = np.array(input.landmarks).reshape(1, -1)
    prediction = model.predict(input_array)
    action = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return {"action": action, "confidence": confidence}

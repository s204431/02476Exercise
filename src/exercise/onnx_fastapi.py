import numpy as np
import onnxruntime
from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
def predict():
    """Predict using ONNX model."""
    # Load the ONNX model
    model = onnxruntime.InferenceSession("mymodel.onnx")

    # Prepare the input data
    input_data = {"input": np.random.rand(1, 1, 28, 28).astype(np.float32)}

    # Run the model
    output = model.run(None, input_data)

    return {"output": output[0].tolist()}
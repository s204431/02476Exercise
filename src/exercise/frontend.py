import os

import pandas as pd
import streamlit as st
from google.cloud import run_v2
import numpy as np
from PIL import Image
import io
import bentoml


def get_backend_url():
    """Get the URL of the backend service."""
    return "https://backend-438187440265.europe-west1.run.app"


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    with bentoml.SyncHTTPClient(backend) as client:
        response = client.predict(image=image)
    return response


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        image = Image.open(io.BytesIO(image))
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to match the minimum input size of the model
        image = np.array(image)
        image = np.expand_dims(image, 2)
        image = np.transpose(image, (2, 0, 1))  # Change to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        result = classify_image(image, backend=backend)
        image = image.squeeze(0).squeeze(0)

        if result is not None:
            result = result.squeeze(0)
            result = np.exp(result - max(result))/np.sum(np.exp(result - max(result)))
            #prediction = result["prediction"]
            #probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", np.argmax(result))

            # make a nice bar chart
            data = {"Class": [f"Class {i}" for i in range(10)], "Probability": result}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
import bentoml
import numpy as np
from PIL import Image

if __name__ == "__main__":
    image = Image.open("img_10007.jpg")
    image = image.resize((28, 28))  # Resize to match the minimum input size of the model
    image = np.array(image)
    image = np.expand_dims(image, 2)
    image = np.transpose(image, (2, 0, 1))  # Change to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    with bentoml.SyncHTTPClient("https://backend-438187440265.europe-west1.run.app") as client:
        resp = client.predict(image=image)
        print(resp)
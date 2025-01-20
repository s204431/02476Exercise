import requests

url = "http://localhost:8000/predict"
review = "This is a great app, I love it!"
response = requests.post(url, json={"review": review})
print(response.json())
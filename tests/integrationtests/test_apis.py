from fastapi.testclient import TestClient
from exercise.api import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status-code": 200, "message": "OK"}

def test_read_root():
    response = client.get("/items/5")
    assert response.status_code == 200
    assert response.json() == {"item_id": 5}
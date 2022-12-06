from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict():
    response = client.post("/predict/", json={"row": "69,1,0,160,234,1,2,131,0,0.1,1,1,0"})
    print(response)
    assert response.status_code == 200
    assert response.json() == {"Result": 0}
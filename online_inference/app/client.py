import requests
import json

if __name__ == "__main__":
    data = {"row": "69,1,0,160,234,1,2,131,0,0.1,1,1,0"}
    resp = requests.post("http://localhost:80/predict/", json=data)
    resp = requests.get("http://localhost:80/health/")
import os
import pandas as pd
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response, Body, status
from utils import init_model

app = FastAPI()

global model
model = None
if model is None:
    model = init_model('./model.pkl')

@app.post("/predict/")
async def predict(row: str = Body(embed=True), status_code=200):
    data = [list(map(float, row.split(',')))]
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
               "ca", "thal"]
    series = pd.DataFrame(data, columns=columns)
    predict = model.predict(series)
    return {"Result": int(predict[0])}

@app.get("/health/", status_code=200)
async def health(response: Response):
    if model is not None:
        return {"response": "Model is initialized"}
    else:
        response.status_code = status.HTTP_201_CREATED
        return {"response": "Model is not still initialized"}

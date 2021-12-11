from typing import Optional

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

@app.get("/api/hello")
def hello():
    return {"hello hello world"}

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    print(image.filename)
    return {"result": 1}


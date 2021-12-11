from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    username: str


app = FastAPI()

@app.get("/api/hello")
def hello():
    return {"hello hello world"}

@app.post("/api/predict")
async def predict(item: Item):
    print(item)
    return item
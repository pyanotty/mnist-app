from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

app = FastAPI()

@app.get("/api/hello")
def hello():
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    model_info = client.get_latest_versions("mnist_model", stages=["None"])[0]
    model_uri = f"models:/{model_info.name}/{model_info.version}"
    model_info = {
        "model": model_info.name,
        "version": model_info.version,
        "source": model_info.source,
        "model_uri": model_uri
    }
    model = mlflow.pytorch.load_model(model_uri)
    return model_info

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    model_info = client.get_latest_versions("mnist_model", stages=["None"])[0]
    model_uri = f"models:/{model_info.name}/{model_info.version}"
    print("model_uri:", model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    print("file name:", image.filename)
    print("content type:", image.content_type)

    contents = image.file.read()
    with open("./image/sample.png", "wb") as out_file:
        out_file.write(contents)
    
    image = Image.open("./image/sample.png")
    # image = image.convert("RGB")
    
    image = image.resize((28, 28))
    # image = image.convert("L") # grayscale化
    
    image = np.array(image)[:, :, 3]
    print("value", image[:, 10:15,])

    Image.fromarray(image).save("./image/converted.png")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0, std=0.5),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    input = transform(image)
    print(input.shape)
    output = model(input)
    output = output.detach().cpu().numpy()
    print("output", output)

    output = output.argmax()
    output = int(output)

    print("output", output)

    # return model_info
    return {"result": output}

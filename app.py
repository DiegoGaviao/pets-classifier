from fastai.vision.all import *
from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI()

learn = load_learner('model.pkl')

@app.post("/predict")
async def predict(file: UploadFile):
    img = PILImage.create(await file.read())
    pred, pred_idx, probs = learn.predict(img)
    return {"prediction": str(pred), "probability": probs[pred_idx].item()}

from fastai.vision.all import *
from fastapi import FastAPI, UploadFile
from io import BytesIO

app = FastAPI()

def is_cat(x): return x[0].isupper()  # <<<<<< ADICIONAR ISSO AQUI!

learn = load_learner('model.pkl')

@app.post("/analyze")
async def analyze(file: UploadFile):
    img = PILImage.create(BytesIO(await file.read()))
    prediction, idx, probs = learn.predict(img)
    return {"result": prediction}

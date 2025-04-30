from fastai.vision.all import *
from fastapi import FastAPI, UploadFile
from io import BytesIO

# ✅ A função is_cat precisa estar igual à usada no treinamento
def is_cat(x): return x[0].isupper()

# ✅ Criação da API
app = FastAPI()

# ✅ Carregar o modelo
learn = load_learner('model.pkl')

# ✅ Endpoint para inferência
@app.post("/analyze")
async def analyze(file: UploadFile):
    img = PILImage.create(BytesIO(await file.read()))
    prediction, idx, probs = learn.predict(img)
    return {"result": prediction}

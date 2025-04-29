from fastai.vision.all import *
from fastapi import FastAPI, UploadFile
from io import BytesIO

# Função usada no treinamento — precisa estar exatamente aqui!
def is_cat(x): return x[0].isupper()

# Criação da API
app = FastAPI()

# Carregar modelo
learn = load_learner('model.pkl')

# Definir endpoint
@app.post("/analyze")
async def analyze(file: UploadFile):
    img = PILImage.create(BytesIO(await file.read()))
    prediction, idx, probs = learn.predict(img)
    return {"result": prediction}

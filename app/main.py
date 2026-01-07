from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib

from app.schemas import EmailInput

app = FastAPI(title="Spam Email Classifier")

BASE_DIR = Path(__file__).resolve().parent.parent

# Static & templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

# Load model
model = joblib.load(BASE_DIR / "model" / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "model" / "vectorizer.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_spam(data: EmailInput):
    message_vec = vectorizer.transform([data.message])
    prediction = model.predict(message_vec)[0]

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam"
    }

from fastapi import FastAPI, HTTPException
from googletrans import Translator
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Request model
class PredictionRequest(BaseModel):
    headline: str
    lang: str = "EN"

# Translator and model loading
translator = Translator()
model = joblib.load("LogisticModel.joblib")

ALLOWED_ORIGINS = [
    "chrome-extension://nekipfkffnlcjiakboiknjfnmlpemopb",  # Chrome extension ID
    "https://news-check-backend-production.up.railway.app",  # Backend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Sadece belirtilen kaynaklara izin ver
    allow_credentials=True,  # Credential (ör. cookie) gönderimine izin ver
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver
    allow_headers=["*"],  # Tüm headerlara izin ver
)

@app.post("/predict")
async def predict_news(request: PredictionRequest):
    try:
        # Translate headline if necessary
        if request.lang.upper() == "TR":
            translated_text = request.headline
        else:
            translated_text = translator.translate(request.headline, src="en", dest="tr").text
        
        # Predict using the model
        prediction = model.predict([translated_text])
        confidence = model.predict_proba([translated_text])
        
        # Convert numpy types to Python types
        prediction_result = int(prediction[0])  # Convert to int
        confidence_score = float(np.max(confidence[0])) * 100  # Convert to float and percentage

        # Return the response
        return {
            "original_text": request.headline,
            "translated_text": translated_text,
            "prediction": prediction_result,
            "confidence": f"{confidence_score:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load model
model = joblib.load('credit_risk_model.pkl')
features = joblib.load('model_features.pkl')

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk assessment API | Built by Kayalvizhi S | M.S. FinTech, SRM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data structure
class ApplicantData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

@app.get("/")
def home():
    return {
        "message": "Credit Risk Scoring API is live",
        "author": "Kayalvizhi S | M.S. FinTech, SRM",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost Credit Risk Model v1.0"}

@app.post("/predict")
def predict(data: ApplicantData):
    input_data = np.array([[
        data.RevolvingUtilizationOfUnsecuredLines,
        data.age,
        data.NumberOfTime30_59DaysPastDueNotWorse,
        data.DebtRatio,
        data.MonthlyIncome,
        data.NumberOfOpenCreditLinesAndLoans,
        data.NumberOfTimes90DaysLate,
        data.NumberRealEstateLoansOrLines,
        data.NumberOfTime60_89DaysPastDueNotWorse,
        data.NumberOfDependents
    ]])

    risk_probability = float(model.predict_proba(input_data)[0][1])  # ← fixed
    risk_score = int(round((1 - risk_probability) * 850 + 150))       # ← fixed

    if risk_probability < 0.2:
        risk_category = "LOW RISK"
        recommendation = "APPROVE"
    elif risk_probability < 0.5:
        risk_category = "MEDIUM RISK"
        recommendation = "MANUAL REVIEW"
    else:
        risk_category = "HIGH RISK"
        recommendation = "DECLINE"

    return {
        "risk_score": risk_score,
        "default_probability": round(risk_probability * 100, 2),
        "risk_category": risk_category,
        "recommendation": recommendation,
        "score_range": "150-850 (higher is better)",
        "model": "XGBoost | Trained on 150,000 borrower records"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost Credit Risk Model v1.0"}

@app.get("/dashboard")
def dashboard():
    return FileResponse("index.html")
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# =========================
# INPUT MODEL
# =========================
class AutismData(BaseModel):
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int

    Speech_Delay: str
    Learning_disorder: str
    Genetic_Disorders: str
    Depression: str
    Global_developmental_delay: str
    Social_Behavioural_Issues: str

    Age_Years: float
    Sex: str


# =========================
# AI LOGIC
# =========================
def predict_autism(data: AutismData):

    # 🔥 حساب المجموع
    score = (
        data.A1 + data.A2 + data.A3 + data.A4 + data.A5 +
        data.A6 + data.A7 + data.A8 + data.A9 + data.A10
    )

    # 🔥 نسبة التوحد
    probability = (score / 10) * 100

    # 🔥 القرار
    prediction = 1 if score >= 6 else 0

    # 🔥 تحديد الشدة
    if prediction == 0:
        severity = "لا يوجد توحد"
    else:
        if score <= 3:
            severity = "خفيف"
        elif score <= 7:
            severity = "متوسط"
        else:
            severity = "شديد"

    return {
        "prediction": prediction,
        "score": score,
        "probability": round(probability, 2),
        "severity": severity
    }


# =========================
# ENDPOINT
# =========================
@app.post("/predict")
def predict(data: AutismData):

    result = predict_autism(data)

    return {
        "status": "success",
        "result": result
    }


# =========================
# ROOT (اختياري)
# =========================
@app.get("/")
def home():
    return {"message": "Autism AI API is running 🚀"}
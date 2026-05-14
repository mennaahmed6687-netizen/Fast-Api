from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AutismData(BaseModel):
    A1: int; A2: int; A3: int; A4: int; A5: int
    A6: int; A7: int; A8: int; A9: int; A10: int
    Speech_Delay: str
    Learning_disorder: str
    Genetic_Disorders: str
    Depression: str
    Global_developmental_delay: str
    Social_Behavioural_Issues: str
    Age_Years: float
    Sex: str


def predict_logic(data: AutismData):

    score = sum([
        data.A1, data.A2, data.A3, data.A4, data.A5,
        data.A6, data.A7, data.A8, data.A9, data.A10
    ])

    probability = round((score / 10) * 100, 2)

    prediction = 1 if score >= 6 else 0

    if score <= 3:
        severity = "خفيف"
    elif score <= 7:
        severity = "متوسط"
    else:
        severity = "شديد"

    return {
        "prediction": prediction,
        "score": score,
        "probability": probability,
        "severity": severity
    }


@app.post("/predict")
def predict(data: AutismData):
    return {
        "status": "success",
        "result": predict_logic(data)
    }
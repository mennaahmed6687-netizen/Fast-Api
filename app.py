from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# WEIGHTS
# =========================

QUESTION_WEIGHTS = {

    "A1": 10,
    "A2": 15,
    "A3": 10,
    "A4": 10,
    "A5": 15,
    "A6": 10,
    "A7": 10,
    "A8": 5,
    "A9": 10,
    "A10": 5,

    "Speech_Delay": 20,
    "Learning_disorder": 15,
    "Genetic_Disorders": 15,
    "Depression": 10,
    "Global_developmental_delay": 20,
    "Social_Behavioural_Issues": 20
}

# =========================
# TOTAL WEIGHT
# =========================

TOTAL_WEIGHT = sum(QUESTION_WEIGHTS.values())

# =========================
# AI LOGIC
# =========================

def predict_autism(data: AutismData):

    score = 0

    # =====================
    # QUESTIONS
    # =====================

    questions = [
        "A1", "A2", "A3", "A4", "A5",
        "A6", "A7", "A8", "A9", "A10"
    ]

    for question in questions:

        value = getattr(data, question)

        if value == 1:
            score += QUESTION_WEIGHTS[question]

    # =====================
    # MEDICAL DATA
    # =====================

    medical_fields = [
        "Speech_Delay",
        "Learning_disorder",
        "Genetic_Disorders",
        "Depression",
        "Global_developmental_delay",
        "Social_Behavioural_Issues"
    ]

    for field in medical_fields:

        value = getattr(data, field)

        if value.lower() == "yes":
            score += QUESTION_WEIGHTS[field]

    # =====================
    # AGE EFFECT
    # =====================

    if data.Age_Years < 5:
        score += 5

    # =====================
    # PROBABILITY
    # =====================

    probability = (score / TOTAL_WEIGHT) * 100

    # =====================
    # PREDICTION
    # =====================

    prediction = 1 if probability >= 50 else 0

    # =====================
    # SEVERITY
    # =====================

    if probability < 30:
        severity = "لا يوجد توحد"

    elif probability < 50:
        severity = "خفيف"

    elif probability < 75:
        severity = "متوسط"

    else:
        severity = "شديد"

    # =====================
    # RETURN
    # =====================

    return {

        "prediction": prediction,
        "score": score,
        "probability": round(probability, 2),
        "severity": severity
    }

# =========================
# PREDICT ENDPOINT
# =========================

@app.post("/predict")
def predict(data: AutismData):

    try:

        result = predict_autism(data)

        return {

            "status": "success",
            "result": result
        }

    except Exception as e:

        return {

            "status": "error",
            "message": str(e)
        }

# =========================
# ROOT
# =========================

@app.get("/")
def home():

    return {

        "message": "Autism AI API is running 🚀"
    }
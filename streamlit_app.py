import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="كشف التوحد", page_icon="🧩")

st.title("🧩 نظام كشف التوحد")

st.write("ادخل البيانات واضغط تحليل")

# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    a1 = st.selectbox("السؤال 1", [0, 1])
    a2 = st.selectbox("السؤال 2", [0, 1])
    a3 = st.selectbox("السؤال 3", [0, 1])
    a4 = st.selectbox("السؤال 4", [0, 1])
    a5 = st.selectbox("السؤال 5", [0, 1])
    a6 = st.selectbox("السؤال 6", [0, 1])
    a7 = st.selectbox("السؤال 7", [0, 1])
    a8 = st.selectbox("السؤال 8", [0, 1])
    a9 = st.selectbox("السؤال 9", [0, 1])
    a10 = st.selectbox("السؤال 10", [0, 1])

    age = st.number_input("العمر", 1, 50, 10)

with col2:
    speech = st.selectbox("تأخر النطق", ["No", "Yes"])
    learning = st.selectbox("صعوبات التعلم", ["No", "Yes"])
    genetic = st.selectbox("أمراض وراثية", ["No", "Yes"])
    depression = st.selectbox("اكتئاب", ["No", "Yes"])
    global_delay = st.selectbox("تأخر نمائي", ["No", "Yes"])
    social = st.selectbox("مشاكل اجتماعية", ["No", "Yes"])
    sex = st.selectbox("النوع", ["m", "f"])


# ================= BUTTON =================
if st.button("تحليل"):

    payload = {
        "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5,
        "A6": a6, "A7": a7, "A8": a8, "A9": a9, "A10": a10,

        "Speech_Delay": speech,
        "Learning_disorder": learning,
        "Genetic_Disorders": genetic,
        "Depression": depression,
        "Global_developmental_delay": global_delay,
        "Social_Behavioural_Issues": social,

        "Age_Years": float(age),
        "Sex": sex
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()["result"]

        st.divider()

        st.success("تم التحليل بنجاح 🎯")

        st.write("🧠 النتيجة:", "توحد" if result["prediction"] == 1 else "لا يوجد")
        st.write("📊 الدرجة:", result["score"])
        st.write("📈 النسبة:", str(result["probability"]) + "%")
        st.write("⚠️ الحالة:", result["severity"])

    except Exception as e:
        st.error("خطأ في الاتصال: " + str(e))
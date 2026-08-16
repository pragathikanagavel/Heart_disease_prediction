import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def predict_heart_disease(patient_data):
    """Load the saved model and predict whether the patient is likely to have heart disease."""
    model = joblib.load(MODEL_PATH)
    patient_df = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)
    prediction = model.predict(patient_df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(patient_df)[0][1]

    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description="Educational heart disease prediction tool.")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--trestbps", type=int, required=True)
    parser.add_argument("--chol", type=int, required=True)
    parser.add_argument("--fbs", type=int, required=True)
    parser.add_argument("--restecg", type=int, required=True)
    parser.add_argument("--thalach", type=int, required=True)
    parser.add_argument("--exang", type=int, required=True)
    parser.add_argument("--oldpeak", type=float, required=True)
    parser.add_argument("--slope", type=int, required=True)
    parser.add_argument("--ca", type=float, required=True)
    parser.add_argument("--thal", type=float, required=True)

    args = parser.parse_args()

    patient = {
        "age": args.age,
        "sex": args.sex,
        "cp": args.cp,
        "trestbps": args.trestbps,
        "chol": args.chol,
        "fbs": args.fbs,
        "restecg": args.restecg,
        "thalach": args.thalach,
        "exang": args.exang,
        "oldpeak": args.oldpeak,
        "slope": args.slope,
        "ca": args.ca,
        "thal": args.thal,
    }

    prediction, probability = predict_heart_disease(patient)
    print("\nEducational machine-learning prediction; not a medical diagnosis.")
    print(f"Predicted class: {prediction}")
    print(f"Interpretation: {'Heart disease present' if prediction == 1 else 'No heart disease'}")

    if probability is not None:
        print(f"Probability of heart disease: {probability:.2%}")


if __name__ == "__main__":
    main()

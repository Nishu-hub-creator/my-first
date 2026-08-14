from functools import lru_cache
from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PLACEMENT_MODEL_PATH = BASE_DIR / "placement_model.pkl"
SALARY_MODEL_PATH = BASE_DIR / "salary_model.pkl"

app = Flask(__name__)
app.config["SECRET_KEY"] = "student-placement-predictor"

DEFAULT_VALUES = {
    "cgpa": 7.5,
    "internships": 1,
    "projects": 2,
    "certifications": 2,
    "aptitude_score": 70,
    "communication_skills": 70,
    "technical_skills": 70,
    "backlogs": 0,
}

FIELD_RULES = {
    "cgpa": {"label": "CGPA", "type": float, "min": 0, "max": 10},
    "internships": {"label": "Internships", "type": int, "min": 0, "max": 20},
    "projects": {"label": "Projects", "type": int, "min": 0, "max": 20},
    "certifications": {"label": "Certifications", "type": int, "min": 0, "max": 20},
    "aptitude_score": {"label": "Aptitude Score", "type": float, "min": 0, "max": 100},
    "communication_skills": {
        "label": "Communication Skills",
        "type": float,
        "min": 0,
        "max": 100,
    },
    "technical_skills": {
        "label": "Technical Skills",
        "type": float,
        "min": 0,
        "max": 100,
    },
    "backlogs": {"label": "Backlogs", "type": int, "min": 0, "max": 20},
}


@lru_cache(maxsize=2)
def load_pickle_artifact(file_path: str):
    path = Path(file_path)
    with open(path, "rb") as file:
        return pickle.load(file)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    transformed["experience_score"] = (
        transformed["internships"] + transformed["projects"] + transformed["certifications"]
    )
    transformed["overall_skill_score"] = (
        transformed["aptitude_score"]
        + transformed["communication_skills"]
        + transformed["technical_skills"]
    ) / 3.0
    return transformed


def validate_input(form_data):
    cleaned_values = {}

    for field_name, rules in FIELD_RULES.items():
        raw_value = form_data.get(field_name, "").strip()
        if raw_value == "":
            raise ValueError(f"{rules['label']} is required.")

        try:
            value = rules["type"](raw_value)
        except ValueError as error:
            raise ValueError(f"{rules['label']} must be a valid number.") from error

        if value < rules["min"] or value > rules["max"]:
            raise ValueError(
                f"{rules['label']} must be between {rules['min']} and {rules['max']}."
            )

        cleaned_values[field_name] = value

    return cleaned_values


def build_feature_frame(input_values: dict, feature_columns):
    feature_frame = pd.DataFrame([input_values])
    feature_frame = add_engineered_features(feature_frame)
    return feature_frame[feature_columns]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None
    form_values = DEFAULT_VALUES.copy()

    if request.method == "POST":
        try:
            placement_artifact = load_pickle_artifact(str(PLACEMENT_MODEL_PATH))
            salary_artifact = load_pickle_artifact(str(SALARY_MODEL_PATH))
        except FileNotFoundError:
            error_message = (
                "Model files are missing. Please run train_model.py before starting the app."
            )
            return render_template("index.html", result=result, error_message=error_message, form_values=form_values)
        except Exception as error:
            error_message = f"Unable to load model files: {error}"
            return render_template("index.html", result=result, error_message=error_message, form_values=form_values)

        try:
            form_values = validate_input(request.form)

            placement_features = build_feature_frame(
                form_values,
                placement_artifact["feature_columns"],
            )
            salary_features = build_feature_frame(
                form_values,
                salary_artifact["feature_columns"],
            )

            placement_prediction = int(placement_artifact["model"].predict(placement_features)[0])
            placement_probability = float(
                placement_artifact["model"].predict_proba(placement_features)[0][1] * 100
            )

            result = {
                "is_placed": placement_prediction == 1,
                "placement_probability": round(placement_probability, 2),
                "salary_lpa": None,
            }

            if result["is_placed"]:
                predicted_salary = float(salary_artifact["model"].predict(salary_features)[0])
                result["salary_lpa"] = round(max(predicted_salary, 0.0), 2)

        except ValueError as error:
            error_message = str(error)
        except Exception as error:
            error_message = f"Prediction failed: {error}"

    return render_template(
        "index.html",
        result=result,
        error_message=error_message,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)

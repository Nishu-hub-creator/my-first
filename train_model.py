from pathlib import Path
import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
PLACEMENT_MODEL_PATH = BASE_DIR / "placement_model.pkl"
SALARY_MODEL_PATH = BASE_DIR / "salary_model.pkl"

PLACEMENT_TARGET = "placement_status"
SALARY_TARGET = "salary_package_lpa"

COLUMN_RENAME_MAP = {
    "cgpa": "cgpa",
    "internships_count": "internships",
    "projects_count": "projects",
    "certifications_count": "certifications",
    "aptitude_score": "aptitude_score",
    "communication_skill_score": "communication_skills",
    "coding_skill_score": "technical_skills",
    "backlogs": "backlogs",
}

INPUT_FEATURE_COLUMNS = list(COLUMN_RENAME_MAP.values())
ENGINEERED_FEATURE_COLUMNS = ["experience_score", "overall_skill_score"]
MODEL_FEATURE_COLUMNS = INPUT_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def load_and_clean_dataset(file_path: Path) -> pd.DataFrame:
    """Load the dataset, remove duplicates, and handle missing values."""
    dataframe = pd.read_csv(file_path)
    original_rows = len(dataframe)

    dataframe = dataframe.drop_duplicates().copy()
    duplicates_removed = original_rows - len(dataframe)

    required_columns = list(COLUMN_RENAME_MAP.keys()) + [PLACEMENT_TARGET, SALARY_TARGET]
    dataframe = dataframe[required_columns].rename(columns=COLUMN_RENAME_MAP)

    dataframe[PLACEMENT_TARGET] = dataframe[PLACEMENT_TARGET].astype(str).str.strip()
    dataframe = dataframe[dataframe[PLACEMENT_TARGET].isin(["Placed", "Not Placed"])].copy()

    numeric_columns = INPUT_FEATURE_COLUMNS + [SALARY_TARGET]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    for column in INPUT_FEATURE_COLUMNS:
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())

    dataframe[SALARY_TARGET] = dataframe[SALARY_TARGET].fillna(0.0)
    dataframe = dataframe.dropna(subset=[PLACEMENT_TARGET]).reset_index(drop=True)

    print(f"Dataset loaded from: {file_path}")
    print(f"Rows after cleaning: {len(dataframe)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print("Missing values after cleaning:")
    print(dataframe[INPUT_FEATURE_COLUMNS + [PLACEMENT_TARGET, SALARY_TARGET]].isna().sum())
    print()

    return dataframe


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a few simple features that combine experience and skill signals."""
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


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def train_classification_models(features: pd.DataFrame, target: pd.Series):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    except Exception:
        print("XGBoost is not installed. Skipping XGBoost classification model.")
        print()

    best_model_name = ""
    best_model = None
    best_metrics = None
    best_accuracy = -1.0
    all_results = []

    for model_name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        result = {
            "model_name": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
        all_results.append(result)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = pipeline
            best_metrics = result

    return best_model_name, best_model, best_metrics, all_results


def train_regression_models(features: pd.DataFrame, target: pd.Series):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
        ),
    }

    best_model_name = ""
    best_model = None
    best_metrics = None
    best_rmse = float("inf")
    all_results = []

    for model_name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        result = {
            "model_name": model_name,
            "rmse": round(rmse, 4),
        }
        all_results.append(result)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name
            best_model = pipeline
            best_metrics = result

    return best_model_name, best_model, best_metrics, all_results


def save_pickle_file(file_path: Path, data: dict):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

    dataframe = load_and_clean_dataset(DATASET_PATH)
    prepared_dataframe = add_engineered_features(dataframe)

    classification_features = prepared_dataframe[MODEL_FEATURE_COLUMNS]
    classification_target = prepared_dataframe[PLACEMENT_TARGET].map(
        {"Not Placed": 0, "Placed": 1}
    )

    best_classifier_name, best_classifier, classification_metrics, classification_results = (
        train_classification_models(classification_features, classification_target)
    )

    regression_dataframe = prepared_dataframe[
        (prepared_dataframe[PLACEMENT_TARGET] == "Placed")
        & (prepared_dataframe[SALARY_TARGET] > 0)
    ].copy()

    if regression_dataframe.empty:
        raise ValueError("No placed students with salary information were found.")

    regression_features = regression_dataframe[MODEL_FEATURE_COLUMNS]
    regression_target = regression_dataframe[SALARY_TARGET]

    best_regressor_name, best_regressor, regression_metrics, regression_results = (
        train_regression_models(regression_features, regression_target)
    )

    placement_artifact = {
        "model": best_classifier,
        "model_name": best_classifier_name,
        "input_feature_columns": INPUT_FEATURE_COLUMNS,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "target_name": PLACEMENT_TARGET,
        "metrics": classification_metrics,
        "all_model_results": classification_results,
    }

    salary_artifact = {
        "model": best_regressor,
        "model_name": best_regressor_name,
        "input_feature_columns": INPUT_FEATURE_COLUMNS,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "target_name": SALARY_TARGET,
        "metrics": regression_metrics,
        "all_model_results": regression_results,
    }

    save_pickle_file(PLACEMENT_MODEL_PATH, placement_artifact)
    save_pickle_file(SALARY_MODEL_PATH, salary_artifact)

    print("Classification model results:")
    for result in classification_results:
        print(result)
    print()
    print("Regression model results:")
    for result in regression_results:
        print(result)
    print()
    print(f"Best placement model: {best_classifier_name}")
    print(f"Best salary model: {best_regressor_name}")
    print(f"Saved: {PLACEMENT_MODEL_PATH.name}")
    print(f"Saved: {SALARY_MODEL_PATH.name}")


if __name__ == "__main__":
    main()

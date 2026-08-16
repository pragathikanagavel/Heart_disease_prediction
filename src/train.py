import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_heart_disease_data
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, transform_target

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_model.pkl"
COMPARE_PATH = MODEL_DIR / "model_comparison.csv"


def prepare_training_data():
    """Load the dataset, clean target and return features X and y."""
    data = load_heart_disease_data()

    if "target" not in data.columns:
        data = data.rename(columns={"num": "target"})

    data = transform_target(data)
    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return X, y


def build_model_pipelines():
    """Create a dictionary of model pipelines."""
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model using key classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return metrics, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])


def cross_validate_models(X, y):
    """Estimate cross-validation ROC-AUC for each model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in build_model_pipelines().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "mean": round(scores.mean(), 4),
            "std": round(scores.std(), 4),
        }

    return results


def tune_models(X_train, y_train):
    """Tune a small set of promising models using GridSearchCV."""
    tuned_models = {}

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    logistic_param_grid = {
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear", "lbfgs"],
    }
    tuned_models["Logistic Regression"] = GridSearchCV(
        logistic_pipeline,
        param_grid=logistic_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=None,
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )
    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
    }
    tuned_models["Random Forest"] = GridSearchCV(
        rf_pipeline,
        param_grid=rf_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=None,
    )

    for name, model in tuned_models.items():
        model.fit(X_train, y_train)
        print(f"Best params for {name}: {model.best_params_}")
        print(f"Best CV ROC-AUC for {name}: {model.best_score_:.4f}")

    return tuned_models


def save_final_model(model_pipeline):
    """Save the trained pipeline using joblib."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    return str(MODEL_PATH)


def main():
    X, y = prepare_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Dataset shape:", X.shape)
    print("Target distribution:\n", y.value_counts().to_string())
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nCross-validation ROC-AUC summary:")
    cv_results = cross_validate_models(X, y)
    for name, result in cv_results.items():
        print(f"{name}: mean={result['mean']}, std={result['std']}")

    print("\nTuning promising models...")
    tuned_models = tune_models(X_train, y_train)

    final_model_name = "Logistic Regression"
    final_model = tuned_models[final_model_name].best_estimator_
    final_model.fit(X_train, y_train)

    metrics, cm, report = evaluate_model(final_model, X_test, y_test)
    print("\nFinal model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    save_final_model(final_model)
    print(f"\nSaved final model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

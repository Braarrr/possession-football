
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from models.prepare_dataset import prepare_training_data


def train_hierarchical(X_train, y_train):
    live_train = y_train != "D"

    y_stage1 = (y_train == "D").astype(int)

    model_stage1 = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None,
    )

    model_stage1.fit(X_train, y_stage1)

    model_stage2 = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None,
    )

    model_stage2.fit(
        X_train.loc[live_train],
        y_train.loc[live_train],
    )

    return model_stage1, model_stage2


def predict_hierarchical(model_stage1, model_stage2, X_test):
    stage1_predictions = model_stage1.predict(X_test)

    predictions = np.full(len(X_test), "D", dtype=object)

    live_test = stage1_predictions == 0

    predictions[live_test] = model_stage2.predict(
        X_test.loc[live_test]
    )

    return predictions


def evaluate_model(y_test, predictions):
    print("\n── HIERARCHICAL MODEL PERFORMANCE ────────")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Macro F1:", f1_score(
        y_test,
        predictions,
        average="macro",
    ))

    print("\n── CLASSIFICATION REPORT ─────────────────")
    print(classification_report(
        y_test,
        predictions,
        labels=["H", "A", "D", "X"],
        zero_division=0,
    ))

    print("\n── CONFUSION MATRIX ──────────────────────")
    print("Labels: H, A, D, X")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["H", "A", "D", "X"],
    ))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_training_data()

    print("Training Hierarchical Random Forest...")

    model_stage1, model_stage2 = train_hierarchical(
        X_train,
        y_train,
    )

    predictions = predict_hierarchical(
        model_stage1,
        model_stage2,
        X_test,
    )

    evaluate_model(y_test, predictions)

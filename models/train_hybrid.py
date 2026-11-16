import pandas as pd
import numpy as np
from possession_pipeline import (
    parse_smart_tagging,
    build_dead_ball_mask,
    make_submission,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from models.prepare_dataset import prepare_training_data
from possession_pipeline import (
    parse_smart_tagging,
    build_dead_ball_mask,
)


def train_hybrid(X_train, y_train, dead_train):
    live_train = ~dead_train

    if not np.all(y_train[dead_train] == "D"):
        raise ValueError("Dead-ball mask does not match training labels.")

    if np.any(y_train[live_train] == "D"):
        raise ValueError("D label found outside dead-ball mask.")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None,
    )

    model.fit(X_train.loc[live_train], y_train.loc[live_train])

    return model


def predict_hybrid(model, X_test, dead_test):
    predictions = np.full(len(X_test), "D", dtype=object)

    live_test = ~dead_test

    predictions[live_test] = model.predict(
        X_test.loc[live_test]
    )

    return predictions


def evaluate_model(y_test, predictions):
    print("\n── HYBRID MODEL PERFORMANCE ──────────────")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Macro F1:", f1_score(
        y_test,
        predictions,
        labels=["H", "A", "D", "X"],
        average="macro",
        zero_division=0,
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

    smart_df = parse_smart_tagging()
    dead_mask = build_dead_ball_mask(smart_df)

    if len(dead_mask) != len(X_train) + len(X_test):
        raise ValueError("Dead-ball mask and dataset lengths do not match.")

    dead_train = dead_mask[:len(X_train)]
    dead_test = dead_mask[len(X_train):]

    print("Training Hybrid Random Forest...")

    model = train_hybrid(
        X_train,
        y_train,
        dead_train,
    )

    # Preserve the original holdout evaluation.
    holdout_predictions = predict_hybrid(
        model,
        X_test,
        dead_test,
    )

    evaluate_model(y_test, holdout_predictions)

    # Predict all frames using the model trained on the first 80%.
    X_all = pd.concat(
        [X_train, X_test],
        ignore_index=True,
    )

    predictions = predict_hybrid(
        model,
        X_all,
        dead_mask,
    )

    # Recover the corresponding 1-based ATD frame IDs.
    frame_df = pd.read_csv(
        "data/processed_tracking.csv"
    ).sort_values("frame_id").reset_index(drop=True)

    frame_ids = frame_df["frame_id"].to_numpy()

    if len(frame_ids) != len(predictions):
        raise ValueError("Frame IDs and predictions have different lengths.")

    if not np.array_equal(
        frame_ids,
        np.arange(1, len(frame_ids) + 1),
    ):
        raise ValueError("Unexpected ATD frame ID sequence.")

    make_submission(
        predictions,
        frame_ids,
        out_path="submission_hybrid.csv",
    )
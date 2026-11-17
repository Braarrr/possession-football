
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance

from models.prepare_dataset import prepare_training_data
from possession_pipeline import parse_smart_tagging, build_dead_ball_mask


def train_and_evaluate(X_train, X_test, y_train, y_test, dead_train, dead_test):
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

    predictions = np.full(len(X_test), "D", dtype=object)
    live_test = ~dead_test
    predictions[live_test] = model.predict(X_test.loc[live_test])

    importance = permutation_importance(
        model,
        X_test.loc[live_test],
        y_test.loc[live_test],
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            labels=["H", "A", "D", "X"],
            average="macro",
            zero_division=0,
        ),
        "predictions": predictions,
        "importance": importance_df,
    }


def main():
    X_train, X_test, y_train, y_test = prepare_training_data()

    if "nearest_player_team" not in X_train.columns:
        raise ValueError("nearest_player_team feature not found.")

    smart_df = parse_smart_tagging()
    dead_mask = build_dead_ball_mask(smart_df)

    if len(dead_mask) != len(X_train) + len(X_test):
        raise ValueError("Dead-ball mask and dataset lengths do not match.")

    dead_train = dead_mask[:len(X_train)]
    dead_test = dead_mask[len(X_train):]

    # Experiment A: original 88 features
    X_train_88 = X_train.drop(columns=["nearest_player_team"])
    X_test_88 = X_test.drop(columns=["nearest_player_team"])

    # Experiment B: all 89 features
    X_train_89 = X_train.copy()
    X_test_89 = X_test.copy()

    print("\nTraining model with 88 features...")
    result_88 = train_and_evaluate(
        X_train_88, X_test_88, y_train, y_test, dead_train, dead_test
    )

    print("\nTraining model with 89 features...")
    result_89 = train_and_evaluate(
        X_train_89, X_test_89, y_train, y_test, dead_train, dead_test
    )

    print("\n=== FEATURE SET COMPARISON ===")
    print(f"{'Metric':<20}{'88 features':>15}{'89 features':>15}{'Difference':>15}")
    print("-" * 65)

    for metric in ["accuracy", "macro_f1"]:
        a = result_88[metric]
        b = result_89[metric]
        print(f"{metric:<20}{a:>15.4f}{b:>15.4f}{b-a:>+15.4f}")

    print("\n=== CLASSIFICATION REPORT: 88 FEATURES ===")
    print(classification_report(
        y_test,
        result_88["predictions"],
        labels=["H", "A", "D", "X"],
        zero_division=0,
    ))

    print("\n=== CLASSIFICATION REPORT: 89 FEATURES ===")
    print(classification_report(
        y_test,
        result_89["predictions"],
        labels=["H", "A", "D", "X"],
        zero_division=0,
    ))

    print("\n=== H/A PERFORMANCE ANALYSIS ===")

    results = {
        "88 features": result_88,
        "89 features": result_89,
    }

    print("\n=== PERMUTATION FEATURE IMPORTANCE (89 FEATURES) ===")

    importance = result_89["importance"]

    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))

    print("\nNearest player feature:")
    print(
        importance[
            importance["feature"] == "nearest_player_team"
        ].to_string(index=False)
    )

    for name, result in results.items():
        predictions = result["predictions"]

        print(f"\n--- {name} ---")

        # H/A Macro F1 across all holdout frames
        ha_f1_all = f1_score(
            y_test,
            predictions,
            labels=["H", "A"],
            average="macro",
            zero_division=0,
        )

        # H/A Macro F1 on live frames only
        live_mask = ~dead_test

        ha_f1_live = f1_score(
            y_test.iloc[live_mask],
            predictions[live_mask],
            labels=["H", "A"],
            average="macro",
            zero_division=0,
        )

        print(f"H/A Macro F1 (all frames):  {ha_f1_all:.4f}")
        print(f"H/A Macro F1 (live frames): {ha_f1_live:.4f}")

        # Confusion matrix for actual H/A frames
        ha_mask = y_test.isin(["H", "A"]).to_numpy()

        matrix = confusion_matrix(
            y_test.to_numpy()[ha_mask],
            predictions[ha_mask],
            labels=["H", "A", "X"],
        )

        print("\nConfusion Matrix (actual H/A frames)")
        print("Rows = Actual | Columns = Predicted")
        print("Labels: H, A, X")
        print(matrix)


if __name__ == "__main__":
    main()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from models.prepare_dataset import prepare_training_data


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=None,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\n── MODEL PERFORMANCE ──────────────────────")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Macro F1:", f1_score(y_test, predictions, average="macro"))

    print("\n── CLASSIFICATION REPORT ─────────────────")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["H", "A", "D", "X"],
            zero_division=0,
        )
    )

    print("\n── CONFUSION MATRIX ──────────────────────")
    print("Labels: H, A, D, X")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["H", "A", "D", "X"],
    ))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_training_data()

    print("Training Random Forest...")
    model = train_random_forest(X_train, y_train)

    evaluate_model(model, X_test, y_test)
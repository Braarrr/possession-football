
import pandas as pd

from features.distance_features import calculate_distance_to_ball
from features.velocity_features import extract_velocity_features
from features.visibility_features import calculate_ball_visibility
from models.generate_pseudo_labels import generate_pseudo_labels


def prepare_features(frame_df):
    distance = calculate_distance_to_ball(frame_df)
    velocity = extract_velocity_features(frame_df)
    visibility = calculate_ball_visibility(frame_df)

    features = distance.merge(velocity, on="frame_id")
    features = features.merge(visibility, on="frame_id")

    return features


def merge_labels(features, labels):
    if features["frame_id"].duplicated().any():
        raise ValueError("Duplicate frame_id found in features.")

    if labels["frame_id"].duplicated().any():
        raise ValueError("Duplicate frame_id found in pseudo-labels.")

    dataset = features.merge(
        labels,
        on="frame_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = dataset["_merge"] != "both"

    if unmatched.any():
        raise ValueError(
            f"Frame alignment error: {unmatched.sum()} unmatched frames."
        )

    dataset = dataset.drop(columns="_merge")
    dataset = dataset.sort_values("frame_id").reset_index(drop=True)

    if dataset["pseudo_label"].isna().any():
        raise ValueError("Missing pseudo-label found.")

    return dataset


def chronological_split(dataset, train_ratio=0.8):
    dataset = dataset.sort_values("frame_id").reset_index(drop=True)

    split_index = int(len(dataset) * train_ratio)

    train = dataset.iloc[:split_index].copy()
    test = dataset.iloc[split_index:].copy()

    return train, test


def impute_missing_values(train, test):
    feature_columns = [
        col for col in train.columns if col != "frame_id"
    ]

    medians = train[feature_columns].median()
    valid_features = medians[medians.notna()].index.tolist()

    train = train[["frame_id"] + valid_features].copy()
    test = test[["frame_id"] + valid_features].copy()

    train[valid_features] = train[valid_features].fillna(
        medians[valid_features]
    )

    test[valid_features] = test[valid_features].fillna(
        medians[valid_features]
    )

    return train, test

def prepare_training_data():
    frame_df = pd.read_csv("data/processed_tracking.csv")

    features = prepare_features(frame_df)
    labels = generate_pseudo_labels()

    dataset = merge_labels(features, labels)

    train, test = chronological_split(dataset)

    y_train = train["pseudo_label"].copy()
    y_test = test["pseudo_label"].copy()

    X_train = train.drop(columns="pseudo_label")
    X_test = test.drop(columns="pseudo_label")

    X_train, X_test = impute_missing_values(X_train, X_test)

    X_train = X_train.drop(columns="frame_id")
    X_test = X_test.drop(columns="frame_id")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_training_data()

    print("\n── FINAL DATASET ──────────────────────────")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\n── LABEL DISTRIBUTION ─────────────────────")
    print("Training:")
    print(y_train.value_counts())

    print("\nTesting:")
    print(y_test.value_counts())

    print("\n── VALIDATION ─────────────────────────────")
    print("NaN in X_train:", X_train.isna().sum().sum())
    print("NaN in X_test:", X_test.isna().sum().sum())
    print("Feature count:", X_train.shape[1])
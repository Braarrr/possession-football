
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from models.prepare_dataset import prepare_training_data
from models.train_hybrid import train_hybrid, predict_hybrid
from possession_pipeline import parse_smart_tagging, build_dead_ball_mask
from starter.baseline import smooth


X_train, X_test, y_train, y_test = prepare_training_data()

dead_mask = build_dead_ball_mask(parse_smart_tagging())

if len(dead_mask) != len(X_train) + len(X_test):
    raise ValueError("Dead-ball mask and dataset lengths do not match.")

n_train = len(X_train)
dead_train = dead_mask[:n_train]
dead_test = dead_mask[n_train:]
live_test = ~dead_test

# Train and predict using the existing Hybrid RF
model = train_hybrid(X_train, y_train, dead_train)
raw_predictions = predict_hybrid(model, X_test, dead_test)

print("\n=== BASELINE HYBRID RF ===")
print("Accuracy:", accuracy_score(y_test, raw_predictions))
print("Macro F1:", f1_score(
    y_test, raw_predictions,
    labels=["H", "A", "D", "X"],
    average="macro",
    zero_division=0,
))

# Find contiguous live segments
live_indices = np.flatnonzero(live_test)
segments = np.split(
    live_indices,
    np.where(np.diff(live_indices) != 1)[0] + 1,
)

code_map = {"H": 0, "A": 1, "X": -1}

for window in [25, 50, 125]:
    smoothed = raw_predictions.copy()

    for segment in segments:
        if len(segment) == 0:
            continue

        codes = np.array(
            [code_map[label] for label in raw_predictions[segment]],
            dtype=np.int8,
        )

        labels = smooth(codes, window)

        # Baseline's D means no H/A vote.
        # In a live segment, preserve that meaning as X.
        labels[labels == "D"] = "X"

        smoothed[segment] = labels

    print(f"\n=== HYBRID RF + SMOOTH {window} ===")
    print("Accuracy:", accuracy_score(y_test, smoothed))
    print("Macro F1:", f1_score(
        y_test, smoothed,
        labels=["H", "A", "D", "X"],
        average="macro",
        zero_division=0,
    ))

    print("D frames preserved:", np.array_equal(
        smoothed[dead_test],
        raw_predictions[dead_test],
    ))

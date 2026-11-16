
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from games import GAME, GAMES
from models.prepare_dataset import prepare_training_data
from possession_pipeline import parse_smart_tagging, build_dead_ball_mask


# Prepare existing dataset
X_train, X_test, y_train, y_test = prepare_training_data()

df = pd.read_csv("data/processed_tracking.csv")
df = df.sort_values("frame_id").reset_index(drop=True)

L, W = GAMES[GAME]["pitch"]
fps = GAMES[GAME]["fps"]
n_train = len(X_train)

# Derive ball speed from consecutive coordinates
xy = df[["ball_x", "ball_y"]]
valid = xy.notna().all(axis=1)
consecutive = (
    valid
    & valid.shift(1, fill_value=False)
    & df["frame_id"].diff().eq(1)
)

speed = (
    np.sqrt(
        (xy["ball_x"].diff() * L) ** 2
        + (xy["ball_y"].diff() * W) ** 2
    ) * fps
).where(consecutive)

# Causal rolling mean: current frame + previous 4 frames
rolling_speed = speed.rolling(window=5, min_periods=5).mean()

# Impute using training medians only
speed_median = speed.iloc[:n_train].median()
rolling_median = rolling_speed.iloc[:n_train].median()

X_train_speed = X_train.copy()
X_test_speed = X_test.copy()

X_train_speed["derived_ball_speed"] = (
    speed.iloc[:n_train].fillna(speed_median).to_numpy()
)
X_test_speed["derived_ball_speed"] = (
    speed.iloc[n_train:].fillna(speed_median).to_numpy()
)

X_train_rolling = X_train_speed.copy()
X_test_rolling = X_test_speed.copy()

X_train_rolling["rolling_ball_speed"] = (
    rolling_speed.iloc[:n_train].fillna(rolling_median).to_numpy()
)
X_test_rolling["rolling_ball_speed"] = (
    rolling_speed.iloc[n_train:].fillna(rolling_median).to_numpy()
)

# Same dead-ball mask and chronological split
dead_mask = build_dead_ball_mask(parse_smart_tagging())
dead_train = dead_mask[:n_train]
dead_test = dead_mask[n_train:]

live_train = ~dead_train
live_test = ~dead_test

experiments = [
    ("Baseline", X_train, X_test),
    ("Derived Speed", X_train_speed, X_test_speed),
    ("Derived + Rolling", X_train_rolling, X_test_rolling),
]

seeds = [42, 21, 7, 123, 2026]
results = []

for name, Xtr, Xte in experiments:
    for seed in seeds:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
            class_weight=None,
        )

        model.fit(Xtr.loc[live_train], y_train.loc[live_train])

        predictions = model.predict(Xte.loc[live_test])
        y_true = y_test.to_numpy()[live_test]

        h_f1, a_f1, x_f1 = f1_score(
            y_true,
            predictions,
            labels=["H", "A", "X"],
            average=None,
        )

        macro = f1_score(
            y_true,
            predictions,
            labels=["H", "A", "X"],
            average="macro",
        )

        results.append({
            "experiment": name,
            "seed": seed,
            "H_f1": h_f1,
            "A_f1": a_f1,
            "X_f1": x_f1,
            "live_macro_f1": macro,
        })

        print(
            f"{name:18} seed={seed:4} | "
            f"H={h_f1:.4f} A={a_f1:.4f} X={x_f1:.4f} "
            f"Macro={macro:.6f}"
        )

results_df = pd.DataFrame(results)

print("\n=== MEAN AND STD ===")
print(
    results_df.groupby("experiment")[
        ["H_f1", "A_f1", "X_f1", "live_macro_f1"]
    ].agg(["mean", "std"]).round(6)
)

print("\n=== PAIRED MACRO F1 DIFFERENCES ===")
pivot = results_df.pivot(
    index="seed",
    columns="experiment",
    values="live_macro_f1",
)

print("\nDerived Speed vs Baseline:")
print((pivot["Derived Speed"] - pivot["Baseline"]).round(6))

print("\nDerived + Rolling vs Baseline:")
print((pivot["Derived + Rolling"] - pivot["Baseline"]).round(6))

print("\nRolling contribution vs Derived Speed:")
print((pivot["Derived + Rolling"] - pivot["Derived Speed"]).round(6))


# Feature importance for Derived + Rolling
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    class_weight=None,
)

model.fit(
    X_train_rolling.loc[live_train],
    y_train.loc[live_train],
)

importance = pd.Series(
    model.feature_importances_,
    index=X_train_rolling.columns,
).sort_values(ascending=False)

print("\n=== DERIVED + ROLLING FEATURE IMPORTANCE ===")
print(importance.head(20).to_string())
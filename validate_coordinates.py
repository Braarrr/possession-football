import numpy as np

from starter.load_atd import load


dataset = load()
df = dataset.to_df()

# Identify player coordinate pairs
player_slots = sorted(
    {
        col[:-2]
        for col in df.columns
        if col.endswith("_x")
        and col[:-2].isdigit()
        and f"{col[:-2]}_y" in df.columns
    },
    key=int
)

coordinate_pairs = [
    ("ball_x", "ball_y"),
    *[
        (f"{slot}_x", f"{slot}_y")
        for slot in player_slots
    ]
]

print("=== COORDINATE VALIDATION ===")
print(f"Total frames: {len(df)}")
print(f"Player tracking slots: {len(player_slots)}")

for x_col, y_col in coordinate_pairs:
    if x_col not in df.columns or y_col not in df.columns:
        print(f"{x_col}/{y_col}: columns missing")
        continue

    x = df[x_col]
    y = df[y_col]

    x_missing = x.isna()
    y_missing = y.isna()

    missing_pair = (x_missing | y_missing).sum()
    partial_pair = (x_missing ^ y_missing).sum()

    x_out = (
        x.notna()
        & np.isfinite(x)
        & ~x.between(0, 1)
    ).sum()

    y_out = (
        y.notna()
        & np.isfinite(y)
        & ~y.between(0, 1)
    ).sum()

    non_finite = (
    np.isinf(x).sum()
    + np.isinf(y).sum()
    )

    print(
        f"{x_col}/{y_col}: "
        f"missing_pairs={missing_pair}, "
        f"partial_pairs={partial_pair}, "
        f"out_of_range={x_out + y_out}, "
        f"non_finite={non_finite}"
    )
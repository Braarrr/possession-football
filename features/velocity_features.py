
import numpy as np

from games import GAME, GAMES


def extract_velocity_features(frame_df, game=GAME):
    """
    Extract player speed and derive ball speed from consecutive ATD coordinates.
    Player speeds are provided in m/s.
    Derived ball speed is calculated in m/s.
    Missing values remain unchanged.
    """
    required = {"frame_id", "ball_x", "ball_y"}
    missing = required - set(frame_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    velocity = frame_df[["frame_id"]].copy()

    player_speed_cols = sorted(
        [
            col for col in frame_df.columns
            if col.endswith("_s") and col[:-2].isdigit()
        ],
        key=lambda col: int(col[:-2])
    )

    for col in player_speed_cols:
        velocity[f"{col[:-2]}_player_speed"] = frame_df[col]

    pitch_length, pitch_width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]

    xy = frame_df[["ball_x", "ball_y"]]
    valid = xy.notna().all(axis=1)

    consecutive = (
        valid
        & valid.shift(1, fill_value=False)
        & frame_df["frame_id"].diff().eq(1)
    )

    derived_speed = (
        np.sqrt(
            (xy["ball_x"].diff() * pitch_length) ** 2
            + (xy["ball_y"].diff() * pitch_width) ** 2
        ) * fps
    ).where(consecutive)

    velocity["derived_ball_speed"] = derived_speed.to_numpy()

    return velocity

import numpy as np

from games import GAME, GAMES


def calculate_distance_to_ball(frame_df, game=GAME):
    """
    Calculate each player slot's distance to the ball per frame.

    Input coordinates are normalized to 0-1.
    Output distances are in metres.
    Missing coordinates remain NaN.
    """
    pitch_length, pitch_width = GAMES[game]["pitch"]

    required = {"frame_id", "ball_x", "ball_y"}
    missing = required - set(frame_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ball_x = frame_df["ball_x"] * pitch_length
    ball_y = frame_df["ball_y"] * pitch_width

    distances = frame_df[["frame_id"]].copy()

    player_slots = sorted({
        col[:-2]
        for col in frame_df.columns
        if col.endswith("_x")
        and col[:-2].isdigit()
        and f"{col[:-2]}_y" in frame_df.columns
    }, key=int)

    for slot in player_slots:
        player_x = frame_df[f"{slot}_x"] * pitch_length
        player_y = frame_df[f"{slot}_y"] * pitch_width

        distances[f"{slot}_distance_to_ball"] = np.sqrt(
            (player_x - ball_x) ** 2
            + (player_y - ball_y) ** 2
        )

    return distances
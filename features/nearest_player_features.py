
import numpy as np
from games import GAME, GAMES


def calculate_nearest_player_team(frame_df, team_of_track, game=GAME):
    """
    Find the team of the player closest to the ball per frame.

    H = Home
    A = Away
    None = No available answer
    """
    required = {"frame_id", "ball_x", "ball_y"}
    missing = required - set(frame_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    player_slots = sorted(
        [
            col[:-2]
            for col in frame_df.columns
            if col.endswith("_x")
            and col[:-2].isdigit()
            and f"{col[:-2]}_y" in frame_df.columns
        ],
        key=int
    )

    if len(player_slots) != len(team_of_track):
        raise ValueError(
            f"Player slot mismatch: {len(player_slots)} slots, "
            f"{len(team_of_track)} team assignments"
        )

    if player_slots != [str(i) for i in range(1, len(player_slots) + 1)]:
        raise ValueError("Player slots must be numbered consecutively from 1")

    pitch_length, pitch_width = GAMES[game]["pitch"]

    ball_x = frame_df["ball_x"].to_numpy() * pitch_length
    ball_y = frame_df["ball_y"].to_numpy() * pitch_width

    best_distance = np.full(len(frame_df), np.inf)
    nearest_team = np.full(len(frame_df), None, dtype=object)

    for index, slot in enumerate(player_slots):
        player_x = frame_df[f"{slot}_x"].to_numpy() * pitch_length
        player_y = frame_df[f"{slot}_y"].to_numpy() * pitch_width

        valid = (
            np.isfinite(ball_x) &
            np.isfinite(ball_y) &
            np.isfinite(player_x) &
            np.isfinite(player_y)
        )

        distance = (
            (player_x - ball_x) ** 2 +
            (player_y - ball_y) ** 2
        )

        closer = valid & (distance < best_distance)

        team = "H" if team_of_track[index] == 0 else "A"

        nearest_team[closer] = team
        best_distance[closer] = distance[closer]

    result = frame_df[["frame_id"]].copy()
    result["nearest_player_team"] = nearest_team

    return result

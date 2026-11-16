
def calculate_ball_visibility(frame_df):
    """
    Determine whether the ball coordinates are available per frame.
    1 = visible, 0 = missing.
    """
    required = {"frame_id", "ball_x", "ball_y"}
    missing = required - set(frame_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    visibility = frame_df[["frame_id"]].copy()

    visibility["ball_visible"] = (
        frame_df["ball_x"].notna() &
        frame_df["ball_y"].notna()
    ).astype("int8")

    return visibility

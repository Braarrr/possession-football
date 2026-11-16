
import pandas as pd

from possession_pipeline import (
    parse_smart_tagging,
    build_dead_ball_mask,
    parse_atd_txt,
    nearest_player_labels,
    majority_vote_smooth,
)


def generate_pseudo_labels():
    smart_df = parse_smart_tagging()
    dead_mask = build_dead_ball_mask(smart_df)

    data = parse_atd_txt(
        max_frames=None,
        skip_frames=0,
    )

    raw_labels = nearest_player_labels(data, dead_mask)

    pseudo_labels = majority_vote_smooth(
        raw_labels,
        window=50,
    )

    labels = pd.DataFrame({
        "frame_id": data["frame_ids"],
        "pseudo_label": pseudo_labels,
    })

    return labels


if __name__ == "__main__":
    labels = generate_pseudo_labels()

    print("\nPseudo-label dataset:")
    print(labels.shape)
    print(labels["pseudo_label"].value_counts())
    print(labels.head())

from starter.load_atd import load
from features.distance_features import calculate_distance_to_ball

dataset = load()
frame_df = dataset.to_df()

distance_df = calculate_distance_to_ball(frame_df)

print(distance_df.shape)
print(distance_df.columns.tolist())
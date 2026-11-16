from starter.load_atd import load
from features.velocity_features import extract_velocity_features

dataset = load()
frame_df = dataset.to_df()

velocity_df = extract_velocity_features(frame_df)

print(velocity_df.shape)
print(velocity_df.columns.tolist())
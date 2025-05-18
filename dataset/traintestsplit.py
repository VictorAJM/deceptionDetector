import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. CONFIG: paths
INPUT_CSV     = "C:/Users/victo/deceptionDetector/dataset/DOLOS/dolos_timestamps.csv"     # your original CSV
VIDEOS_DIR    = "C:/Users/victo/deceptionDetector/dataset/face_frames/"               # path to the folder containing sub-folders named by column 2
TRAIN_CSV     = "train.csv"
TEST_CSV      = "test.csv"

# 2. Read the full CSV (no header, 5 columns)
df = pd.read_csv(
    INPUT_CSV,
    header=None,
    names=["channel", "folder", "start", "end", "label"]
)

# 3. Filter to only those rows whose folder exists
existing = set(os.listdir(VIDEOS_DIR))
df = df[df["folder"].isin(existing)].reset_index(drop=True)

# 4. Split into 80% train / 20% test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,      # for reproducibility
    shuffle=True
)

# 5. Write out WITHOUT headers or indexes to match original format
train_df.to_csv(TRAIN_CSV, index=False, header=False)
test_df.to_csv(TEST_CSV,  index=False, header=False)

print(f"Filtered {len(df)} segments → {len(train_df)} train / {len(test_df)} test")

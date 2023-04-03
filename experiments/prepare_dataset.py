import os
import pandas as pd

FILENAME = "data/fulldataset.csv"
OUTP_DIR = "data/fulldataset_notna_targets_vnum0.csv"

dataset = pd.read_csv(FILENAME, sep = ";")

# Use only the first two targets in the dataset and remove unnecessary columns.
# columns = ["Unnamed: 0", "ROW_ID"]

columns = [
    "target_00"+str(i)
    for i in range(2, 6)
]

dataset.drop(
    columns = columns,
    inplace = True,
)


# For now, I'll only consider rows where both targets are available.
targets = dataset[["target_000", "target_001"]]
dataset = dataset[
    targets.notna().all(1)
]

# Remove features with missing values.
dataset.dropna(
    axis="columns",
    how="any",
    inplace=True,
)

os.makedirs("data", exist_ok=True)
if os.path.exists(OUTP_DIR):
    os.remove(OUTP_DIR)

dataset.to_csv(
    OUTP_DIR,
    sep = ",",
    header = True,
    index = False,
)

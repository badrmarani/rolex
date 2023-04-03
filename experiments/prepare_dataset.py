import os
import pandas as pd

FILENAME = "cont_features.csv"
OUTP_DIR = "data/cont_features_tr.csv"

dataset = pd.read_csv(FILENAME, sep = ",")

# Use only the first two targets in the dataset and remove unnecessary columns.
# columns = ["Unnamed: 0", "ROW_ID"]
# columns += [
#     col for idx, col in enumerate(dataset.columns)
#     if (
#         col.startswith("target") and idx <= 1
#     )
# ]

# dataset.drop(
#     columns = columns,
#     inplace = True,
# )

# For now, I'll only consider rows where both targets are available.
targets = dataset[["target_000", "target_001"]]
notna_dataset = dataset[
    targets.notna().all(1)
]

if not os.path.exists(OUTP_DIR):
    os.makedirs("data", exist_ok=True)
    notna_dataset.to_csv(
        OUTP_DIR,
        sep = ",",
        header = True,
        index = False,
    )

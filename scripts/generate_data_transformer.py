import argparse
import os
import pickle

import pandas as pd
from ctgan.data_transformer import DataTransformer

from uqvae.utils import reproduce


def load(filename):
    data = pd.read_csv(filename, sep=",", index_col=0)
    data.drop(
        columns=[
            "ROW_ID",
        ],
        inplace=True,
    )

    x = data.iloc[:, :135]
    n_unique_vals = pd.DataFrame(
        data={col: len(x[col].unique()) for col in x},
        index=[
            "item",
        ],
    ).T.sort_values(by="item")

    categorical_columns = n_unique_vals[n_unique_vals["item"] == 2].index.to_list()
    categorical_columns += [
        "data_028",
        "data_007",
        "data_030",
        "data_023",
        "data_031",
    ]
    return x, categorical_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/", required=False)
    args = parser.parse_args()

    print("Data transformer")
    production_data, discrete_columns = load("data/fulldataset.csv")
    print("\t+ Train data transformer from scratch.")
    with reproduce(seed=42):
        data_transformer = DataTransformer()
        data_transformer.fit(production_data, discrete_columns)

    print(
        "\t+ From {} columns to {} columns.".format(
            production_data.shape[-1], data_transformer.output_dimensions
        )
    )

    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, "data_transformer.pkl")
    print("\t+ End of script: save data transformer {}".format(out))
    with open(out, "wb") as fout:
        pickle.dump(data_transformer, fout)

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def highly_correlated_features(data, features, correlation_threshold):
    corr_data = abs(data[features].corr())
    correlated_features = np.where(np.abs(corr_data) > correlation_threshold)
    correlated_features = [
        (corr_data.iloc[x, y], x, y)
        for x, y in zip(*correlated_features)
        if x != y and x < y
    ]
    s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0]))

    if s_corr_list != []:
        for idx, (v, i, j) in enumerate(s_corr_list):
            s_corr_list[idx] = (corr_data.index[i], corr_data.columns[j], v)
    out = set(col[0] for col in s_corr_list)
    return list(set(out))


def oversample(data: pd.DataFrame, columns: str = None, oversample_quantile=0.5):
    temp_data = data.copy()
    df_valid_pos_info = pd.DataFrame(
        columns=["column", "value", "index", "diff_quantile", "diff_q75"]
    )
    df_valid_pos_info = df_valid_pos_info.astype(
        {
            "column": "object",
            "value": "float64",
            "index": "int64",
            "diff_quantile": "float64",
            "diff_q75": "float64",
        }
    )

    if columns is None:
        iterator = tqdm(temp_data.columns)
    else:
        iterator = tqdm(columns)

    for col in iterator:
        vc = temp_data[col].value_counts()
        quantile = np.quantile(vc, oversample_quantile, method="lower")

        t = vc.where(vc < quantile).dropna()
        for val, counts in zip(t.index, t):
            diff_quantile = quantile - counts

            val_position_df = np.array(temp_data[temp_data[col] == val].index)
            _tmp_df_valid_pos_info = pd.DataFrame(
                {
                    "column": np.array([col] * len(val_position_df)),
                    "value": np.array([val] * len(val_position_df)),
                    "index": val_position_df,
                    "diff_quantile": diff_quantile.repeat(len(val_position_df)),
                }
            )
            df_valid_pos_info = df_valid_pos_info.append(_tmp_df_valid_pos_info)
            # df_valid_pos_info = pd.concat(
            #     (df_valid_pos_info, _tmp_df_valid_pos_info),
            #     axis=0,
            #     ignore_index=True,
            # )
        df_valid_pos_info.drop_duplicates(subset=["value"], inplace=True)

        unique_val_per_col = df_valid_pos_info[df_valid_pos_info["column"] == col][
            "value"
        ].unique()

        for unique_val in unique_val_per_col:
            t = df_valid_pos_info[df_valid_pos_info["value"] == unique_val]
            idx_oversample = np.random.choice(
                t["index"], t["diff_quantile"].unique()[0].astype("int64")
            )
            # temp_data = pd.concat(
            #     (temp_data, temp_data.iloc[idx_oversample, :]),
            #     axis=0,
            #     ignore_index=True,
            # )
            temp_data = temp_data.append(temp_data.iloc[idx_oversample, :])

        iterator.set_description(f"column:{col}/ new_shape:{temp_data.shape[0]}")
    return temp_data


def get_categorical_columns(data, qq_threshold):
    iterator = tqdm(data.columns, ascii=True, desc="Determine categorical columns")
    categorical_columns = []
    for col in iterator:
        _, fit = stats.probplot(data[col], dist="norm")
        fit = fit[2]
        if fit < qq_threshold:
            categorical_columns += [col]
    return categorical_columns

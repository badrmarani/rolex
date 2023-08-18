import os
import re
from ast import literal_eval
from glob import glob
from typing import Any

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def parse_list(raw: str) -> Any:
    """
    Parses a string representation of a list and returns the corresponding Python object.

    Args:
        raw (str): The string representation of the list.

    Returns:
        Any: The parsed Python object.
    """
    pattern = raw.replace('"', "").replace("\\'", "'")
    return literal_eval(pattern)


def read_tensorboard_logs(pathname: str, save: bool = True) -> None:
    """
    Reads TensorBoard logs from the specified directory and processes them.

    Args:
        pathname (str): The directory path containing the TensorBoard event files.
        save (bool, optional): Whether to save the processed data to CSV files. Defaults to True.
    """
    pathname = os.path.join(pathname, "*/event*")
    for path in glob(pathname):
        item = re.search(r"[-+]?(?:\d*\.*\d+)", path).group()
        ea = event_accumulator.EventAccumulator(
            path,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        _absorb_print = ea.Reload()

        logs = {k: pd.DataFrame(ea.Scalars(k)) for k in ea.Tags()["scalars"]}
        tol = min([v.shape[0] for k, v in logs.items()])
        for k, v in logs.items():
            vv = v.value.values
            if vv.shape[0] == tol:
                logs[k] = vv
            else:
                logs.pop(k)
        df = pd.DataFrame(logs)
        df["quality_score/mean"] = (
            df["quality_score/TVComplement"] + df["quality_score/KSComplement"]
        ) / 2
        if save:
            df.to_csv(path + f"item_{item}.csv", index=False)

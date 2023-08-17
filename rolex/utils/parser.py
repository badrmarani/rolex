import os
import re
from ast import literal_eval
from glob import glob
from typing import Any

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def parse_list(raw: str) -> Any:
    pattern = raw.replace('"', "").replace("\\'", "'")
    return literal_eval(pattern)


def read_tensorboard_logs(pathname: str, save: bool = True) -> None:
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

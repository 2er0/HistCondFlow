from typing import Generator

import numpy as np
import pandas as pd

from dataset.DatasetUtils.GHL.main import load_ghl_g
from dataset.DatasetUtils.Metro.main import load_metro
from dataset.DatasetUtils.calit2.main import load_calit2
from dataset.DatasetUtils.smd.main import load_smd_g

from dataset.DatasetUtils.swat.main import load_swat

generators = ["smd", "occupancy"]  # 'msl', 'smap',
files_with_label = [
    "swat",
    "calit2",
    "ghl",  # old
    "metro",
    # "occupancy1", old
    # "occupancy2", old
]  # 'wadi_new', [28, 33]
files_without_label = ["wadi_ori"]  # 'telenor' [34]


def remove_zero_columns(param, train, test):
    for col in train.columns:
        if col in ["timestamp", "is_anomaly"]:
            continue
        if train[col].nunique() == 1:
            print(
                f"Removing column {col} from train and test set "
                f"because it has only one unique value before normalization"
            )
            train = train.drop(columns=col)
            test = test.drop(columns=col)

    cols = (
            ["timestamp"]
            + [f"value-{coli}" for coli in range(train.shape[1] - 2)]
            + ["is_anomaly"]
    )
    train.columns = cols
    test.columns = cols
    param["channels"] = train.shape[1] - 2
    return param, train, test


def provide_all_datasets():
    for t, dataset in [
        ("with_label", files_with_label),
        ("gen", generators),
        ("without_label", files_without_label),
    ]:
        if t == "gen":
            for d in dataset:
                for g_ in get_generator_by_name(d):
                    yield g_
        elif t == "with_label":
            for d in dataset:
                yield get_files_with_label_by_name(d)
        else:
            raise ValueError("Dataset name not found")


def load_all_real_datasets():
    for x in provide_all_datasets():
        # param, train, test = remove_zero_columns(param, train, test)
        name, param, train, test = x
        yield name, param, train, test


def get_dataset_type(name: str):
    if name in generators:
        return "generator", get_generator_by_name
    elif name in files_with_label:
        return "with_label", get_files_with_label_by_name
    else:
        raise ValueError("Dataset name not found")


def get_dataset_by_name(name: str):
    t, f = get_dataset_type(name)
    if t == "generator":
        for g in f(name):
            if g[0] == name:
                return g
    return f(name)


def get_generator_by_name(name: str) -> Generator:
    # if name == generators[0]:
    #    return load_msl_g()
    # elif name == generators[1]:
    #    return load_smap_g()
    # if name == generators[0]:
    #     return load_ghl_g()
    if name == generators[0]:
        return load_smd_g()
    else:
        raise ValueError(f"No dataset with the given name available")


def get_files_with_label_by_name(name: str) -> (pd.DataFrame, pd.DataFrame):
    if name == files_with_label[0]:
        return load_swat()
    # elif name == files_with_label[1]:
    #     return load_wadi_new()
    elif name == files_with_label[1]:
        return load_calit2()
    elif name == files_with_label[2]:
        return load_ghl_g()
    elif name == files_with_label[3]:
        return load_metro()
    # elif name == files_with_label[4]:
    #     return load_occupancy_1()
    # elif name == files_with_label[5]:
    #     return load_occupancy_2()
    else:
        raise ValueError(f"No dataset with the given name available")


if __name__ == "__main__":
    i = 0
    for sequence in load_all_real_datasets():
        i += 1
        pass
    print(i)
    pass

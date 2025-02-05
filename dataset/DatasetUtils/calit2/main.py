import pathlib
from pathlib import Path
from typing import Generator, List
import datetime as dt

import numpy
import numpy as np
import pandas as pd

from dataset.DatasetUtils.core import core_load, fix_dataset_columns


def load_calit2_old() -> (str, dict, pd.DataFrame, pd.DataFrame):
    base = str(pathlib.Path(__file__).parent.resolve())

    df = pd.read_csv(base + '/output/CalIt2.data', header=None)

    df = df.pivot_table(index=[1, 2], columns=[0], values=[3]).reset_index()
    df.columns = ['day', 'time', 'value-0', 'value-1']

    df['timestamp'] = df['day'].astype(str) + " " + df['time'].astype(str)
    format = '%m/%d/%y %H:%M:%S'
    df['timestamp'] = pd.to_datetime(df['timestamp'], format=format)

    df = pd.concat([df['timestamp'], df['value-0'], df['value-1']], axis=1)
    df['is_anomaly'] = 0

    df.set_index('timestamp', inplace=True)

    events = pd.read_csv(base + '/output/CalIt2.events', header=None)
    for i, row in events.iterrows():
        start = pd.to_datetime(f'{row[0]} {row[1]}', format=format)
        end = pd.to_datetime(f'{row[0]} {row[2]}', format=format)

        current = start
        delta = dt.timedelta(minutes=30)
        while current <= end:
            df.loc[current, 'is_anomaly'] = 1
            current += delta

    df.reset_index(inplace=True)
    # split into train:test
    train_count = int(df.shape[0] * 0.8)
    df_train = df.iloc[:train_count]
    df_test = df.iloc[train_count:]

    return 'calit', {'channels': df_train.shape[1] - 2}, df_train, df_test


def load_calit2() -> (str, dict, pd.DataFrame, pd.DataFrame):
    base = str(pathlib.Path(__file__).parent.resolve())

    df = pd.read_csv(base + '/output/CalIt2-traffic.test.csv')

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # split into train:test
    train_count = int(df.shape[0] * 0.8)
    df_train = df.iloc[:train_count]
    df_test = df.iloc[train_count:]

    columns = ["timestamp"] + [f"value-{i}" for i in range(df_train.shape[1] - 2)] + ["is_anomaly"]
    df_train.columns = columns
    df_test.columns = columns

    return 'calit_v2', {'channels': df_train.shape[1] - 2}, df_train, df_test


if __name__ == '__main__':
    # n, p, train, test = load_calit2_old()
    n, p, train, test = load_calit2()
    pass

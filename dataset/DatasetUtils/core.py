from pathlib import Path

import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def core_load(path, func):
    for p in list(sorted(Path(path).glob('*'))):
        yield p, func(p)


def min_max_scale(ser: pd.Series):
    sc = MinMaxScaler()
    sc.fit(ser.to_numpy().reshape(1, -1))

    def scaler(ser: pd.Series):
        if (ser == ser[0]).all():
            return ser
        return sc.transform(ser.to_numpy().reshape(1, -1)).flatten()

    return scaler


def fix_dataset_columns(train, test, test_label):
    res = []
    empty_columns = []
    for df in [train, test]:
        df.drop('Timestamp', axis=1, inplace=True)
        df = df.index.to_frame(name='timestamp').join(df)
        df.columns = ['timestamp'] + [f'value-{i}' for i in range(df.shape[1] - 1)]
        empty_columns.append([col for col in df.columns if (df[col] == 0).all()])
        df['is_anomaly'] = 0
        res.append(df)

    # Remove columns that are empty in both datasets
    useless_columns = list(set(empty_columns[0]) & set(empty_columns[1]))
    print(f"Empty columns: {useless_columns}")
    train, test = res
    res = []
    for df in [train, test]:
        for col in useless_columns:
            df[col] = 0.5
        # df.drop(useless_columns, axis=1, inplace=True)
        # df.columns = ['timestamp'] + [f'value-{i}' for i in range(df.shape[1] - 2)] + ['is_anomaly']
        res.append(df)

    train, test = res
    test['is_anomaly'] = test_label

    return train, test

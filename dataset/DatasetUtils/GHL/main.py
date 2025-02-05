import pathlib
from typing import Generator

import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from dataset.DatasetUtils.core import min_max_scale


def load_ghl_old() -> (str, dict, pd.DataFrame, Generator):
    base = str(pathlib.Path(__file__).parent.resolve())
    df = pd.read_csv(base + '/output/train_1500000_seed_11_vars_23.csv')

    df.columns = ['timestamp'] + [f'value-{i}' for i in range(df.shape[1] - 1)]
    df['is_anomaly'] = 0

    df_test = df.copy(deep=True)
    # RT Level anomaly
    # unauthorized change of max RT level
    # increase the RT level locally by the factor of the standard deviation
    increase_factor = df_test['value-3'].std()
    for i in [141520, 604538, 716419, 866941, 791956]:
        df_test.loc[i - 5:i + 5, 'value-3'] += increase_factor
        df_test.loc[i - 5:i + 5, 'is_anomaly'] = 1

    return 'ghl', {'channels': df.shape[1] - 2}, df, df_test


def load_ghl_g() -> (str, dict, pd.DataFrame, Generator):
    base = str(pathlib.Path(__file__).parent.resolve())
    df_train = pd.read_csv(base + '/output/train_1500000_seed_11_vars_23.csv')

    tests = {}
    match_list = set(df_train.columns)
    for test in pathlib.Path(base + '/output').glob("*.test.csv"):
        logger.info(f"Loading test data: {test}")
        df_test = pd.read_csv(test)
        number = test.name.split('_')[0]
        tests[number] = df_test

        match_list = match_list & set(df_test.columns)

    match_list = list(match_list)
    train_match_list = [df_train.columns[0]] + match_list
    df_train_current = df_train[train_match_list]
    df_train_current.loc[:, "is_anomaly"] = 0
    columns = ["timestamp"] + [f"value-{i}" for i in range(df_train_current.shape[1] - 2)] + ["is_anomaly"]
    df_train_current.columns = columns

    test_list = []
    for number, df_test in tqdm(tests.items(), desc="Preparing test data"):
        test_match_list = [df_test.columns[0]] + match_list + ["is_anomaly"]
        df_test_current = df_test[test_match_list]
        df_test_current.columns = columns
        test_list.append((number, df_test_current))

    return 'ghl_v2', {'channels': df_train_current.shape[1] - 2,
                      'testconstruct': "generator"}, df_train_current, test_list


if __name__ == '__main__':
    # n, p, train, test = load_ghl_old()
    for n, p, train, test in tqdm(load_ghl_g()):
        print(n)
    pass

import pathlib

import pandas as pd


def load_metro_old() -> (str, dict, pd.DataFrame, pd.DataFrame):
    base = str(pathlib.Path(__file__).parent.resolve())
    df = pd.read_csv(base + '/output/Metro_Interstate_Traffic_Volume.csv', parse_dates=['date_time'])

    df = df[['date_time', 'holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description',
             'traffic_volume']]

    # convert categorical to float
    df['holiday'] = df['holiday'].astype('category').cat.codes.astype('int')
    to_convert = ['weather_main', 'weather_description']
    for col in to_convert:
        df[col] = df[col].astype('category').cat.codes

    # change header names
    df.columns = ['timestamp'] + [f'value-{i}' for i in range(df.shape[1] - 1)]
    df['is_anomaly'] = 0

    # anomalies are missing values
    # they are in the training and test set
    # get time diffs bigger than 6 hours
    diff = df[df['timestamp'].diff().dt.total_seconds() > 3600 * 6]
    # set anomalies
    for i, row in diff.iterrows():
        df.loc[i - 5:i + 5, 'is_anomaly'] = 1

    return 'metro', {'channels': df.shape[1] - 2}, df, df.copy(deep=True)


def load_metro() -> (str, dict, pd.DataFrame, pd.DataFrame):
    base = str(pathlib.Path(__file__).parent.resolve())
    df = pd.read_csv(base + '/output/metro-traffic-volume.test.csv', parse_dates=['timestamp'])

    # split into train:test
    train_count = int(df.shape[0] * 0.8)
    df_train = df.iloc[:train_count]
    df_test = df.iloc[train_count:]

    columns = ["timestamp"] + [f"value-{i}" for i in range(df_train.shape[1] - 2)] + ["is_anomaly"]
    df_train.columns = columns
    df_test.columns = columns

    return 'metro_v2', {'channels': df_train.shape[1] - 2}, df_train, df_test


if __name__ == '__main__':
    # n, p, train, test = load_metro_old()
    n, p, train, test = load_metro()
    pass

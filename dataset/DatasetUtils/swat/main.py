import pathlib

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from dataset.DatasetUtils.core import min_max_scale, fix_dataset_columns


def load_ori():
    with open('input/SWaT_Dataset_Attack_v0.xlsx', 'rb') as file:
        ori_excel = pd.read_excel(file, header=1)
    return ori_excel


def load_from_csv():
    return pd.read_csv('input/SWaT_Dataset_Normal_v1.csv', sep=';', low_memory=False), \
           pd.read_csv('input/SWaT_Dataset_Attack_v0.csv', sep=';', low_memory=False)


def transform_to_float(data: pd.DataFrame):
    # Transform all columns into float64
    for i in list(data):
        data[i] = data[i].apply(lambda x: str(x).replace(",", "."))
    return data.astype(float)


def prepare_dataset():
    normal, attack = load_from_csv()
    normal_values = normal.drop(['Timestamp', 'Normal/Attack'], axis=1)
    normal_values = transform_to_float(normal_values)
    normal = normal.drop(['Normal/Attack'], axis=1)
    attack_label = attack['Normal/Attack']
    attack_values = attack.drop(['Timestamp', 'Normal/Attack'], axis=1)
    attack_values = transform_to_float(attack_values)
    attack = attack.drop(['Normal/Attack'], axis=1)
    joined = pd.concat([normal_values, attack_values])
    scaler = MinMaxScaler()
    scaler.fit(joined)
    normal[normal_values.columns] = scaler.transform(normal_values[normal_values.columns])
    attack[attack_values.columns] = scaler.transform(attack_values[attack_values.columns])

    attack_label = attack_label.apply(lambda x: float(x != 'Normal'))
    normal.to_csv('output/SWaT_Dataset_Normal_v1.csv', index=False)
    attack.to_csv('output/SWaT_Dataset_Attack_v0.csv', index=False)
    attack_label.to_csv('output/SWaT_Dataset_Attack_labels_v0.csv', index=False)


def load_swat() -> (str, dict, pd.DataFrame, pd.DataFrame):
    base = str(pathlib.Path(__file__).parent.resolve())
    train = pd.read_csv(base + '/output/SWaT_Dataset_Normal_v1.csv')
    test = pd.read_csv(base + '/output/SWaT_Dataset_Attack_v0.csv')
    test_label = pd.read_csv(base + '/output/SWaT_Dataset_Attack_labels_v0.csv')

    train, test = fix_dataset_columns(train, test, test_label)

    return 'SWat', {'channels': train.shape[1] - 2}, train, test


if __name__ == '__main__':
    prepare_dataset()
    n, train, test = load_swat()
    pass

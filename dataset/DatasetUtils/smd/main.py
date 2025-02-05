import pathlib
from pathlib import Path
from typing import Generator, List

import numpy
import numpy as np
import pandas as pd

from dataset.DatasetUtils.core import core_load


def load_in_path(base):
    for p in Path(base).glob('*'):
        data = np.loadtxt(p, delimiter=',')
        pass


def load_handler(p):
    train = numpy.loadtxt(p, delimiter=',')
    test = numpy.loadtxt(str(p).replace('/train/', '/test/'), delimiter=',')
    test_label = numpy.loadtxt(str(p).replace('/train/', '/test_label/'), delimiter=',')
    return train, test, test_label


def load_smd_g() -> Generator:
    base = str(pathlib.Path(__file__).parent.resolve())
    set = core_load(base + '/output/train', load_handler)

    for p, s in set:
        train = pd.DataFrame(s[0])
        test = pd.DataFrame(s[1])
        test_label = pd.DataFrame(s[2])
        res = []
        for df in [train, test]:
            df = df.index.to_frame(name='timestamp').join(df)
            df.columns = ['timestamp'] + [f'value-{i}' for i in range(df.shape[1] - 1)]
            df['is_anomaly'] = 0
            res.append(df)
        train, test = res
        test['is_anomaly'] = test_label

        yield 'smd_' + str(p).split('/')[-1].split('.')[0], {'channels': train.shape[1] - 2}, train, test


if __name__ == '__main__':
    # load_in_path('input/ServerMachineDataset/train')
    set = load_smd_g()
    x = next(set)
    x = next(set)
    pass

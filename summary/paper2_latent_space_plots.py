import json

import numpy as np
import pandas as pd
from loguru import logger

import plotly.io as pio
from numba.core.utils import benchmark

pio.kaleido.scope.mathjax = None

from data_handling import prepare_train_data, create_all_test_sets
from dataset.mtads_loading import load_all_stored_datasets
from global_utils import generator_seek, keys_to_numeric, flow_parse_args, restore_parameters, format_exception, \
    restore_config
from models.executor_factory import ExecutorFactory
from plot_utils import plot_parallel_coordinates, plot_sequence_and_latent_space


def reduce_dimensions_by_correlation(train: np.ndarray, test: np.ndarray):
    # reduce dimensions by correlation
    # drop columns with constant values
    drop_columns = np.std(test, axis=0) > 0.1
    test = test[:, drop_columns]
    train = train[:, drop_columns]
    # calculate pearson correlation between all columns
    df = pd.DataFrame(test, columns=[f"{i}" for i in range(test.shape[1])])
    corr = df.corr()
    # calc average correlation per column
    corr = corr.abs().mean(axis=1)
    # sort corr matrix from the lowest to the highest
    corr = corr.sort_values()
    # select the lowest 6 channels
    idx = [int(x) for x in corr.index[:6]]
    keep_train = train[:, idx]
    keep_test = test[:, idx]
    return keep_train, keep_test


if __name__ == "__main__":

    create_paracoord = False
    create_latent_plot = True

    source = "real"
    if source == "fsb":
        results = pd.read_csv("fsb/results-TFselfopt-fsb.csv")
        picks = ["run-20241030_231548-4g4wf37e", "offline-run-20241119_083321-v7u390ec"]
        ranges = [None, None]
        highlights = [{}, {}]
        channels = [None, None]
    elif source == "srb":
        results = pd.read_csv("srb/results-TFselfopt-srb.csv")
        picks = ["run-20250109_221459-ntvswq1x"]
        ranges = [[100_000, 150_000], None, None, None]
        highlights = [{1: [(119040, 119062), (144285, 144306)], 3: [(131083, 131103)]},
                      {}, {}, {}]
        channels = [None, None, None, None]
    elif source == "real":
        results = pd.read_csv("real/results-TFselfopt-real.csv")
        picks = ["run-20241119_135351-oulp81d2", "run-20241121_130848-br9o70a2"]
        ranges = [[9_000, 11_000], [12_600, 12_900]]
        highlights = [{}, {}]
        channels = [(5, 6, 10, 13, 14, 22), (5, 8, 11, 19, 22, 25)]
    else:
        raise ValueError("Unknown source")

    for pick, clip, highlight, channel in zip(picks, ranges, highlights, channels):
        row = results[results["path"].str.contains(pick)]
        config_path = f"{row['path'].values[0]}/config.json"

        with open(config_path, "r") as f:
            args = json.load(f, object_hook=keys_to_numeric)

        # create parallel coordinates plot
        if create_paracoord:
            fig = plot_parallel_coordinates(args["optimization_trace"],
                                            row["dataset"].values[0],
                                            args["parameters"]["model_type"])
            fig.update_layout(margin=dict(r=0, l=40))
            fig.write_image(f"{source}_{pick}_paramsearch.pdf", width=1000, height=300, scale=2)

        # create paired plot with sequence, anomaly and latent representation
        if create_latent_plot:
            full_dataset = row["dataset_full"].values[0]
            benchmark_iter = generator_seek(load_all_stored_datasets(source), 0, drop=False)
            for dataset in benchmark_iter:
                if dataset[0] == full_dataset:
                    break
            else:
                raise ValueError("Dataset not found")
            g, p, trains, tests = dataset
            probs = np.load(f"{row['path'].values[0]}/0_raw_all_prob.npy")

            base_args = flow_parse_args()
            base_args["load_pretrained"] = True
            base_args["pretrained_model"] = row["path"].values[0]
            base_args = restore_config(base_args)
            p = restore_parameters(p, base_args)
            p["device"] = "cpu"

            try:
                (samples, sample_hist, data_w_dates, sample_anomalies, add_noise,
                 normalize_factors) = prepare_train_data(p, trains)
            except (IndexError, TypeError) as e:
                logger.info("Not enough data for training")
                logger.error("BaseException: An exception occurred: {}".format(e))
                print(format_exception(e))
                exit(1)

            test_sequences = create_all_test_sets(tests, p, args["parameters"]["model_type"], normalize_factors)
            executor = ExecutorFactory.create_executor(base_args, p)

            train_individual_probs = executor.predict_individual(None, samples, sample_hist, sample_anomalies)

            name, test, test_hist, test_anomaly, test_dates = next(test_sequences)
            test_prob = executor.predict(None, test, test_hist, test_anomaly, save_output_tensor=False)
            test_individual_probs = executor.predict_individual(None, test, test_hist, test_anomaly)

            if samples.shape[1] > 6:
                # reduce to 6 dimensions
                if channel is None:
                    samples, test = reduce_dimensions_by_correlation(samples, test)
                elif isinstance(channel, tuple):
                    samples, test = samples[:, channel], test[:, channel]
                else:
                    logger.info("Channel is not a tuple, skipping dimension reduction")
                    channel = None

            if clip is not None:
                samples = samples[clip[0]:clip[1]]
                test = test[clip[0]:clip[1]]
                data_w_dates = data_w_dates[clip[0]:clip[1]]
                test_dates = test_dates[clip[0]:clip[1]]
                train_individual_probs = [x[clip[0]:clip[1]] for x in train_individual_probs]
                test_individual_probs = [x[clip[0]:clip[1]] for x in test_individual_probs]
                test_prob = [x[clip[0]:clip[1]] for x in test_prob]
                # rerange test_prob between 0 and 1
                test_prob[0] = (test_prob[0] - test_prob[0].min()) / (test_prob[0].max() - test_prob[0].min())
                test_anomaly = test_anomaly[clip[0]:clip[1]]
            else:
                clip = [0, len(samples)]

            fig = plot_sequence_and_latent_space(samples, data_w_dates,
                                                 test, test_dates,
                                                 train_individual_probs[1], test_individual_probs[1],
                                                 test_prob[0], test_anomaly,
                                                 channel,
                                                 f"{dataset[0]} | {row['model_type'].values[0]}")
            highlight_list = []
            for row, marks in highlight.items():
                for (start, end) in marks:
                    mid = (start + end) / 2 / 1000
                    e = dict(type="rect",
                             xref=f"x{row if row > 1 else ''}",
                             yref=f"y{row if row > 1 else ''}",
                             x0=mid, x1=mid + 2,
                             y0=-1.1, y1=1.1,
                             fillcolor="lightgray",
                             opacity=0.8,
                             layer="below",
                             line_width=0)
                    highlight_list.append(e)

            fig.update_layout(shapes=highlight_list, overwrite=False)
            fig.update_layout(margin=dict(r=0, l=0, b=50, t=50), overwrite=False)
            fig.write_image(f"{source}_{pick}_sequence_latent.pdf", width=900, height=600, scale=1)

    print("Done")

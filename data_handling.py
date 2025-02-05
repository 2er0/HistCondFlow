import os
from types import GeneratorType, FunctionType

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocessing import create_windowed_dataset_from_dict, create_windowed_dataset_from_dict_list, \
    create_dataset_from_dict_matrix, create_windowed_dataset, normalize, \
    create_dataset_from_dataset_with_end_as_validation, prepend_noise, create_dataset_with_past_from_dataset, \
    normalize_complete
from global_utils import format_exception
from models.real_nvp import update_device, OpenDualDataProvider


def reduce_data_for_debug(parameters, sequence, test=False):
    # if debug mode is enabled, reduce the dataset size
    if os.getenv("DATAREDUCE", False) == "True":
        logger.debug("DATAREDUCE MODE")
        # reduce the dataset size for faster debugging
        if "construct" in parameters and parameters["construct"] in ["json", "jsonlist", "jsonmatrix"]:
            # if the dataset is json based, reduce the size via iterator
            train_sequence = {k: sequence[k] for k in list(sequence.keys())[:1000]}
            if test:
                for k in list(sequence.keys())[500:510]:
                    sequence[k]["is_anomaly"] = 1
        else:
            # if the dataset is pandas based, reduce the size via slicing
            sequence = sequence[:1000]
            if test:
                sequence.loc[500:510, 'is_anomaly'] = 1

        logger.warning("Reduced dataset size for debugging and reduced epochs")
        parameters["epochs"] = 12

    return sequence


def window_data(parameters, sequence):
    # TODO fix length of anomaly array
    add_noise = True
    # create windowed dataset from the given training and test dataset
    if "construct" in parameters and parameters["construct"] == "json":
        # handle json based dataset for all models except LSTM
        samples, sample_hist, data_w_dates, anomalies = create_windowed_dataset_from_dict(
            sequence, parameters["past"], parameters["names"]
        )
    elif "construct" in parameters and parameters["construct"] == "jsonlist":
        samples, sample_hist, data_w_dates, anomalies = create_windowed_dataset_from_dict_list(
            sequence, parameters["past"]
        )
    elif "construct" in parameters and parameters["construct"] == "jsonmatrix":
        samples, sample_hist, data_w_dates, anomalies = create_dataset_from_dict_matrix(
            sequence, parameters["past"]
        )
        parameters["past"] = sample_hist.shape[1]
        add_noise = False
    else:
        # handle normal tabular source input data by creating a windowed dataset
        samples, sample_hist, data_w_dates, anomalies = create_windowed_dataset(
            sequence, parameters["channels"], parameters.get("date_column", "timestamp"), parameters["past"]
        )
        # anomalies = sequence["is_anomaly"]
        # anomalies = np.array(anomalies)[parameters["past"]:]

    return samples, sample_hist, data_w_dates, anomalies, add_noise


def prepare_train_data(parameters, train_sequence):
    logger.info("Prepare training data")
    # reduce the dataset size for debugging
    train_sequence = reduce_data_for_debug(parameters, train_sequence)

    logger.info(f"Length of train_sequence: {len(train_sequence)}")

    # create windowed dataset for training and test data
    samples, sample_hist, data_w_dates, anomalies, add_noise = window_data(parameters, train_sequence)

    # normalize the dataset
    if "normalized" not in parameters or not parameters["normalized"]:
        samples, sample_hist, normalize_factors = normalize(samples, sample_hist)
    elif "normalize_factors" in parameters and len(parameters["normalize_factors"]) > 0:
        normalize_factors = parameters["normalize_factors"]
        samples, sample_hist, _ = normalize(samples, sample_hist, **normalize_factors)
    elif "normalize_factors" in parameters and len(parameters["normalize_factors"]) == 0:
        samples, sample_hist, normalize_factors = normalize(samples, sample_hist)
    else:
        normalize_factors = {}

    return samples, sample_hist, data_w_dates, anomalies, add_noise, normalize_factors


# TODO this is not finished, don't know if needed
@DeprecationWarning
def prepare_train_data_with_normalization_first(parameters, train_sequence):
    logger.info("Prepare training data")
    # reduce the dataset size for debugging
    train_sequence = reduce_data_for_debug(parameters, train_sequence)

    # normalize the dataset
    if "normalized" not in parameters or not parameters["normalized"]:
        train_sequence, normalize_factors = normalize_complete(parameters, train_sequence)
    elif "normalize_factors" in parameters and len(parameters["normalize_factors"]) > 0:
        normalize_factors = parameters["normalize_factors"]
        train_sequence, _ = normalize_complete(parameters, train_sequence, **normalize_factors)
    elif "normalize_factors" in parameters and len(parameters["normalize_factors"]) == 0:
        train_sequence, normalize_factors = normalize_complete(parameters, train_sequence)
    else:
        normalize_factors = {}

    # create windowed dataset for training and test data
    samples, sample_hist, data_w_dates, anomalies, add_noise = window_data(parameters, train_sequence)

    return samples, sample_hist, data_w_dates, anomalies, add_noise, normalize_factors


def prepare_test_data(parameters, model_type, test_sequence, normalize_factors):
    # reduce the dataset size for debugging
    test_sequence = reduce_data_for_debug(parameters, test_sequence, test=True)

    # create windowed dataset for training and test data
    test, test_hist, test_data_w_dates, anomalies, _ = window_data(parameters, test_sequence)

    if "normalized" not in parameters or not parameters["normalized"]:
        test, test_hist, _ = normalize(test, test_hist, **normalize_factors)
    elif "normalize_factors" in parameters:
        normalize_factors = parameters["normalize_factors"]
        test, test_hist, _ = normalize(test, test_hist, **normalize_factors)

    return test, test_hist, test_data_w_dates, anomalies


def create_all_test_sets(test_sequence, parameters, model_type, normalize_factors):
    logger.info("Prepare test data")
    # test_sequences = []
    if isinstance(test_sequence, FunctionType):
        test_iter = test_sequence()
    elif isinstance(test_sequence, list):
        test_iter = test_sequence
    else:
        test_iter = [("0", test_sequence)]

    for test_name, test_data in tqdm(test_iter, desc="Preprocessing test data"):
        logger.info(f"Length of test_data: {len(test_data)}")
        test, test_hist, test_data_w_dates, anomalies = prepare_test_data(parameters, model_type, test_data,
                                                                          normalize_factors)
        yield test_name, test, test_hist, anomalies, test_data_w_dates


@DeprecationWarning
def prepare_data(run_args, parameters, model_type, train_sequence, test_sequence):
    # reduce the dataset size for debugging
    train_sequence, test_sequence = reduce_data_for_debug(parameters, train_sequence)

    # prepare the dataset for the training process
    if model_type == 3:
        # sequential data loading for stateful LSTM
        global DEVICE
        DEVICE = update_device("cpu")
        batch_size = 1
        past = 1
        parameters["past"] = past
    else:
        batch_size = run_args.batch_size

    # create windowed dataset for training and test data
    samples, sample_hist, data_w_dates, _, add_noise = window_data(parameters, train_sequence)
    test, test_hist, test_data_w_dates, anomalies, _ = window_data(parameters, test_sequence)

    # TODO return normalize factors for test data as well
    samples, sample_hist = normalize(samples, sample_hist)
    test, test_hist = normalize(test, test_hist)

    if model_type == 3:
        # handle json based dataset for LSTM
        # use the end as validation set
        (
            (samples, sample_hist),
            (train_samples, train_hist_samples),
            (validation_samples, validation_hist_samples),
            data_w_dates,
            (selection, validation_selection)
        ) = create_dataset_from_dataset_with_end_as_validation(samples, sample_hist, data_w_dates,
                                                               parameters["past"], add_noise)

        if "construct" in parameters and parameters["construct"] in ["json", "jsonlist"]:
            # handel json based dataset for LSTM stateful model
            # by adding noice to the first samples
            test, test_hist, test_data_w_dates = prepend_noise(
                test, test_hist, test_data_w_dates
            )
            # update the anomaly tracker with the newly added sample
            anomalies = np.concatenate([[0], anomalies], axis=0)
        elif "construct" in parameters and parameters["construct"] == "jsonmatrix":
            logger.info("No noise to add")
        else:
            # handle normal tabular source input data by creating a windowed dataset
            # and adding noise to the first samples
            test, test_hist, test_data_w_dates = prepend_noise(
                test, test_hist, test_data_w_dates
            )
    else:
        # create training and validation set for all other models
        # by splitting randomly picking sections as validation and add buffers around the validation sets
        try:
            (
                (samples, sample_hist),
                (train_samples, train_hist_samples),
                (validation_samples, validation_hist_samples),
                data_w_dates,
                (selection, validation_selection)
            ) = create_dataset_with_past_from_dataset(samples, sample_hist, data_w_dates,
                                                      parameters["past"])
        except IndexError as e:
            logger.error(f"IndexError: An exception occurred: {e}")
            print(format_exception(e))
            return

    if model_type == 0:
        logger.info("Vanilla flow, provide the main input and past as full main input")
        train_hist_flat = train_hist_samples.reshape((train_hist_samples.shape[0], -1))
        train_samples = np.hstack([train_samples, train_hist_flat])
        validation_hist_flat = validation_hist_samples.reshape((validation_hist_samples.shape[0], -1))
        validation_samples = np.hstack([validation_samples, validation_hist_flat])
        test_hist_flat = test_hist.reshape((test_hist.shape[0], -1))
        test = np.hstack([test, test_hist_flat])

    # prepare the training data loaders for extended input data
    train_provider = OpenDualDataProvider(
        train_samples.astype(np.float32),
        train_hist_samples.astype(np.float32)
    )
    train_loader = DataLoader(train_provider, batch_size=batch_size, shuffle=False)
    train_batches = len(train_loader)
    if train_batches == 0:
        raise ValueError("No training data available, past requirement is too high")
    # prepare the validation data loader for extended input data
    validation_provider = OpenDualDataProvider(
        validation_samples.astype(np.float32),
        validation_hist_samples.astype(np.float32)
    )
    validation_loader = DataLoader(
        validation_provider, batch_size=batch_size, shuffle=False
    )
    validation_batches = len(validation_loader)

    return (train_loader, train_batches, validation_loader, validation_batches,
            test, test_hist, test_data_w_dates, anomalies,
            batch_size)

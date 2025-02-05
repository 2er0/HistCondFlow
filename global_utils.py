import argparse
import inspect
import json
import sys
import traceback
from copy import deepcopy
from dataclasses import make_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from models.flow_factory import model_types_parameters

global_config = {
    "code_version": 1
}


def flow_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="timeflow-base",
                        help="Name of the project tracking on a W&B server")
    parser.add_argument("-u", "--user", type=str, default="timeflow",
                        help="Name of the user for tracking on a W&B server")
    parser.add_argument("-wm", "--wandb_mode", type=str, default="offline",
                        help="Run W&B in online or offline mode")
    parser.add_argument("--slurm_id", type=int, default=0,
                        help="Slurm job id")

    parser.add_argument("--code_version", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="Code version to run")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--early_stopping", type=int, default=50,
                        help="Early stopping patience")

    parser.add_argument("--past_range", type=int, default=[3], nargs="+",
                        help="List of number indicating the steps into the past")
    parser.add_argument("--nruns", type=int, default=2,
                        help="Number of runs per configuration pair")
    parser.add_argument("--self_optimization", type=bool, default=False,
                        help="Run self optimization for the model")

    parser.add_argument("--model_types", type=str, default=["RealNVP"], nargs="+",
                        help="Model type one or list of ['RealNVP', 'RealNVP-extended', …, 'tcNF-extended', …]")
    parser.add_argument("--coupling_layers", type=int, default=[7], nargs="+",  # [3, 5, 7, 9]
                        help="List with odd numbers indicating the amount coupling layers to use.\n"
                             "One final coupling layer will be added automatically")

    parser.add_argument("-d", "--dataset", type=str, default="fsb",
                        choices=["fsb", "srb", "real", "statnett", "aneo", "aneo_complex", "aneo_dynamic"],
                        help="Name name of the benchmark suite to generate: 'fsb' or 'srb'")
    parser.add_argument("--generator_seek", type=int, default=0,
                        help="Dataset generator seek to start with a different dataset")
    parser.add_argument("--generator_stop", type=int, default=1e10,
                        help="Dataset generator stop at a specific dataset")

    parser.add_argument("--load_pretrained", type=bool, default=False,
                        help="Load pretrained model")
    parser.add_argument("--pretrained_model", type=str, default="",
                        help="Path to the pretrained model")

    args = parser.parse_args()
    vargs = vars(args)
    run_arguments = make_dataclass("RunArguments", vargs.keys())(**vargs)

    global_config["code_version"] = run_arguments.code_version

    if run_arguments.self_optimization:
        run_arguments.nruns = 1
    if run_arguments.load_pretrained and run_arguments.pretrained_model == "":
        raise ValueError("Pretrained model path is missing")

    args = run_arguments.__dict__

    if run_arguments.load_pretrained:
        args = restore_config(args)

    return args


def format_exception(e):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(exception_list)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str


def data_guard(func):
    """
    Decorator to guard the data from unwanted changes.
    :param func:
    :return:
    """

    def inner(*args, **kwargs):
        args_ = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                args_.append(arg.copy())
            elif isinstance(arg, (list, dict)):
                args_.append(deepcopy(arg))
            else:
                args_.append(arg)

        args_ = tuple(args_)
        return func(*args_, **kwargs)

    return inner


def re_range(nll_prob):
    if np.min(nll_prob) == np.max(nll_prob):
        return nll_prob
    return (nll_prob - np.min(nll_prob)) / (np.max(nll_prob) - np.min(nll_prob))


def drop_time_features(data: np.ndarray, parameters: dict):
    if np.all(data[:, -1] == data[0, -1]):
        less = 1
    else:
        less = 0
    if "number_of_time_features" in parameters:
        data = data[:, :-(parameters["number_of_time_features"] + less)]
    else:
        data = data
    return data


def generator_seek(gen, seek=0, stop=1e10, drop=False):
    for i, g in enumerate(gen):
        if drop:
            if "no-anomaly" not in g[0]:
                continue
        if seek <= i < stop:
            yield g


class Encoders(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def keys_to_numeric(x):
    obj = {}
    for k, v in x.items():
        if k.lstrip('-').isdigit():
            obj[int(k)] = v
        elif k.strip().replace(".", "").isnumeric():
            obj[float(k)] = v
        else:
            obj[k] = v
    return obj


def save_config(config, filename):
    with open(filename, "w") as f:
        json.dump(config, f, indent=4, cls=Encoders)


def build_known_args_dict(func, params):
    specs = inspect.getfullargspec(func)
    build_args = {}
    for i, arg in enumerate(specs.args):
        if arg not in params:
            if len(specs.defaults) != len(specs.args):
                raise ValueError(f"Not enough default arguments")
            if specs.defaults[i] is not None:
                build_args[arg] = specs.defaults[i]
            else:
                raise ValueError(f"Argument {arg} is missing in the parameters")
        else:
            build_args[arg] = params[arg]
    return build_args


def handle_graceful_stop(t, p):
    if (datetime.now() - t).seconds > 2700:  # 45min run time max per process
        logger.warning(f"Terminating process {p.pid} gracefully")
        p.join(timeout=60)
        p.terminate()
    elif (datetime.now() - t).seconds > 3000:  # 30min run time max per process
        logger.error(f"Killing process {p.pid}")
        p.join(timeout=60)
        p.kill()


def restore_config(args):
    logger.info("Restore config from pretrained model")

    model_folder = Path(args["pretrained_model"])
    if not model_folder.exists():
        raise ValueError(f"Model folder {model_folder} does not exist")

    config_file = model_folder / "config.json"
    if not config_file.exists():
        raise ValueError(f"Config file {config_file} does not exist")

    with open(config_file, "r") as f:
        config = json.load(f, object_hook=keys_to_numeric)

    # load the config as backup and later use into the run arguments
    args["loaded_config"] = config
    # update the run arguments with the run arguments in the saved config
    args["dataset"] = config["run_args"]["dataset"]
    parameters = config["parameters"]
    args["model_types"] = [parameters["model_type"]]
    args["coupling_layers"] = [parameters["coupling_layers"]]
    args["past_range"] = [parameters["past"]]
    args["code_version"] = config["run_args"]["code_version"]

    return args


def restore_parameters(dataset_params, config):
    logger.info("Restore base/general parameters from pretrained model")
    parameters = config["loaded_config"]["parameters"]

    dataset_params["code_version"] = parameters["code_version"]
    dataset_params["model_type"] = parameters["model_type"]
    dataset_params["group"] = parameters["group"]
    dataset_params["epochs"] = parameters["epochs"]
    dataset_params["past"] = parameters["past"]
    dataset_params["fixed_past"] = parameters["fixed_past"]
    dataset_params["coupling_layers"] = parameters["coupling_layers"]
    dataset_params["input_shape"] = parameters["input_shape"]
    dataset_params["hist_shape"] = parameters["hist_shape"]
    dataset_params["normalized"] = parameters["normalized"]
    dataset_params["z_score"] = parameters["z_score"]
    dataset_params["normalize_factors"] = parameters["normalize_factors"] if "normalize_factors" in parameters else {}

    logger.info("Restore model specific parameters from pretrained model")
    model_params = set(model_types_parameters["default"].keys()) | set(
        model_types_parameters[dataset_params["model_type"]].keys())

    for k in model_params:
        if k == "seed" and "seed" not in parameters:
            dataset_params["seed"] = 42
        elif k in parameters:
            dataset_params[k] = parameters[k]
        else:
            raise ValueError(f"Parameter {k} is missing in the pretrained model")

    dataset_params["pretrained_model_path"] = Path(config["pretrained_model"]) / "best_model.pth"

    return dataset_params

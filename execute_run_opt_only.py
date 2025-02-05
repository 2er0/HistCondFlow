import inspect
import os
import time
from datetime import datetime, timedelta
from functools import partial
from types import GeneratorType, FunctionType

from torch.multiprocessing import Process, set_start_method
from typing import Union

import torch
from loguru import logger
from tqdm import tqdm

import wandb
from data_handling import prepare_train_data, create_all_test_sets
from dataset.mtads_loading import load_all_stored_datasets
from global_utils import (
    format_exception,
    generator_seek,
    flow_parse_args, handle_graceful_stop,
    restore_parameters
)
from models.executor_factory import ExecutorFactory
from models.flow_factory import model_types
from models.flow_param_optimizer import CMAParamOptimizer
from models.real_nvp import DEVICE, update_device


def load_data(generator=(), past=3):
    # get the dataset and prepare the training and testing data
    if isinstance(generator, GeneratorType) or isinstance(generator, tuple):
        _generator = generator
    elif isinstance(generator, FunctionType):
        # check if the generator is a function and call it with the needed arguments
        args = inspect.getfullargspec(generator).kwonlyargs
        kargs = {}
        if "past_size" in args:
            kargs["past_size"] = past
        _generator = generator(**kargs)
    else:
        _generator = generator

    if _generator[1].get("callback", False):
        logger.info("Callback function collected, disable callback for other uses")
        group, parameters, train_sequence, test_sequence, callback = _generator
        parameters["callback"] = False
    else:
        group, parameters, train_sequence, test_sequence = _generator
        callback = None
    return group, parameters.copy(), train_sequence, test_sequence, callback


def run(run_args, model_type: Union[int, str] = 1, generator=(), past=3, coupling_number=3, device="cpu"):
    group, parameters, train_sequence, test_sequence, _ = load_data(generator, past)
    # clean up memory
    del generator

    if DEVICE == torch.device("cpu"):
        logger.info("No GPU available - running on CPU")
        device = update_device("cpu")
    elif isinstance(device, str):
        device = update_device(device)
    logger.info(f"Running with device: {device}")

    logger.info(f"START EXPERIMENT | {group}")
    if past is not None:
        parameters["past"] = past
    else:
        parameters["past"] = 25

    try:
        (samples, sample_hist, data_w_dates, sample_anomalies, add_noise,
         normalize_factors) = prepare_train_data(parameters, train_sequence)
        # clean up memory
        del train_sequence
    except (IndexError, TypeError) as e:
        logger.info("Not enough data for training")
        logger.error("BaseException: An exception occurred: {}".format(e))
        print(format_exception(e))
        return

    test_sequences = create_all_test_sets(test_sequence, parameters, model_type, normalize_factors)

    # convert model_type int to model_type string
    if model_type in model_types:
        model_type = model_types[model_type]
    parameters["code_version"] = run_args["code_version"]
    parameters["model_type"] = model_type
    parameters["group"] = group
    parameters["device"] = device

    # prepare configuration for running and logging
    parameters["epochs"] = run_args["epochs"]
    parameters["coupling_layers_"] = coupling_number
    parameters["coupling_layers"] = coupling_number + 1
    parameters["input_shape"] = samples.shape
    parameters["hist_shape"] = sample_hist.shape
    parameters["normalized"] = True
    parameters["z_score"] = normalize_factors["z_score"] if "z_score" in normalize_factors else False
    parameters["normalize_factors"] = normalize_factors

    # setup self optimization parameter search
    if run_args["self_optimization"]:
        # optimize the parameters for the model
        optimization_test_sequence = None
        logger.info(f"Checking for optimization test sequence on dataset '{run_args['dataset']}'")
        if run_args["dataset"] in ["fsb", "srb"]:
            # Find the corresponding sequence pair for the optimization testing
            if "-no-anomaly" in group:
                find_supervised_training_sequence = group.replace("-no-anomaly", "")
                # use a balanced metric between auc and vus for the optimization guidance
                optimization_metric = "auc_vus_balance"
            else:
                find_supervised_training_sequence = f"{group}-no-anomaly"
                # use a balanced metric between the validation and evaluation losses for the optimization guidance
                optimization_metric = "losses"

            for option in tqdm(load_all_stored_datasets(run_args["dataset"]), desc="Searching for sequence"):
                if find_supervised_training_sequence == option[0] and group != option[0]:
                    logger.info(f"Found sequence: {option[0]}")
                    _, _, optimization_test_sequence, _, _ = load_data(option)
                    break

            # prepare the test sequences for the optimization
            optimization_goal_sequences = create_all_test_sets(optimization_test_sequence, parameters, model_type,
                                                               normalize_factors)
            _, opt_eval, opt_eval_hist, opt_eval_anomalies, _ = next(optimization_goal_sequences)[0]
        else:
            logger.warning("Using best validation loss to guide the optimization")
            opt_eval, opt_eval_hist, opt_eval_anomalies = None, None, None
            # use the validation loss as the optimization metric
            optimization_metric = "val_loss"

        _, opt_test, opt_test_hist, opt_test_anomalies, _ = next(test_sequences)

        optimizer = CMAParamOptimizer(device, run_args, parameters, samples, sample_hist, sample_anomalies,
                                      opt_eval, opt_eval_hist, opt_eval_anomalies,
                                      opt_test, opt_test_hist, opt_test_anomalies,
                                      metric=optimization_metric)
        optimizer.optimize()
        best_model_params, optimization_trace, best_model_weights = optimizer.get_results()
        parameters.update(best_model_params)
    else:
        raise ValueError("Self optimization is not enabled, needs to be enabled to run this script")

    executor = ExecutorFactory.create_executor(run_args, parameters)
    executor.load_model_from_weights(best_model_weights)
    executor.start_wandb_logging()
    run_dir = executor.get_run_dir()
    executor.update_config({"optimization_trace": optimization_trace})
    executor.save_model()

    try:
        # run scoring for the train and test dataset
        scores = {}

        test_sequences = create_all_test_sets(test_sequence, parameters, model_type, normalize_factors)

        for test_name, _test_, _test_hist_, _is_anomaly_, test_data_w_dates in test_sequences:
            logger.info(f"Testing on {test_name}")

            re_ranged_nll_prob, all_probs_np = executor.predict(test_name, _test_, _test_hist_, _is_anomaly_)
            individual_probs, latent = executor.predict_individual(test_name, _test_, _test_hist_, _is_anomaly_)

            scores[test_name] = executor.score(re_ranged_nll_prob, _is_anomaly_)
            logger.info(scores[test_name])

            executor.time_line_plot(test_name, samples, _test_, all_probs_np, _is_anomaly_, re_ranged_nll_prob,
                                    data_w_dates, test_data_w_dates)

            executor.latent_space_plot(test_name, samples, _test_, latent, individual_probs,
                                       sample_anomalies, _is_anomaly_)

            executor.roc_curve_plot(test_name, re_ranged_nll_prob, _is_anomaly_)

            executor.update_config({"test": scores})
            executor.save_config()

        logger.info("Calculate mean scores")
        mean_score = None
        for _, score in scores.items():
            if mean_score is None:
                mean_score = score
            else:
                for key, value in score.items():
                    mean_score[key] += value
        for key, value in mean_score.items():
            mean_score[key] /= len(scores)

        scores["mean"] = mean_score
        logger.info(f"mean-score: {mean_score}")

        executor.update_config({"test": scores})
        executor.save_config()
        executor.log_score_to_wandb()

    # catch all exceptions and continue with the next run
    except (ValueError, AttributeError, BaseException, RuntimeError) as e:
        logger.error("An exception occurred: {}".format(e))
        print(format_exception(e))
    except:
        logger.error("Unknown exception occurred")
        print(format_exception(None))
    finally:
        logger.info("END EXPERIMENT", group)
        wandb.finish()
        log_dir = run_dir.replace("/files", "")
        return log_dir


if __name__ == "__main__":
    args = flow_parse_args()
    # all_iter = generator_seek(load_all_stored_datasets("fsb"), 1, 3, drop=True)
    all_iter = generator_seek(load_all_stored_datasets("fsb"), 0, drop=False)
    args["dataset"] = "fsb"
    # all_iter = generator_seek(load_all_stored_datasets("srb"), 0, drop=True)
    # args["dataset"] = "srb"
    # all_iter = generator_seek(load_all_stored_datasets("srb"), 0)
    # args["dataset"] = "srb"
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # args["epochs"] = 120
    args["project"] = "TFselfopt_fsb"
    # args["wandb_mode"] = "online"
    args["self_optimization"] = True
    # unimportant parameter - auc-vus-score balance if possible - otherwise val-loss
    # args["self_optimization_goal"] = "val_loss_and_scores"

    process_list = []
    for i, gen in enumerate(all_iter):
        logger.debug(f"============ RUN {i} ============")
        g, p, trains, tests, callback = load_data(gen)

        _gen = (g, p, trains, tests)

        process_func_list_1 = [
            partial(run, run_args=args, generator=_gen, model_type="RealNVP", past=1, coupling_number=3,
                    device="cuda:0"),
            partial(run, run_args=args, generator=_gen, model_type="RealNVP-extended", past=51, coupling_number=3,
                    device="cuda:0"),
            partial(run, run_args=args, generator=_gen, model_type="tcNF-stateful", past=1, coupling_number=3,
                    device="cuda:0"),
        ]
        process_func_list_2 = [
            partial(run, run_args=args, generator=_gen, model_type="tcNF-base", past=51, coupling_number=3,
                    device="cuda:1"),
            partial(run, run_args=args, generator=_gen, model_type="tcNF-mlp", past=51, coupling_number=3,
                    device="cuda:1"),
        ]
        process_func_list_3 = [
            partial(run, run_args=args, generator=_gen, model_type="tcNF-cnn", past=51, coupling_number=3,
                    device="cuda:2"),
            partial(run, run_args=args, generator=_gen, model_type="tcNF-stateless", past=51, coupling_number=3,
                    device="cuda:2"),
        ]

        # if debug then run sequential
        if os.getenv("DEBUG", False) == "True":
            process_list = process_func_list_1 + process_func_list_2 + process_func_list_3
            for run_func in process_list:
                run_func = partial(run_func, device="cuda:0")
                p = Process(target=run_func)
                p.start()
                p.join()
            exit(0)

        # normal run with parallel execution
        process_dict = {i: f for i, f in enumerate([
            process_func_list_1,
            process_func_list_2,
            process_func_list_3,
        ])}

        current_iteration = {}
        wait_for = {}
        while_guard = True
        while while_guard:
            for g in range(len(process_dict.keys())):
                if g not in current_iteration:
                    current_iteration[g] = 0
                    p = Process(target=process_dict[g][current_iteration[g]])
                    p.start()
                    wait_for[g] = (datetime.now(), p)
                else:
                    t, p = wait_for[g]
                    if datetime.now() - t > timedelta(days=1):
                        try:
                            p.terminate()
                        except Exception as e:
                            logger.error("Some Termination issue")
                            logger.error(e)
                    elif p.is_alive():
                        p.join(timeout=10)
                    elif not p.is_alive():
                        current_iteration[g] += 1
                        if current_iteration[g] < len(process_dict[g]):
                            p = Process(target=process_dict[g][current_iteration[g]])
                            p.start()
                            wait_for[g] = (datetime.now(), p)

            status = [current_iteration[g] >= len(process_dict[g]) for g in range(len(process_dict.keys()))]
            logger.info(f"Status: {status}")
            if all(status):
                logger.info("All processes finished")
                while_guard = False

            logger.info("Waiting for 60 seconds")
            time.sleep(60)

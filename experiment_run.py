import gc
import shutil
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from models.real_nvp import DEVICE
# from models.real_nvp import update_device
# disable GPU use only for local debugging
# update_device('cpu')

from wandb import UsageError

from global_utils import generator_seek, flow_parse_args, format_exception
from dataset.mtads_loading import load_all_stored_datasets
from execute_run_opt_only import run


def create_generator_for_runs(generators: Iterator, run_args: dict):
    for generator in generators:
        # test different past information
        for past in run_args["past_range"]:
            # make run for each defined model_type
            for model_type in run_args["model_types"]:
                # test the number of coupling layers
                for coupling_number in run_args["coupling_layers"]:
                    # make n runs for each type
                    for i in range(run_args["nruns"]):
                        yield generator, past, coupling_number, i, model_type


def run_one_exp(run_args, generator, past, coupling_number, i_run, model_type):
    try:
        # run one experiment combination
        device = DEVICE
        logger.info("START", generator, "\n#", i_run + 1,
                    "| ModelType:", model_type, "CL:", coupling_number + 1, "Past:", past)

        log_path = run(run_args=run_args, model_type=model_type, generator=generator,
                       past=past, coupling_number=coupling_number, device=device)

        time.sleep(2)
        # cleanup unnecessary information that consume too much storage
        logger.info("cleanup", log_path)
        shutil.rmtree(log_path, ignore_errors=True)

        # catch all exceptions and continue with the next run
    except ValueError as e:
        logger.warning('ValueError: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except BrokenPipeError as e:
        logger.warning('BrokenPipeError: An exception occurred: {}'.format(e))
        print(format_exception(e))
        time.sleep(60)
    except FileExistsError as e:
        logger.warning('FileExistsError: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except OSError as e:
        logger.warning('OSError: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except UsageError as e:
        logger.warning('UsageError: An exception occurred: {}'.format(e))
        print(format_exception(e))
        time.sleep(60)
    except BaseException as e:
        logger.warning('BaseException: An exception occurred: {}'.format(e))
        print(format_exception(e))
    finally:
        logger.info("END", generator, "\n#", i_run + 1,
                    "| ModelType:", model_type, "CL:", coupling_number + 1, "Past:", past)
        # time.sleep(5)


def run_exp(run_args: dataclass, generators: Iterator):
    for generator, past, coupling_number, i_run, model_type in create_generator_for_runs(generators,
                                                                                         run_args):
        # run one experiment provided by the generator and the run arguments
        run_one_exp(run_args, generator, past, coupling_number, i_run, model_type)
        gc.collect()


if __name__ == "__main__":
    # parse the arguments and run the experiment
    args = flow_parse_args()
    logger.info(args)
    drop = False
    if args["dataset"] in ["fsb"]:  # run srb with noisy data in training as well
        drop = True

    # run the experiment
    run_exp(args, generator_seek(load_all_stored_datasets(args["dataset"]),
                                 args["generator_seek"], args["generator_stop"], drop))

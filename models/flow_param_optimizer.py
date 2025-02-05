import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime
import signal
from functools import partial, wraps

import cma
import numpy as np
import pingouin as pg
import torch
from loguru import logger
from torch.multiprocessing import set_start_method, Process, Manager

from global_utils import format_exception, handle_graceful_stop, data_guard
from models.executor import Executor
from models.executor_factory import ExecutorFactory
from models.flow_factory import model_types_parameters
from models.real_nvp import update_train_test_seed_value

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class ParamOptimizer(ABC):

    def __init__(self, device, args, parameters, samples, sample_hist, sample_anomalies,
                 eval_data, eval_hist, eval_anomalies,
                 test, test_hist, test_anomalies,
                 metric="val_loss"):
        super().__init__()
        self.device = device
        self.args = args
        self.parameters = parameters
        # store optimization data
        self.samples = samples
        self.sample_hist = sample_hist
        self.sample_anomalies = sample_anomalies
        # store eval data
        self.eval = eval_data
        self.eval_hist = eval_hist
        self.eval_anomalies = eval_anomalies
        # store test data
        self.test = test
        self.test_hist = test_hist
        self.test_anomalies = test_anomalies
        # performance metric
        self.metric = metric
        self.metric_func = optimization_metric_funcs[metric]

    @abstractmethod
    def optimize(self):
        pass


class CMAParamOptimizer(ParamOptimizer):
    def __init__(self, device, args, parameters, samples, sample_hist, sample_anomalies,
                 eval_data, eval_hist, eval_anomalies,
                 test, test_hist, test_anomalies,
                 metric="val_loss"):
        super().__init__(device, args, parameters, samples, sample_hist, sample_anomalies,
                         eval_data, eval_hist, eval_anomalies,
                         test, test_hist, test_anomalies,
                         metric)
        self.params = model_types_parameters["default"]
        self.executor_base_params = {"device": self.device,
                                     "model_type": self.parameters["model_type"],
                                     "group": self.parameters["group"],
                                     "code_version": self.parameters["code_version"],
                                     "input_shape": self.parameters["input_shape"],
                                     "hist_shape": self.parameters["hist_shape"],
                                     "past": self.parameters["past"],
                                     "fixed_past": self.parameters.get("fixed_past", False)}
        self.model_type = self.parameters["model_type"]
        self.params.update(model_types_parameters[self.model_type])
        self.param_order = list(self.params.keys())
        # max range for past
        if "past" in self.params:
            self.params["past"] = [self.params["past"][0], min(self.params["past"][1], self.parameters["past"])]
        # check if the past can be optimized
        if "fixed_past" not in self.parameters:
            self.parameters["fixed_past"] = False
        if self.parameters["fixed_past"]:
            self.param_order.remove("past")

        # TODO make this configurable
        # generation to evaluate
        self.max_iterations = 10
        # candidates to evaluate per generation
        self.candidates = 12
        self.sigma = 0.5
        self.in_opts = {'bounds': [0, 1],
                        'popsize': self.candidates}
        self.init_params = [0.5] * len(self.param_order)
        self.best_params = None
        self.best_weights = None
        self.optimizer_trace = None

        # extend runtime for non FSB runs
        if "fsb" == self.args["dataset"]:
            self.run_one_func = _run_one_short  # 49min
        else:
            self.run_one_func = _run_one_long  # 1.5h

    def get_results(self):
        return self.best_params, self.optimizer_trace, pickle.loads(self.best_weights)

    def optimize(self):
        es = cma.CMAEvolutionStrategy(self.init_params, self.sigma, inopts=self.in_opts)

        # increase the learning rate and decrease the epochs for faster convergence check
        run_args = self.args.copy()
        # TODO make this configurable
        run_args["epochs"] = 1000
        run_args["early_stopping"] = 50
        run_args["lr"] = 2e-4

        optimization_trace = {}

        # make iterations configurable
        best_candidate = None
        best_candidate_value = np.inf
        best_candidate_weights = None
        iteration = 0
        while not es.stop() and iteration < self.max_iterations:
            logger.info(f"Starting optimization search iteration {iteration}/{self.max_iterations - 1} "
                        f"with {self.candidates} candidates and optimization target {self.metric}\n")
            candidates = es.ask()
            candidates_mapped = [self._map_params(x) for x in candidates]
            manager = Manager()
            result_dict = manager.dict()
            candidates_pool_funcs = [partial(self.run_one_func, run_args=run_args, parameters=x,
                                             samples=self.samples,
                                             sample_hist=self.sample_hist,
                                             sample_anomalies=self.sample_anomalies,
                                             eval_data=self.eval,
                                             eval_hist=self.eval_hist,
                                             eval_anomalies=self.eval_anomalies,
                                             test=self.test,
                                             test_hist=self.test_hist,
                                             test_anomalies=self.test_anomalies,
                                             metric_func=self.metric_func,
                                             gen=iteration + 1,
                                             candidate=f"{iteration}.{i}",
                                             result_list=result_dict) for i, x in enumerate(candidates_mapped)]
            if False and self.device == torch.device("cpu"):
                logger.warning("Running sequentially on CPU")
                # run sequentially on cpu
                for run_func in candidates_pool_funcs:
                    run_func()
            else:
                # run in parallel on gpu
                wait_for = []
                for run_func in candidates_pool_funcs:
                    if len(wait_for) < 2:
                        p = Process(target=run_func)
                        p.start()
                        wait_for.append((datetime.now(), p))
                    else:
                        status = [p[1].is_alive() for p in wait_for]
                        while all(status):
                            time.sleep(5)
                            status = [p[1].is_alive() for p in wait_for]
                            logger.info(status)

                        wait_for = [p for p in wait_for if p[1].is_alive()]
                        p = Process(target=run_func)
                        p.start()
                        wait_for.append((datetime.now(), p))

                status = [p[1].is_alive() for p in wait_for]
                while any(status):
                    time.sleep(5)
                    status = [p[1].is_alive() for p in wait_for]
                    logger.info(status)

            # results are (value, details, test_details, model_weights)
            result_list = [result_dict.get(f"{iteration}.{i}", (1000, None, None, None)) for i in
                           range(len(candidates))]
            loss_only = [rl[0] for rl in result_list]
            es.tell(candidates, loss_only)
            best_loss_in_iteration = np.min(loss_only)
            if best_loss_in_iteration < best_candidate_value:
                best_candidate_in_interation = np.where(loss_only == best_loss_in_iteration)[0]
                best_candidate = (iteration, best_candidate_in_interation)
                best_candidate_value = best_loss_in_iteration
                best_candidate_weights = result_list[best_candidate_in_interation[0]][3]

            optimization_trace[iteration] = {}
            for i, x in enumerate(candidates_mapped):
                r = {k: x[k] for k in self.param_order}
                r["opt_goal"] = result_list[i][0]
                r["details"] = result_list[i][1:-1]
                optimization_trace[iteration][i] = r
            iteration += 1

        optimization_trace["iterations"] = iteration
        best_params = self._map_params(es.result.xbest)
        self.best_params = {k: best_params[k] for k in self.param_order}
        self.best_weights = best_candidate_weights
        logger.info(f"Best candidate: {best_candidate}, Best params: {self.best_params}")
        if "past" in self.best_params:
            self.best_params["hist_shape"] = (self.parameters["hist_shape"][0],
                                              self.best_params["past"],
                                              self.parameters["hist_shape"][-1])
        else:
            self.best_params["hist_shape"] = self.parameters["hist_shape"]

        optimization_trace["best_params"] = self.best_params
        optimization_trace["best_candidate"] = best_candidate
        self.optimizer_trace = optimization_trace

    def _map_params(self, current_set):
        current_params = self.executor_base_params.copy()
        for k, v in zip(self.param_order, current_set):
            to_range = self.params[k]
            v_mapped = v * (to_range[1] - to_range[0]) + to_range[0]
            if isinstance(to_range[0], int):
                v_mapped = np.round(v_mapped).astype(int)
            elif isinstance(to_range[0], float):
                v_mapped = np.round(v_mapped * 10).astype(int) / 10
            else:
                raise NotImplementedError("This type is not yet supported")
            current_params[k] = v_mapped
        return current_params


def timeout(seconds, default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def signal_handler(signum, frame):
                raise TimeoutError("Timed out!")

            # Set up the signal handler for timeout
            signal.signal(signal.SIGALRM, signal_handler)

            # Set the initial alarm for the integer part of seconds
            signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                return default
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator


def val_loss_metric(gen: int, executor: Executor, best_val_loss: float,
                    eval_data: np.ndarray, eval_hist: np.ndarray, eval_anomalies: np.ndarray) -> (float, str):
    return best_val_loss, {'values': {'best_val_loss': best_val_loss},
                           'print': f"best val_loss: {best_val_loss:.3f}"}


def score_metric(gen: int, executor: Executor, best_val_loss: float,
                 eval_data: np.ndarray, eval_hist: np.ndarray, eval_anomalies: np.ndarray, metric: str) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_anomalies, save_output_tensor=False)
    optimization_score = executor.score(re_range_nll_prob, eval_anomalies)[metric]
    return 1 - optimization_score, {'values': {f"{metric}_score": optimization_score},
                                    'print': f"1-({metric} score: {optimization_score:.3f})"}


auc_roc_metric = partial(score_metric, metric="AUC_ROC")
vus_roc_metric = partial(score_metric, metric="VUS_ROC")
r_auc_roc_metric = partial(score_metric, metric="R_AUC_ROC")
f_metric = partial(score_metric, metric="F")
rf_metric = partial(score_metric, metric="RF")


def score_balance_metric(gen: int, executor: Executor, best_val_loss: float,
                         eval_data: np.ndarray, eval_hist: np.ndarray, eval_anomalies: np.ndarray) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_anomalies, save_output_tensor=False)
    scores = executor.score(re_range_nll_prob, eval_anomalies)
    return ((1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.7,
            {"values": {"AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"]},
             "print": f"(1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + (1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.7"})


score_factor = {0: (0.75, 0.25),
                1: (0.5, 0.5),
                2: (0.1, 0.8)}


def combined_metric(gen: int, executor: Executor, best_val_loss: float,
                    eval_data: np.ndarray, eval_hist: np.ndarray, eval_anomalies: np.ndarray,
                    metric: str) -> (float, str):
    re_range_prob, _ = executor.predict(None, eval_data, eval_hist, eval_anomalies, save_output_tensor=False)
    optimization_score = executor.score(re_range_prob, eval_anomalies)[metric]
    factor = score_factor.get(gen // 4, (0.1, 0.8))
    # score = np.where(best_val_loss > 0, best_val_loss, best_val_loss * 0.01) - optimization_score * factor
    # return score, f" | leakyRelu(val_loss: {best_val_loss:.3f}) - {metric} score: {optimization_score:.3f} * {factor}"
    score = np.abs(best_val_loss) * factor[0] + 1 - optimization_score * factor[1]
    return (score, {"values": {"val_loss": best_val_loss, metric: optimization_score},
                    "print": (f"np.abs(val_loss: {best_val_loss:.3f}) * {factor[0]} + 1 - "
                              f"{metric} score: {optimization_score:.3f} * {factor[1]}")})


auc_roc_val_loss_metric = partial(combined_metric, metric="AUC_ROC")
vus_roc_val_loss_metric = partial(combined_metric, metric="VUS_ROC")
r_auc_roc_val_loss_metric = partial(combined_metric, metric="R_AUC_ROC")
f_val_loss_metric = partial(combined_metric, metric="F")
rf_val_loss_metric = partial(combined_metric, metric="RF")


def val_loss_and_scores(gen: int, executor: Executor, best_val_loss: float,
                        eval_data: np.ndarray, eval_hist: np.ndarray, eval_anomalies: np.ndarray) -> (float, str):
    re_range_nll_prob, _ = executor.predict(None, eval_data, eval_hist, eval_anomalies, save_output_tensor=False)
    scores = executor.score(re_range_nll_prob, eval_anomalies)
    if best_val_loss < 1:
        return ((1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.7,
                {"values": {"AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"]},
                 "print": f"(1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + (1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.7"})
    else:
        return (best_val_loss * 0.1 + (1 - scores["AUC_ROC"]) * 0.3 + (1 - scores["VUS_ROC"]) * 0.6,
                {"values": {"val_loss": best_val_loss, "AUC_ROC": scores["AUC_ROC"], "VUS_ROC": scores["VUS_ROC"]},
                 "print": f"val_loss: {best_val_loss:.3f} * 0.1 + (1 - AUC_ROC: {scores['AUC_ROC']:.3f}) * 0.3 + "
                          f"(1 - VUS_ROC: {scores['VUS_ROC']:.3f}) * 0.6"})


def gaussian_convergence_metric(gen: int, executor: Executor, best_val_loss: float,
                                eval_data: np.ndarray, eval_hist: np.ndarray,
                                eval_anomalies: np.ndarray) -> (float, str):
    _, latent = executor.predict_individual(None, eval_data, eval_hist, eval_anomalies)
    mardia_result = pg.multivariate_normality(latent, alpha=0.05)
    score = 1 - mardia_result.pval
    return score, {"values": {"p-value": mardia_result.pval},
                   "print": f"1 - p-value: {mardia_result.pval:.3f}"}


def losses_metric(gen: int, executor: Executor, best_val_loss: float,
                  eval_data: np.ndarray, eval_hist: np.ndarray, eval_anomalies: np.ndarray) -> (float, str):
    _, probs = executor.predict(None, eval_data, eval_hist, eval_anomalies, save_output_tensor=False)
    test_loss = np.mean(probs[:, 0])  # nll loss
    return (best_val_loss * 0.3 + test_loss * 0.7,
            {"values": {"val_loss": best_val_loss, "test_loss": test_loss},
             "print": f"val_loss: {best_val_loss:.3f} * 0.5 + test_loss: {test_loss:.3f} * 0.5"})


optimization_metric_funcs = {
    "val_loss": val_loss_metric,
    "auc_roc": auc_roc_metric,
    "vus_roc": vus_roc_metric,
    "r_auc_roc": r_auc_roc_metric,
    "f": f_metric,
    "rf": rf_metric,
    "auc_roc_val_loss": auc_roc_val_loss_metric,
    "vus_roc_val_loss": vus_roc_val_loss_metric,
    "r_auc_roc_val_loss": r_auc_roc_val_loss_metric,
    "f_val_loss": f_val_loss_metric,
    "rf_val_loss": rf_val_loss_metric,
    "auc_vus_balance": score_balance_metric,
    "val_loss_and_scores": val_loss_and_scores,
    "gaussian_convergence": gaussian_convergence_metric,
    "losses": losses_metric
}


@timeout(2940)  # 49min timeout
def _run_one_short(*args, **kwargs):
    return _run_one_optimization(*args, **kwargs)


@timeout(5400)  # 1.5h timeout
def _run_one_long(*args, **kwargs):
    return _run_one_optimization(*args, **kwargs)


def _run_one_optimization(run_args, parameters, samples, sample_hist, sample_anomalies,
                          eval_data, eval_hist, eval_anomalies,
                          test, test_hist, test_anomalies,
                          metric_func,
                          gen, candidate, result_list):
    sample_hist = sample_hist[:, -parameters["past"]:, :]
    # reduce the dataset size if it is too big
    if eval_data is None:
        eval_data = samples
        eval_hist = sample_hist
        eval_anomalies = sample_anomalies
    samples, sample_hist, sample_anomalies = size_reducer(samples, sample_hist, sample_anomalies)
    # test, test_hist, test_anomalies = size_reducer(test, test_hist, test_anomalies, 100_000)

    parameters["input_shape"] = samples.shape
    parameters["hist_shape"] = sample_hist.shape

    logger.info(f"Start candidate {candidate}: {parameters}")
    update_train_test_seed_value(parameters["seed"])

    executor = ExecutorFactory.create_executor(run_args, parameters)
    try:
        best_val_loss, epoch, model_weights = executor.fit_light(samples, sample_hist, sample_anomalies)
        optimization_loss, details = metric_func(gen, executor, best_val_loss, eval_data, eval_hist, eval_anomalies)
        details["epoch"] = epoch
        # allways run auc & vus test on the test set but don't use it for optimization
        # TODO exclude areas with anomaly
        test_loss, test_details = score_balance_metric(gen, executor, best_val_loss, test, test_hist, test_anomalies)
        model_weights = pickle.dumps(model_weights)
    except Exception as e:
        logger.warning(f"Exception occurred: {e}")
        print(format_exception(e))
        optimization_loss = 1000
        model_weights = None
        details = {"print": "Exception occurred", "values": {}}
        test_loss = 1000
        test_details = {"print": "Exception occurred", "values": {}}
    logger.info(f"End candidate {candidate}")
    logger.info(f"Opti Loss: {optimization_loss:.3f} | {details['print']}")
    logger.info(f"Test loss: {test_loss:.3f} | {test_details['print']}")

    result_list[candidate] = (optimization_loss, details, test_details, model_weights)


def size_reducer(data, data_hist, data_anomalies, max_size: int = 100_000):
    if data is None:
        return data, data_hist, data_anomalies
    if data.shape[0] > max_size:
        # stepping solution instead of reducing the amount of data
        # this should be a parameter and sequence depending
        stepping = data.shape[0] // max_size
        data = data[::stepping]
        data_hist = data_hist[::stepping]
        data_anomalies = data_anomalies[::stepping]
    return data, data_hist, data_anomalies

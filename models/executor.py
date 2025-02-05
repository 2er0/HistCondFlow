from abc import ABCMeta, abstractmethod, ABC
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import wandb
from loguru import logger
from sklearn.metrics import roc_curve

from global_utils import save_config, drop_time_features
from plot_utils import plot_roc_curve, plot_all_multiple_detection, plot_2d_latent_dist_space
from vus.metrics import get_metrics
from vus.utils.slidingWindows import find_length


class Executor(ABC):
    """
    Abstract base class for an Executor that defines the interface for fitting and predicting.
    """

    def __init__(self, run_args, parameters):
        """
        Initialize the Executor with run arguments and parameters.

        :param run_args: Arguments for the run.
        :param parameters: Parameters for the Executor.
        """
        super().__init__()
        self.run_args = run_args
        self.parameters = parameters
        self.model_type = parameters["model_type"]
        self.device = parameters["device"]
        self.group = parameters["group"]
        self.config = {
            "dataset": self.group,
            "run_args": run_args,
            "parameters": parameters
        }
        self.run_dir = None
        self.offline_dir = None
        self.load_pretrained = run_args.get("load_pretrained", False)
        if self.load_pretrained:
            logger.info("Run with pretrained model")
            self.pretrained_model_path = parameters["pretrained_model_path"]
        self.chunk_size = 300_000

    def get_config(self):
        """
        Get the current configuration.

        :return: The current configuration as a dictionary.
        """
        return self.config

    def update_config(self, config):
        """
        Update the current configuration with new values.

        :param config: Dictionary containing new configuration values.
        """
        self.config.update(config)

    def save_config(self):
        """
        Save the current configuration to a file if offline directory is set.
        """
        if self.offline_dir is not None:
            save_config(self.get_config(), f"{self.offline_dir}/config.json")

    def get_run_dir(self):
        """
        Get the current run directory.

        :return: The current run directory as a string.
        """
        return self.run_dir

    def start_wandb_logging(self):
        """
        Start logging to Weights and Biases (wandb) and locally to disk.
        """
        # prepare logging to wand and locally to disk
        run_store_path = f"./wandb/{self.run_args['project']}"
        Path(run_store_path).mkdir(parents=True, exist_ok=True)
        wandb.init(project=self.run_args["project"],
                   entity=self.run_args["user"],
                   group=self.group,
                   reinit=True,
                   mode=self.run_args["wandb_mode"],
                   dir=run_store_path,
                   config=self.config)
        self.run_dir = wandb.run.dir
        run_name = self.run_dir.split("/")[-2]
        self.offline_dir = f"{run_store_path}/offline/{run_name}"
        Path(self.offline_dir).mkdir(parents=True, exist_ok=True)

        # save run configuration to local disk
        self.save_config()

    def log_score_to_wandb(self):
        if "test" not in self.config:
            return
        scores = self.config["test"]["mean"]
        wandb.summary["AUC_ROC"] = scores["AUC_ROC"]
        wandb.summary["VUS_ROC"] = scores["VUS_ROC"]
        wandb.summary["F"] = scores["F"]
        wandb.summary["RF"] = scores["RF"]
        wandb.summary["sliding_window"] = scores["sliding_window"]

    @abstractmethod
    def get_model(self):
        """
        Get the model instance.

        :return: The model instance.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Path):
        """
        Load the model from the provided path.

        :param model_path: Path to the model file.
        """
        pass

    @abstractmethod
    def fit(self, samples: np.ndarray, sample_hist: np.ndarray, anomalies: Union[list, np.ndarray]) -> None:
        """
        Fit the model to the provided samples.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param anomalies: The anomalies in the data, can be a list or numpy array.
        """
        pass

    @abstractmethod
    def predict(self, test_name: Union[str, None], test: np.ndarray, test_hist: np.ndarray,
                anomalies: Union[list, np.ndarray], save_output_tensor: bool = True) -> (
            np.ndarray, Union[np.ndarray, List[np.ndarray]]):
        """
        Predict the outcomes for the provided test sequences.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :param save_output_tensor:  If True, the output tensor will be saved to the offline directory.
        :return: A tuple containing:
                 - full negative log probability
                 - log probability
                 - log determinant
        """
        pass

    @abstractmethod
    def predict_individual(self, test_name: Union[str, None], test: np.ndarray, test_hist: np.ndarray,
                           anomalies: Union[list, np.ndarray]) -> (
            Union[np.ndarray, List[np.ndarray]],
            Union[np.ndarray, List[np.ndarray]],
    ):
        """
        Predict individual outcomes for visualization purposes.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :return: A tuple containing:
                 - full log probability
                 - output
        """
        pass

    def score(self, nll_log: np.ndarray, is_anomaly: np.ndarray) -> dict:
        """
        Calculate the score based on negative log likelihood and anomaly status.

        :param nll_log: Negative log likelihood values.
        :param is_anomaly: Boolean array indicating anomaly status.
        :return: Dictionary containing scores and sliding window length.
        """
        estimated_sliding_window = find_length(is_anomaly)
        scores = get_metrics(nll_log, is_anomaly, metric='all', slidingWindow=estimated_sliding_window)
        scores["sliding_window"] = estimated_sliding_window
        return scores

    def time_line_plot(self, name, train, test, all_probs_np, is_anomaly, nll_prob, train_dates,
                       test_dates, save_to_disk=True, show=False, title=None):

        """
        Create a timeline plot for the given data.

        :param name: Name of the plot.
        :param train: Training data.
        :param test: Test data.
        :param all_probs_np: All probabilities as a numpy array.
        :param is_anomaly: Boolean array indicating anomaly status.
        :param nll_prob: Negative log likelihood probabilities.
        :param train_dates: Dates corresponding to the training data.
        :param test_dates: Dates corresponding to the test data.
        :param save_to_disk: If True, the plot will be saved to disk.
        :param show: If True, the plot will be shown.
        :param title: Title of the plot.
        """
        cs = self.chunk_size
        for chunk in range(test.shape[0] // cs + 1):
            logger.info("Create time plot: {}".format(chunk))
            _train, _test = self._drop_time_features(train, test)
            _train = _train[chunk * cs:(chunk + 1) * cs]
            _test = _test[chunk * cs:(chunk + 1) * cs]
            _all_probs_np = all_probs_np[chunk * cs:(chunk + 1) * cs] if all_probs_np is not None else None
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _nll_prob = nll_prob[chunk * cs:(chunk + 1) * cs] if nll_prob is not None else None
            _test_dates = test_dates[chunk * cs:(chunk + 1) * cs]
            _train_dates = train_dates[chunk * cs:(chunk + 1) * cs]

            if title is None:
                title = f"{self.group}, {name}, model type: {self.model_type}"

            # create plot with continues result
            fig = plot_all_multiple_detection(
                _train,
                _test,
                _all_probs_np,
                (_is_anomaly, _nll_prob),
                _test_dates,
                _train_dates,
                title
            )
            if save_to_disk:
                img_byte = fig.to_image(format="png", width=2000, height=1200)
                with open(f"{self.offline_dir}/{name}_{chunk}_overview.png", "wb+") as destination:
                    destination.write(img_byte)
                # fig.write_html(f"{self.offline_dir}/{name}_{chunk}_overview.html")
            if show:
                fig.show()

    def latent_space_plot(self, name, train, test, transformed_latent_space, individual_probs, sample_anomalies,
                          is_anomaly, title=None):
        """
        Create a latent space plot for the given data.

        :param name: Name of the plot.
        :param train: Training data.
        :param test: Test data.
        :param transformed_latent_space: Transformed latent space data.
        :param individual_probs: Individual probabilities.
        :param sample_anomalies: Anomalies in the sample data.
        :param is_anomaly: Boolean array indicating anomaly status.
        """

        cs = self.chunk_size
        for chunk in range(test.shape[0] // cs + 1):
            logger.info("Create time plot: {}".format(chunk))
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _sample_anomalies = sample_anomalies[chunk * cs:(chunk + 1) * cs]

            _train, _test = self._drop_time_features(train, test)
            _train = _train[chunk * cs:(chunk + 1) * cs]
            _test = _test[chunk * cs:(chunk + 1) * cs]
            _transformed_latent_space = transformed_latent_space[chunk * cs:(chunk + 1) * cs]
            _individual_probs = individual_probs[chunk * cs:(chunk + 1) * cs]

            latent_fig = plot_2d_latent_dist_space(_train, _test, _transformed_latent_space,
                                                   _individual_probs, _sample_anomalies, _is_anomaly,
                                                   f"{self.group}, {name}, model type: {self.model_type}")
            img_byte = latent_fig.to_image(format="png", width=2000, height=1200)
            with open(f"{self.offline_dir}/{name}_{chunk}_latent_space.png", "wb+") as destination:
                destination.write(img_byte)
            # latent_fig.write_html(f"{self.offline_dir}/{name}_{chunk}_latent_space.html")

    def roc_curve_plot(self, name, prob, is_anomaly):
        """
        Create a ROC plot.

        :param name: Name of the plot.
        :param prob: Probabilities.
        :param is_anomaly: Boolean array in         dicating anomaly status.
        """
        cs = self.chunk_size
        for chunk in range(prob.shape[0] // cs + 1):
            logger.info("Create roc curve plot: {}".format(chunk))
            _is_anomaly = is_anomaly[chunk * cs:(chunk + 1) * cs]
            _prob = prob[chunk * cs:(chunk + 1) * cs]
            fig = plot_roc_curve(_prob, _is_anomaly,
                                 f"{self.group}, {name}, model type: {self.parameters['model_type']}")
            img_byte = fig.to_image(format="png", width=500, height=500)
            with open(f"{self.offline_dir}/{name}_{chunk}_roc_curve.png", "wb+") as destination:
                destination.write(img_byte)
            # fig.write_html(f"{self.offline_dir}/{name}_{chunk}_roc_curve.html")

    def _drop_time_features(self, train, test):
        _train = drop_time_features(train, self.parameters)
        _test = drop_time_features(test, self.parameters)
        return _train, _test

from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from data_preprocessing import create_dataset_with_past_from_dataset, create_dataset_from_dataset_with_end_as_validation
from global_utils import re_range, format_exception, build_known_args_dict
from models.executor import Executor
from models.flow_factory import flow_factory
from models.real_nvp import OpenDualDataProvider


def checkpoint(model, filename):
    """
    Save the model state to a file.

    :param model: The model to save.
    :param filename: The filename to save the model state to.
    """
    torch.save(model.state_dict(), filename)


def resume(model, filename, device):
    """
    Load the model state from a file.

    :param model: The model to load the state into.
    :param filename: The filename to load the model state from.
    :param device: The device to load the model state into.
    """
    save_state = torch.load(filename, map_location=device)
    current_state = model.state_dict()
    model.load_state_dict(save_state)


class FlowExecutor(Executor, ABC):
    """
    Executor class for flow-based models, inheriting from the Executor abstract base class.
    """

    def __init__(self, run_args, parameters):
        """
        Initialize the FlowExecutor with run arguments, parameters, and device.

        :param run_args: Arguments for the run.
        :param parameters: Parameters for the Executor.
        """
        super().__init__(run_args, parameters)

        self.lr = run_args["lr"]
        self.batch_size = run_args["batch_size"]
        self.epochs = run_args["epochs"]
        self.early_stop_thresh = run_args["early_stopping"]

        # request and prime flow factory to acquire the correct model
        self.flow_factory_args = build_known_args_dict(flow_factory, parameters)
        flow_builder = flow_factory(**self.flow_factory_args)

        # build the flow model from the builder
        self.base_build_args = build_known_args_dict(flow_builder, parameters)
        self.flow = flow_builder(**self.base_build_args).to(self.device)

        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr)

        if self.load_pretrained:
            logger.info(f"Loading pretrained model from {self.pretrained_model_path}")
            self.load_model(self.pretrained_model_path)

    def get_model(self):
        """
        Get the flow model instance.

        :return: The flow model instance.
        """
        return self.flow

    def load_model(self, model_path: Union[Path, str, None] = None):
        """
        Load the model from the provided path.

        :param model_path: The path to the model file.
        """
        if model_path is None:
            model_path = f"{self.offline_dir}/best_model.pth"
        resume(self.flow, model_path, self.device)

    def load_model_from_weights(self, model_weights: dict):
        """
        Load the model from the provided weights.

        :param model_weights: The weights of the model.
        """
        logger.info("Loading model from weights")
        self.flow.load_state_dict(model_weights)

    def save_model(self):
        """
        Save the model to the provided path.

        """
        checkpoint(self.flow, f"{self.offline_dir}/best_model.pth")

    def start_wandb_logging(self):
        super().start_wandb_logging()
        wandb.watch(self.get_model())

    def fit(self, samples: np.ndarray, sample_hist: np.ndarray, anomalies: Union[list, np.ndarray]) -> None:
        """
        Fit the flow model to the provided samples.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :param anomalies: The anomalies in the data, can be a list or numpy array.
        """
        samples, sample_hist = self._pre_data_processing(samples, sample_hist)
        (train, train_hist), (val, val_hist) = self._create_train_validation_split(samples, sample_hist)
        train_loader, train_batches = self.__create_loader(train, train_hist)
        val_loader, val_batches = self.__create_loader(val, val_hist)

        best_val_loss = None
        best_epoch = None

        for epoch in range(self.epochs):
            # training
            self.flow.train()
            t_loss, t_log_prob, t_log_det = self._one_epoch(train_loader, train_batches)
            # validation
            self.flow.eval()
            v_loss, v_log_prob, v_log_det = self._one_eval_epoch(val_loader, val_batches)

            s = ("{m} | Epoch: {e:04d} | Loss Train: {tl:.4f}, Val: {vl:.4f} | "
                 "Log Prob Train: {tlp:.4f}, Val: {vlp:.4f} | "
                 "Log Det Train: {tld:.4f}, Val: {vld:.4f}").format(sep=" ", m=self.model_type, e=epoch + 1,
                                                                    tl=t_loss, vl=v_loss,
                                                                    tlp=t_log_prob, vlp=v_log_prob,
                                                                    tld=t_log_det, vld=v_log_det)
            logger.info(s)
            if self.run_args["wandb_mode"] == "online":
                wandb.log({
                    "Train Loss": t_loss,
                    "Val Loss": v_loss,
                    "Train Log Prob": t_log_prob,
                    "Val Log Prob": v_log_prob,
                    "Train Log Det": t_log_det,
                    "Val Log Det": v_log_det
                })

            # early stopping with 10 epochs grace period
            if epoch > 3:
                if best_val_loss is None:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    self.save_model()
                elif best_val_loss > v_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    self.save_model()
                elif epoch - best_epoch > self.early_stop_thresh:
                    logger.info("Early stopped training at epoch %d" % (best_epoch + 1))
                    break  # terminate the training loop

        logger.info(f"Training finished, best model at epoch {best_epoch + 1}, loading best model")
        self.load_model()

    def fit_light(self, samples: np.ndarray, sample_hist: np.ndarray, anomalies: Union[list, np.ndarray]):
        """
        Fit the flow model to the provided samples with a lighter version of the training loop.
        :param samples:
        :param sample_hist:
        :param anomalies:
        :return:
        """
        samples, sample_hist = self._pre_data_processing(samples, sample_hist)
        (train, train_hist), (val, val_hist) = self._create_train_validation_split(samples, sample_hist)
        train_loader, train_batches = self.__create_loader(train, train_hist)
        val_loader, val_batches = self.__create_loader(val, val_hist)

        best_val_loss = None
        best_epoch = None
        best_weights = None

        for epoch in range(self.epochs):
            # training
            self.flow.train()
            t_loss, t_log_prob, t_log_det = self._one_epoch(train_loader, train_batches)
            # validation
            self.flow.eval()
            v_loss, v_log_prob, v_log_det = self._one_eval_epoch(val_loader, val_batches)

            # early stopping with 10 epochs grace period
            if epoch > 3:
                if best_val_loss is None:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    best_weights = self.flow.state_dict()
                elif best_val_loss > v_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    best_weights = self.flow.state_dict()
                elif epoch - best_epoch > self.early_stop_thresh:
                    # logger.info("Early stopped training at epoch %d" % (best_epoch + 1))
                    break  # terminate the training loop

        return best_val_loss, epoch, best_weights

    def __predict_with_func(self, func=None, test=None, test_hist=None) -> (
            List[Union[np.ndarray, List[np.ndarray]]]):
        """
        Predict using a specified function on the test data.

        :param func: The function to use for prediction.
        :param test: The test samples as a numpy array.
        :param test_hist: The historical data of the test samples as a numpy array.
        :return: List of prediction results.
        """
        self.flow.eval()
        self._pre_test_model_preparing()

        test_loader, test_batches = self.__create_loader(test, test_hist)
        with torch.no_grad():
            batched_probs = []
            for i, (x, past) in enumerate(test_loader):
                x = x.to(self.device)
                past = past.to(self.device)
                prob_outputs = func(x, past)
                batched_probs.append(prob_outputs)

        return batched_probs

    def predict(self, test_name: Union[str, None], test: np.ndarray, test_hist: np.ndarray,
                anomalies: Union[list, np.ndarray],
                save_output_tensor: bool = True) -> (
            np.ndarray, Union[np.ndarray, List[np.ndarray]]):
        """
        Predict the outcomes for the provided test sequences.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :param save_output_tensor: Save the output tensor.
        :return: A tuple containing:
                 - re-ranged negative log likelihood probability
                 - all probabilities as a numpy array
                    - negative log likelihood
                    - log probability
                    - log determinant
        """
        if test_name is not None:
            logger.info(f"Testing on {test_name} | Predict")
        test, test_hist = self._pre_data_processing(test, test_hist)
        # get probs
        batched_probs = self.__predict_with_func(self._calc_log_prob, test, test_hist)

        all_probs = []
        for prob in zip(*batched_probs):
            prob = torch.hstack(prob)
            prob = prob.cpu().numpy()
            all_probs.append(prob)

        # negative log likelihood
        all_probs[0] = -all_probs[0]
        # scale the negative log likelihood to a range between 0 and 1
        re_ranged_nll_prob = re_range(all_probs[0])

        # save probability scores (model output)
        all_probs_np = np.vstack(all_probs).T
        if save_output_tensor:
            with open(f"{self.offline_dir}/{test_name}_raw_all_prob.npy", "wb") as f:
                np.save(f, all_probs_np)

        return re_ranged_nll_prob, all_probs_np

    def predict_individual(self, test_name: str, test: np.ndarray, test_hist: np.ndarray,
                           anomalies: Union[list, np.ndarray]) -> (Union[np.ndarray, List[np.ndarray]],
                                                                   Union[np.ndarray, List[np.ndarray]]):
        """
        Predict individual outcomes for visualization purposes.

        :param test_name: Name of the test.
        :param test: Test samples as a numpy array.
        :param test_hist: Historical data of the test samples as a numpy array.
        :param anomalies: Anomalies in the test data, can be a list or numpy array.
        :param dates: Dates corresponding to the test samples.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: A tuple containing:
                 - individual probabilities
                 - latent space representation
        """
        if test_name is not None:
            logger.info(f"Testing on {test_name} | Predict Individual")
        test, test_hist = self._pre_data_processing(test, test_hist)
        batched_probs = self.__predict_with_func(self._calc_individual_log_prob, test, test_hist)

        individual_probs_and_latent = []
        for prob in zip(*batched_probs):
            prob = torch.vstack(prob)
            prob = prob.cpu().numpy()
            individual_probs_and_latent.append(prob)

        individual_probs = -(individual_probs_and_latent[0] + individual_probs_and_latent[1])

        return individual_probs, individual_probs_and_latent[2]

    def sample(self, n: int = 128, past: np.ndarray = None) -> np.ndarray:
        """
        Sample from the flow model.

        :param n: Number of samples to generate.
        :param past: Historical data.
        :return: The generated samples.
        """
        past = torch.from_numpy(past).to(self.device)
        self.flow.eval()
        with torch.no_grad():
            samples, z_probs = self.flow.sample(n, past)

        samples = samples.cpu().numpy()
        z_probs = z_probs.cpu().numpy()
        return samples, z_probs

    @abstractmethod
    def _calc_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Abstract method to calculate the log probability.

        :param x: Input data.
        :param past: Historical data.
        :return: A tuple containing:
                 - log probability
                 - log determinant
                 - additional tensor
        """
        pass

    @abstractmethod
    def _calc_individual_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Abstract method to calculate the individual log probability.

        :param x: Input data.
        :param past: Historical data.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        pass

    def _pre_data_processing(self, samples: np.ndarray, sample_hist: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Pre-process before the training and validation split.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :return: A tuple containing:
                 - processed samples
                 - processed historical samples
        """
        if self.parameters["past"] != sample_hist.shape[1]:
            sample_hist = sample_hist[:, -self.parameters["past"]:, :]
        return samples, sample_hist

    @abstractmethod
    def _pre_test_model_preparing(self):
        """
        Abstract method to prepare the model for testing.
        """
        pass

    @abstractmethod
    def _create_train_validation_split(self, samples: np.ndarray, sample_hist: np.ndarray) -> (
            (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        """
        Abstract method to create the training and validation split.

        :param samples: The input samples as a numpy array.
        :param sample_hist: The historical data of the samples as a numpy array.
        :return: A tuple containing:
                 - training samples and historical samples
                 - validation samples and historical samples
        """
        pass

    @abstractmethod
    def _one_epoch(self, loader: DataLoader, batches: int):
        """
        Abstract method to perform one training epoch.

        :param loader: DataLoader for the training data.
        :param batches: Number of batches.
        """
        pass

    @abstractmethod
    def _one_eval_epoch(self, loader: DataLoader, batches: int):
        """
        Abstract method to perform one evaluation epoch.

        :param loader: DataLoader for the evaluation data.
        :param batches: Number of batches.
        """
        pass

    def __create_loader(self, samples, hist_samples) -> (DataLoader, int):
        """
        Create a DataLoader for the given samples and historical samples.

        :param samples: The input samples.
        :param hist_samples: The historical samples.
        :return: A tuple containing:
                 - DataLoader instance
                 - number of batches
        """
        provider = OpenDualDataProvider(
            samples.astype(np.float32),
            hist_samples.astype(np.float32)
        )
        loader = DataLoader(provider, batch_size=self.batch_size, shuffle=False)
        batches = len(loader)
        if batches == 0:
            raise ValueError("No training data available, past requirement is too high")

        return loader, batches


class FlowBatchedExecutor(FlowExecutor):
    """
    Executor class for flow-based models with batched processing, inheriting from FlowExecutor.
    """

    def __init__(self, run_args, parameters):
        super().__init__(run_args, parameters)

    def _pre_test_model_preparing(self):
        pass

    def _one_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one training epoch.

        :param loader: DataLoader for the training data.
        :param batches: Number of batches.
        :return: A tuple containing:
                 - average loss
                 - average log probability
                 - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        for i, (x, past) in enumerate(loader):
            x = x.to(self.device)
            past = past.to(self.device)
            self.optimizer.zero_grad()
            loss, dist_log_prob, log_det = self.flow.loss(x, past)

            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach().cpu().item()
            dist_log_prob_sum += dist_log_prob.detach().cpu().item()
            log_det_sum += log_det.detach().cpu().item()

        return loss_sum / batches, dist_log_prob_sum / batches, log_det_sum / batches

    def _one_eval_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one evaluation epoch.

        :param loader: DataLoader for the evaluation data.
        :param batches: Number of batches.
        :return: A tuple containing:
                 - average loss
                 - average log probability
                 - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        with torch.no_grad():
            self.optimizer.zero_grad()
            for i, (x, past) in enumerate(loader):
                x = x.to(self.device)
                past = past.to(self.device)
                loss, dist_log_prob, log_det = self.flow.loss(x, past)

                loss_sum += loss.detach().cpu().item()
                dist_log_prob_sum += dist_log_prob.detach().cpu().item()
                log_det_sum += log_det.detach().cpu().item()

        return loss_sum / batches, dist_log_prob_sum / batches, log_det_sum / batches

    def _calc_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.flow.log_prob(x, past)

    def _calc_individual_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        probs = self.flow.individual_log_prob(x, past)
        # un-squeeze the second tensor to match the shape of the first tensor
        probs[1].unsqueeze_(1)

        return probs[0], probs[1], probs[2]

    def _create_train_validation_split(self, samples, sample_hist) -> (
            (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        # create training and validation set for all other models
        # by splitting randomly picking sections as validation and add buffers around the validation sets
        try:
            (
                (train_samples, train_hist_samples),
                (validation_samples, validation_hist_samples),
                (_, _)
            ) = create_dataset_with_past_from_dataset(samples, sample_hist,
                                                      self.parameters["past"])
            return (train_samples, train_hist_samples), (validation_samples, validation_hist_samples)
        except IndexError as e:
            logger.error(f"IndexError: An exception occurred: {e}")
            print(format_exception(e))
            raise e


class FlowNonBatchedExecutor(FlowExecutor):
    """
    Executor class for flow-based models with non-batched (sequential) processing, inheriting from FlowExecutor.
    """

    def __init__(self, run_args, parameters):
        super().__init__(run_args, parameters)
        self.batch_size = 1  # run in non-batched mode - sequential training

    def _one_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one training epoch in non-batched mode.

        :param loader: DataLoader for the training data.
        :param batches: Number of batches.
        :return: A tuple containing:
                 - average loss
                 - average log probability
                 - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        # tcNF-stateful - requires no batching and therefore sequential training
        self.flow.reset_rnn_hidden()
        self.optimizer.zero_grad()
        loss_list = []

        for i, (x, past) in enumerate(loader):
            x = x.to(self.device)
            past = past.to(self.device)
            loss, dist_log_prob, log_det = self.flow.loss(x, past)
            loss_list.append(loss)

            loss_sum += loss.detach().cpu().item()
            dist_log_prob_sum += dist_log_prob.detach().cpu().item()
            log_det_sum += log_det.detach().cpu().item()

            # back propagate every 256 steps
            if i > 0 and i % 256 == 0:
                loss_for_backward = torch.sum(torch.stack(loss_list)) / 256
                loss_for_backward.backward()
                self.flow.detach()
                self.optimizer.step()
                loss_list = []

        if len(loss_list) > 0:
            loss_for_backward = torch.sum(torch.stack(loss_list)) / len(loss_list)
            loss_for_backward.backward()
            self.flow.detach()
            self.optimizer.step()

        divider = batches  # batches * self.batch_size
        return loss_sum / divider, dist_log_prob_sum / divider, log_det_sum / divider

    def _one_eval_epoch(self, loader: DataLoader, batches: int) -> (float, float, float):
        """
        Perform one evaluation epoch in non-batched mode.

        :param loader: DataLoader for the evaluation data.
        :param batches: Number of batches.
        :return: A tuple containing:
                    - average loss
                    - average log probability
                    - average log determinant
        """
        loss_sum = 0.0
        dist_log_prob_sum = 0.0
        log_det_sum = 0.0
        with torch.no_grad():
            self.optimizer.zero_grad()
            self.flow.reset_rnn_hidden()
            for i, (x, past) in enumerate(loader):
                x = x.to(self.device)
                past = past.to(self.device)
                loss, dist_log_prob, log_det = self.flow.loss(x, past)

                loss_sum += loss.detach().cpu().item()
                dist_log_prob_sum += dist_log_prob.detach().cpu().item()
                log_det_sum += log_det.detach().cpu().item()

        return loss_sum / batches, dist_log_prob_sum / batches, log_det_sum / batches

    def _pre_test_model_preparing(self):
        self.flow.reset_rnn_hidden()

    def _calc_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        all_probs = [self.flow.log_prob(s_t[None], s_h_t[None])
                     for (s_t, s_h_t) in zip(x, past)]
        all_probs = [torch.as_tensor(p) for p in zip(*all_probs)]
        return all_probs[0], all_probs[1], all_probs[2]

    def _calc_individual_log_prob(self, x, past) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        probs = [self.flow.individual_log_prob(s_t[None], s_h_t[None])
                 for (s_t, s_h_t) in zip(x, past)]
        probs = [torch.vstack(p) for p in zip(*probs)]

        return probs[0], probs[1], probs[2]

    def _create_train_validation_split(self, samples: np.ndarray, sample_hist: np.ndarray) -> (
            (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        # handle json based dataset for LSTM
        # use the end as validation set
        (
            (train_samples, train_hist_samples),
            (validation_samples, validation_hist_samples),
            (_, _)
        ) = create_dataset_from_dataset_with_end_as_validation(samples, sample_hist,
                                                               self.parameters["past"], noise=True)

        return (train_samples, train_hist_samples), (validation_samples, validation_hist_samples)

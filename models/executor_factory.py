from models.flow_executor import FlowBatchedExecutor, FlowNonBatchedExecutor

# Dictionary mapping model types to their corresponding executor classes
executors = {
    "RealNVP": FlowBatchedExecutor,
    "tcNF-base": FlowBatchedExecutor,
    "tcNF-cnn": FlowBatchedExecutor,
    "tcNF-mlp": FlowBatchedExecutor,
    "tcNF-stateless": FlowBatchedExecutor,
    "tcNF-stateful": FlowNonBatchedExecutor,
}


class ExecutorFactory:
    """
    Factory class to create executor instances based on the model type.
    """

    @staticmethod
    def create_executor(run_args, parameters):
        """
        Create an executor instance based on the provided model type.

        :param run_args: Arguments for the run.
        :param parameters: Parameters for the Executor.
        :return: An instance of the appropriate executor class.
        """
        executor = executors[parameters["model_type"]](run_args, parameters)
        return executor

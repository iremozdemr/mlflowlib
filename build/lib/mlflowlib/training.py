#Delete all files in the dist folder
#Update the version number in the setup.py file
#Re-create the wheels:
#python3 setup.py sdist bdist_wheel
#Re-upload the new files:
#twine upload dist/*

# import mlflow
# import tensorflow as tf
# from tensorflow.keras.callbacks import Callback
# from contextlib import contextmanager

# class LearningRateLogger(Callback):
#     """Callback to log learning rate at the end of each epoch."""
#     def on_epoch_end(self, epoch, logs=None):
#         lr = self.model.optimizer.learning_rate
#         if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
#             lr = lr(self.model.optimizer.iterations)
#         logs = logs or {}
#         mlflow.log_metric('learning_rate', float(lr.numpy()), step=epoch)

# class LossAndErrorPrintingCallback(Callback):
#     """Callback to log metrics at the end of each epoch."""
#     def on_epoch_end(self, epoch, logs=None):
#         mlflow.log_metrics(logs, step=epoch)

# @contextmanager
# def start_mlflow(tracking_uri: str, experiment_name: str, run_name: str, params: dict, autolog: bool = False):
#     """Context manager for starting an MLflow run.
    
#     This function sets up MLflow tracking, starts a new run, logs parameters, and optionally enables autologging.
    
#     Args:
#         tracking_uri (str): The URI for the MLflow tracking server.
#         experiment_name (str): The name of the experiment to log the run under.
#         run_name (str): The name of the run.
#         params (dict): A dictionary of parameters to log.
#         autolog (bool): If True, enables MLflow autologging for TensorFlow. Defaults to False.
        
#     Yields:
#         run: The MLflow run object. Can be used to access run information if needed.
#     """
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_experiment(experiment_name)
    
#     with mlflow.start_run(run_name=run_name) as run:
#         if autolog:
#             mlflow.tensorflow.autolog(log_models=True)
#         else:
#             mlflow.tensorflow.autolog(log_models=False)

#         for key, value in params.items():
#             mlflow.log_param(key, value)

#         yield run  

import mlflow
from contextlib import contextmanager
from typing import Dict

@contextmanager
def start_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: Dict = None,
    metrics: Dict = None,
    metrics_prefix: str = "",
    autolog: bool = False,
    using_tensorflow: bool = False,
    using_xgboost: bool = True,
):
    """
    Context manager for starting an MLflow run.

    Args:
        tracking_uri (str): The URI for the MLflow tracking server.
        experiment_name (str): The name of the experiment to log the run under.
        run_name (str): The name of the run.
        params (dict): A dictionary of parameters to log.
        metrics (dict): A dictionary of metrics to log.
        metrics_prefix (str): Prefix to add to metric names when logging.
        autolog (bool): If True, enables MLflow autologging for XGBoost or TensorFlow. Defaults to False.
        using_tensorflow (bool): If True, enable autologging for TensorFlow. Defaults to False.
        using_xgboost (bool): If True, enable autologging for XGBoost. Defaults to True.

    Yields:
        run: The MLflow run object.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Autolog desteği
        if autolog:
            if using_tensorflow:
                mlflow.tensorflow.autolog(log_models=True)
            if using_xgboost:
                mlflow.xgboost.autolog(log_models=True)
        else:
            mlflow.tensorflow.autolog(log_models=False)
            mlflow.xgboost.autolog(log_models=False)

        # Parametreleri logla
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # Metrikleri logla
        if metrics:
            for metric_name, value in metrics.items():
                metric_full_name = f"{metrics_prefix}{metric_name}" if metrics_prefix else metric_name
                mlflow.log_metric(metric_full_name, value)

        # Context'i dışarı aktar
        yield run
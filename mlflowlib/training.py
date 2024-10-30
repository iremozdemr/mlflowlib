# #Delete all files in the dist folder
# #Update the version number in the setup.py file
# #Re-create the wheels:
# #python3 setup.py sdist bdist_wheel
# #Re-upload the new files:
# #twine upload dist/*

import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class LearningRateLogger(Callback):
    """Callback to log learning rate at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        logs = logs or {}
        mlflow.log_metric('learning_rate', float(lr.numpy()), step=epoch)

class LossAndErrorPrintingCallback(Callback):
    """Callback to log metrics at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)

def setup_mlflow(
    tracking_uri: str,
    experiment_name: str,
    model: 'tf.keras.Model',
    batch_size: int,
    epochs: int,
    run_name: str,
    autolog: bool = False,
    additional_params: dict = None,
):
    """
    Sets up MLflow tracking, logs parameters, and prepares for logging model training metrics.

    Args:
        tracking_uri (str): MLflow tracking URI.
        experiment_name (str): Name of the experiment in MLflow.
        model (tf.keras.Model): The Keras model to be logged.
        batch_size (int): Size of batches during training.
        epochs (int): Number of epochs for training.
        run_name (str): Name of the run in MLflow.
        autolog (bool): Whether to enable autologging.
        additional_params (dict): Additional parameters to log.

    Returns:
        None.
    """
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if autolog:
        mlflow.keras.autolog(log_models=False)

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        # Log additional parameters if any
        if additional_params:
            for key, value in additional_params.items():
                mlflow.log_param(key, value)

        # Log the model
        mlflow.keras.log_model(model, "model")

        print("####################")
        print("run info: ")
        print("####################")
        print("run id: ", run.info.run_id)
        print("start time:", run.info.start_time)
        print("end time:", run.info.end_time)
        print("status:", run.info.status)
        print("user id:", run.info._user_id)
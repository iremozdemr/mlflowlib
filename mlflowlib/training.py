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

def log_experiment(model: tf.keras.Model, 
                   batch_size: int = 32,  # Varsayılan değer
                   epochs: int = 10,  # Varsayılan değer
                   run_name: str = "my_run",  # Varsayılan değer
                   tracking_uri: str = "http://localhost:5000",
                   experiment_name: str = "my_experiment",
                   autolog: bool = False,
                   additional_params: dict = None):
    """Logs the MLflow experiment and prepares the model for training.

    Args:
        model (tf.keras.Model): The Keras model to be trained.
        batch_size (int): Size of batches during training. Defaults to 32.
        epochs (int): Number of epochs for training. Defaults to 10.
        run_name (str): Name of the run in MLflow. Defaults to "Default_Run".
        tracking_uri (str): MLflow tracking URI. Defaults to "http://localhost:5000".
        experiment_name (str): Name of the experiment in MLflow. Defaults to "Default_Experiment".
        autolog (bool): Whether to enable autologging. Defaults to False.
        additional_params (dict): Additional parameters to log.

    Returns:
        None. The function prepares the model for training and logs necessary parameters.
    """
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if autolog:
        mlflow.keras.autolog(log_models=False)

    additional_params = additional_params or {}

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        # Log additional parameters if any
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
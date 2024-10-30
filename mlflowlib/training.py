# #Delete all files in the dist folder
# #Update the version number in the setup.py file
# #Re-create the wheels:
# #python3 setup.py sdist bdist_wheel
# #Re-upload the new files:
# #twine upload dist/*

# import mlflow
# import tensorflow as tf
# from tensorflow.keras.callbacks import Callback

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

# def start_mlflow(tracking_uri: str, experiment_name: str, run_name: str, params: dict, autolog: bool = False):
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_experiment(experiment_name)

#     if autolog:
#         mlflow.tensorflow.autolog(log_models=True)
#     else:
#         mlflow.tensorflow.autolog(log_models=False)

#     mlflow.start_run(run_name=run_name) 

#     for key, value in params.items():
#         mlflow.log_param(key, value)

import mlflow
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from contextlib import contextmanager

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

@contextmanager
def start_mlflow(tracking_uri: str, experiment_name: str, run_name: str, params: dict, autolog: bool = False):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Run başlatılıyor
    with mlflow.start_run(run_name=run_name) as run:
        if autolog:
            mlflow.tensorflow.autolog(log_models=True)
        else:
            mlflow.tensorflow.autolog(log_models=False)

        for key, value in params.items():
            mlflow.log_param(key, value)

        yield run  # Context manager burada devam ediyor

        # Eğitim sonunda run otomatik kapanacak
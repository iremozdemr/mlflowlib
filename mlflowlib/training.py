# #Delete all files in the dist folder
# #Update the version number in the setup.py file
# #Re-create the wheels:
# #python3 setup.py sdist bdist_wheel
# #Re-upload the new files:
# #twine upload dist/*

# import mlflow
# import tensorflow as tf
# # from tensorflow.keras import Input, Model
# # from tensorflow.keras.layers import Conv3D, Flatten, Concatenate, Dense, Reshape
# from tensorflow.keras.callbacks import Callback
# import time
# from mlflow import keras as mlflow_keras
# import argparse

# # Callback to log learning rate
# class LearningRateLogger(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         lr = self.model.optimizer.learning_rate
#         if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
#             lr = lr(self.model.optimizer.iterations)
#         logs = logs or {}
#         mlflow.log_metric('learning_rate', float(lr.numpy()), step=epoch)

# # Callback to log loss and errors
# class LossAndErrorPrintingCallback(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         mlflow.log_metrics(logs, step=epoch)

# # Custom callback to track and print progress
# class ProgressLogger(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         print(f"Epoch {epoch+1}/{self.params['epochs']} started.")
#         self.start_time = time.time()

#     def on_epoch_end(self, epoch, logs=None):
#         duration = time.time() - self.start_time
#         print(f"Epoch {epoch+1}/{self.params['epochs']} completed in {duration:.2f} seconds.")
#         print(f"Metrics: {logs}")
#         progress = ((epoch+1) / self.params['epochs']) * 100
#         print(f"Progress: {progress:.2f}% completed.")

#     def on_batch_end(self, batch, logs=None):
#         steps = self.params['steps']
#         progress = ((batch+1) / steps) * 100 if steps else 0
#         print(f"Batch {batch+1}/{steps} - {progress:.2f}% completed.")

# # General model training function
# def train_model(
#     run_name="example",
#     tracking_uri="http://your_mlflow_server_uri:5000",
#     experiment_name='using mlflowlib',
#     batch_size=32,
#     epochs=7,
#     callbacks=None,
#     model=None
# ):
#     """
#     Args:
#         run_name (str): Name of the run in MLflow. This is used to distinguish different 
#                         model training sessions within the same experiment.
#         tracking_uri (str, optional): MLflow tracking URI. This is the URI where the MLflow 
#                                       server is hosted. Defaults to "http://your_mlflow_server_uri:5000".
#         experiment_name (str, optional): Name of the experiment in MLflow. If it does not exist, 
#                                          it will be created. Defaults to 'using mlflowlib'.
#         batch_size (int, optional): Size of batches during training. Determines how many samples 
#                                     will be processed before updating the model weights. 
#                                     Defaults to 32.
#         epochs (int, optional): Number of epochs for training. An epoch is one full cycle through 
#                                 the entire training dataset. Defaults to 7.
#         model (tf.keras.Model, optional): The Keras model object that will be trained. This model 
#                                           should be compiled with an optimizer, loss function, and 
#                                           metrics before being passed into this function. Defaults to None.

#     Returns:
#         None. The function logs all relevant metrics and the model to MLflow during and after training.
#     """

#     # Set up MLflow
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_experiment(experiment_name)
#     mlflow.tensorflow.autolog(log_models=False)

#     lr_logger_callback = LearningRateLogger()
#     get_metrics_callback = LossAndErrorPrintingCallback()

#     with mlflow.start_run(run_name=run_name) as run:

#         mlflow.log_param("batch_size", batch_size)
#         mlflow.log_param("epochs", epochs)

#         if callbacks is None:
#                 lr_logger_callback = LearningRateLogger()
#                 get_metrics_callback = LossAndErrorPrintingCallback()
#                 callbacks = [lr_logger_callback, get_metrics_callback]
#                 progress_logger_callback = ProgressLogger()
#                 callbacks = [lr_logger_callback, get_metrics_callback, progress_logger_callback]
                
#         model.fit(callbacks=callbacks)
#         mlflow.keras.log_model(model, "model")

#         # print("####################")
#         # print("run info: ")
#         # print("####################")
#         # print("run id: ",run.info.run_id)
#         # print("start time:",run.info.start_time)
#         # print("end time:",run.info.end_time)
#         # print("status:",run.info.status)
#         # print("status:",run.info._user_id)

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
                   train_data=None, 
                   test_data=None, 
                   batch_size: int = 32,  # Varsayılan değer eklenmiştir
                   epochs: int = 10,  # Varsayılan değer eklenmiştir
                   run_name: str = "Default_Run",  # Varsayılan değer eklenmiştir
                   tracking_uri: str = "http://localhost:5000",
                   experiment_name: str = "Default_Experiment",
                   autolog: bool = False,
                   additional_params: dict = None):
    """Logs the MLflow experiment and trains the model.

    Args:
        model (tf.keras.Model): The Keras model to be trained.
        train_data: The training data generator or a tuple of (X, y).
        test_data: The validation data generator or a tuple of (X, y).
        batch_size (int): Size of batches during training. Defaults to 32.
        epochs (int): Number of epochs for training. Defaults to 10.
        run_name (str): Name of the run in MLflow. Defaults to "Default_Run".
        tracking_uri (str): MLflow tracking URI. Defaults to "http://localhost:5000".
        experiment_name (str): Name of the experiment in MLflow. Defaults to "Default_Experiment".
        autolog (bool): Whether to enable autologging. Defaults to False.
        additional_params (dict): Additional parameters to log.

    Returns:
        history: The training history returned by model.fit.
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

        # Train the model
        if train_data is not None and test_data is not None:
            # If generators are provided, use them
            history = model.fit(
                train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=test_data,
                callbacks=[
                    LearningRateLogger(),
                    LossAndErrorPrintingCallback()
                ]
            )
        else:
            # If data is provided as numpy arrays
            history = model.fit(
                train_data[0],  # X
                train_data[1],  # y
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(test_data[0], test_data[1]) if test_data else None,
                callbacks=[
                    LearningRateLogger(),
                    LossAndErrorPrintingCallback()
                ]
            )

        # Log the model
        mlflow.keras.log_model(model, "model")
        return history
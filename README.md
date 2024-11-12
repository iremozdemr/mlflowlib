# MLFlow Model Training Package

This Python package simplifies MLflow integration with TensorFlow models by providing custom callbacks and a context manager for efficient experiment tracking and logging.

## Features
- **LearningRateLogger**: A custom TensorFlow callback to log the learning rate at the end of each epoch. This is helpful for tracking learning rate changes when using learning rate schedules or optimizers that adapt during training.
- **LossAndErrorPrintingCallback**: A callback that logs the training metrics at the end of each epoch, ensuring that metrics like loss, accuracy, and custom metrics are recorded in MLflow after each epoch.
- **start_mlflow**: A context manager to set up and manage an MLflow run. This utility function allows you to:
+ Specify the MLflow tracking URI, experiment name, and run name.
+ Log model parameters easily by passing a dictionary of parameters.
+ Optionally enable MLflow’s autologging for TensorFlow models to automatically log training details, including model architecture, optimizer configurations, and metrics.
+ Provides custom callbacks for logging metrics and learning rate to MLflow.

## Installation

To install the package, use `pip`:

```bash
pip install mlflowlib
```

```bash
pip install mlflowlib==[version-number]
```

## Example Usage

Below is an example of how to use the `start_mlflow` function from the package 

```python
params = {
    "batch_size": 16,  
    "epochs": 10       
}
#parametreleri belirle
    
model = Model(inputs=[input_images, input_wind_speeds], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    
lr_logger_callback = training.LearningRateLogger()
get_metrics_callback = training.LossAndErrorPrintingCallback()
#LearningRateLogger() ve LossAndErrorPrintingCallback() kullanımı:

with training.start_mlflow(
    tracking_uri="http://your-url",
    experiment_name="experiment_a",
    run_name='run_b',
    params=params,
    autolog=True
) as run:
#start_mlflow fonksiyonunu çağır
        
    history = model.fit(
        train_generator, 
        epochs=5, 
        steps_per_epoch=len(train_generator), 
        validation_data=test_generator, 
        callbacks=[
                tensorboard_callback, 
                checkpoint_callback, 
                earlystop_callback, 
                lr_scheduler_callback, 
                lr_logger_callback, 
                get_metrics_callback
        ]
    )

    mlflow.keras.log_model(model, "model")
```
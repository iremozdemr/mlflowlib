# MLFlow Model Training Package

This package provides a general-purpose deep learning model training function, integrated with MLflow for logging and tracking experiments. The model supports TensorFlow-based training and allows easy configuration of various hyperparameters such as input shapes, dense layers, and callbacks. Additionally, it logs important metrics like loss, learning rate, and model performance using MLflow.

## Features
- **MLflow Integration**: Automatically tracks and logs experiments, parameters, and metrics.
- **Flexible Model Design**: Easily configure the model architecture (input layers, dense layers, and output layer).
- **Callbacks**: Provides custom callbacks for logging metrics and learning rate to MLflow.
- **Device Selection**: Choose whether to train on CPU or GPU.
- **Default Parameters**: Provides sensible defaults for common use cases, while allowing full customization.

## Installation

To install the package, use `pip`:

```bash
pip install gtek_mlflow
```

## Example Usage

Below is an example of how to use the `train_model` function from the package with a simple TensorFlow data generator.

```python
from mlflow_utils import log_experiment
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Reshape, Concatenate
from tensorflow.keras.models import Model

# Example usage
if __name__ == "__main__":
    # Define parameters
    run_name = "run_name_demo"
    tracking_uri = "http:your/url"
    experiment_name = 'experiment_name_demo'
    batch_size = 32
    epochs = 7

    # Define model architecture
    input_images = Input(shape=(8, 670, 1413, 3), name='image_input')
    x = Conv3D(64, kernel_size=(3, 3, 3), strides=(4, 4, 4), activation='relu', padding='same')(input_images)
    x = Conv3D(32, kernel_size=(3, 3, 3), strides=(4, 4, 4), activation='relu', padding='same')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), strides=(4, 4, 4), activation='relu', padding='same')(x)
    x = Flatten()(x)

    input_wind_speeds = Input(shape=(turbine_total_count, 24), name='wind_speed_input')
    flattened_wind_speeds = Flatten()(input_wind_speeds)  

    combined = Concatenate()([x, flattened_wind_speeds])
    combined = Dense(128, activation='relu')(combined)
    combined = Dense(64, activation='relu')(combined)
    output = Dense(turbine_total_count * 24, activation='linear')(combined)
    output = Reshape((turbine_total_count, 24))(output)
    
    model = Model(inputs=[input_images, input_wind_speeds], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Train the model using the log_experiment function
    log_experiment(model=model, 
                   train_generator=train_generator, 
                   test_generator=test_generator, 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   run_name=run_name,
                   tracking_uri=tracking_uri,
                   experiment_name=experiment_name,
                   additional_params={'param1': 'value1'})
```
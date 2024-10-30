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
if __name__ == "__main__":
    run_name = "run_name_demo"
    tracking_uri = "http://192.168.13.79:5000"
    experiment_name = 'experiment_name_demo'
    batch_size = 32
    epochs = 7
    turbine_total_count = 2
    samples = 100  # Eğitim verisi örnek sayısı

    # Örnek veri oluşturma
    (X_images, X_wind_speeds), y = generate_data(samples, input_shape=(8, 100, 100, 3), turbine_count=turbine_total_count)

    # Model mimarisini tanımlama
    model = ...  # Modelinizi burada tanımlayın

    # Modeli eğitmek için log_experiment fonksiyonunu kullan
    log_experiment(model=model, 
                   train_data=(X_images, y),  # Sayısal veri kullanıyoruz
                   test_data=(X_wind_speeds, y),  # Test verisi
                   batch_size=batch_size, 
                   epochs=epochs, 
                   run_name=run_name,
                   tracking_uri=tracking_uri,
                   experiment_name=experiment_name,
                   additional_params={'param1': 'value1'})
```
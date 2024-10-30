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
pip install mlflowlib
```

```bash
pip install mlflowlib==[version-number]
```

## Example Usage

Below is an example of how to use the `train_model` function from the package with a simple TensorFlow data generator.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from mlflowlib.training import log_experiment

# Rastgele sayısal veri üreten fonksiyon
def generate_data(samples, input_shape, output_shape):
    """Rastgele sayısal veri üreten fonksiyon."""
    # Girdi verisi
    X = np.random.rand(samples, *input_shape).astype(np.float32)
    # Hedef çıktı verisi
    y = np.random.rand(samples, *output_shape).astype(np.float32)  # Hedef çıktı
    return X, y

# Örnek kullanım
if __name__ == "__main__":
    # Parametreleri tanımla
    run_name = "example_run"
    tracking_uri = "http://localhost:5000"
    experiment_name = 'example_experiment'
    batch_size = 32
    epochs = 10
    samples = 1000  # Eğitim verisi örnek sayısı
    input_shape = (10,)  # Giriş verisi boyutu (örneğin 10 özellik)
    output_shape = (1,)  # Çıktı boyutu (örneğin 1 hedef)

    # Veri setini oluştur
    X, y = generate_data(samples, input_shape=input_shape, output_shape=output_shape)

    # Model mimarisini tanımla
    input_data = Input(shape=input_shape)
    x = Flatten()(input_data)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(output_shape[0], activation='linear')(x)  # Çıktı katmanı
    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Modeli eğitmek için log_experiment fonksiyonunu kullan
    log_experiment(model=model, 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   run_name=run_name,
                   tracking_uri=tracking_uri,
                   experiment_name=experiment_name,
                   additional_params={'param1': 'value1'})  # İsteğe bağlı ek parametreler
```
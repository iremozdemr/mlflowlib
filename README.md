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

Below is an example of how to use the `start_mlflow` function from the package 

```python
```
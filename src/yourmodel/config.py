"""
Generally speaking, writing a `Config` is the first step in implementing a pre-trained model,
because the `transformers` library in Huggingface requires a specific `Config` class to build the model.

When creating a model, we usually only input a specific `Config` to adapt various parameters, etc.
Here, for a model, we typically create two `Config` classes: one for model building and one for model training.
"""

from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig


@dataclass
class YourModelConfig(PretrainedConfig):
    """
    The Huggingface transformer-style pre-trained model config for `xxx` model
    """

    pass


@dataclass
class YourModelTrainingConfig(PretrainedConfig):
    """
    The training configuration for `xxx` model
    """

    pass

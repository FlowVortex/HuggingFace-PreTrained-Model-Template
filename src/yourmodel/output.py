"""
We typically create an `output.py` file to implement the model output class.
This class usually inherits from the `ModelOutput` class in Huggingface.
Generally, we can choose to write only the Output for the entire model.

However, to further standardize the input and output within the model framework,
it is recommended to define the input and output content for the model's backbone and multiple model layers as well.
"""

from dataclasses import dataclass

import torch
from transformers.file_utils import ModelOutput


@dataclass
class AttentionOutput(ModelOutput):
    """
    The output of the attention mechanism in each layer of the Transformer model
    """

    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class EncoderBlockOutput(ModelOutput):
    """
    The output of the encoder model in each layer of the Transformer model
    """

    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class EncoderOutput(ModelOutput):
    """
    The output of the entire encoder backbone of the Transformer model.
    """

    last_hidden_state: torch.Tensor | None = None
    all_attn_weights: tuple[torch.Tensor, ...] | None = None


@dataclass
class YourModelOutput(ModelOutput):
    """
    The model's final output.
    Here, `YourModel` refers to the name of the specific model.
    We will name this class after the specific model.

    This class will return the model's specific output and the loss during training.
    """

    loss: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    all_attn_weights: tuple[torch.Tensor, ...] | None = None

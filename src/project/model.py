"""
This file is primarily used to implement the specific structure and logic of the model.
The model needs to be passed a specific Huggingface configuration file to build the model.
For models with multiple different layer architectures, it is recommended to create an
additional `layers.py` file to implement the specific layer architecture of the model.
And only put the core parts of the model, such as the `backbone`, in this file.

We also need to further consider the compatibility of the model itself with the `pipeline` and `trainer`.
"""

import copy
from dataclasses import dataclass
from typing import cast, Union, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel

from project.config import ProjectConfig
from project.output import ProjectOutput


class ProjectModel(PreTrainedModel):
    config_class = ProjectConfig  # type: ignore[assignment]
    # _supports_long_horizon: bool = True
    # _supports_sdpa: bool = True
    # _supports_flash_attn_2 = True

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__(config)

        # This is where you create the model's internal modules and perform initialization.

        # Initialize weights and apply final processing
        # self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _validate_input(
        self,
    ) -> None:
        """
        This method is used to check if the input data meets the requirements.

        This method does not perform any data processing and does not forcibly return anything.
        It only checks if the input data meets the requirements.

        When the input data does not meet the requirements, this method will throw an exception.
        """

    def _compute_loss(
        self, *args, **kwargs
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor, float]:
        """
        This method is used to calculate the model's loss and returns a floating-point number.

        To simplify the downstream model training process, we usually calculate the model loss during the forward propagation of the pre-trained model.
        """

    def forward(self, *args, **kwargs) -> ProjectOutput:
        pass

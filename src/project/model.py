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

        # 在这里创建模型的内部module和执行初始化

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
        这个方法用于检验输入的数据是否符合要求，
        这个方法并不会对数据进行任何的处理也不会强制性的返回任何东西
        它只会用于检验输入的数据是否符合要求
        当输入的数据不符合要求时，这个方法会抛出异常
        """

    def _compute_loss(
        self, *args, **kwargs
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor, float]:
        """
        这个方法用于计算模型的损失
        这个方法会返回一个浮点数，这个浮点数是模型的损失
        这个损失会随着模型的output被一起返回
        """

    def forward(self, *args, **kwargs) -> ProjectOutput:
        pass

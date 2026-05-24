"""
通常来说编写`Config`是我们实现预训练模型的第一步，
因为对于Huggingface中`transformers`库来说，
它需要一个具体的`Config`类来构建模型。
我们在创建模型的时候往往都只输入一个具体的Config进行各种参数的适配等。
在这里对于一个模型我们通常会创建两个`Config`类，
一个用于模型的构建，一个用于模型的训练。
"""

from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig


@dataclass
class ProjectConfig(PretrainedConfig):
    """
    The Huggingface transformer-style pre-trained model config for `xxx` model
    """

    pass


@dataclass
class ProjectTrainingConfig(PretrainedConfig):
    """
    The training configuration for `xxx` model
    """

    pass

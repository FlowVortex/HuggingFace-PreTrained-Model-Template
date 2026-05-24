from dataclasses import dataclass

import torch
from transformers.file_utils import ModelOutput


@dataclass
class AttentionOutput(ModelOutput):
    """
    Transformer模型每一层中注意力机制的输出
    """

    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class EncoderBlockOutput(ModelOutput):
    """
    Transformer模型每一层中编码器模型的输出
    """

    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class EncoderOutput(ModelOutput):
    """
    Transformer模型整个编码器的backbone的输出
    """

    last_hidden_state: torch.Tensor | None = None
    all_attn_weights: tuple[torch.Tensor, ...] | None = None


@dataclass
class ProjectOutput(ModelOutput):
    """
    模型最终的输出
    这里的`Project`是指具体模型的名字
    我们会根据具体模型的名字来命名这个类
    这个类将会返回模型具体的一些输出内容以及训练过程中的损失
    """

    loss: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    all_attn_weights: tuple[torch.Tensor, ...] | None = None

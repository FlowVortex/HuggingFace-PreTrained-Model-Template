"""
`trainer.py` is typically used to store the model's training logic,
which can be categorized into different training methods such as supervised training,
large-scale pre-training, and unsupervised training.
Here, we usually write a `BaseTrainer` class to implement some general and basic methods,
and then implement more specific training modules through inheritance.

Our `Trainer` class typically has three implementation methods:

First, it's generally based on the `Accelerate` library to implement multi-GPU training logic.
The advantage of this method is that all training logic and key steps are completely transparent,
and it allows for easy debugging and optimization.

Second, it's based on the `PyTorch Lightning` library to implement the training logic.
The advantage of this method is that it provides a higher-level training interface
and more flexible training methods, and it facilitates distributed training.

Third, it's based on the `Trainer` class of the `Huggingface` library to implement the training logic.
This method is more adaptable to our `PreTrainedModel` and can be used to train the model in a more flexible way.
"""


class BaseTrainer(object):
    pass

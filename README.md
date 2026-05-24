# Hugging Face Pre-trained Model Template <img width="12%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/main/docs/source/_static/S2Generator_logo.png?raw=true">

A modular, production-ready template for building Hugging Face Transformers-compatible pre-trained models. This template follows Hugging Face ecosystem conventions and provides a structured foundation for model development, training, and deployment.

## YourModel Structure

```
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration notebook
├── scripts/
│   └── pre-training.sh         # Shell script for launching pre-training jobs
├── src/
│   ├── yourmodel/              # Main project package
│   │   ├── __init__.py         # Package initialization and versioning
│   │   ├── config.py           # Model and training configuration classes
│   │   ├── dataset.py          # Custom dataset implementations
│   │   ├── model.py            # Core model architecture
│   │   ├── output.py           # Typed model output classes
│   │   ├── pipeline.py         # High-level inference pipelines
│   │   ├── trainer.py          # Training logic and trainer classes
│   │   └── utils.py            # Reusable utility functions
│   └── main.py                 # Entry point for training and fine-tuning
├── .gitignore
├── LICENSE
└── README.md
```

## Core Modules

### `src/yourmodel/__init__.py`
- Defines the package version (`__version__ = "0.0.1"`)
- Exposes public API through `__all__` list
- Imports key classes (config, model, pipeline, trainer) for easy access

### `src/yourmodel/config.py`
- First step in implementing a Hugging Face model
- Contains two primary configuration classes:
  - `YourModelConfig`: Inherits from `PretrainedConfig`, defines model architecture parameters
  - `YourModelTrainingConfig`: Defines training-specific hyperparameters
- Fully compatible with Hugging Face Transformers configuration system

### `src/yourmodel/dataset.py`
- Implements `Dataset` and `IterableDataset` classes for training, fine-tuning, and evaluation
- Handles data parsing, tokenization, batching, and augmentation
- Follows Hugging Face Datasets API standards
- Compatible with Hugging Face Trainer API and standard PyTorch DataLoader

### `src/yourmodel/model.py`
- Implements the core model architecture
- Inherits from `PreTrainedModel` for full Hugging Face compatibility
- Includes weight initialization (`_init_weights`)
- Provides input validation (`_validate_input`)
- Computes loss internally during forward pass (`_compute_loss`)
- Returns typed outputs defined in `output.py`
- For complex models, layer implementations should be moved to a separate `layers.py` file

### `src/yourmodel/output.py`
- Defines strongly-typed output classes inheriting from `ModelOutput`
- Standardizes outputs across different model components:
  - `AttentionOutput`: Output of individual attention layers
  - `EncoderBlockOutput`: Output of individual encoder blocks
  - `EncoderOutput`: Output of the entire encoder backbone
  - `YourModelOutput`: Final model output including loss and hidden states
- Improves code readability and type safety

### `src/yourmodel/pipeline.py`
- Provides high-level, user-friendly inference interfaces
- Encapsulates the complete model usage workflow:
  - Model loading from Hugging Face Hub or local files
  - Configuration setup
  - Input preprocessing
  - Inference execution
  - Result post-processing
- Serves as the primary entry point for end-users
- Maintains compatibility with Hugging Face pipeline conventions

### `src/yourmodel/trainer.py`
- Contains all training logic
- Supports multiple training paradigms:
  - Supervised training
  - Large-scale pre-training
  - Unsupervised/self-supervised training
- Base `BaseTrainer` class implements common functionality
- Supports three implementation approaches:
  1. Accelerate-based (full transparency and control)
  2. PyTorch Lightning-based (high-level interface)
  3. Hugging Face Trainer-based (best ecosystem integration)

### `src/yourmodel/utils.py`
- Reusable helper functions used across the codebase
- Includes:
  - Visualization tools
  - Data processing helpers
  - Logging utilities
  - Common mathematical operations
- Designed to be generic and architecture-agnostic

### `src/main.py`
- Entry point script for model pre-training and fine-tuning
- Called by shell scripts in the `scripts/` directory
- Parses command-line arguments
- Initializes config, model, dataset, and trainer
- Executes the training loop

## Development Guide

### Adding a New Model Architecture

1. Create a new configuration class in `config.py`
2. Implement the model architecture in `model.py` (or create a new file)
3. Define appropriate output classes in `output.py`
4. Create a corresponding pipeline in `pipeline.py`
5. Update `__init__.py` to expose the new classes

### Adding a New Dataset

1. Implement a new `Dataset` or `IterableDataset` class in `dataset.py`
2. Add data loading and processing logic
3. Update the training script in `main.py` to support the new dataset

### Adding a New Training Method

1. Create a new trainer class in `trainer.py` inheriting from `BaseTrainer`
2. Implement the training loop and any custom logic
3. Add command-line arguments in `main.py` to select the new trainer

**Note:** Replace all instances of `YourModel`, `yourmodel`, and `Your` with your actual project name throughout the codebase and documentation.
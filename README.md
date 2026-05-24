# Hugging Face Pre-trained Model Template

A modular, production-ready template for building Hugging Face Transformers-compatible pre-trained models. This template follows Hugging Face ecosystem conventions and provides a structured foundation for model development, training, and deployment.

## Project Structure

```
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration notebook
├── scripts/
│   └── pre-training.sh         # Shell script for launching pre-training jobs
├── src/
│   ├── project/                # Main project package
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

### `src/project/__init__.py`
- Defines the package version (`__version__ = "0.0.1"`)
- Exposes public API through `__all__` list
- Imports key classes (config, model, pipeline, trainer) for easy access

### `src/project/config.py`
- First step in implementing a Hugging Face model
- Contains two primary configuration classes:
  - `ProjectConfig`: Inherits from `PretrainedConfig`, defines model architecture parameters
  - `ProjectTrainingConfig`: Defines training-specific hyperparameters
- Fully compatible with Hugging Face Transformers configuration system

### `src/project/dataset.py`
- Implements `Dataset` and `IterableDataset` classes for training, fine-tuning, and evaluation
- Handles data parsing, tokenization, batching, and augmentation
- Follows Hugging Face Datasets API standards
- Compatible with Hugging Face Trainer API and standard PyTorch DataLoader

### `src/project/model.py`
- Implements the core model architecture
- Inherits from `PreTrainedModel` for full Hugging Face compatibility
- Includes weight initialization (`_init_weights`)
- Provides input validation (`_validate_input`)
- Computes loss internally during forward pass (`_compute_loss`)
- Returns typed outputs defined in `output.py`
- For complex models, layer implementations should be moved to a separate `layers.py` file

### `src/project/output.py`
- Defines strongly-typed output classes inheriting from `ModelOutput`
- Standardizes outputs across different model components:
  - `AttentionOutput`: Output of individual attention layers
  - `EncoderBlockOutput`: Output of individual encoder blocks
  - `EncoderOutput`: Output of the entire encoder backbone
  - `ProjectOutput`: Final model output including loss and hidden states
- Improves code readability and type safety

### `src/project/pipeline.py`
- Provides high-level, user-friendly inference interfaces
- Encapsulates the complete model usage workflow:
  - Model loading from Hugging Face Hub or local files
  - Configuration setup
  - Input preprocessing
  - Inference execution
  - Result post-processing
- Serves as the primary entry point for end-users
- Maintains compatibility with Hugging Face pipeline conventions

### `src/project/trainer.py`
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

### `src/project/utils.py`
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

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-project.git
cd your-project

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Basic Inference

```python
from project import YourModelPipeline

# Load the pipeline
pipeline = YourModelPipeline.from_pretrained("your-username/your-model")

# Run inference
result = pipeline("Your input text here")
print(result)
```

## Training

### Pre-training

```bash
# Launch pre-training using the provided script
bash scripts/pre-training.sh
```

The `pre-training.sh` script calls `main.py` with appropriate arguments. You can modify the script to adjust training parameters or specify different configuration files.

### Fine-tuning

```bash
# Fine-tune on your custom dataset
python src/main.py \
  --do_finetune \
  --model_name_or_path your-username/your-model \
  --dataset_path ./data/your-dataset \
  --output_dir ./finetune-output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16
```

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

**Note:** Replace all instances of `Project`, `project`, and `Your` with your actual project name throughout the codebase and documentation.
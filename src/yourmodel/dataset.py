"""
dataset.py - Custom dataset implementations for model training and fine-tuning.

This module contains specialized Dataset and IterableDataset classes for loading, processing,
and formatting data used during model pre-training, fine-tuning, and evaluation. It handles
data parsing, tokenization, batching preparation, and data augmentation specific to our
model's input requirements.

Dataset classes follow Hugging Face Datasets API standards, ensuring compatibility with
Hugging Face Trainer API and standard PyTorch data loading pipelines.

"""

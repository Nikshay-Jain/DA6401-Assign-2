import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
import math

from A_classes import *

def train_best_model(config, data_dir='/kaggle/input/inaturalist/inaturalist_12K', project_name="inaturalist_cnn_final"):
    """
    Train the best model based on sweep results.
    This addresses Question 4: Training and evaluating on test data
    
    Args:
        config (dict): Best hyperparameter configuration
        data_dir (str): Path to dataset directory
        project_name (str): Name of the wandb project
    """
    # Initialize wandb
    wandb.init(project=project_name, config=config)
    
    # Generate filter counts based on strategy
    if config["filter_counts_strategy"] == 'same':
        filter_counts = [config["base_filters"]] * 5
    elif config["filter_counts_strategy"] == 'doubling':
        filter_counts = [config["base_filters"] * (2**i) for i in range(5)]
    elif config["filter_counts_strategy"] == 'halving':
        filter_counts = [config["base_filters"] * (2**(4-i)) for i in range(5)]
    
    # Generate filter sizes
    filter_sizes = [config["filter_size"]] * 5
    
    # Create data module
    data_module = iNaturalistDataModule(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        augmentation=config["augmentation"]
    )
    data_module.setup()
    
    # Create model with best hyperparameters
    model = CustomCNN(
        num_classes=10,  # Assuming 10 classes in iNaturalist subset
        filter_counts=filter_counts,
        filter_sizes=filter_sizes,
        activation=config["activation"],
        dense_neurons=config["dense_neurons"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
        batch_norm=config["batch_norm"]
    )
    
    # Log model information
    wandb.log({
        'total_params': model.total_params,
        'total_computations': model.total_computations,
        'model_summary': str(model)
    })
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.4f}',
            save_top_k=1,
            mode='max'
        )
    ]
    
    # Setup wandb logger
    wandb_logger = WandbLogger(project=project_name)
    
    # Create trainer
    trainer = Trainer(
        max_epochs=30,  # Train longer for final model
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    test_results = trainer.test(model, data_module.test_dataloader())
    
    return model, test_results[0]['test_acc']

def display_model_architecture(model):
    """
    Display the architecture of the model with parameter counts
    This helps answer Question 1 about parameter and computation counts
    """
    print(f"Model Architecture Summary:")
    print(f"===========================")
    print(f"Total parameters: {model.total_params:,}")
    print(f"Total computations: {model.total_computations:,}")
    print(f"===========================")
    
    # Print the formulas for parameter and computation counts
    input_size = 244  # Adjust if using a different size
    base_filter = 32  # Example value, adjust as needed
    k = 3  # Example filter size, adjust as needed
    n = 512  # Example dense neurons, adjust as needed
    
    print(f"Formula for parameter count (with m={base_filter}, k={k}, n={n}):")
    print(f"Layer 1: m * (3 * k * k + 1) = {base_filter * (3 * k * k + 1)}")
    print(f"Layers 2-5: 4 * m * (m * k * k + 1) = {4 * base_filter * (base_filter * k * k + 1)}")
    
    # Calculate feature map size after 5 pooling layers (size/32)
    final_feature_size = input_size // 32
    flattened_size = base_filter * final_feature_size * final_feature_size
    
    print(f"Dense layer: flattened_size * n + n = {flattened_size * n + n}")
    print(f"Output layer: n * num_classes + num_classes = {n * 10 + 10}")
    
    print(f"\nFormula for computation count:")
    print(f"Layer 1: m * 3 * k * k * input_size * input_size = {m * 3 * k * k * input_size * input_size}")
    print(f"Layers 2-5: Sum of m * m * k * k * (input_size/(2^i)) * (input_size/(2^i)) = {m * m * k * k * (input_size/(2^i)) * (input_size/(2^i))}")
    print(f"Dense layer: flattened_size * n = {flattened_size * n}")
    print(f"Output layer: n * num_classes = {n * 10}")
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def setup_wandb_sweep():
    """Define sweep configuration for hyperparameter tuning"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'filter_counts_strategy': {
                'values': ['same', 'doubling', 'halving']  # Different filter count strategies
            },
            'base_filters': {
                'values': [16, 32, 64]  # Base number of filters
            },
            'filter_size': {
                'values': [3, 5]  # Filter sizes
            },
            'activation': {
                'values': ['relu', 'gelu', 'silu', 'mish']  # Different activation functions
            },
            'dense_neurons': {
                'values': [128, 256, 384, 512]  # Number of neurons in dense layer
            },
            'dropout_rate': {
                'values': [0.2, 0.3, 0.5]  # Dropout rate
            },
            'learning_rate': {
                'values': [0.0001, 0.001]  # Learning rate
            },
            'batch_norm': {
                'values': [True, False]  # Whether to use batch normalization
            },
            'batch_size': {
                'values': [16, 32]  # Batch size
            },
            'augmentation': {
                'values': [True, False]  # Whether to use data augmentation
            }
        }
    }
    
    return sweep_config

def train_model_sweep():
    """Training function for sweep"""
    # Initialize wandb
    wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Generate filter counts based on strategy
    if config.filter_counts_strategy == 'same':
        filter_counts = [config.base_filters] * 5
    elif config.filter_counts_strategy == 'doubling':
        filter_counts = [config.base_filters * (2**i) for i in range(5)]
    elif config.filter_counts_strategy == 'halving':
        filter_counts = [config.base_filters * (2**(4-i)) for i in range(5)]
    
    # Generate filter sizes
    filter_sizes = [config.filter_size] * 5
    
    # Create data module
    data_module = iNaturalistDataModule(
        batch_size=config.batch_size,
        augmentation=config.augmentation
    )
    data_module.setup()
    
    # Create model with hyperparameters
    model = CustomCNN(
        num_classes=10,  # Assuming 10 classes in iNaturalist subset
        filter_counts=filter_counts,
        filter_sizes=filter_sizes,
        activation=config.activation,
        dense_neurons=config.dense_neurons,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
        batch_norm=config.batch_norm
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.4f}',
            save_top_k=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_acc',
            patience=5,
            mode='max'
        )
    ]
    
    # Setup wandb logger
    wandb_logger = WandbLogger(project="inaturalist_cnn_sweep")
    
    # Create trainer
    trainer = Trainer(
        max_epochs=15,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    
    # Get best validation accuracy
    best_val_acc = trainer.callback_metrics.get('val_acc', 0)
    
    # Log metrics
    wandb.log({
        'best_val_acc': best_val_acc,
        'total_params': model.total_params,
        'total_computations': model.total_computations
    })
    
    return model, best_val_acc

def run_sweep():
    """Run the sweep"""
    # Initialize wandb
    wandb.login(key="e030007b097df00d9a751748294abc8440f932b1")
    
    # Setup sweep
    sweep_config = setup_wandb_sweep()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="inaturalist_cnn_sweep")
    
    # Run sweep
    wandb.agent(sweep_id, function=train_model_sweep, count=5)
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

def train_final_model(config):
    """Train final model with best hyperparameters"""
    # Initialize wandb
    wandb.init(project="inaturalist_cnn_final", config=config)
    
    # Generate filter counts based on strategy
    if config['filter_counts_strategy'] == 'same':
        filter_counts = [config['base_filters']] * 5
    elif config['filter_counts_strategy'] == 'doubling':
        filter_counts = [config['base_filters'] * (2**i) for i in range(5)]
    elif config['filter_counts_strategy'] == 'halving':
        filter_counts = [config['base_filters'] * (2**(4-i)) for i in range(5)]
    
    # Generate filter sizes
    filter_sizes = [config['filter_size']] * 5
    
    # Create data module
    data_module = iNaturalistDataModule(
        batch_size=config['batch_size'],
        augmentation=config['augmentation']
    )
    data_module.setup()
    
    # Create model with hyperparameters
    model = CustomCNN(
        num_classes=10,
        filter_counts=filter_counts,
        filter_sizes=filter_sizes,
        activation=config['activation'],
        dense_neurons=config['dense_neurons'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate'],
        batch_norm=config['batch_norm']
    )
    
    # Log model summary
    wandb.watch(model, log="all")
    
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
    wandb_logger = WandbLogger(project="inaturalist_cnn_final")
    
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
    
    # Test model
    test_results = trainer.test(model, dataloader=data_module.test_dataloader())
    
    # Log test results
    wandb.log({
        'test_acc': test_results[0]['test_acc'],
        'test_loss': test_results[0]['test_loss']
    })
    
    # Log model architecture
    wandb.log({
        'total_params': model.total_params,
        'total_computations': model.total_computations
    })
    
    return model, test_results

def visualize_test_samples(model, data_module, num_samples=30):
    """Visualize test samples with predictions"""
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Get a batch of test data
    images, labels = next(iter(test_loader))
    
    # Move to device
    model = model.to(device)
    images = images.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy for visualization
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    # Get class names
    class_names = data_module.test_dataset.classes
    
    # Visualize images with predictions
    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # Transpose image from (C, H, W) to (H, W, C)
            img = np.transpose(images[i], (1, 2, 0))
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Plot image
            ax.imshow(img)
            
            # Get true and predicted labels
            true_label = class_names[labels[i]]
            pred_label = class_names[predicted[i]]
            
            # Set title
            if labels[i] == predicted[i]:
                ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='green')
            else:
                ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
            
            ax.axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"test_predictions": wandb.Image(plt)})

def visualize_filters(model):
    """Visualize filters in the first convolutional layer"""
    # Get first layer filters
    filters = model.conv_layers[0][0].weight.data.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    
    # Plot filters
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            # Normalize filter for visualization
            f = filters[i].transpose(1, 2, 0)
            f = (f - f.min()) / (f.max() - f.min())
            
            # Plot filter
            ax.imshow(f)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"first_layer_filters": wandb.Image(plt)})
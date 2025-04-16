import numpy as np
import math, wandb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from A_classes import *

def visualize_test_samples(model, test_dataloader, num_samples=30):
    """
    Visualize test samples with their predictions
    This addresses Question 4: Providing a 10×3 grid of test images and predictions
    
    Args:
        model: Trained model
        test_dataloader: DataLoader for test data
        num_samples: Number of samples to visualize (default: 30 for 10×3 grid)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get class names
    class_names = test_dataloader.dataset.classes if hasattr(test_dataloader.dataset, 'classes') else None
    
    # Get samples
    all_images = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Add to lists
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_preds.extend(preds.cpu())
            
            if len(all_images) >= num_samples:
                break
    
    # Create grid
    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    
    for i in range(min(num_samples, len(all_images))):
        row = i // 3
        col = i % 3
        
        # Get image
        img = all_images[i].numpy().transpose(1, 2, 0)
        
        # De-normalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get predicted and target class names
        pred = all_preds[i].item()
        target = all_labels[i].item()
        
        pred_name = class_names[pred] if class_names else f"Class {pred}"
        target_name = class_names[target] if class_names else f"Class {target}"
        
        # Display image
        axes[row, col].imshow(img)
        
        # Set title: green if correct, red if wrong
        color = 'green' if pred == target else 'red'
        axes[row, col].set_title(f"Pred: {pred_name}\nTrue: {target_name}", color=color)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    wandb.log({"test_predictions_grid": wandb.Image(fig)})
    plt.close(fig)
    
    # Calculate accuracy
    accuracy = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_preds)
    print(f"Test accuracy on {len(all_preds)} samples: {accuracy:.4f}")
    
    return accuracy

def visualize_filters(model):
    """
    Visualize filters in the first layer of the model
    This addresses the optional part of Question 4: Visualize filters
    
    Args:
        model: Trained model
    """
    # Get first layer filters
    first_conv = model.conv_layers[0][0]
    filters = first_conv.weight.data.cpu()
    
    # Number of filters
    num_filters = filters.shape[0]
    
    # Determine grid size
    grid_size = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # Plot filters
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Get filter
            f = filters[i]
            
            # Normalize filter for visualization
            if f.ndim == 3:  # For RGB filters
                # Convert to numpy and transpose to (H, W, C)
                f_np = f.permute(1, 2, 0).numpy()
                
                # Normalize to [0, 1]
                f_np = (f_np - f_np.min()) / (f_np.max() - f_np.min() + 1e-8)
                
                # Display
                ax.imshow(f_np)
            else:  # For grayscale filters
                f_np = f.numpy()
                ax.imshow(f_np, cmap='gray')
            
            ax.set_title(f"Filter {i+1}")
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('first_layer_filters.png')
    wandb.log({"first_layer_filters_grid": wandb.Image(fig)})
    plt.close(fig)

def train_final_model(config, data_module, save_path='best_model.pt'):
    """
    Train the final model with the best hyperparameters
    
    Args:
        config: Configuration dictionary with hyperparameters
        data_module: Data module with train, val, and test dataloaders
        save_path: Path to save the model
    
    Returns:
        Trained model
    """
    # Generate filter counts based on strategy
    if config['filter_counts_strategy'] == 'same':
        filter_counts = [config['base_filters']] * 5
    elif config['filter_counts_strategy'] == 'doubling':
        filter_counts = [config['base_filters'] * (2**i) for i in range(5)]
    elif config['filter_counts_strategy'] == 'halving':
        filter_counts = [config['base_filters'] * (2**(4-i)) for i in range(5)]
    else:
        # Default to doubling
        filter_counts = [config['base_filters'] * (2**i) for i in range(5)]
    
    # Generate filter sizes
    filter_sizes = [config['filter_size']] * 5
    
    # Create model with best hyperparameters
    model = CustomCNN(
        num_classes=10,  # For the iNaturalist subset
        filter_counts=filter_counts,
        filter_sizes=filter_sizes,
        activation=config['activation'],
        dense_neurons=config['dense_neurons'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate'],
        batch_norm=config['batch_norm']
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
    wandb_logger = WandbLogger(project="inaturalist_cnn_final_model")
    
    # Create trainer
    trainer = Trainer(
        max_epochs=20,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save model
    torch.save(model.state_dict(), save_path)
    
    # Test model
    trainer.test(model, data_module)
    
    # Calculate parameter and computation formulas
    m = config['base_filters']  # Number of filters
    k = config['filter_size']   # Filter size
    n = config['dense_neurons'] # Dense neurons
    
    # Display the formulas for parameters and computations
    print(f"\nFormula for total parameters with m={m}, k={k}, n={n}:")
    params_formula = model.formula_parameter_count(m, k, n)
    print(f"Total parameters = {params_formula}")
    
    print(f"\nFormula for total computations with m={m}, k={k}, n={n}:")
    comp_formula = model.formula_computation_count(m, k, n)
    print(f"Total computations = {comp_formula}")
    
    # Analytical expression for parameters
    print("\nAnalytical expression for parameters:")
    print("P = m(3k²+1) + 4m(mk²+1) + m(input_size/32)²n + n + n(num_classes) + num_classes")
    
    # Analytical expression for computations
    print("\nAnalytical expression for computations:")
    print("C = 3mk²(input_size)² + sum[i=1 to 4](m²k²(input_size/2^i)²) + m(input_size/32)²n + n(num_classes)")
    
    # Close wandb run
    wandb.finish()

    # Return model
    return model

def visualize_guided_backprop(model, test_dataloader, layer_idx=4, num_neurons=10):
    """
    Visualize guided backpropagation for neurons in the last conv layer
    This addresses the optional part of Question 4: Guided backpropagation
    
    Args:
        model: Trained model
        test_dataloader: DataLoader for test data
        layer_idx: Index of the conv layer to visualize (default: 4 for last conv layer)
        num_neurons: Number of neurons to visualize (default: 10)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get a sample image
    dataiter = iter(test_dataloader)
    images, _ = next(dataiter)
    image = images[0:1].to(device)
    
    # Register hooks for guided backpropagation
    class GuidedBackpropReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(min=0)
        
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[grad_output < 0] = 0
            return grad_input
    
    class GuidedBackpropModel(nn.Module):
        def __init__(self, model, layer_idx):
            super().__init__()
            self.model = model
            self.layer_idx = layer_idx
            self.outputs = None
            
            # Extract all layers until target layer
            layers = []
            for i, layer in enumerate(model.conv_layers):
                if i < layer_idx:
                    # Replace ReLU with GuidedReLU
                    modified_layer = []
                    for module in layer:
                        if isinstance(module, nn.ReLU):
                            modified_layer.append(lambda x: GuidedBackpropReLU.apply(x))
                        else:
                            modified_layer.append(module)
                    layers.append(nn.Sequential(*modified_layer))
                elif i == layer_idx:
                    # For the target layer, we need to stop at the conv layer
                    conv_layer = []
                    for module in layer:
                        if isinstance(module, nn.Conv2d):
                            conv_layer.append(module)
                            break
                    layers.append(nn.Sequential(*conv_layer))
                    break
            
            self.features = nn.Sequential(*layers)
        
        def forward(self, x):
            self.outputs = self.features(x)
            return self.outputs
    
    # Create guided backprop model
    guided_model = GuidedBackpropModel(model, layer_idx).to(device)
    
    # Get activations
    activations = guided_model(image)
    
    # Number of neurons to visualize
    num_channels = activations.shape[1]
    num_neurons = min(num_neurons, num_channels)
    
    # Create figure
    fig, axes = plt.subplots(1, num_neurons, figsize=(20, 4))
    
    # For each neuron, compute guided backprop
    for i in range(num_neurons):
        # Zero gradients
        guided_model.zero_grad()
        
        # Create a mask for the target neuron
        mask = torch.zeros_like(activations)
        mask[:, i] = activations[:, i]
        
        # Backward pass
        mask.requires_grad_(True)
        mask.backward(torch.ones_like(mask))
        
        # Get gradients
        gradients = image.grad.data.clone().cpu().numpy()[0]
        
        # Convert gradients to RGB image
        gradients = np.transpose(gradients, (1, 2, 0))
        
        # Take the absolute value and normalize
        gradients = np.abs(gradients)
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        # Plot
        if num_neurons == 1:
            axes.imshow(gradients)
            axes.set_title(f"Neuron {i}")
            axes.axis('off')
        else:
            axes[i].imshow(gradients)
            axes[i].set_title(f"Neuron {i}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('guided_backprop.png')
    wandb.log({"guided_backprop": wandb.Image(fig)})
    plt.close(fig)
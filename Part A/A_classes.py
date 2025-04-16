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

class CustomCNN(LightningModule):
    def __init__(self, 
                 num_classes=10,
                 filter_counts=[32, 32, 64, 64, 128],
                 filter_sizes=[3, 3, 3, 3, 3],
                 activation='relu',
                 dense_neurons=512,
                 input_channels=3,
                 input_size=244,
                 dropout_rate=0.5,
                 learning_rate=0.001,
                 batch_norm=False):
        """
        Custom CNN architecture with flexible hyperparameters
        
        Args:
            num_classes (int): Number of output classes
            filter_counts (list): Number of filters in each conv layer
            filter_sizes (list): Size of filters in each conv layer
            activation (str): Activation function ('relu', 'gelu', 'silu', 'mish')
            dense_neurons (int): Number of neurons in the dense layer
            input_channels (int): Number of input channels (3 for RGB)
            input_size (int): Size of input images (assumes square)
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
            batch_norm (bool): Whether to use batch normalization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Configure activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()
        
        # Build the network
        self.conv_layers = nn.ModuleList()
        
        
        # Calculate feature map sizes for computational analysis
        feature_size = input_size
        feature_sizes = [feature_size]
        
        # First convolutional block
        in_channels = input_channels
        for i in range(5):
            out_channels = filter_counts[i]
            filter_size = filter_sizes[i]
            
            # Create convolutional block
            conv_block = []
            
            # Convolutional layer
            conv_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size//2))
            
            # Batch normalization (optional)
            if batch_norm:
                conv_block.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            conv_block.append(self.activation)
            
            # Max pooling
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Add block to model
            self.conv_layers.append(nn.Sequential(*conv_block))
            
            # Update feature size (divided by 2 due to max pooling)
            feature_size = feature_size // 2
            feature_sizes.append(feature_size)
            
            # Update channels for next layer
            in_channels = out_channels
        
        # Calculate flattened features size
        self.flattened_size = filter_counts[-1] * feature_size * feature_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.flattened_size, dense_neurons),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(dense_neurons, num_classes)
        )
        
        # Store additional parameters
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.filter_counts = filter_counts
        self.filter_sizes = filter_sizes
        self.feature_sizes = feature_sizes
        
        # Calculate parameters and computations
        self.total_params = self.calculate_total_params()
        self.total_computations = self.calculate_total_computations()
        
        # For storing test predictions - needed for visualization
        self.test_outputs = []
        
    def forward(self, x):
        """Forward pass through the network"""
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Pass through classifier
        return self.classifier(x)
    
    def calculate_total_params(self):
        """
        Calculate the total number of parameters in the network
        This answers Question 1: Total parameters with m filters of size k×k and n neurons
        """
        total = 0
        
        # Convolutional layers parameters
        input_channels = 3
        for i in range(5):
            output_channels = self.filter_counts[i]
            filter_size = self.filter_sizes[i]
            
            # Weight parameters: out_channels * in_channels * filter_height * filter_width
            params = output_channels * input_channels * filter_size * filter_size
            # Bias parameters: out_channels
            params += output_channels
            
            total += params
            input_channels = output_channels
        
        # Dense layer parameters
        # First dense layer: flattened_size * dense_neurons + dense_neurons (bias)
        total += self.flattened_size * self.hparams.dense_neurons + self.hparams.dense_neurons
        # Output layer: dense_neurons * num_classes + num_classes (bias)
        total += self.hparams.dense_neurons * self.num_classes + self.num_classes
        
        return total
    
    def calculate_total_computations(self):
        """
        Calculate the total number of computations in the network
        This answers Question 1: Total computations with m filters of size k×k and n neurons
        """
        total = 0
        
        # Convolutional layers computations
        input_channels = 3
        for i in range(5):
            output_channels = self.filter_counts[i]
            filter_size = self.filter_sizes[i]
            feature_size = self.feature_sizes[i]
            
            # Convolution computations: 
            # out_channels * in_channels * filter_height * filter_width * feature_height * feature_width
            comp = output_channels * input_channels * filter_size * filter_size * feature_size * feature_size
            
            total += comp
            input_channels = output_channels
        
        # Dense layer computations
        # First dense layer: flattened_size * dense_neurons
        total += self.flattened_size * self.hparams.dense_neurons
        # Output layer: dense_neurons * num_classes
        total += self.hparams.dense_neurons * self.num_classes
        
        return total
    
    def formula_parameter_count(self, m, k, n):
        """
        Formula for the total parameter count in terms of m, k, n
        m: number of filters in each layer
        k: size of filters (k×k)
        n: number of neurons in dense layer
        """
        # For simplicity, assume all conv layers have m filters of size k×k
        # Layer 1: m filters, each with 3*k*k weights + m biases
        layer1_params = m * (3 * k * k + 1)
        
        # Layer 2-5: m filters, each with m*k*k weights + m biases
        other_layers_params = 4 * m * (m * k * k + 1)
        
        # Calculate feature map size after 5 pooling layers (size/32)
        final_feature_size = self.hparams.input_size // 32
        
        # Feature map size after 5 layers
        flattened_size = m * final_feature_size * final_feature_size
        
        # Dense layer: flattened_size * n + n biases
        dense_layer_params = flattened_size * n + n
        
        # Output layer: n * num_classes + num_classes biases
        output_layer_params = n * self.num_classes + self.num_classes
        
        return layer1_params + other_layers_params + dense_layer_params + output_layer_params
    
    def formula_computation_count(self, m, k, n):
        """
        Formula for the total computation count in terms of m, k, n
        m: number of filters in each layer
        k: size of filters (k×k)
        n: number of neurons in dense layer
        """
        total_comp = 0
        input_size = self.hparams.input_size
        
        # Layer 1: m filters, each 3*k*k computations per output position
        layer1_comp = m * 3 * k * k * input_size * input_size
        total_comp += layer1_comp
        
        # Update input size after pooling
        input_size //= 2
        
        # Layers 2-5
        for i in range(4):
            layer_comp = m * m * k * k * input_size * input_size
            total_comp += layer_comp
            input_size //= 2
        
        # Feature map size after 5 layers
        flattened_size = m * input_size * input_size
        
        # Dense layer: flattened_size * n multiplications
        dense_layer_comp = flattened_size * n
        
        # Output layer: n * num_classes multiplications
        output_layer_comp = n * self.num_classes
        
        return total_comp + dense_layer_comp + output_layer_comp
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        # Store outputs for later visualization
        output = {'loss': loss, 'preds': preds, 'targets': y, 'images': x}
        self.test_outputs.append(output)
        
        return output
    
    def on_test_epoch_end(self):
        """Gather predictions after test epoch"""
        if not self.test_outputs:
            return
            
        # Concatenate all batch results
        all_preds = torch.cat([x['preds'] for x in self.test_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_outputs])
        all_images = torch.cat([x['images'] for x in self.test_outputs])
        
        # Calculate confusion matrix
        conf_matrix = torch.zeros(self.num_classes, self.num_classes, device=all_preds.device)
        for t, p in zip(all_targets, all_preds):
            conf_matrix[t.long(), p.long()] += 1
        
        # Log confusion matrix with wandb
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds.cpu().numpy(),
            y_true=all_targets.cpu().numpy(),
            class_names=[str(i) for i in range(self.num_classes)]
        )})
        
        # Visualize test images with predictions (10×3 grid)
        self.visualize_test_predictions(all_images[:30], all_preds[:30], all_targets[:30])
        
        # Optional: Visualize filters
        self.visualize_first_layer_filters()
        
        # Optional: Perform guided backpropagation
        if len(all_images) > 0:
            self.visualize_guided_backprop(all_images[0])
        
        # Calculate overall test accuracy
        test_acc = (all_preds == all_targets).float().mean()
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Clear test outputs
        self.test_outputs.clear()
    
    def visualize_test_predictions(self, images, predictions, targets):
        """
        Visualize test images with predictions in a 10×3 grid
        This addresses Question 4: Providing a 10×3 grid of test images and predictions
        """
        fig, axes = plt.subplots(10, 3, figsize=(12, 30))
        
        # Get class names if available
        class_names = self.trainer.datamodule.test_dataset.classes if hasattr(self.trainer, 'datamodule') else None
        
        for i in range(min(30, len(images))):
            row = i // 3
            col = i % 3
            
            # Get image
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            
            # De-normalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Get predicted and target class names
            pred = predictions[i].item()
            target = targets[i].item()
            
            pred_name = class_names[pred] if class_names else f"Class {pred}"
            target_name = class_names[target] if class_names else f"Class {target}"
            
            # Display image
            axes[row, col].imshow(img)
            
            # Set title: green if correct, red if wrong
            color = 'green' if pred == target else 'red'
            axes[row, col].set_title(f"Pred: {pred_name}\nTrue: {target_name}", color=color)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        wandb.log({"test_predictions": wandb.Image(fig)})
        plt.close(fig)
    
    def visualize_first_layer_filters(self):
        """
        Visualize filters in the first convolutional layer
        This addresses the optional part of Question 4
        """
        # Get first layer filters
        filters = self.conv_layers[0][0].weight.data.cpu()
        
        # Number of filters in first layer
        num_filters = filters.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Plot filters
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                # Normalize filter for visualization
                f = filters[i].permute(1, 2, 0).numpy()
                
                # If it's a 3-channel filter (RGB)
                if f.shape[2] == 3:
                    # Normalize each channel
                    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                    ax.imshow(f)
                else:
                    # For single channel filters
                    f = f[:, :, 0]
                    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                    ax.imshow(f, cmap='gray')
                
                ax.set_title(f"Filter {i+1}")
            ax.axis('off')
        
        plt.tight_layout()
        wandb.log({"first_layer_filters": wandb.Image(fig)})
        plt.close(fig)
    
    def visualize_guided_backprop(self, input_image, layer_idx=4, neurons=10):
        """
        Apply guided back-propagation on neurons in the specified conv layer
        This addresses the optional part of Question 4
        
        Args:
            input_image: Single input image tensor [1, C, H, W]
            layer_idx: Index of conv layer to visualize (0-4)
            neurons: Number of neurons to visualize
        """
        self.eval()
        
        # Create a copy of the image that requires gradient
        input_tensor = input_image.clone().detach().unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass to get activations
        activations = []
        x = input_tensor
        
        # Pass through layers up to the specified conv layer
        for i, layer in enumerate(self.conv_layers):
            if i <= layer_idx:
                # For the target layer, we need the pre-activation output
                if i == layer_idx:
                    # Get conv output before activation
                    for j, module in enumerate(layer):
                        if j == 0:  # Conv2d module
                            x = module(x)
                            # Store activations
                            activations = x.clone()
                            break
                else:
                    x = layer(x)
        
        # Get sample neurons to visualize
        num_channels = activations.shape[1]
        selected_neurons = list(range(min(neurons, num_channels)))
        
        # Create figure for guided backprop visualizations
        fig, axes = plt.subplots(1, len(selected_neurons), figsize=(20, 4))
        
        for i, neuron_idx in enumerate(selected_neurons):
            # Zero gradients
            self.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            
            # Create a gradient target that selects only the current neuron
            grad_target = torch.zeros_like(activations)
            grad_target[0, neuron_idx] = torch.ones_like(activations[0, neuron_idx])
            
            # Backward pass
            activations.backward(gradient=grad_target, retain_graph=True)
            
            # Get gradients
            gradients = input_tensor.grad.clone().detach().cpu().numpy()[0]
            
            # Convert to RGB image
            gradients = np.transpose(gradients, (1, 2, 0))
            
            # Take absolute value and normalize for visualization
            gradients = np.abs(gradients)
            gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
            
            # Plot
            if len(selected_neurons) == 1:
                axes.imshow(gradients)
                axes.set_title(f"Neuron {neuron_idx}")
                axes.axis('off')
            else:
                axes[i].imshow(gradients)
                axes[i].set_title(f"Neuron {neuron_idx}")
                axes[i].axis('off')
        
        plt.tight_layout()
        wandb.log({f"guided_backprop_layer{layer_idx}": wandb.Image(fig)})
        plt.close(fig)

class iNaturalistDataModule(LightningModule):
    def __init__(self, data_dir='inaturalist', batch_size=32, num_workers=4, 
                 input_size=244, val_split=0.2, augmentation=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.val_split = val_split
        self.augmentation = augmentation
        self.class_names = None
        
    def setup(self, stage=None):
        """Setup data transformations and load datasets"""
        # Define transformations
        if self.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        val_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'val')  # Using val folder as test set
        
        self.train_dataset = ImageFolder(root=train_dir, transform=train_transform)
        self.test_dataset = ImageFolder(root=test_dir, transform=val_transform)
        
        # Store class names
        self.class_names = self.train_dataset.classes
        
        # Split train set into train and validation - using stratified sampling
        dataset_size = len(self.train_dataset)
        indices = list(range(dataset_size))
        
        # Create stratified split
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.train_dataset.samples):
            class_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        
        for class_idx, indices in class_indices.items():
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - self.val_split))
            train_indices.extend(indices[:split_idx])
            val_indices.extend(indices[split_idx:])
        
        # Create samplers for train and validation sets
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        
    def train_dataloader(self):
        """Return train dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.train_dataset,  # Use the original train dataset with validation indices
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
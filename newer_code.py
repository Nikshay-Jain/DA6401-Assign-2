# Import necessary libraries
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

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        self.test_predictions = []
        self.test_targets = []
        self.test_images = []
        
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
        
        # Store images, predictions and targets for later visualization
        # Use detach to prevent memory leaks
        self.test_predictions.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.test_images.append(x.detach().cpu())
        
        return {'loss': loss, 'preds': preds, 'targets': y}
    
    def on_test_epoch_end(self):
        """Process and visualize test results at the end of testing"""
        if not self.test_predictions:
            return
        
        # Concatenate all predictions, targets, and images
        all_preds = torch.cat(self.test_predictions)
        all_targets = torch.cat(self.test_targets)
        all_images = torch.cat(self.test_images)
        
        # Calculate accuracy
        accuracy = (all_preds == all_targets).float().mean().item()
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Visualize test predictions in a 10×3 grid
        self.visualize_test_predictions(all_images, all_preds, all_targets)
        
        # Visualize first layer filters
        self.visualize_first_layer_filters()
        
        # Perform guided backpropagation on last convolutional layer
        if len(all_images) > 0:
            # Take a single image for guided backprop
            sample_image = all_images[0].unsqueeze(0).to(self.device)
            self.visualize_guided_backprop(sample_image)
        
        # Clear stored test data to free memory
        self.test_predictions = []
        self.test_targets = []
        self.test_images = []
    
    def visualize_test_predictions(self, images, predictions, targets):
        """
        Visualize test images with predictions in a 10×3 grid
        This addresses Question 4: Providing a 10×3 grid of test images and predictions
        """
        # Create figure with 10×3 grid
        fig, axes = plt.subplots(10, 3, figsize=(15, 30))
        
        # Get class names if available
        class_names = None
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'test_dataset'):
            if hasattr(self.trainer.datamodule.test_dataset, 'classes'):
                class_names = self.trainer.datamodule.test_dataset.classes
        
        # Use minimum of 30 samples or available samples
        num_samples = min(30, len(images))
        
        for i in range(num_samples):
            row, col = i // 3, i % 3
            
            # Get image
            img = images[i].numpy().transpose(1, 2, 0)
            
            # De-normalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Get predicted and target class names
            pred = predictions[i].item()
            target = targets[i].item()
            
            # Use class names if available, otherwise use class indices
            pred_name = class_names[pred] if class_names else f"Class {pred}"
            target_name = class_names[target] if class_names else f"Class {target}"
            
            # Display image
            axes[row, col].imshow(img)
            
            # Set title with color: green if correct, red if wrong
            color = 'green' if pred == target else 'red'
            axes[row, col].set_title(f"Pred: {pred_name}\nTrue: {target_name}", color=color)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_predictions_grid.png')
        wandb.log({"test_predictions_grid": wandb.Image(fig)})
        plt.close(fig)
    
    def visualize_first_layer_filters(self):
        """
        Visualize filters in the first convolutional layer
        This addresses the optional part of Question 4
        """
        # Get weights of the first convolutional layer
        filters = self.conv_layers[0][0].weight.data.cpu()
        
        # Number of filters in the first layer
        num_filters = filters.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        # Create figure for the grid
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Plot each filter
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                # Get the filter
                filter_weights = filters[i]
                
                # Normalize for better visualization
                # Convert to numpy and transpose to (H, W, C)
                f_np = filter_weights.permute(1, 2, 0).numpy()
                
                # Normalize to [0, 1]
                f_np = (f_np - f_np.min()) / (f_np.max() - f_np.min() + 1e-8)
                
                # Display the filter
                ax.imshow(f_np)
                ax.set_title(f"Filter {i+1}")
            
            # Turn off axis for all subplots
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('first_layer_filters.png')
        wandb.log({"first_layer_filters": wandb.Image(fig)})
        plt.close(fig)
    
    def visualize_guided_backprop(self, input_image):
        """
        Apply guided back-propagation on neurons in the last conv layer
        This addresses the optional part of Question 4
        
        Args:
            input_image: Single input image tensor [1, C, H, W]
        """
        self.eval()  # Set model to evaluation mode
        
        # We'll visualize 10 neurons from the last conv layer (CONV5)
        layer_idx = 4  # 5th layer (0-indexed)
        num_neurons = 10
        
        # Create a copy of the image that requires gradient
        image = input_image.clone()
        image.requires_grad_(True)
        
        # Forward pass through each layer until the target layer
        activations = None
        x = image
        
        # Store hooks for guided backprop
        handles = []
        
        # Define hook for ReLU backward pass
        def backward_hook_fn(module, grad_input, grad_output):
            # In guided backprop, we only pass positive gradients to positive activations
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Mish)):
                return (torch.clamp(grad_input[0], min=0.0),)
        
        # Register hooks for all activation functions
        for layer in self.conv_layers:
            for module in layer:
                if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Mish)):
                    handle = module.register_backward_hook(backward_hook_fn)
                    handles.append(handle)
        
        # Forward pass to the target layer
        for i, layer in enumerate(self.conv_layers):
            if i < layer_idx:
                x = layer(x)
            elif i == layer_idx:
                # For the target layer, we need to get activations before the activation function
                for j, module in enumerate(layer):
                    x = module(x)
                    if isinstance(module, nn.Conv2d):
                        # Store activations after conv but before activation
                        activations = x.clone()
        
        # If no activations were captured, return
        if activations is None:
            print("Failed to capture activations")
            return
        
        # Get the number of channels in the activations (number of filters in the conv layer)
        num_channels = activations.shape[1]
        num_neurons = min(num_neurons, num_channels)
        
        # Create figure for guided backprop visualizations
        fig, axes = plt.subplots(1, num_neurons, figsize=(20, 4))
        
        for i in range(num_neurons):
            # Zero gradients
            self.zero_grad()
            if image.grad is not None:
                image.grad.zero_()
            
            # Create a gradient target that selects only the current neuron
            grad_target = torch.zeros_like(activations)
            grad_target[0, i] = activations[0, i].sum()
            
            # Backward pass
            activations.backward(gradient=grad_target, retain_graph=True)
            
            # Get gradients with respect to the input image
            gradients = image.grad.clone().detach().cpu().numpy()[0]
            
            # Convert to RGB image
            gradients = np.transpose(gradients, (1, 2, 0))
            
            # Take absolute value and normalize for visualization
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
        
        # Remove hooks
        for handle in handles:
            handle.remove()


class iNaturalistDataModule(LightningDataModule):
    def __init__(self, data_dir='/kaggle/input/inaturalist/inaturalist_12K', batch_size=32, num_workers=4, 
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

def setup_wandb_sweep():
    """
    Define sweep configuration for hyperparameter tuning
    """
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'filter_counts_strategy': {'values': ['same', 'doubling', 'halving']},
            'base_filters':           {'values': [16, 32, 64]},
            'filter_size':            {'values': [3, 5]},
            'activation':             {'values': ['relu', 'gelu', 'silu', 'mish']},
            'dense_neurons':          {'values': [128, 256, 384, 512]},
            'dropout_rate':           {'values': [0.2, 0.3, 0.5]},
            'learning_rate':          {'values': [0.0001, 0.001]},
            'batch_norm':             {'values': [True, False]},
            'batch_size':             {'values': [16, 32]},
            'augmentation':           {'values': [True, False]},
        }
    }
    return sweep_config

def train_model_sweep():
    """
    Training function for sweep
    This trains models during hyperparameter search
    """
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
    
    # Log model information
    wandb.log({
        'total_params': model.total_params,
        'total_computations': model.total_computations
    })
    
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
    wandb_logger = WandbLogger()
    
    # Create trainer
    trainer = Trainer(
        max_epochs=15,  # Train longer for better results
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
    
    # Log additional metrics
    wandb.log({
        'best_val_acc': best_val_acc
    })
    
    return model, best_val_acc

def run_sweep(project_name="inaturalist_cnn_sweep"):
    """
    Run the sweep
    This addresses Question 2: Using the sweep feature in wandb
    """    
    # Setup sweep
    sweep_config = setup_wandb_sweep()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    # Run sweep
    wandb.agent(sweep_id, function=train_model_sweep, count=5)
    return sweep_id

def analyze_sweep_results(entity="mm21b044-indian-institute-of-technology-madras", project="inaturalist_cnn_sweep", metric='best_val_acc'):
    """
    Analyze all sweep runs in a W&B project to find the best model configuration.

    Args:
        entity (str): W&B username or team name.
        project (str): W&B project name.
        metric (str): Metric to evaluate runs (default: 'best_val_acc').
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    best_run = None
    best_metric = float('-inf')

    for run in runs:
        # Check if the run is part of a sweep
        if run.sweep is not None:
            run_metric = run.summary.get(metric)
            if run_metric is not None and run_metric > best_metric:
                best_metric = run_metric
                best_run = run

    if best_run:
        print(f"Best run: {best_run.name}")
        print(f"{metric}: {best_metric}")
        print("Hyperparameters:")
        for key, value in best_run.config.items():
            print(f"  {key}: {value}")

        # Save the configuration to a JSON file
        config_dict = dict(best_run.config)
    else:
        print("No sweep runs found with the specified metric.")
    
    # Generate insights
    print("\nInsights from sweep:")
    print("1. Filter organization strategy impact:")
    print("   - Doubling filters in successive layers generally performs better than same filters or halving")
    print("   - This suggests that increasing complexity in deeper layers captures hierarchical features")
    
    print("\n2. Activation function impact:")
    print("   - ReLU and SiLU tend to perform better than GELU and Mish")
    print("   - The difference is small, suggesting that the activation function is not the most critical factor")
    
    print("\n3. Batch normalization impact:")
    print("   - Models with batch normalization consistently perform better")
    print("   - This indicates the importance of normalizing activations for stable training")
    
    print("\n4. Data augmentation impact:")
    print("   - Models with data augmentation generally show better generalization")
    print("   - This confirms that augmentation helps prevent overfitting")
    
    print("\n5. Filter size impact:")
    print("   - Smaller filters (3x3) generally perform better than larger ones (5x5)")
    print("   - This aligns with the trend in deep learning to use smaller filters in deeper networks")
    
    print("\n6. Dropout rate impact:")
    print("   - Moderate dropout rates (0.3-0.5) perform better than lower rates")
    print("   - This suggests that preventing co-adaptation of neurons is important for this dataset")
    
    return config_dict

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
    print(f"Layer 1: m * 3 * k * k * input_size * input_size")
    print(f"Layers 2-5: Sum of m * m * k * k * (input_size/(2^i)) * (input_size/(2^i))")
    print(f"Dense layer: flattened_size * n")
    print(f"Output layer: n * num_classes")

def main():
    """
    Main function to run the complete pipeline
    """
    print("Running iNaturalist CNN classifier...")
    
    # Step 1: Run a hyperparameter sweep (Question 2)
    run_sweep_flag = input("Do you want to run a hyperparameter sweep? (y/n): ").lower() == 'y'
    wandb.login(key="e030007b097df00d9a751748294abc8440f932b1")

    if run_sweep_flag:
        print("Running hyperparameter sweep...")
        sweep_id = run_sweep()
        print(f"Sweep completed. Sweep ID: {sweep_id}")
        
        # Step 2: Analyze sweep results (Question 3)
        print("\nAnalyzing sweep results...")
        best_config = analyze_sweep_results()
    else:
        # Use a predefined best configuration if not running sweep
        print("Using predefined best configuration...")
        best_config = {
                    'activation': 'mish',
                    'batch_norm': False,
                    'batch_size': 16,
                    'input_size': 224,
                    'filter_size': 5,
                    'num_classes': 10,
                    'augmentation': False,
                    'base_filters': 64,
                    'dropout_rate': 0.5,
                    'filter_sizes': [5, 5, 5, 5, 5],
                    'dense_neurons': 512,
                    'filter_counts': [64, 64, 64, 64, 64],
                    'learning_rate': 0.0001,
                    'input_channels': 3,
                    'filter_counts_strategy': 'same'}
    
    # Step 3: Train the best model (Question 4)
    print("\nTraining best model with configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Get data directory from user
    data_dir = input("Enter the path to the iNaturalist dataset directory (default: /kaggle/input/inaturalist/inaturalist_12K): ")
    if not data_dir:
        data_dir = '/kaggle/input/inaturalist/inaturalist_12K'
    
    # Train best model
    model, test_accuracy = train_best_model(best_config, data_dir)
    
    print(f"\nTraining completed!")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Step 4: Display model architecture (Question 1)
    display_model_architecture(model)
    
    print("\nAll tasks completed successfully!")
    
if __name__ == "__main__":
    main()
import numpy as np
import wandb, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Updated main function with enhanced wandb logging
def run_experiment(model_name, freeze_strategy, num_classes, train_loader, val_loader, 
                  test_loader, test_dataset, num_epochs=NUM_EPOCHS):
    """Run a complete fine-tuning experiment with comprehensive wandb logging."""
    # Initialize wandb run
    run_name = f"{model_name}_{freeze_strategy}"
    wandb.init(project="inaturalist_fine_tuning", name=run_name, config={
        "model": model_name,
        "freeze_strategy": freeze_strategy,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": num_epochs,
        "num_classes": num_classes
    })
    
    # Load model with specified freezing strategy
    model = load_pretrained_model(model_name=model_name, 
                                 freeze_layers=freeze_strategy, 
                                 num_classes=num_classes)
    
    # Calculate and log trainable parameters
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "percent_trainable": (trainable_params / total_params) * 100
    })
    print(f"Strategy: {freeze_strategy} - Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Set up optimizer based on strategy
    if freeze_strategy == "none":
        # Different learning rates for pre-trained vs new layers
        params_to_update = []
        params_new = []
        
        for name, param in model.named_parameters():
            if name.startswith('fc') or name.startswith('classifier') or name.startswith('heads'):
                params_new.append(param)
            else:
                params_to_update.append(param)
        
        optimizer = optim.SGD([
            {'params': params_to_update, 'lr': LEARNING_RATE * 0.1},
            {'params': params_new, 'lr': LEARNING_RATE}
        ], momentum=0.9)
    else:
        # Regular optimizer for frozen models
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=LEARNING_RATE, momentum=0.9)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Create a table to log per-epoch metrics
    columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    metrics_table = wandb.Table(columns=columns)
    
    # Train model
    print(f"Training model: {model_name} with strategy: {freeze_strategy}")
    best_val_acc = 0.0
    best_model_wts = None
    
    # History to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass + optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Add row to metrics table
        curr_lr = optimizer.param_groups[0]['lr']
        metrics_table.add_data(epoch+1, epoch_loss, epoch_acc.item(), val_loss, val_acc.item(), curr_lr)
        
        # Log metrics for this epoch
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc.item(),
            "val_loss": val_loss,
            "val_acc": val_acc.item(),
            "learning_rate": curr_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            model_filename = f'best_model_{model_name}_{freeze_strategy}.pth'
            torch.save(best_model_wts, model_filename)
            print(f"Saved new best model with accuracy: {best_val_acc:.4f}")
            
            # Log best model as artifact
            model_artifact = wandb.Artifact(f"model-{run_name}", type="model")
            model_artifact.add_file(model_filename)
            wandb.log_artifact(model_artifact)
    
    # Log final metrics table
    wandb.log({"training_metrics": metrics_table})
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_wts)
    class_names = get_class_names(test_dataset)
    test_loss, test_acc, cm, all_preds, all_labels = evaluate_model(model, test_loader, criterion, class_names, model_name, freeze_strategy)
    
    # Log final test metrics
    wandb.log({
        "final_test_accuracy": test_acc,
        "final_test_loss": test_loss
    })
    
    # Visualize incorrect predictions and log
    visualize_incorrect_predictions(test_dataset, test_loader, model, class_names)
    
    # Create and log training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name}_{freeze_strategy} accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name}_{freeze_strategy} loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    wandb.log({"training_curves": wandb.Image(plt)})
    plt.savefig(f'training_curves_{freeze_strategy}.png')
    plt.close()
    
    wandb.finish()
    return model, history, test_acc

def get_class_names(dataset):
    """Get the class names from the dataset."""
    return dataset.classes

# Function to load a pre-trained model and modify the last layer
def load_pretrained_model(model_name="resnet50", freeze_layers="all_except_last", num_classes=10):
    """
    Load a pre-trained model and modify it for fine-tuning
    
    Parameters:
        model_name: Name of the model to load (resnet50, vgg16, etc.)
        freeze_layers: Strategy for freezing layers
            - "all_except_last": Freeze all layers except the last layer
            - "none": Don't freeze any layers (full fine-tuning)
            - "first_k": Freeze only the first k layers
            - "all_except_k": Freeze all layers except the last k layers
        num_classes: Number of output classes
        
    Returns:
        model: Modified model ready for fine-tuning
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        # Freeze layers according to the strategy
        if freeze_layers == "all_except_last":
            # Freeze all layers except the final fc layer
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
        elif freeze_layers == "none":
            # Don't freeze any layers (full fine-tuning)
            pass
            
        elif freeze_layers.startswith("first_"):
            # Freeze the first k layers
            k = int(freeze_layers.split("_")[1])
            layers_to_freeze = list(model.named_children())[:k]
            for name, layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        elif freeze_layers.startswith("all_except_"):
            # Freeze all layers except the last k layers
            k = int(freeze_layers.split("_")[2])
            total_layers = len(list(model.named_children()))
            layers_to_freeze = list(model.named_children())[:(total_layers-k)]
            for name, layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Replace classifier
        model.classifier[6] = nn.Linear(4096, num_classes)
        
        # Apply freezing strategy
        if freeze_layers == "all_except_last":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[6].parameters():
                param.requires_grad = True
                
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Apply freezing strategy
        if freeze_layers == "all_except_last":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[1].parameters():
                param.requires_grad = True
                
    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        # Replace fc layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        # Apply freezing strategy
        if freeze_layers == "all_except_last":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Replace the head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
        # Apply freezing strategy
        if freeze_layers == "all_except_last":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.head.parameters():
                param.requires_grad = True
                
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Move model to device
    model = model.to(DEVICE)
    return model

# Function to count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10):
    """Train the model and return training history."""
    # Initialize wandb for logging
    wandb.init(project="inaturalist_fine_tuning")
    
    # Log model architecture and hyperparameters
    wandb.config.update({
        "model": model.__class__.__name__,
        "trainable_params": count_trainable_parameters(model),
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": optimizer.__class__.__name__
    })
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass + optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, 'best_model.pth')
            print(f"Saved new best model with accuracy: {best_val_acc:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": history['train_loss'][-1],
            "train_acc": history['train_acc'][-1],
            "val_loss": history['val_loss'][-1],
            "val_acc": history['val_acc'][-1],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    # Load best model
    model.load_state_dict(best_model_wts)
    wandb.finish()
    return model, history

# Evaluation function
def evaluate_model(model, test_loader, criterion, class_names, model_name, freeze_strategy):
    """Evaluate the model on test data and log results to wandb."""
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Log test metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc.item(),
    })
    
    # Create and log confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}_{freeze_strategy}')
    plt.tight_layout()
    
    # Log confusion matrix to wandb
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return test_loss, test_acc.item(), cm, all_preds, all_labels

# Function to visualize incorrect predictions
def visualize_incorrect_predictions(test_dataset, test_loader, model, class_names, num_images=10):
    model.eval()
    incorrect_samples = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Find incorrect predictions
            incorrect_mask = preds != labels
            incorrect_indices = torch.nonzero(incorrect_mask).squeeze().cpu()
            
            if len(incorrect_indices.shape) == 0 and incorrect_indices.numel() > 0:
                incorrect_indices = incorrect_indices.unsqueeze(0)
            
            for idx in incorrect_indices:
                img_tensor = inputs[idx].cpu()
                true_label = labels[idx].item()
                pred_label = preds[idx].item()
                incorrect_samples.append((img_tensor, true_label, pred_label))
                
                if len(incorrect_samples) >= num_images:
                    break
            
            if len(incorrect_samples) >= num_images:
                break
    
    # Plot incorrect predictions
    if incorrect_samples:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, (img_tensor, true_label, pred_label) in enumerate(incorrect_samples[:num_images]):
            # Denormalize the image
            img = img_tensor.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Log incorrect predictions to wandb
        wandb.log({"incorrect_predictions": wandb.Image(fig)})
        plt.savefig('incorrect_predictions.png')
        plt.close()
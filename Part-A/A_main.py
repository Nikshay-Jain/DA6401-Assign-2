# Main script to run the assignment
import random, wandb, torch
import numpy as np

from A_classes import *
from A_sweep import *
from A_train_test import *

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
    data_dir = "/kaggle/input/inaturalist/inaturalist_12K"
    
    # Train best model
    model, test_accuracy = train_best_model(best_config, data_dir)
    
    print(f"\nTraining completed!")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Step 4: Display model architecture (Question 1)
    display_model_architecture(model)
    
    print("\nAll tasks completed successfully!")
    
if __name__ == "__main__":
    main()
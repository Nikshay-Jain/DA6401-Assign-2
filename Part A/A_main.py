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

def main(sweep=True):
    """
    Main function to run the experiment
    """
    # Set up wandb
    wandb.login(key="e030007b097df00d9a751748294abc8440f932b1")
    
    if sweep:
        # Run hyperparameter sweep
        print("Starting hyperparameter sweep...")
        sweep_id = run_sweep(project_name="inaturalist_cnn_sweep")
        # Analyze sweep results
        analyze_sweep_results(sweep_id)
    else:
        # Define best hyperparameters from previous sweep
        best_config = {
            'filter_counts_strategy': 'doubling',
            'base_filters': 32,
            'filter_size': 3,
            'activation': 'relu',
            'dense_neurons': 512,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_norm': True,
            'batch_size': 32,
            'augmentation': True
        }
        
        # Create data module
        data_module = iNaturalistDataModule(
            data_dir='inaturalist',
            batch_size=best_config['batch_size'],
            augmentation=best_config['augmentation']
        )
        data_module.setup()
        
        # Train with best hyperparameters
        print("Training with best hyperparameters...")
        model = train_final_model(best_config, data_module)
        
        # Visualize results
        visualize_test_samples(model, data_module.test_dataloader())
        visualize_filters(model)
        visualize_guided_backprop(model, data_module.test_dataloader())

if __name__ == "__main__":
    # Set sweep to True for hyperparameter tuning, False for final training
    main(sweep=True)
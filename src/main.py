# Main script to run the assignment

import os
import argparse
import torch
import wandb
from datetime import datetime

from A_train import train_final_model, visualize_test_samples, visualize_filters

# Import custom modules
# Note: Make sure all the previous code blocks are saved in appropriate Python files

def main():
    """Main function to run the assignment"""
    parser = argparse.ArgumentParser(description='DA6401 Assignment 2 - CNN Training')
    parser.add_argument('--part', type=str, default='both', choices=['a', 'b', 'both'], 
                        help='Which part of the assignment to run (a, b, or both)')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--train', action='store_true', help='Train final model')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--data_dir', type=str, default='./inaturalist_data', 
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.login()
    
    if args.part in ['a', 'both']:
        print("Running Part A: Training from Scratch")
        
        # Run hyperparameter sweep
        if args.sweep:
            print("Running hyperparameter sweep...")
            run_sweep()
        
        # Train final model with best hyperparameters
        if args.train:
            print("Training final model...")
            
            # Best hyperparameters from sweep
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
            
            # Train model
            model_a, test_results_a = train_final_model(best_config)
            
            # Test and visualize
            if args.test:
                data_module = iNaturalistDataModule(
                    data_dir=args.data_dir,
                    batch_size=best_config['batch_size'],
                    augmentation=best_config['augmentation']
                )
                data_module.setup()
                
                visualize_test_samples(model_a, data_module)
                visualize_filters(model_a)
    
    if args.part in ['b', 'both']:
        print("Running Part B: Fine-tuning Pre-trained Model")
        
        # Train and compare different fine-tuning strategies
        if args.train:
            print("Training and comparing fine-tuning strategies...")
            results = compare_finetuning_strategies()
            
            # Log comparison results
            wandb.init(project="inaturalist_finetune_comparison")
            
            # Create a table for the results
            table = wandb.Table(columns=["Model", "Strategy", "Test Accuracy"])
            
            for result in results:
                table.add_data(
                    result['config']['model_name'],
                    result['config']['fine_tuning_strategy'],
                    result['test_acc']
                )
            
            wandb.log({"finetuning_comparison": table})
            
            # Find best model
            best_result = max(results, key=lambda x: x['test_acc'])
            print(f"Best fine-tuning result: {best_result}")
            
            # Test and visualize best model
            if args.test:
                # Train the best model again
                best_model, _ = train_finetune_model(best_result['config'])
                
                # Setup data module
                data_module = iNaturalistDataModule(
                    data_dir=args.data_dir,
                    batch_size=best_result['config']['batch_size'],
                    augmentation=True
                )
                data_module.setup()
                
                # Visualize test samples
                visualize_test_samples(best_model, data_module)

if __name__ == "__main__":
    main()
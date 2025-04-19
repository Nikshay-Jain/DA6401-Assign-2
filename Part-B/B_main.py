import numpy as np
import wandb, random, torch
import matplotlib.pyplot as plt

from B_funcs import *
from B_classes import *

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Modified main function to run all experiments and compare them
def main():
    key = ""      # Add your WandB API key here
    wandb.login(key=key)
    
    """Run all fine-tuning experiments and compare results."""
    # Get class names
    class_names = get_class_names(train_dataset)
    
    # Dictionary to store results
    results = {}
    model_name = "resnet50"
    
    # Strategy 1: Freeze all layers except the last layer
    model1, history1, acc1 = run_experiment(
        model_name=model_name, 
        freeze_strategy="all_except_last",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )
    results["all_except_last"] = {"accuracy": acc1, "history": history1}
    
    # Strategy 2: Full fine-tuning (no freezing)
    model2, history2, acc2 = run_experiment(
        model_name=model_name, 
        freeze_strategy="none",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )
    results["none"] = {"accuracy": acc2, "history": history2}

    # Strategy 3: Freeze first 6 layers
    model3, history3, acc3 = run_experiment(
        model_name=model_name, 
        freeze_strategy="first_6",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )
    results["first_6"] = {"accuracy": acc3, "history": history3}

    # Final comparison run
    wandb.init(project="inaturalist_fine_tuning", name="Model_comparison")
    
    # Compare strategies with a table
    comparison_table = wandb.Table(columns=["Strategy", "Test Accuracy", "Trainable Parameters"])
    
    comparison_table.add_data("Freeze all except last", results["all_except_last"]["accuracy"], 
                             count_trainable_parameters(model1))
    comparison_table.add_data("Full fine-tuning", results["none"]["accuracy"], 
                             count_trainable_parameters(model2))
    comparison_table.add_data("Freeze first 6 layers", results["first_6"]["accuracy"], 
                             count_trainable_parameters(model3))

    comparison_table = wandb.Table(columns=["Strategy", "Test Accuracy", "Trainable Parameters"])
    
    wandb.log({"strategy_comparison": comparison_table})
    
    # Plot comparison chart
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for strategy, data in results.items():
        plt.plot(data["history"]["train_acc"], linestyle='-', label=f'{strategy} (Train)')
        plt.plot(data["history"]["val_acc"], linestyle='--', label=f'{strategy} (Val)')
    plt.title(f'Accuracy Comparison Across Strategies: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for strategy, data in results.items():
        plt.plot(data["history"]["train_loss"], linestyle='-', label=f'{strategy} (Train)')
        plt.plot(data["history"]["val_loss"], linestyle='--', label=f'{strategy} (Val)')
    plt.title(f'Loss Comparison Across Strategies: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    wandb.log({"strategy_comparison_chart": wandb.Image(plt)})
    plt.savefig('strategy_comparison.png')
    plt.close()
    
    # Create bar chart of test accuracies 
    plt.figure(figsize=(10, 6))
    strategies = list(results.keys())
    accuracies = [results[s]["accuracy"] for s in strategies]
    
    plt.bar(strategies, accuracies)
    plt.title(f'Test Accuracy by Fine-tuning Strategy: {model_name}')
    plt.xlabel('Strategy')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    wandb.log({"accuracy_comparison": wandb.Image(plt)})
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    wandb.finish()
    
    print("All experiments completed!")
    print("\nTest Accuracies:")
    for model_key, data in results.items():
        print(f"- {model_key}: {data['accuracy']:.4f}")

if __name__=="__main__":
    main()
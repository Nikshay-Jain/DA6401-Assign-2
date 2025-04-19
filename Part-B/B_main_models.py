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

def main_model_comp():
    wandb.login(key="e030007b097df00d9a751748294abc8440f932b1")
    
    """Run all fine-tuning experiments and compare results."""
    # Get class names
    class_names = get_class_names(train_dataset)
    
    # Dictionary to store results
    results = {}
    model_name = "resnet50"
    
    model2, history2, acc2 = run_experiment(
        model_name="vgg16", 
        freeze_strategy="all_except_last",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )
    results["vgg16"] = {"accuracy": acc2, "history": history2}

    
    model3, history3, acc3 = run_experiment(
        model_name="efficientnet_v2_s", 
        freeze_strategy="all_except_last",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )
    results["efficientnet_v2_s"] = {"accuracy": acc3, "history": history3}

    
    model4, history4, acc4 = run_experiment(
        model_name="vit_b_16", 
        freeze_strategy="all_except_last",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )
    results["vit_b_16"] = {"accuracy": acc4, "history": history4}
    
    # Final comparison run
    wandb.init(project="inaturalist_fine_tuning", name="Model_comparison")
    
    comparison_table = wandb.Table(columns=["Model", "Test Accuracy", "Trainable Parameters"])
    
    comparison_table.add_data("VGG16", results["vgg16"]["accuracy"], 
                             count_trainable_parameters(model2))
    comparison_table.add_data("EfficientNet", results["efficientnet_v2_s"]["accuracy"], 
                             count_trainable_parameters(model3))
    comparison_table.add_data("vit_b_16", results["vit_b_16"]["accuracy"], 
                             count_trainable_parameters(model4))
    
    wandb.log({"models_comparison": comparison_table})
    
    # Plot comparison chart
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for model_name, data in results.items():
        plt.plot(data["history"]["train_acc"], linestyle='-', label=f'{model_name} (Train)')
        plt.plot(data["history"]["val_acc"], linestyle='--', label=f'{model_name} (Val)')
    plt.title(f'Accuracy Comparison Across Strategies: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for model_name, data in results.items():
        plt.plot(data["history"]["train_loss"], linestyle='-', label=f'{model_name} (Train)')
        plt.plot(data["history"]["val_loss"], linestyle='--', label=f'{model_name} (Val)')
    plt.title(f'Loss Comparison Across Strategies: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    wandb.log({"model_comparison_chart": wandb.Image(plt)})
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Create bar chart of test accuracies 
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[s]["accuracy"] for s in models]
    
    plt.bar(models, accuracies)
    plt.title(f'Test Accuracy by Fine-tuning Strategy: {"all_except_last"}')
    plt.xlabel('Model')
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
    main_model_comp()
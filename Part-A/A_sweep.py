import torch, wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from A_classes import *

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
        
    return config_dict
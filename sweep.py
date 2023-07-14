import json
import wandb
import os


def train():
    # Load the default configuration
    config_defaults = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "learning_rate": 5e-5,
        "max_seq_length": 128,
        "config_file": "./configs/bert_base_uncased.json",  # or any default config file path
    }

    # Initialize a new wandb run
    run = wandb.init(config=config_defaults)

    # config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # Call your training script with parameters as environment variables
    os.system(
        f"num_train_epochs={config['num_train_epochs']} "
        f"per_device_train_batch_size={config['per_device_train_batch_size']} "
        f"learning_rate={config['learning_rate']} max_seq_length={config['max_seq_length']} "
        f"config_file={config['config_file']} python run_ner_sweep.py")


if __name__ == "__main__":
    with open('sweep_config.json') as f:
        sweep_config = json.load(f)

    sweep_id = wandb.sweep(sweep_config, project="RTB-NER-Transfer-Learning-Evaluation-Sweep")
    wandb.agent(sweep_id, train, count=50)  # adjust count as needed

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

    # Set environment variables
    os.environ['num_train_epochs'] = str(config['num_train_epochs'])
    os.environ['per_device_train_batch_size'] = str(config['per_device_train_batch_size'])
    os.environ['learning_rate'] = str(config['learning_rate'])
    os.environ['max_seq_length'] = str(config['max_seq_length'])
    os.environ['config_file'] = config['config_file']

    # Call your training script
    os.system("python run_ner.py")


if __name__ == "__main__":
    with open('sweep_config.json') as f:
        sweep_config = json.load(f)

    sweep_id = wandb.sweep(sweep_config, project="RTB-NER-Transfer-Learning-Evaluation-Sweep")
    wandb.agent(sweep_id, train, count=50)  # adjust count as needed

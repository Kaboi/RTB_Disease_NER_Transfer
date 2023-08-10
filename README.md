# NER Using Transfer Learning for RTB Crops

## Original Code
The original code is build from the legacy examples from the HuggingFace
https://github.com/huggingface/transformers/tree/main/examples/legacy

## Data Format
labels available at ./data/labels.txt
config files in specific model folders e.g. ./data_xyz/model_xyz
Data files with max 128 or 256 tokens after tokenization are in ./data_xyz/model_xyz/128 and ./data_xyz/model_xyz/256

## Configuration
Config files hold the hyperparemeters
See notebook for example of usage

Please note: Data_30 the 30 is legacy in the name will be changed
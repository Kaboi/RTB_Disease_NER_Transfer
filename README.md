# NER Using Transfer Learning for RTB Crops

## Original Code
The code uses the  HuggingFace legacy example below as the starting point
https://github.com/huggingface/transformers/tree/main/examples/legacy

## Data Format
Labels available at ./data/labels.txt

Config files in specific model folders e.g. ./data_xyz/model_xyz

Data files with max 128 or 256 tokens after tokenization are in ./data_xyz/model_xyz/128 and ./data_xyz/model_xyz/256

## Configuration
Config files hold the hyperparemeters
See notebook for example of usage

Please note: Data_30 the 30 is legacy in the name will be changed

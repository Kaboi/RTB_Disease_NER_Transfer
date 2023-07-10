# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch import nn
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
import datetime
import wandb
import itertools

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import is_main_process

from dataclasses import asdict


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # log training to wandb
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{model_args.model_name_or_path}-{timestamp}"

    wandb_config = {**asdict(model_args), **asdict(data_args), **asdict(training_args)}

    wandb.init(project='RTB-NER-Transfer-Learning', name=run_name, tags=['BERT', 'train'],
               config=wandb_config)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use"
            " --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = dict(enumerate(labels))
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)

        fig, ax = plt.subplots(figsize=(14, 12))  # Adjust the figsize parameter

        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)  # Use the mappable object 'im' for colorbar creation
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=60)
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        return fig  # Return the figure object

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)

        accuracy = accuracy_score(out_label_list, preds_list)
        precision = precision_score(out_label_list, preds_list, average='micro')
        recall = recall_score(out_label_list, preds_list, average='micro')
        f1 = f1_score(out_label_list, preds_list, average='micro')

        # Classification report for macro-averaged per-class metrics
        report_data = classification_report(out_label_list, preds_list, digits=4, output_dict=True)
        wandb.log({"classification_report_data": wandb.Table(dataframe=pd.DataFrame(report_data).transpose())})

        report = classification_report(out_label_list, preds_list, digits=4)
        logger.info("*** Classification report ***")
        logger.info("\n%s", report)

        # Calculating Non-O accuracy
        non_o_true_labels = []
        non_o_pred_labels = []
        for true_labels, pred_labels in zip(out_label_list, preds_list):
            for true_label, pred_label in zip(true_labels, pred_labels):
                # Only include instances where the true label is not 'O'
                if true_label != 'O':
                    non_o_true_labels.append(true_label)
                    non_o_pred_labels.append(pred_label)

        non_o_accuracy = accuracy_score(non_o_true_labels, non_o_pred_labels)

        # # Flattening the lists for confusion matrix
        # flat_true_labels = [label for sublist in out_label_list for label in sublist]
        # flat_pred_labels = [label for sublist in preds_list for label in sublist]
        #
        # # Compute confusion matrix
        # cm = confusion_matrix(flat_true_labels, flat_pred_labels, labels=labels)
        #
        # # Plot confusion matrix
        # # Call the plot_confusion_matrix function here and pass the computed confusion matrix
        # fig = plot_confusion_matrix(cm, classes=labels, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues)

        # Log metrics and confusion matrix to wandb
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "non_O_accuracy": non_o_accuracy  # ,
            # "confusion_matrix": [wandb.Image(fig, caption="Confusion Matrix")]
        })

        return {
            "accuracy": accuracy,  # Changed from "accuracy_score" to "accuracy"
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "non_O_accuracy": non_o_accuracy
        }

    # Data collator
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    ) if training_args.fp16 else None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        predicted_outputs = trainer.predict(test_dataset)
        metrics = predicted_outputs.metrics
        preds_list_out, out_label_list_out = align_predictions(predicted_outputs.predictions,
                                                               predicted_outputs.label_ids)

        if trainer.is_world_process_zero():

            # Custom order
            custom_order = ['B-CROP', 'I-CROP', 'B-PLANT_PART', 'I-PLANT_PART', 'B-PATHOGEN', 'I-PATHOGEN',
                            'B-DISEASE', 'I-DISEASE', 'B-SYMPTOM', 'I-SYMPTOM', 'B-GPE', 'I-GPE',
                            'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE', 'B-ORG', 'I-ORG', 'O']

            # Flattening the lists for confusion matrix
            flat_true_labels = [label for sublist in out_label_list_out for label in sublist]
            flat_pred_labels = [label for sublist in preds_list_out for label in sublist]

            # Compute ordered confusion matrix using the custom order
            ordered_cm = confusion_matrix(flat_true_labels, flat_pred_labels, labels=custom_order)

            # Compute unordered confusion matrix
            unordered_cm = confusion_matrix(flat_true_labels, flat_pred_labels)

            # Plot ordered confusion matrix
            fig1 = plot_confusion_matrix(ordered_cm, classes=custom_order, normalize=True,
                                         title='Ordered Confusion Matrix',
                                         cmap=plt.cm.Blues)

            # Plot unordered confusion matrix
            fig2 = plot_confusion_matrix(unordered_cm, classes=labels, normalize=True,
                                         title='Unordered Confusion Matrix',
                                         cmap=plt.cm.Blues)

            # Log confusion matrices to wandb
            wandb.log({
                "ordered_confusion_matrix": [wandb.Image(fig1, caption="Ordered Confusion Matrix")],
                "unordered_confusion_matrix": [wandb.Image(fig2, caption="Unordered Confusion Matrix")]
            })

            # Write test results to file
            output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            logger.info(f"preds_list_out - training_args BEFORE SAVE PREDICTIONS: {preds_list_out[:5]}")

            # Save predictions
            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                    token_classification_task.write_predictions_to_file(writer, f, preds_list_out)

            logger.info(f"preds_list_out - training_args AFTER SAVE PREDICTIONS: {preds_list_out[:5]}")

            wandb.log({
                "Accuracy": metrics.get("test_accuracy", None) * 100 if metrics.get(
                    "test_accuracy") is not None else None,
                "Precision": metrics.get("test_precision", None) * 100 if metrics.get(
                    "test_precision") is not None else None,
                "Recall": metrics.get("test_recall", None) * 100 if metrics.get(
                    "test_recall") is not None else None,
                "F1": metrics.get("test_f1", None) * 100 if metrics.get("test_f1") is not None else None,
                "Non_O_accuracy": metrics.get("test_non_O_accuracy", None) * 100 if metrics.get(
                    "test_non_O_accuracy") is not None else None,
            })

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

wandb.finish()

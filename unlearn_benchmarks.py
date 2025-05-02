#!/usr/bin/env python3

import sys
import os
import math

from glob_vars import DOMAINS
from process_dset import load_processed_dataset
from learn_router import LearnRouter
from det_router import DeterministicRouter
from model_args import ModelArguments
from data_args import DataTrainingArguments

import datasets
import evaluate

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    default_data_collator,
    HfArgumentParser,
    is_torch_xla_available,
    set_seed
)

# Benchmark paper:
# https://arxiv.org/pdf/2304.04934

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"): 
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # TT: changed sys.argv[1] to argv[-1]
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

def accuracy(
        eval_dataset: Dataset,
        model: LearnRouter | DeterministicRouter,
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast,   
        max_eval_samples: int,
        ):
   
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )

    metrics = trainer.evaluate()

    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    pass

def load_dataset_wrapper(
        dataset_name: str, 
        forget_set_name: str,
        mode : str,
        max_eval_samples: int = 1000,
    ):
    if mode == "retain_train":
        eval_dataset = list()
        for domain in DOMAINS:
            if domain != forget_set_name:
                dataset = load_processed_dataset(dataset_name, forget_set_name, split = "train", num_samples = 1000, seed = training_args.seed)
                eval_dataset.append(dataset)
        eval_dataset = datasets.concatenate_datasets(eval_dataset)
    elif mode == "forget_train":
        eval_dataset = load_processed_dataset(dataset_name, forget_set_name, split = "train")
    elif mode == "retain_test":
        eval_dataset = list()
        for domain in DOMAINS:
            if domain != forget_set_name:
                dataset = load_processed_dataset(dataset_name, forget_set_name, split = "train")
                eval_dataset.append(dataset)
        eval_dataset = datasets.concatenate_datasets(eval_dataset)
    else:
        raise(ValueError("`mode` must be one of retain_train, forget_train, or retain_test"))
    
    eval_dataset = eval_dataset.shuffle(seed = training_args.seed)
    eval_dataset = eval_dataset.select(range(max_eval_samples))
    return eval_dataset

def mia(
        dataset_name: str, 
        forget_set_name: str,
        max_eval_samples: int = 1000,
):
    pass

if __name__ == "__main__":
    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
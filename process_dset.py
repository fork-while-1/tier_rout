#!/usr/bin/env python3.8
from __future__ import annotations

from typing import List
import datasets
from datasets import load_dataset

from transformers.utils import logging
from itertools import chain
import sys, os, multiprocessing

import numpy as np

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast, 
    TrainingArguments
    )

from model_args import ModelArguments
from data_args import DataTrainingArguments

from transformers import HfArgumentParser, AutoTokenizer, TrainingArguments



def load_processed_dataset(
    dataset_name: str | None = None, 
    dataset_config: List[str] | None = None,
    dataset_path: str | None = None,
    split : str | None = None,
    num_samples : int | None = None, 
    seed : int | None = None
    ) -> datasets.DatasetDict | datasets.Dataset:
    tok_logger = logging.get_logger("transformers.tokenization_utils_base")
    try:
        if dataset_path != None:
            tokenized_datasets = datasets.load_from_disk(dataset_path, keep_in_memory = True)
        else:
            dataset_list = list()
            if split == None:
                for domain in dataset_config:
                    tokenized_datasets = datasets.load_from_disk(f"/share/suh-scrap2/tier_rout/datasets/{dataset_name}/{domain}", keep_in_memory = True)
            else:
                for domain in dataset_config:
                    tokenized_datasets = datasets.load_from_disk(f"/share/suh-scrap2/tier_rout/datasets/{dataset_name}/{domain}/{split}", keep_in_memory = True)
            dataset_list.append(tokenized_datasets)
            tokenized_datasets["train"] = datasets.concatenate_datasets([i["train"] for i in dataset_list])
            tokenized_datasets["validation"] = datasets.concatenate_datasets([i["validation"] for i in dataset_list])
        if num_samples != None:
            if seed == None:
                raise(ValueError("`seed` must not be None if num_samples is not None"))
            
            # This is extremely slow because it generates a layer of indirection, and 
            # now the dataset is no longer sequential access. 
            # To mititgate this, for now we keep the entire dataset in memory
            tokenized_datasets = tokenized_datasets.shuffle(seed = seed, keep_in_memory = True)
            tokenized_datasets = tokenized_datasets.select(range(num_samples), keep_in_memory = True)
        if seed != None:
            tokenized_datasets = tokenized_datasets.shuffle(seed = seed, keep_in_memory = True)
        return tokenized_datasets
    except FileNotFoundError:
        tok_logger.error("run process_dset.py first")
        raise FileNotFoundError(f"No tokenized datasets found for {dataset_name}/{dataset_config} or {dataset_path}")

def concat_datasets(path : str) -> None:
    dataset_chunks = os.listdir(path)
    train_dataset = list()
    valid_dataset = None
    for i in dataset_chunks:
        if "." in i:
            continue
        print("Reading chunk", os.path.join(path,i))
        tokenized_datasets = datasets.load_from_disk(os.path.join(path,i))
        train_dataset.append(tokenized_datasets["train"])
        if valid_dataset is None:
            valid_dataset = tokenized_datasets["validation"]
    train_dataset = datasets.concatenate_datasets(train_dataset)
    concatenated_dataset = datasets.DatasetDict({"train" : train_dataset, "validation": valid_dataset})
    print(f"Successfully concatenated {path}")
    concatenated_dataset.save_to_disk(path)
    return

def create_router_dataset(dataset_name : str, dataset_config_name : str) -> None:
    print(f"Processing {dataset_config_name}")
    if os.path.exists(f"/share/suh-scrap2/tier_rout/router_dataset/{dataset_name}/{dataset_config_name}/validation/dataset_info.json"):
        return
    tokenized_datasets = datasets.load_from_disk(f"/share/suh-scrap2/tier_rout/datasets/{dataset_name}/{dataset_config_name}/")
    router_dataset = tokenized_datasets["train"].select(range(5000))
    remaining_dataset = tokenized_datasets["train"].select(range(5000, len(tokenized_datasets["train"])))
    remaining_datasetdict = datasets.DatasetDict({"train" : remaining_dataset, "validation": tokenized_datasets["validation"]})
    # this takes a while so we don't want to repeat it if not necessary
    if not os.path.exists(f"/share/suh-scrap2/tier_rout/datasets/{dataset_name}/{dataset_config_name}/main_dataset/validation/dataset_info.json"):
        remaining_datasetdict.save_to_disk(f"/share/suh-scrap2/tier_rout/datasets/{dataset_name}/{dataset_config_name}/main_dataset/")
    split_router_dataset = router_dataset.train_test_split(0.1)
    split_router_dataset = datasets.DatasetDict({"train" : split_router_dataset["train"], "validation": split_router_dataset["test"]})
    if not os.path.exists(f"/share/suh-scrap2/tier_rout/router_dataset/{dataset_name}/{dataset_config_name}/validation/dataset_info.json"):
        split_router_dataset.save_to_disk(f"/share/suh-scrap2/tier_rout/router_dataset/{dataset_name}/{dataset_config_name}")
    return


def save_processed_dataset(
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast,            
        training_args : TrainingArguments, 
        data_args : DataTrainingArguments, 
        dataset_name : str,
        dataset_config : str, 
        raw_datasets : datasets.Dataset | datasets.DatasetDict
        ):

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "body" if "body" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = logging.get_logger("transformers.tokenization_utils_base")

    # reduce the size of the dataset
    tok_logger.info(f"Truncating dataset to include {data_args.sample_range[0]} to {data_args.sample_range[1]}")
    
    try:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.sample_range[0], data_args.sample_range[1]))
    except IndexError:
        pass

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name]) 
        return output
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched = True,
            writer_batch_size = 50000,
            num_proc = data_args.preprocessing_num_workers,
            remove_columns = column_names,
            load_from_cache_file = True,
            desc = "Running tokenizer on dataset",
        )
    
    def group_texts(examples):
        # Concatenate all texts.
        unk_threshold = int(0.1 * data_args.block_size)
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // data_args.block_size) * data_args.block_size
        # Split by chunks of max_len.        
        result = {}
        result["input_ids"] = [concatenated_examples["input_ids"][i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size) 
                               if concatenated_examples["input_ids"][i : i + data_args.block_size].count(tokenizer.unk_token_id) < unk_threshold]                
        result["attention_mask"] = [[1, ] *  data_args.block_size for _ in range(len(result["input_ids"]))]
        result["token_type_ids"] = [[0, ] *  data_args.block_size for _ in range(len(result["input_ids"]))]
        
        # result["labels"] = result["input_ids"].copy()        
        return result
    
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched = True,
                writer_batch_size = 50000,
                num_proc = data_args.preprocessing_num_workers,
                load_from_cache_file = True,
                desc = f"Grouping texts in chunks of {data_args.block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    
    lm_datasets.save_to_disk(f"/share/suh-scrap2/tier_rout/datasets/{ dataset_name }/{ dataset_config }/{ data_args.sample_range[0]}")
    tok_logger.info(f"Saved to /share/suh-scrap2/tier_rout/datasets/{ dataset_name }/{ dataset_config }")

if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"): 
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # TT: changed sys.argv[1] to argv[-1]
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # create_router_dataset("/share/suh-scrap2/tier_rout/datasets/allenai/c4/pa/")
    
    # get raw dataset
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        streaming=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    # init tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    tokenized_datasets = save_processed_dataset(
        tokenizer, 
        training_args, 
        data_args, 
        data_args.dataset_name, 
        data_args.dataset_config_name, 
        raw_datasets)
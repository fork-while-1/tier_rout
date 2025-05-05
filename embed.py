#!/usr/bin/env python3.8

import os, sys
import torch
from process_dset import load_processed_dataset
from train_det_router import DecoderModel, DecoderConfig

from model_args import ModelArguments
from data_args import DataTrainingArguments

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)



def embed_tokens(
        embedder_path: str,
        config : DecoderConfig,
        dataset_name : str, 
        dataset_config : str,
        batch_size: int,
        seed : int,
        ):
    
    embedder = DecoderModel(config).from_pretrained(
        os.path.join(
            embedder_path, 
            dataset_name, 
            "embed_model")
            )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embedder.to(device)
    embedder.eval()

    dataset = load_processed_dataset(
        dataset_name, 
        dataset_config, 
        split = "train", 
        num_samples = 100000, 
        seed = seed)
    
    seq_len = len(dataset["input_ids"][0])

    average_embedding = torch.zeros(batch_size, seq_len, config.embedding_dim)

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = torch.tensor(dataset["input_ids"][start: end], device=device)
        average_embedding += embedder(batch)["hidden_states"].sum(0)

    average_embedding = average_embedding / len(dataset)
    
    with open(os.path.join(embedder_path, dataset_name, "embed_model", dataset_config), 'wb') as fdesc:
        torch.save(average_embedding, fdesc)

if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    config = config = DecoderConfig(
            vocab_size = len(tokenizer), 
            embedding_dim = model_args.hidden_size, 
            num_heads = 2,
            padding_idx = tokenizer.pad_token_id
        )

    embed_tokens(
        "output",
        config,
        model_args.dataset_name,
        model_args.dataset_config_name,
        8,
        training_args.seed
        )

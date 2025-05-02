#!/usr/bin/env python3

import torch
import os
import sys

from data_args import DataTrainingArguments
from model_args import ModelArguments
from modeling_monet import MonetForCausalLM, MonetConfig
from process_dset import load_processed_dataset

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
)

torch.set_grad_enabled(False)

def track_experts(
        eval_batch_size : int,
        seed : int,
        hidden_size : int,
        moe_experts : int,
        vocab_size : int,
        checkpoint_pth : str,
        dataset_name : str,
        domain_name : str,
        threshold: float = 0.1,
) -> None:
    config = MonetConfig(
            vocab_size = vocab_size, 
            hidden_size = hidden_size,
            moe_experts= moe_experts,
            output_router_probs=True)
    
    # model = MonetForCausalLM(config = config)
    model = MonetForCausalLM.from_pretrained(checkpoint_pth, config = config, weights_only = True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    dataset = load_processed_dataset(
        dataset_name, 
        domain_name, 
        num_samples=1000, 
        seed=seed,
        split="train",
        )

    total_router_probs = torch.zeros(config.num_hidden_layers // config.moe_groups, config.moe_experts, 2, device=device)
    
    for batch_start in range(0, len(dataset), eval_batch_size):
        if batch_start % 100 == 0:
            print(f"Batch: {batch_start}")

        batch_end = min(batch_start + eval_batch_size, len(dataset))
        outputs = model(torch.tensor(dataset["input_ids"][batch_start : batch_end], device=device), return_dict = True)

        # (0) layers, (1) 2, [(2) batch, (3) seq_len, (4) heads, (5) experts]
        # since routing is grouped, first dimension is either layer probs or None (if routing was reused.)
        # transform router probs to batch x seq_len x layers x heads x num_experts x 2
        probs = torch.stack([torch.stack(outputs.router_probs[i]) for i in range(0, len(outputs.router_probs), config.moe_groups)])
        probs = probs.permute(2,3,0,4,5,1)
        probs = probs.sum(dim = 0) # reduce over all batches
        probs = probs.mean(dim = 0) # reduce over all tokens
        probs = probs.mean(dim = 1) # reduce over all heads
        
        total_router_probs += probs
    
    # normalize across length of dataset
    total_router_probs = total_router_probs / len(dataset)
    print(total_router_probs)

    # filter important experts based on threshold
    imp_expert_mask = total_router_probs >= threshold

    # torch.save(imp_expert_mask, f"{checkpoint_pth}/{domain_name}.experts")

def determine_ckpt_pth(
        num_partitions: int,
        domain_name : str,
) -> str:
    ckpt_pth = f"/home/tt544/tier_rout/output/{num_partitions}/"
    ls_out = os.listdir(ckpt_pth)
    for f in ls_out:
        if domain_name in f:
            return os.path.join(ckpt_pth, f)
    return ""


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
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    vocab_size = len(tokenizer)
    path = determine_ckpt_pth(16, data_args.dataset_config_name)
    
    track_experts(
        training_args.eval_batch_size,
        training_args.seed,
        model_args.hidden_size,
        model_args.moe_experts,
        vocab_size,
        path,
        data_args.dataset_name,
        data_args.dataset_config_name,
    )

        
    
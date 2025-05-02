#!/usr/bin/env python3.8

from __future__ import annotations

import os
import json
import torch
from typing import List, Dict
from modeling_monet import MonetForCausalLM
from glob_vars import DOMAINS

from train_det_router import DecoderModel, DecoderModelOutput
from process_dset import load_processed_dataset
from sklearn.metrics.pairwise import cosine_similarity

from model_args import ModelArguments
from data_args import DataTrainingArguments
from transformers import HfArgumentParser, AutoTokenizer, TrainingArguments
import sys



def load_partition_to_domain(model_paths_dir: str) -> Dict[int, str]:
    with open(os.path.join(model_paths_dir, "mapping.json")) as fdesc:
        return json.load(fdesc)

def load_domain_to_partition(model_paths_dir: str) -> Dict[int, str]:
    reverse_mapping = dict()
    mapping = load_partition_to_domain(model_paths_dir)
    for partition, domains in mapping.items():
        for domain in domains:
            reverse_mapping[domain] = partition
    return reverse_mapping

def load_models(model_paths_dir: str) -> List[MonetForCausalLM]:
    mapping = load_partition_to_domain()
    models = list()
    for partition, domains in mapping.items():
        model = MonetForCausalLM.from_pretrained(os.path.join(model_paths_dir, domains.join("-")))
        models.append(model)
    return models

def load_partition_embeddings(embedding_dir: str) -> torch.Tensor:
   partition_embeddings = list()
   for domain in DOMAINS:
       partition_embedding = MonetForCausalLM.from_pretrained(os.path.join(embedding_dir, domain))
       partition_embeddings.append(partition_embedding)
   return torch.stack(partition_embeddings)
    
# model = AutoregressiveTransformer(num_tokens, embedding_dim, num_heads, num_layers)

# # Generate a sample target sequence
# target_seq = torch.randint(0, num_tokens, (seq_length,))

# # Create a causal mask to prevent the model from attending to future tokens
# target_mask = torch.tril(torch.ones((seq_length, seq_length)))

# # Pass the target sequence and mask through the model
# output = model(target_seq, target_mask=target_mask)

# print(output.shape)

def eval_embedder(
        embedding_dir : str, 
        dataset_name: str,
        embedder_params : Dict[str, int],
        domain_names : List[str],
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = DecoderModel(
        embedder_params["vocab_size"], 
        embedder_params["embedding_dim"], 
        embedder_params["num_heads"], 
        embedder_params["padding_idx"],
        output_hidden_states=True
    )
    
    
    embedder.from_pretrained(
            os.path.join(
                embedding_dir, "/checkpoint-5500/"), 
            )
    
    embedder.eval()
    embedder.to(device)
    
    dataset_list = list()
    for domain in domain_names:
        domain_samples = load_processed_dataset(
            dataset_name = dataset_name, 
            dataset_config = domain,
            num_samples = 2,
            seed = 42
            )
        dataset_list.append(domain_samples)
    
    output_embeddings = list()
    for domain_samples in dataset_list:
        input_ids = domain_samples["input_ids"].to(device)
        embeddings = embedder(input_ids).hidden_states
        output_embeddings.append(embeddings)

    print(output_embeddings)
    

class DeterministicRouter():
   def __init__(self, config):
       self.config = config
       self.embedder = DecoderModel()
       self.embedder.load_state_dict(
           torch.load(
               os.path.join(
                   self.config.embedding_dir, "embed_model"), 
                   weights_only=True)
                   )
       self.models = load_models(self.config.model_paths_dir)
       self.partition_embeddings = load_partition_embeddings(self.config.embedding_dir)
       self.delete_models = self.config.delete_models
       self.num_partitions = len(self.models)
       self.domain_to_partititon = load_domain_to_partition(self.config.model_paths_dir)
       self.cosine = torch.nn.CosineSimilarity(dim=-1)
       self.loss = torch.nn.CrossEntropyLoss()



   def forward(
       self,
       input_ids: torch.LongTensor = None,
       attention_mask: torch.Tensor | None = None,
       position_ids: torch.LongTensor | None = None,
       inputs_embeds: torch.FloatTensor | None = None,
       labels: torch.LongTensor | None = None,
       use_cache: bool | None = None,
       output_attentions: bool | None = None,
       output_hidden_states: bool | None = None,
       output_router_probs: bool | None = None,
       return_dict: bool | None = None,
       cache_position: torch.LongTensor | None = None,
   ):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[-1]

        # shape : batch x seq_len x embedding dim
        embeddings = torch.tensor(self.embedder(input_ids))["hidden_staes"]
        embed_dim = embeddings.shape[-1]

        # get cumulative sums to account for context from prior tokens
        embeddings = embeddings.cumsum(1)

        # convert cum sum to mean
        embeddings = embeddings / torch.arange(1, seq_len + 1).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, embed_dim)

        class_probs = torch.zeros(embeddings.shape[0], len(self.partition_embeddings))

        for idx, embedding in enumerate(embeddings):
           class_probs[idx] = self.cosine(embedding.unsqueeze(0).repeat(self.num_partitions, 1), self.partition_embeddings)
        
        if self.delete_models != None:
            for del_model in self.delete_models:
                class_probs[...,del_model] = -float('inf')
        
        classes = torch.argmax(class_probs, dim = -1)
        uniq_classes = set(torch.flatten(classes).sort())
        uniq_partitions = set([self.domain_to_partition[i] for i in uniq_classes])
        
        chosen_class_idx = classes.clone()
        for class_id in uniq_classes:
            chosen_class_idx[chosen_class_idx == class_id] = self.domain_to_partition[class_id]

        model_logits = list()
        for model_idx in uniq_partitions:
            output = self.models[model_idx].forward(
                input_ids,
                attention_mask,
                position_ids,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                output_router_probs,
                return_dict,
                cache_position
            )
            model_logits.append(output["logits"])

        model_logits = torch.stack(model_logits)
        # collect the output of the right model for each token
        final_model_logits = torch.gather(model_logits, 0, chosen_class_idx)

        loss = self.loss(model_logits, input_ids)

        return DecoderModelOutput(
            logits=final_model_logits,
            loss=loss
        )

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"): 
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # TT: changed sys.argv[1] to argv[-1]
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
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


    embedder_params = {
        "vocab_size" : len(tokenizer), 
        "embedding_dim" : model_args.hidden_size, 
        "num_heads" : 2,
        "padding_idx" : tokenizer.pad_token_id
    }
    eval_embedder(
        "/home/tt544/tier_rout/output/embed_model/",
        data_args.dataset_name,
        embedder_params,
        ["ur", "pa", "si"]
        )


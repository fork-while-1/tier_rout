#!/usr/bin/env python3.8

import os
import torch
from typing import Dict
from process_dset import load_processed_dataset
from train_det_router import DecoderModel



def embed_tokens(
        embedder_path: str,
        embedder_params : Dict[str, int],
        dataset_name : str, 
        dataset_config : str,
        batch_size: int
        ):
    
    embedder = DecoderModel(
        embedder_params["vocab_size"], 
        embedder_params["embedding_dim"], 
        embedder_params["num_heads"], 
        embedder_params["padding_idx"],
        output_hidden_states=True
        )
    embedder.load_state_dict(torch.load(os.path.join(embedder_path, dataset_name, "embedding_model") , weights_only=True))
    embedder.eval()
    dataset = load_processed_dataset(dataset_name, dataset_config, split = "train")
    seq_len = len(dataset["input_ids"][0])

    average_embedding = torch.zeros(batch_size, seq_len, embedder_params["embedding_dim"])

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = torch.tensor(dataset["input_ids"][start: end])
        average_embedding += embedder(batch)["hidden_states"].sum(0)

    average_embedding = average_embedding / len(dataset)
    
    with open(os.path.join(embedder_path, dataset_name, dataset_config), 'wb') as fdesc:
        torch.save(average_embedding, fdesc)


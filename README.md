## TLDR: 
This is a project that studies the trade-offs between data compartmentalization vs model utility in a Fine-Grained MoE style LLM via *tiered-routing*.

## Introduction:
Unlearning of semantically similar data (e.g., concepts) from a trained model has been of increasing interest. 
Proposed solutions to this range from strictly partitioning data and model weights in ensemble / Mixture of Experts (MoE) architectures to exploring completely self-organizing MoE models where the model learns to limit the influence of semantically similar data in certain model experts. 
While the former provides strict guarantees on data removal, there has been prior work that shows that these types of solutions are suboptimal for model utility. 
In this paper we introduce tiered routing--a combination of the aforementioned strict partitioning and learnt partitioning--as a way to systematically evaluate the tradeoff between unlearnability and model utility.

## Research Questions:
1) What is the tradeoff between data unlearnability of semantically similar data, and model utility as we go from strict partitioning of training data and model parameters to completely self-organized MoE architectures?
2) Is there a middle ground solution that balances between the extremes on data unlearnability and model utility?
3) Does the way the data is partitioned (i.e., random or based on data similarity) affect utility and compartmentalization ability?

## Methodology – Tiered Routing:
**Level 1**: Partition the dataset into n strict partitions:
1. Training: Each partition will have its own MoE model trained in isolation.
2. Inference: During inference, a sample is routed to the best expert(s).

**Level 2**: Within a single partition, train an MoE model with learned routing on all data within the partition.

## Script Descriptions
1. process_dset.py : Processes a chosen dataset (downloading, tokenizing, grouping, and purging low-quality text) and stores it to disk.
2. run_clm.py: Main entry point for training a model, must provide a config file specifiying the model and dataset that needs to be trained
4. train_det_router.py : Defining a deterministic router (Level 1 router). This router is trained on the language modeling objective, however, we really only care about the hidden states produced at the end since these dense embeddings are used to route tokens to different Level 2 models during evaluation.
3. modeling_monet.py: Taken from https://github.com/dmis-lab/Monet , an implementation of a fine-grained MoE Model. The router in this is going to be our Level 2 router.
5. eval_det_router.py: This is where the top-level routing is done to see which Monet Model token should be sent to.
6. unlearn_benchmarks.py: When we unlearn a given domain, this script is used to evaluate how successful the unlearning was, and also how accurate the unlearnt model is on the remaining data.

## Dataset
16 languages in C4/multilingual:
1. hi, si, gu, pa (Indian Languages)
2. zh, vi, ja, ko (East Asian Languages)
3. ar, fa, ur, kk (Middle Eastern Languages)
4. it, en, de, es (Germanic Languages)

<!-- Total number of trainable parameters to hidden dim and moe experts mapping:
1. (01) 766,556,160 = 2048, 512
2. (02) 382,931,320 = 1224, 256
3. (04) 191,639,040 = 0712, 128
4. (08) 099,883,648 = 0384, 064 
5. (16) 048,925,020 = 0196, 032 -->

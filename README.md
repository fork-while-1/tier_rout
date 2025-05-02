## TLDR: 
This is a project that studies the trade-offs between data compartmentalization vs model utility in a Fine-Grained MoE style LLM via *tiered-routing*.

## Introduction:
Unlearning of semantically similar data (e.g., concepts) from a trained model has been of increasing interest. Proposed solutions to this range from strictly partitioning data and model weights in ensemble / Mixture of Experts (MoE) architectures to exploring completely self-organizing MoE models where the model learns to limit the influence of semantically similar data in certain model experts. While the former provides strict guarantees on data removal, there has been prior work that shows that these types of solutions are suboptimal for model utility. In this paper we introduce tiered routing–a combination of the aforementioned strict partitioning and learnt partitioning–as a way to systematically evaluate the tradeoff between unlearnability and model utility.

## Research Questions:
1) What is the tradeoff between data unlearnability of semantically similar data, and model utility as we go from strict partitioning of training data and model parameters to completely self-organized MoE architectures?
2) Is there a middle ground solution that balances between the extremes on data unlearnability and model utility?
    
3) Does the way the data is partitioned (i.e., random or based on data similarity) affect utility and compartmentalization ability?

## Methodology – Tiered Routing:
**Level 1**: Partition the dataset into n strict partitions:
1. Training: Each partition will have its own MoE model trained in isolation.
2. Inference: During inference, a sample is routed to the best expert(s).

**Level 2**: Within a single partition, train an MoE model with learned routing on all data within the partition.


## Dataset
16 languages in C4/multilingual:
hi, si, gu, pa
zh, vi, ja, ko
ar, fa, ur, kk
it, en, de, es

Total number of trainable parameters to hidden dim and moe experts mapping:
1. (1) 766,556,160 = 2048, 512
2. (2)
3. (4)
4. (8)
5. (16) 48,925,020 = 196, 32
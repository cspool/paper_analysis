# **4 Training Efficiency**

In Figure [8,](#page-9-0) we're taking a look at the training process itself, comparing the training dynamic between MoE and dense model candidates, aligned by datasets, steps and hyper-parameters. Specifically, we compare the average language modeling performance between models at training checkpoints ranging from 10k to the full 310k steps.

Comparing the active parameter and total parameter aligned dense models with our best performing MoE model, we corroborate the findings in [Lin et al.](#page-13-5) [\(2024\)](#page-13-5), showing a 5-10x training efficiency gain using MoE models over their active parameter aligned dense candidates. Specifically, our MoE model candidate reaches the best performance of the 1.4B dense model at around 35k steps, while the larger and more powerful 3.6B dense model achieves generally higher scores.


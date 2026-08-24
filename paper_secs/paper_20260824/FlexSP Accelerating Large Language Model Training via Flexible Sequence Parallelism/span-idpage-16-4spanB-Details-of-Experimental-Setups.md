# <span id="page-16-4"></span>**B** Details of Experimental Setups

## <span id="page-16-2"></span>**B.1** Model Configuration

Tab. 5 presents the specific configurations of the GPT-7B, 13B and 30B models used in our experiments. The parameter number of each model in Tab. 5 is under maximum context length of 384K, where the positional embedding contains 1-2 billion parameters.

<span id="page-16-7"></span>**Table 5.** Model configuration (384K max context length).

| Model   | # Layers | # Param | Hidden Dim |
|---------|----------|---------|------------|
| GPT-7B  | 32       | 7.85B   | 4096       |
| GPT-13B | 40       | 14.03B  | 5120       |
| GPT-30B | 60       | 32.72B  | 6656       |

## <span id="page-16-3"></span>**B.2** Protocols

For fair comparison, we manually tune the most efficient parallelism strategies for all baseline systems under different workloads. For DeepSpeed, the optimal strategy is usually among SP=64 or SP=32 with ZeRO-3, and for Megatron-LM, the optimal strategy is usually among TP=8, CP=8, or TP=16, CP=4, or TP=8, CP=4, DP=2 with ZeRO-1.

We also apply activation checkpointing strategies for each system to make sure all systems can fit the models without out-of-memory issues. For GPT-7B, activation checkpointing is unnecessary to support a 384K context length on 64 GPUs.

For GPT-13B, we only checkpoint MLP layers, while for GPT-30B, almost all layers are checkpointed to support 384K context length.


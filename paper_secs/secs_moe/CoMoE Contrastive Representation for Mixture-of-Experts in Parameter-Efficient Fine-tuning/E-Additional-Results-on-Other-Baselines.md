# E Additional Results on Other Baselines

In the main paper, we compared CoMoE with three widely recognized and well-performing baselines (LoRA, DoRA, and MixLoRA) using the

| Datasets   | #train | —#test | Type                | Metrics |
|------------|--------|--------|---------------------|---------|
| BoolQ      | 9,427  | 3,270  | Text Classification | acc     |
| OBQA       | 4,957  | 500    | Question Answering  | acc     |
| ARC-e      | 2,251  | 2,376  | Question Answering  | acc     |
| ARC-c      | 1,119  | 1,172  | Question Answering  | acc     |
| PIQA       | 16,100 | 1,840  | Question Answering  | acc     |
| SIQA       | 33,410 | 1,954  | Question Answering  | acc     |
| HellaSwag  | 39,905 | 10,042 | Sentence Completion | acc     |
| WinoGrande | 9,248  | 1,267  | Fill in the Blank   | acc     |

Table 5: The dataset statistics.

| Hyperparameters    | LoRA/DoRA | MixLoRA/MixDoRA            | OMoE-LoRA    | CoMoE           |  |  |  |  |  |
|--------------------|-----------|----------------------------|--------------|-----------------|--|--|--|--|--|
| Cutoff Length      |           | 512                        |              |                 |  |  |  |  |  |
| Learning Rate      |           | 2e-4                       |              |                 |  |  |  |  |  |
| Optimizer          |           | AdamW                      |              |                 |  |  |  |  |  |
| Batch size         |           | 16                         |              |                 |  |  |  |  |  |
| Accumulation Steps |           | 8                          |              |                 |  |  |  |  |  |
| Dropout            |           | 0.05                       |              |                 |  |  |  |  |  |
| Epochs             |           | 2                          |              |                 |  |  |  |  |  |
| Where              |           | Q, K, V, O, Up, Down, Gate |              |                 |  |  |  |  |  |
| LoRA Rank r        | 80        | 16                         | 16           | 16              |  |  |  |  |  |
| LoRA Alpha a       | 160       | 32                         | 32           | 32              |  |  |  |  |  |
| Experts            | -         | 8                          | 4            |                 |  |  |  |  |  |
| Routing strategy   | -         | Top - 2 routing            | Soft routing | Top - 2 routing |  |  |  |  |  |

Table 6: Hyperparameter configurations of LORA, DoRA, MixLoRA, MixDoRA, OMoE-LoRA and CoMoE for fine-tuning LLaMA-2 7B on datasets.

LLaMA-2 7B model. In addition to the results shown in Table 2, we provide experimental results involving 11 additional strong baselines on the same LLaMA-2 7B backbone, as detailed in Table 12. The results demonstrate that CoMoE achieves significant improvements in both parameter efficiency and overall performance compared to these baselines.


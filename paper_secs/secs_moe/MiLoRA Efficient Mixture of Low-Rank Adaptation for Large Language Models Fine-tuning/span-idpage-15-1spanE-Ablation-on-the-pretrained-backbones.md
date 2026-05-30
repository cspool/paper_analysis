# <span id="page-15-1"></span>E Ablation on the pretrained backbones

Our main experiments are conducted on the LlaMA-2 7B model. To demonstrate that our method works well regardless of the backbone models, we now conduct experiments on the LlaMA-2 13B model and Gemma 2B models. The other experimental settings are kept the same with the main experiments (Table [1\)](#page-5-0). We conduct experiments on

<span id="page-15-0"></span>

| BoolQ<br>(acc)          | PIQA<br>(acc) | MMLU<br>(acc) |  |  |  |  |
|-------------------------|---------------|---------------|--|--|--|--|
| Results for LlaMA-2 13B |               |               |  |  |  |  |
| 73.5                    | 85.8          | 50.5          |  |  |  |  |
| 74.9                    | 86.6          | 51.2          |  |  |  |  |
| Results for Gemma 2B    |               |               |  |  |  |  |
| 62.3                    | 79.4          | 39.8          |  |  |  |  |
| 63.9                    | 80.3          | 40.7          |  |  |  |  |
|                         |               |               |  |  |  |  |

Table 7: Results for different PEFT methods on the BoolQ, PIQA and MMLU benchmarks. The backbone LMs are LlaMA-2 13B, an Gemma 2B.

the BoolQ, PIQA and MMLU tasks. The results are reported in Table [7.](#page-15-0)
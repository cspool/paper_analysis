# <span id="page-17-1"></span>E EFFICIENCY ANALYSIS

Table 4: Efficiency evaluation on Haystack-Ego4D. Baseline results are directly cited from [Ye et al.](#page-11-2) [\(2025\)](#page-11-2). We report the overall latency of temporal search and answering. Evaluations are conducted on the Haystack-Ego4D using A100 GPUs. Temporal search metrics are reported in Tab. [1.](#page-5-0)

| Method                     | Question Grounding | Frame Retrieval | Latency (sec) ↓ |
|----------------------------|--------------------|-----------------|-----------------|
| VideoAgent                 | GPT4               | CLIP-1B         | 34.9            |
| Retrieval-based            | –                  | YOLO-world-110M | 32.2            |
| ∗<br>T<br>(Detector-based) | LLaVA-OV-7B        | YOLO-world-110M | 11.1            |
| TimeSearch-R               | –                  | SigLIP-400M     | 13.4            |

TimeSearch-R attains an end-to-end latency of 13.4 seconds on the Haystack-Ego4D test set, yielding a 61.6% speed-up over the 34.9-second latency of VideoAgent. Despite T<sup>∗</sup> operating with the lightweight YOLO-World-110M detector and completing inference in 11.1 seconds, our method maintains a comparable runtime while avoiding the complexity of hand-crafted scheduling. As shown in Tab. [1,](#page-5-0) TimeSearch-R markedly surpasses these baselines in temporal search metrics and QA accuracy, underscoring the effectiveness of reinforcement-driven temporal policies.


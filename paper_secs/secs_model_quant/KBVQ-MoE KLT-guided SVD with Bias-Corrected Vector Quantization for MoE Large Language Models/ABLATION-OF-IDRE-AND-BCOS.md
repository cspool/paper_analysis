# ABLATION OF IDRE AND BCOS

To further clarify the contribution of each component within the IDRE and BCOS modules, we conduct fine-grained, component-wise ablations at 2-bit on Qwen1.5-MoE-A2.7B. This directly addresses the reviewer's concern regarding the lack of quantitative analysis for the internal steps of each module.

(a) Effect of the KLT step in IDRE. IDRE consists of (i) an input-driven KLT transform and (ii) a subsequent SVD on the stacked expert weights in the KLT space. To isolate the effect of the KLT step, we compare the full IDRE against a variant that performs SVD directly on the original stacked weights (without KLT alignment). The results at 2-bit are summarized in Tab.14.

Table 14: Ablation of KLT in IDRE on Qwen1.5-MoE-A2.7B (2-bit).

<span id="page-20-0"></span>

| Model             | IDRE    | W2    | HE    | WI    |
|-------------------|---------|-------|-------|-------|
| Qwen1.5-MoE-A2.7B | w/o KLT | 11.77 | 35.89 | 63.01 |
|                   | KLT     | 9.61  | 39.59 | 65.11 |

(b) Effect of mean and variance correction in BCOS. BCOS performs channel-wise bias correction using both the mean and variance of the channel outputs. To disentangle their respective roles, we evaluate four variants: (i) no correction, (ii) mean-only, (iii) variance-only, and (iv) full mean+variance correction. The 2-bit results on Qwen1.5-MoE-A2.7B are given in Tab.15.

<span id="page-20-1"></span>Table 15: Ablation of mean and variance correction in BCOS on Qwen1.5-MoE-A2.7B (2-bit).

| Model             | Mean | Variance     | W2    | HE    | WI             |
|-------------------|------|--------------|-------|-------|----------------|
| Qwen1.5-MoE-A2.7B | ×    | ×            | 11.03 | 35.18 | 63.49          |
|                   | ✓    | ×            | 11.01 | 35.35 | 63.49<br>63.40 |
|                   | ×    | $\checkmark$ | 10.38 | 36.70 | 64.23          |
|                   | ✓    | ✓            | 9.61  | 39.59 | 65.11          |

#### A.10 COMPARISON OF MOE COMPRESSION METHODS

The core designs of D2-MoE, SubMoE, and EAC-MoE all revolve around "expert layer structure optimization", which belongs to a different technical branch of MoE compression compared to KBVQ-MoE's "weight quantization". The specific differences can be summarized as follows:

- SubMoE: Merges similar experts through subspace clustering and extracts cross-expert shared low-rank features via SVD to reduce the number of experts. Essentially, it is a structural compression method combining "expert-level pruning + low-rank decomposition". The compression ratio is mainly determined by the expert merging ratio (e.g., 50% means merging half of the experts).
- D2-MoE: Decomposes expert weights into a "shared base matrix + expert-specific delta matrix" and performs low-rank approximation on the delta matrix to reduce the number of parameters. Essentially, it is a structural compression method combining "weight decomposition + low-rank approximation". The compression ratio is determined by the rank truncation ratio of the delta matrix (e.g., retaining 60% of the rank).
- EAC-MoE: Starts from "expert selection behavior" and jointly models two operations: quantization and pruning. It proposes QESC (Quantization with Expert-Selection Calibration), which explicitly calibrates the router during low-bit quantization to mitigate expert selection bias caused by quantization noise. It also proposes PESF (Pruning based on Expert-Selection Frequency), which prunes "cold experts" using their actual selection frequency to further improve inference speed and reduce memory usage while minimizing accuracy loss. Thus, EAC-MoE can be regarded as a "routing-aware quantization + expert pruning hybrid compression scheme".
- KBVQ-MoE (ours): Does not change the number or structure of experts. First, it extracts cross-expert shared low-rank subspaces in the input KLT space via IDRE to explicitly remove redundancy, then performs vector quantization on expert-specific residuals. Essentially, it is an "MoE structure-aware weight quantization method". The compression ratio is determined

by the quantization bit-width (2–3 bits) and codebook reuse efficiency, achieving 80%–90% memory compression under the same structural constraints.

Overall, D2-MoE/SubMoE are more inclined to directly reduce the number of parameters (removing experts / reducing rank), EAC-MoE follows a hybrid route of structural pruning + quantization, while KBVQ-MoE focuses on achieving high compression ratios through structure-aware low-bit quantization while preserving the original expert structure. Different routes may exhibit distinct trade-offs in terms of "expert specificity preservation", "inference speed", and "implementation complexity", which need to be characterized through empirical comparisons.

Fair Comparison Experiments: Settings and Results. To ensure fairness, we uniformly use Mixtral-8×7B as the baseline model (retaining the original expert structure: 8 experts, 32 layers) and adopt identical experimental settings:

- Evaluation tasks: WikiText2 (language modeling, metric: perplexity (PPL), lower is better), ARC-Challenge (ARC\_C), ARC-Easy (ARC\_E), WinoGrande (WinG) (all zero-shot inference, metric: accuracy, higher is better);
- Compression ratio definition: Memory saving is uniformly calculated as "the ratio of compressed parameter storage to the original FP16 model" to avoid misunderstandings caused by inconsistent definitions of "compression ratio" across different methods;
- Experimental environment: NVIDIA RTX A100 GPU, PyTorch 2.1, evaluation tool: LM-Evaluation-Harness (v0.4.0).

The comparison results are shown in the Tab[.16](#page-21-0) (Note: ARC\_C, ARC\_E, and WinG values are accuracy, retained to two decimal places for precision):

<span id="page-21-0"></span>

| Method          | Memory saving | WikiText2 (PPL) | ARC_C | ARC_E | WinG |
|-----------------|---------------|-----------------|-------|-------|------|
| Sub-MoE         | 50%           | 6.97            | 0.45  | 0.75  | 0.72 |
| D2-MoE          | 60%           | 6.46            | 0.38  | 0.72  | 0.71 |
| EAC-MoE         | 84%           | 4.58            | 0.55  | 0.81  | 0.75 |
| KBVQ-MoE (ours) | 87%           | 4.07            | 0.63  | 0.85  | 0.76 |

Table 16: Ablation KBVQ—MoE and other MoE compression methods

In the experiments on Mixtral-8×7B, significant differences exist among the three compression approaches: KBVQ-MoE (quantization compression) achieves an 87% memory saving, which is significantly higher than that of Sub-MoE (50%) and D2-MoE (60%), and slightly higher than that of EAC-MoE (84%). Moreover, it performs optimally across all tasks: its Perplexity (PPL) on WikiText2 is 2.39-2.90 lower than that of Sub-MoE/D2-MoE and 0.51 lower than that of EAC-MoE; its accuracy on ARC\_C is 0.18-0.25 higher than that of Sub-MoE/D2-MoE and 0.08 higher than that of EAC-MoE. These results confirm that KBVQ-MoE can well retain the expressive ability of MoE experts under a high compression ratio (>80%).

EAC-MoE (hybrid compression), which combines routing calibration and expert pruning based on selection frequency, outperforms Sub-MoE and D2-MoE (structural compression) in performance. However, KBVQ-MoE still shows better performance under a similar compression ratio and does not require modifying the expert structure. Additionally, KBVQ-MoE preserves the original expert topology, achieving an accuracy of 0.63 on complex reasoning tasks (e.g., ARC\_C)—far exceeding the 0.38-0.45 of Sub-MoE/D2-MoE and the 0.55 of EAC-MoE. It also features greater flexibility in deployment and migration: it can be directly adapted to various MoE LLMs (e.g., Qwen, DeepSeek) with only one round of offline calibration and codebook training, demonstrating prominent comprehensive advantages.


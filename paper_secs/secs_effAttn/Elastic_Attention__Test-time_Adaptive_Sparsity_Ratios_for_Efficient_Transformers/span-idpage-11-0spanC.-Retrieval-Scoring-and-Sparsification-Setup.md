# <span id="page-11-0"></span>C. Retrieval Scoring and Sparsification Setup

In this section, we provide a detailed description of the retrieval head identification and the progressive sparsification strategy mentioned in Section [2.2.](#page-2-0)

### C.1. Retrieval Score Calculation

Following the methodology proposed by Retrieval Head [\(Wu et al.,](#page-9-7) [2024\)](#page-9-7), we identify and rank attention heads based on their ability to retrieve specific information from long contexts. We employ the Needle-in-a-Haystack probing method using the Llama-3.1-8B-Instruct [\(Grattafiori et al.,](#page-8-4) [2024\)](#page-8-4) model, where specific key information (the "needle") is inserted into a

*Table 5.* Performance retention rates across various model sparsity ratios. The values represent the percentage of performance relative to the Full Attention baseline (Sparsity 0.0), where 100.00 indicates parity. Rows denote the model sparsity ratio, and columns denote the evaluation tasks.

| Sparsity                         | Single-Doc QA | Multi-Hop QA | Summarization | Few-Shot | Synthetic | Code   |
|----------------------------------|---------------|--------------|---------------|----------|-----------|--------|
| Full ( $\Omega_{\rm MSR}$ = 0.0) | 100.00        | 100.00       | 100.00        | 100.00   | 100.00    | 100.00 |
| $\Omega_{\rm MSR} = 0.1$         | 85.37         | 96.23        | 99.38         | 96.54    | 92.27     | 99.07  |
| $\Omega_{\mathrm{MSR}} = 0.2$    | 71.32         | 73.84        | 98.00         | 93.83    | 81.26     | 98.49  |
| $\Omega_{\mathrm{MSR}} = 0.3$    | 61.94         | 69.07        | 96.19         | 91.83    | 59.28     | 98.10  |
| $\Omega_{\mathrm{MSR}} = 0.4$    | 59.36         | 66.48        | 94.75         | 87.01    | 47.61     | 95.58  |
| $\Omega_{\mathrm{MSR}} = 0.5$    | 57.95         | 64.33        | 93.02         | 87.33    | 45.31     | 99.54  |
| $\Omega_{\mathrm{MSR}} = 0.6$    | 56.95         | 59.66        | 94.68         | 87.05    | 44.75     | 98.66  |
| $\Omega_{\mathrm{MSR}} = 0.7$    | 56.08         | 61.91        | 92.52         | 88.51    | 41.25     | 97.73  |
| $\Omega_{\mathrm{MSR}} = 0.8$    | 56.04         | 59.55        | 93.73         | 89.07    | 47.98     | 99.86  |
| $\Omega_{\mathrm{MSR}} = 0.9$    | 56.02         | 63.40        | 94.00         | 90.19    | 48.37     | 98.48  |
| $\Omega_{\mathrm{MSR}} = 1.0$    | 56.46         | 61.45        | 93.42         | 90.19    | 47.98     | 99.46  |

long context (the "haystack").

For a given attention head h in layer  $\ell$ , denoted as  $H_{\ell,h}$ , we calculate its Retrieval Score  $(S_{\ell,h})$  by measuring the attention mass allocated to the needle tokens. Formally, let s be the input sequence length,  $\mathcal{O}^{(\ell,h)} \in \mathbb{R}^{s \times s}$  be the attention matrix, and  $\mathcal{I}_{needle}$  be the set of indices corresponding to the needle tokens. The score is computed as:

$$S_{\ell,h} = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \sum_{j \in \mathcal{I}_{needle}} \mathcal{O}_{T,j}^{(\ell,h)} \tag{9}$$

where  $\mathcal{D}$  represents the validation dataset, and  $\mathcal{O}_{T,j}^{(\ell,h)}$  denotes the attention weight from the last token (query position T) to the needle position j. A higher  $S_{\ell,h}$  indicates that the head frequently and strongly activates on relevant information in long contexts.

## C.2. Progressive Sparsification Strategy

Based on the computed scores  $S_{\ell,h}$ , we rank all  $L \cdot H$  attention heads across the model in descending order. As defined in the main text, the **Model Sparsity Ratio** ( $\Omega_{\rm MSR}$ ) represents the proportion of heads converted to local attention.

To simulate the varying levels of sparsity reported in our experiments (e.g.,  $\Omega_{MSR}=20\%$ ), we employ a thresholding mechanism:

- 1. We determine the number of heads to preserve as Full Attention (FA) via  $k = |(1 \Omega_{MSR}) \cdot (L \cdot H)|$ .
- 2. The top-k heads with the highest retrieval scores are retained as **Retrieval Heads** (keeping FA) to ensure global information integration.
- 3. The remaining heads (the bottom ranked ones) are replaced with **Streaming Sparse Attention (SSA)** heads.

## <span id="page-12-0"></span>**D. Implementation Details**

In this section, we provide a comprehensive overview of the training configurations, baseline implementation details, and system-level optimizations tailored for efficient long-context processing.

## D.1. Training Configuration and Hyperparameters

**Model Architecture.** We validate the scalability of our approach across diverse model sizes, ranging from lightweight architectures like Qwen3-4B to widely-adopted mid-class models such as Qwen3-8B (Yang et al., 2025) and Meta-Llama-3.1-Instruct (Grattafiori et al., 2024). To preserve the pre-trained model's general capabilities, we adhere to a parameter-efficient fine-tuning protocol: the pre-trained backbone is frozen, and only the *Attention Router* parameters are optimized. Regarding task representation, we employ a boundary-pooling strategy that exclusively aggregates the first 100 and last 100 tokens

Table 6. Hyperparameters: General configuration (Left) and Baseline-specific settings (Right).

(a) General Config

(b) Baseline-Specifics

<span id="page-13-0"></span>

| Hyperparameter                                                                  | Value                                                                             | Param                                                  | MoBA                | NSA                          | InfLLMv2                     |  |  |  |  |
|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------|---------------------|------------------------------|------------------------------|--|--|--|--|
| Model & Train                                                                   | ing                                                                               | Structure & Kernel                                     |                     |                              |                              |  |  |  |  |
| Base Model<br>Sequence length<br>Precision<br>Global Batch Size                 | Qwen, Llama<br>65536<br>bfloat16<br>48                                            | Block Size Top-k Window Size Kernel Size Kernel Stride | 1024<br>8<br>-<br>- | 64<br>128<br>512<br>32<br>16 | 64<br>64<br>2048<br>32<br>16 |  |  |  |  |
| Training Steps Mask / Reg. LR Warmup Ratio AdamW Momentum $(\beta_1, \beta_2)$  | $ \begin{array}{c c} 300 \\ 5e^{-4} / 1e^{-3} \\ 0.2 \\ (0.9, 0.95) \end{array} $ | Q/K/V Proj<br>Gate                                     | Params              | 10                           |                              |  |  |  |  |
| Weight Decay<br>Learning Rate Schedule                                          | 0.1<br>Cosine                                                                     | Compress K/V                                           | Extra Co            | nfia                         | -                            |  |  |  |  |
| Sparsity Confi                                                                  | g                                                                                 | Compress Type                                          |                     |                              | Dooling                      |  |  |  |  |
| Sink / Local Size<br>Block / Chunk Size<br>Stride / Threshold<br>Selection Mode | 128/2048<br>64/16384<br>16/0.9<br>Inverse                                         | Compress Type Use NoPE Dense Len                       | Pooling<br>-<br>-   | Linear<br>-<br>-             | Pooling<br>False<br>8192     |  |  |  |  |

of the sequence. These segments are selected as they typically encapsulate critical system instructions and user queries essential for accurate task identification.

**Optimization Setup.** All models are trained with a sequence length of L=65,536 tokens using 'bfloat16' precision to accommodate long-context dependencies. The training utilizes the AdamW optimizer (Loshchilov & Hutter, 2019)  $(\beta_1=0.9,\beta_2=0.95)$  on a distributed cluster employing Fully Sharded Data Parallel (FSDP) with a Hybrid Sharding strategy. We adopt a decoupled learning rate strategy to balance router convergence and sparsity regularization:

- Router Parameters: A learning rate of  $5 \times 10^{-4}$  is applied to the attention router to facilitate rapid adaptation to retrieval patterns.
- Regularization Coefficients: A higher learning rate of  $1 \times 10^{-3}$  is assigned to the sparsity regularization terms. Specifically, the dual regularization coefficients  $\lambda_1$  and  $\lambda_2$  are randomly initialized and optimized alongside the router parameters.

We utilize a cosine decay learning rate schedule following a linear warmup phase spanning the first 20% of total training steps.

## <span id="page-13-1"></span>**D.2. Baseline Implementation Details**

To rigorously evaluate the efficacy of our approach, we benchmark against a comprehensive suite of state-of-the-art sparse attention mechanisms. These are categorized into training-free methods and training-based adaptation methods.

**Training-Free Methods.** We employ XAttention <sup>5</sup> (Xu et al., 2025) as the primary training-free baseline. This category relies on heuristic-based sparsity without parameter updates.

<span id="page-13-2"></span><sup>5</sup>https://github.com/mit-han-lab/x-attention

<span id="page-14-6"></span>Algorithm 1 Comparison of Serial Dispatch (Baseline) vs. Parallel BSA (Ours)

```
(a) PyTorch Baseline
Input: Q, K, V, Router R
 1: r ← R(x)
 2:
    Step 1: Serial Split
    Ifull ← {h | rh = 0}
    Isp ← {h | rh = 1}
    Qfull ← Q[:, Ifull]
    Step 2: Separate Comp.
    Ofull ← FlashAttn(. . .)
    Osp ← SlidingWin(. . .)
    Step 3: Merge Results
    O[:, Ifull] ← Ofull
    O[:, Isp] ← Osp
                                                                          (b) Ours (Parallel via BSA)
                                                            Input: Q, K, V, Router R
                                                             1: r ← R(x)
                                                             2: m ← Map(r)
                                                             3:
                                                                Step 1: Unified Execution
                                                                O ← BSA Krn(Q, K, V, m)
                                                                Inside Kernel:
                                                                par for h do
                                                                   if m[h]==SP then
                                                                      O[h] ← Sparse(. . .);
                                                                   else
                                                                      O[h] ← Full(. . .);
                                                                   end if
                                                                end par for
```

Training-Based Methods. For methods requiring training adaptation, including InfLLM v2 [6](#page-14-1) [\(Zhao et al.,](#page-10-6) [2025\)](#page-10-6), MoBA [7](#page-14-2) [\(Lu et al.,](#page-9-12) [2025a\)](#page-9-12), NSA [8](#page-14-3) [\(Yuan et al.,](#page-10-8) [2025\)](#page-10-8), PruLong [9](#page-14-4) [\(Bhaskar et al.,](#page-8-6) [2025\)](#page-8-6), and DuoAttention [10](#page-14-5) [\(Xiao](#page-10-2) [et al.,](#page-10-2) [2025\)](#page-10-2), we implement a unified fine-tuning protocol to ensure strict fairness. Unlike our method, which freezes the backbone entirely, most competing methods require updating projection layers. For InfLLM v2, MoBA, and NSA, we restrict the trainable scope to the query-key-value projection weights (Wqkv) and their respective method-specific parameters, freezing the remaining backbone. All baselines are trained under the same environment and dataset, and the hyperparameters are strictly adhered to their original setups, such as a block size of 1024 for MoBA versus 64 for NSA, and the specific dense context length (8K) for InfLLM v2. Detailed hyperparameter comparisons are provided in Table [6](#page-13-0) (Right).

### D.3. Sparsity and Kernel Configuration

To achieve efficient streaming inference, we employ Block-Sparse-Attention [\(Guo et al.,](#page-8-5) [2024\)](#page-8-5). This configuration governs the granularity and retention policy of the attention mechanism:

- Block Size: Set to 64, defining the minimum unit of sparsity.
- Chunk Size: Set to 16,384, enabling the processing of ultra-long sequences.
- Sink Token Strategy: We enforce a "sink token" size of 128 to preserve the attention sink phenomenon, ensuring stability during streaming generation.

Specific kernel parameters, including stride, normalization, and selection modes, are detailed in the *Sparsity Config* section of Table [6](#page-13-0) (Left).


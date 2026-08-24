# <span id="page-13-0"></span>B Mathematical Description of Compression

In this section, we provide a detailed formulation of the compression operation introduced in Section 3.2.

**Notation.** During compression, the context can be divided into three segments: 1. The sequence that remains in the context without being compressed, denoted as  $Pre := \{X, \{C^{(1)}, [o]^{(1)}, ..., C^{(i-1)}, [o]^{i-1}\}\}$ , with the number of tokens represented by N; 2. The thought sequence to be compressed, defined as  $Tho := S_i$ , with the number of tokens denoted by T; 3. The sequence storing the compressed content,  $C := C^{(i)}$ , with its length represented by |C|.

**Compression Operation.** Here, we describe the compression operation at a specific layer, focusing on the information passed to the sequence C. According to the definition of self-attention (Vaswani et al., 2017), the attention matrix for the sequence C with respect to other content is calculated as:

$$A = \mathrm{Softmax}(\mathrm{mask}(\frac{Q^C[K^{Pre}:K^{Tho}:K^C]^\top}{\sqrt{d}}))$$

where [:] denotes the concatenation operation,  $\operatorname{mask}(\cdot)$  represents the attention mask corresponding to the "Thought-based Attention Mask Construction" in Section 3.2,  $K^{Pre}, V^{Pre} \in \mathbb{R}^{N \times d}$ ,

 $K^{Tho}, V^{Tho} \in \mathbb{R}^{T \times d}, K^C, V^C \in \mathbb{R}^{|C| \times d}, Q^C \in \mathbb{R}^{|C| \times d}$ , and d is the hidden dimension. The matrix  $A \in \mathbb{R}^{|C| \times (N+T+|C|)}$  describes the attention of sequence C to other content. The *values* of the other sequences are then weighted and summed according to the attention matrix:

$$H = A \times [V^{Pre} : V^{Tho} : V^C]$$

where  $[V^{Pre}:V^{Tho}:V^C]\in\mathbb{R}^{(N+T+|C|)\times d}$ , and thus  $H\in\mathbb{R}^{|C|\times d}$ . At this point, the information from the current Tho is preserved in H. Through training, the model learns to selectively retain useful information from Tho in H. H is then stored in the KV Cache after passing through an MLP and the next layer's projection.

#### <span id="page-13-2"></span>**C** Experiment

#### C.1 Training Data

Examples of training samples are shown in Figure 15.

#### <span id="page-13-1"></span>**C.2** Baseline Details

H2O (Zhang et al., 2023) is a training-free acceleration method that greedily retains tokens with the highest cumulative attention values from historical tokens. It includes two hyper-parameters: the maximum number of tokens and the current window size (i.e., local\_size). The maximum number of tokens for each task is listed in the "Peak" column of Table 1, and the local\_size is set to half of the maximum number of tokens. The experimental code is implemented based on https://github.com/meta-llama/llama-cookbook.

**SepLLM** (Chen et al., 2024) is another training-free acceleration method that considers tokens at punctuation positions as more important. It includes four parameters: the maximum number of tokens is set to 1024, local\_size is set to 256, sep\_cache\_size is set to 64, and init\_cache\_size is set to 384. We also tried another set of parameters (init\_cache\_size=4, sep\_cache\_size=64, local\_size=720, maximum number of tokens=1024), but found that the first set of parameters performed slightly better.

**AnLLM** (Pang et al., 2024) is a training-based method that shares a similar overall approach with LightThinker but accelerates by saving historical content in anchor tokens. The specific differences between the two are detailed in Section E.

## C.3 Training Details

Both Vanilla and AnLLM are trained on the B17K [\(Labs,](#page-10-17) [2025\)](#page-10-17) dataset using the R1- Distill [\(DeepSeek-AI et al.,](#page-8-1) [2025\)](#page-8-1) model for 5 epochs, while LightThinker is trained for 6 epochs. The maximum length is set to 4096, and a cosine warmup strategy is adopted with a warmup\_ratio of 0.05. Experiments are conducted on 4 A800 GPUs with DeepSpeed ZeRo3 offload enabled. The batch size per GPU is set to 5, and the gradient accumulation step is set to 4, resulting in a global batch size of 80. The learning rate for Vanilla is set to 1e-5, while for AnLLM and LightThinker, it is set to 2e-5.


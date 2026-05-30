# 1 Introduction

The pursuit of artificial general intelligence (AGI) has driven the development of large language models (LLMs) to unprecedented scales, with the promise of handling complex tasks that mimic human cognition. A pivotal capability for achieving AGI is the ability to process, understand, and generate long sequences, which is essential for a wide range of applications, from historical data analysis to complex reasoning and decision-making processes. This growing demand for extended context processing can be seen not only in the popularity of long input prompt understanding, as showcased by models like Kimi (MoonshotAI [2023\)](#page-11-0), Claude (Anthropic [2023\)](#page-10-0) and Gemini (Reid et al. [2024\)](#page-12-0), but also in recent explorations of long chain-of-thought (CoT) output capabilities in Kimi k1.5 (Team et al. [2025\)](#page-12-1), DeepSeek-R1 (D. Guo et al. [2025\)](#page-11-1), and OpenAI o1/o3 (Guan et al. [2024\)](#page-11-2).

However, extending the sequence length in LLMs is non-trivial due to the quadratic growth in computational complexity associated with the vanilla attention mechanism (Waswani et al. [2017\)](#page-12-2). This challenge has spurred a wave of research aimed at improving efficiency without sacrificing performance. One prominent direction capitalizes on the inherent sparsity of attention scores. This sparsity arises both mathematically — from the softmax operation, where various sparse attention patterns have be studied (H. Jiang et al. [2024\)](#page-11-3) — and biologically (Watson et al. [2025\)](#page-12-3), where sparse connectivity is observed in brain regions related to memory storage.

<sup>∗∗</sup>zhang mingxing@mail.tsinghua.edu.cn

<sup>†‡</sup>Co-corresponding authors. Xinyu Zhou (zhouxinyu@moonshot.cn), Jiezhong Qiu (jiezhongqiu@outlook.com)

Existing approaches often leverage predefined structural constraints, such as sink-based (G. Xiao et al. 2023) or sliding window attention (Beltagy et al. 2020), to exploit this sparsity. While these methods can be effective, they tend to be highly task-specific, potentially hindering the model's overall generalizability. Alternatively, a range of dynamic sparse attention mechanisms, exemplified by Quest (Tang et al. 2024), Minference (H. Jiang et al. 2024), and RetrievalAttention (Di Liu et al. 2024), select subsets of tokens at inference time. Although such methods can reduce computation for long sequences, they do not substantially alleviate the intensive training costs of long-context models, making it challenging to scale LLMs efficiently to contexts on the order of millions of tokens. Another promising alternative way has recently emerged in the form of linear attention models, such as Mamba (Dao and Gu 2024), RWKV (Peng, Alcaide, et al. 2023; Peng, Goldstein, et al. 2024), and RetNet (Sun et al. 2023). These approaches replace canonical softmax-based attention with linear approximations, thereby reducing the computational overhead for long-sequence processing. However, due to the substantial differences between linear and conventional attention, adapting existing Transformer models typically incurs high conversion costs (Mercat et al. 2024; J. Wang et al. 2024; Bick et al. 2025; M. Zhang et al. 2024) or requires training entirely new models from scratch (A. Li et al. 2025). More importantly, evidence of their effectiveness in complex reasoning tasks remains limited.

Consequently, a critical research question arises: How can we design a robust and adaptable attention architecture that retains the original Transformer framework while **adhering to a "less structure" principle, allowing the model to determine where to attend without relying on predefined biases?** Ideally, such an architecture would transition seamlessly between full and sparse attention modes, thus maximizing compatibility with existing pre-trained models and enabling both efficient inference and accelerated training without compromising performance.

Thus, we introduce Mixture of Block Attention (MoBA), a novel architecture that builds upon the innovative principles of Mixture of Experts (MoE) (Shazeer et al. 2017) and applies them to the attention mechanism of the Transformer model. MoE has been used primarily in the feedforward network (FFN) layers of Transformers (Lepikhin et al. 2020; Fedus et al. 2022; Zoph et al. 2022), but MoBA pioneers its application to long context attention, allowing dynamic selection of historically relevant blocks of key and values for each query token. This approach not only enhances the efficiency of LLMs but also enables them to handle longer and more complex prompts without a proportional increase in resource consumption. MoBA addresses the computational inefficiency of traditional attention mechanisms by partitioning the context into blocks and employing a gating mechanism to selectively route query tokens to the most relevant blocks. This block sparse attention significantly reduces the computational costs, paving the way for more efficient processing of long sequences. The model's ability to dynamically select the most informative blocks of keys leads to improved performance and efficiency, particularly beneficial for tasks involving extensive contextual information.

In this paper, we detail the architecture of MoBA, firstly its block partitioning and routing strategy, and secondly its computational efficiency compared to traditional attention mechanisms. We further present experimental results that demonstrate MoBA's superior performance in tasks requiring the processing of long sequences. Our work contributes a novel approach to efficient attention computation, pushing the boundaries of what is achievable with LLMs in handling complex and lengthy inputs.

## <span id="page-1-0"></span>2 Method

In this work, we introduce a novel architecture, termed Mixture of Block Attention (MoBA), which extends the capabilities of the Transformer model by dynamically selecting historical segments (blocks) for attention computation. MoBA is inspired by techniques of Mixture of Experts (MoE) and sparse attention. The former technique has been predominantly applied to the feedforward network (FFN) layers within the Transformer architecture, while the latter has been widely adopted in scaling Transformers to handle long contexts. Our method is innovative in applying the MoE principle to the attention mechanism itself, allowing for more efficient and effective processing of long sequences.

### 2.1 Preliminaries: Standard Attention in Transformer

We first revisit the standard Attention in Transformers. For simplicity, we revisit the case where a single query token  $q \in \mathbb{R}^{1 \times d}$  attends to the N key and value tokens, denoting  $K, V \in \mathbb{R}^{N \times d}$ , respectively. The standard attention is computed as:

$$Attn(q, K, V) = Softmax(qK^{\top})V,$$
(1)

where d denotes the dimension of a single attention head. We focus on the single-head scenario for clarity. The extension to multi-head attention involves concatenating the outputs from multiple such single-head attention operations.

<span id="page-2-1"></span>Figure 1: Illustration of mixture of block attention (MoBA). (a) A running example of MoBA; (b) Integration of MoBA into Flash Attention.

### 2.2 MoBA Architecture

Different from standard attention where each query tokens attend to the entire context, MoBA enables each query token to only attend to a subset of keys and values:

<span id="page-2-2"></span>
$$MoBA(q, K, V) = Softmax(qK[I]^{\top})V[I],$$
(2)

where I ⊆ [N] is the set of selected keys and values.

The key innovation in MoBA is the block partitioning and selection strategy. We divide the full context of length N into n blocks, where each block represents a subset of subsequent tokens. Without loss of generality, we assume that the context length N is divisible by the number of blocks n. We further denote B = N n to be the block size and

$$I_i = [(i-1) \times B + 1, i \times B] \tag{3}$$

to be the range of the i-th block. By applying the top-k gating mechanism from MoE, we enable each query to selectively focus on a subset of tokens from different blocks, rather than the entire context:

<span id="page-2-0"></span>
$$I = \bigcup_{g_i > 0} I_i. \tag{4}$$

The model employs a gating mechanism, as g<sup>i</sup> in Equation [4,](#page-2-0) to select the most relevant blocks for each query token. The MoBA gate first computes the affinity score s<sup>i</sup> measuring the relevance between query q and the i-th block, and applies a top-k gating among all blocks. More formally, the gate value for the i-th block g<sup>i</sup> is computed by

$$g_i = \begin{cases} 1 & s_i \in \text{Topk}\left(\{s_j | j \in [n]\}, k\right) \\ 0 & \text{otherwise} \end{cases}, \tag{5}$$

where Topk(·, k) denotes the set containing k highest scores among the affinity scores calculated for each block. In this work, the score s<sup>i</sup> is computed by the inner product between q and the mean pooling of K[I<sup>i</sup> ] along the sequence dimension:

<span id="page-2-3"></span>
$$s_i = \langle \boldsymbol{q}, \text{mean\_pool}(\boldsymbol{K}[I_i]) \rangle$$
 (6)

A Running Example. We provide a running example of MoBA at Figure [1a,](#page-2-1) where we have two query tokens and four KV blocks. The router (gating network) dynamically selects the top two blocks for each query to attend. As shown in Figure [1a,](#page-2-1) the first query is assigned to the first and second blocks, while the second query is assigned to the third and fourth blocks.

It is important to maintain causality in autoregressive language models, as they generate text by next-token prediction based on previous tokens. This sequential generation process ensures that a token cannot influence tokens that come before it, thus preserving the causal relationship. MoBA preserves causality through two specific designs:

Causality: No Attention to Future Blocks. MoBA ensures that a query token cannot be routed to any future blocks. By limiting the attention scope to current and past blocks, MoBA adheres to the autoregressive nature of language modeling. More formally, denoting pos(q) as the position index of the query q, we set s<sup>i</sup> = −∞ and g<sup>i</sup> = 0 for any blocks i such that pos(q) < i × B.

Current Block Attention and Causal Masking. We define the "current block" as the block that contains the query token itself. The routing to the current block could also violate causality, since mean pooling across the entire block can inadvertently include information from future tokens. To address this, we enforce that each token must be routed to its respective current block and apply a causal mask during the current block attention. This strategy not only avoids any leakage of information from subsequent tokens but also encourages attention to the local context. More formally, we set g<sup>i</sup> = 1 for the block i where the position of the query token pos(q) is within the interval I<sup>i</sup> . From the perspective of Mixture-of-Experts (MoE), the current block attention in MoBA is akin to the role of shared experts in modern MoE architectures (Dai et al. [2024;](#page-10-3) A. Yang et al. [2024\)](#page-12-11), where static routing rules are added when expert selection.

Next, we discuss some additional key design choices of MoBA, such as its block segmentation strategy and the hybrid of MoBA and full attention.

Fine-Grained Block Segmentation. The positive impact of fine-grained expert segmentation in improving mode performance has been well-documented in the Mixture-of-Experts (MoE) literature (Dai et al. [2024;](#page-10-3) A. Yang et al. [2024\)](#page-12-11). In this work, we explore the potential advantage of applying a similar fine-grained segmentation technique to MoBA. MoBA, inspired by MoE, operates segmentation along the context-length dimension rather than the FFN intermediate hidden dimension. Therefore our investigation aims to determine if MoBA can also benefit when we partition the context into blocks with a finer grain. More experimental results can be found in Section [3.1.](#page-5-0)

Hybrid of MoBA and Full Attention. MoBA is designed to be a substitute for full attention, maintaining the same number of parameters without any addition or subtraction. This feature inspires us to conduct smooth transitions between full attention and MoBA. Specifically, at the initialization stage, each attention layer has the option to select full attention or MoBA, and this choice can be dynamically altered during training if necessary. A similar idea of transitioning full attention to sliding window attention has been studied in previous work (X. Zhang et al. [2024\)](#page-12-12). More experimental results can be found in Section [3.2.](#page-6-0)

Comparing to Sliding Window Attention and Attention Sink. Sliding window attention (SWA) and attention sink are two popular sparse attention architectures. We demonstrate that both can be viewed as special cases of MoBA. For sliding window attention (Beltagy et al. [2020\)](#page-10-1), each query token only attends to its neighboring tokens. This can be interpreted as a variant of MoBA with a gating network that keeps selecting the most recent blocks. Similarly, attention sink (G. Xiao et al. [2023\)](#page-12-4), where each query token attends to a combination of initial tokens and the most recent tokens, can be seen as a variant of MoBA with a gating network that always selects both the initial and the recent blocks. The above discussion shows that MoBA has stronger expressive power than sliding window attention and attention sink. Moreover, it shows that MoBA can flexibly approximate many static sparse attention architectures by incorporating specific gating networks.

Overall, MoBA's attention mechanism allows the model to adaptively and dynamically focus on the most informative blocks of the context. This is particularly beneficial for tasks involving long documents or sequences, where attending to the entire context may be unnecessary and computationally expensive. MoBA's ability to selectively attend to relevant blocks enables more nuanced and efficient processing of information.

### <span id="page-3-0"></span>2.3 Implementation

We provide a high-performance implementation of MoBA, by incorporating optimization techniques from FlashAttention (Dao, D. Fu, et al. [2022\)](#page-11-12) and MoE (Rajbhandari et al. [2022\)](#page-11-13). Figure [2](#page-4-0) demonstrates the high efficiency of MoBA, while we defer the detailed experiments on efficiency and scalability to Section [3.4.](#page-9-0) Our implementation consists of five major steps:

- Determine the assignment of query tokens to KV blocks according to the gating network and causal mask.
- Arrange the ordering of query tokens based on their assigned KV blocks.
- Compute attention outputs for each KV block and the query tokens assigned to it. This step can be optimized by FlashAttention with varying lengths.


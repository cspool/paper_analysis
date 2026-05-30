# 1 Introduction

Large language models (LLMs) have become ubiquitous, revolutionizing both academic research and industrial applications [Brown et al.](#page-10-0) [\(2020\)](#page-10-0); [OpenAI](#page-12-0) [\(2023\)](#page-12-0); [Chowdhery et al.](#page-10-1) [\(2022\)](#page-10-1); [Liu et al.](#page-11-0) [\(2024a\)](#page-11-0); [Guo et al.](#page-11-1) [\(2025\)](#page-11-1). Their success largely stems from pre-training and instruction tuning on vast amounts of data, the power of self-attention in Transformer architectures, and the computational capabilities of GPU accelerators.

Despite their widespread adoption, Transformers face two major challenges in their self-attention mechanism: quadratic computational complexity and high memory demands for KV cache storage, particularly when processing long sequences. Significant efforts have been made to mitigate these challenges by: (a) replacing self-attention with new sub-quadratic architectures such as state-space models (SSMs) [Gu & Dao;](#page-11-2) [Dao & Gu](#page-10-2) [\(2024\)](#page-10-2); [Poli et al.](#page-12-1) [\(2024\)](#page-12-1); (b) enhancing the efficiency of existing Transformer self-attention mechanisms [Arora et al.](#page-10-3) [\(2024\)](#page-10-3); [Ainslie et al.](#page-10-4) [\(2023\)](#page-10-4); [Zhang](#page-12-2) [et al.](#page-12-2) [\(2024b\)](#page-12-2); [Yang et al.](#page-12-3) [\(2024\)](#page-12-3); [Qin et al.](#page-12-4) [\(2024\)](#page-12-4); and (c) developing hybrid solutions that combine the advantages of both quadratic and sub-quadratic models [Lieber et al.](#page-11-3) [\(2024\)](#page-11-3); [Dong et al.](#page-10-5) [\(2024\)](#page-10-5); [Wang et al.](#page-12-5) [\(2024a\)](#page-12-5); [Bick et al.](#page-10-6) [\(2024\)](#page-10-6). Among these approaches, the second category is particularly appealing, as it requires minimal architectural changes while capitalizing on existing hardware optimized for Transformers. This work focuses on improving the efficiency of existing self-attention mechanisms within this category.

Multi-head attention (MHA) is a fundamental mechanism in Transformer architectures. However, during inference, MHA requires the saving of large amount of key-value (KV) cache, resulting in high

<sup>∗</sup>Equal Contribution First Authors, with order determined alphabetically.

![](_page_1_Figure_0.jpeg)

<span id="page-1-1"></span>Figure 1: X-EcoMLA with Different Teacher Sizes: [Right] Our results show that using Llama3.2-1B, 3B, and 8B teacher models enables KV cache compression of Llama3.2-1B by 1.9×, 3.6×, and 6.4× using 3.6B tokens respectively, without compromising average accuracy across multiple tasks on the LM Harness Evaluation benchmark. [Left] With 7B training tokens, we can further compress the KV cache to 10.6× and 12.8x, while maintaining competitive accuracy.

memory consumption. To address this challenge, DeepSeek [\(Liu et al.,](#page-11-0) [2024a;](#page-11-0)[b\)](#page-11-4) recently proposed multi-head latent attention (MLA), a novel approach that compresses the KV cache while maintaining the performance of LLMs. The mainstream training paradigm for MLA has relied on pre-training from scratch using vast amounts of data and computational resources. For example, pre-training the Deepseek-v3 model required 2.664M GPU hours on Nvidia H800 clusters. This highlights a major challenge: developing models with a new attention mechanism demands significant computational resources during pre-training. Given the substantial effort already invested in training Transformer models, a natural question arises—can we transfer the rich pre-training knowledge from trained LLMs into more efficient MLA models without training them from scratch?

Existing approaches showcase strong evidence supporting the feasibility of knowledge transfer with architectural adaptations for LLMs, such as MambaInLlama [\(Wang et al.,](#page-12-5) [2024a\)](#page-12-5), MOHAWK [\(Bick](#page-10-6) [et al.,](#page-10-6) [2024\)](#page-10-6), and HedgeHog [\(Zhang et al.,](#page-12-2) [2024b\)](#page-12-2). Inspired by these solutions, we introduce X-EcoMLA, a cost-effective knowledge transfer approach designed to upcycle pre-trained multi-head attention into MLA. In X-EcoMLA, we initialize MLA from its corresponding pre-trained attention blocks using our static or dynamic SVD approach, followed by distillation from a well-trained teacher model. By leveraging the dark knowledge of a high-quality model, we enhance training accuracy and achieve extreme KV cache compression in MLA without sacrificing performance. Our results demonstrate that an 8B teacher model enables 6.4× KV cache compression of the Llama3.2-1B-Inst baseline while preserving 100% of its average score across multiple tasks on the LM Harness Evaluation benchmark. This requires only 3.4B training tokens and about only 70 GPU hours on AMD MI300, while pre-training the Llama3.2-1B model requires 370K GPU hours[1](#page-1-0) . Furthermore, we achieve a 10.6× compression with 7B training tokens and around 140 GPU hours, with less than 0.1% average score drop. We summarize our major contributions as follows:

- We propose a lightweight post-training approach to upcycle pre-trained attention to MLA, significantly reducing computational costs by eliminating the need for training from scratch.
- We develop static and dynamic SVD-based initialization techniques to improve the convergence and accuracy of MLA adaptation.
- We demonstrate that leveraging a larger teacher model enables extreme KV cache compression while maintaining model performance, achieving up to 10.6× compression with minimal accuracy loss. (see Fig. [1\)](#page-1-1).
- We validate the effectiveness of our approach through extensive experiments on the LM Harness Evaluation benchmark, showcasing its efficiency across various settings and LLMs.

<span id="page-1-0"></span><sup>1</sup>https://huggingface.co/meta-Llama/Llama-3.2-1B

### 2 Related Work

This section gives a brief overview of related work; a detailed version is available in Appendix A.1.

**KV Cache Management in Transformers** Transformers store key-value (KV) vectors for each token and each layer during auto-regressive generation, leading to high memory requirements during inference, especially for long sequences. Several methods have been proposed to reduce the KV cache size in Transformers, broadly divided into training-based and post-training approaches Shi et al. (2024). Post-training techniques are easier to apply but may degrade performance due to information loss. These include KV eviction (e.g., Heavy Hitter Zhang et al. (2023)), sliding window attention Arora et al. (2024); Beltagy et al. (2020), low-rank projection approach for KV-cache compression Chang et al. (2025), and quantization Kang et al. (2024); Zhang et al. (2024a). Some strategies, like Attention Sink Xiao et al. (2023) and KV merging Wang et al. (2024b), aim to mitigate information loss while keeping memory usage low. This paper, however, focuses on training-based solutions, which tend to offer a better trade-off between efficiency and accuracy.

**Training-based KV Cache Management** Training-based methods modify attention mechanisms to reduce memory use during inference. Multi-query attention (MQA)Shazeer (2019) and grouped-query attention (GQA)Ainslie et al. (2023) reduce KV cache size by sharing keys/values among heads. YOCO Sun et al. (2024) reduces redundancy with a shared KV cache across layers. DeepSeek-V2 Liu et al. (2024a) introduces multi-head latent attention (MLA), which compresses hidden states via low-rank projection, reducing cache size while outperforming standard MHA. Inspired by MLA's efficiency, this work explores adapting MLA to pre-trained model and we try to address this question: Can we upcycle pre-trained models to their MLA counterparts without costly retraining?

**Upcycling Attention** Upcycling refers to upgrading pre-trained models with minimal computation Komatsuzaki et al. (2022). In attention upcycling, existing attention blocks are adapted into efficient forms like MLA without full retraining. GQA Ainslie et al. (2023) and Hedgehog Zhang et al. (2024b) achieve this via light fine-tuning or distillation. Hybrid models like MambaInLlama Wang et al. (2024a) and MOHAWK Bick et al. (2024) distill knowledge from Transformer attention into Mamba layers. MHA2MLA Ji et al. (2025) introduces a data-efficient fine-tuning approach for converting MHA to MLA via partial RoPE removal and joint SVD-based low-rank approximation. In contrast, X-EcoMLA adopts a unified RoPE design with a shared Key-RoPE vector across all heads (similar to DeepSeek MLA), and employs structured initialization along with knowledge-distillation-based efficient training to enable effective MLA-based upcycling.

#### 3 Background

In this section, we formalize the mathematical framework of MHA and MLA, following the notation from the original DeepSeek-V2 technical report Liu et al. (2024a) with some slight modifications.

### 3.1 Multi-Head Attention (MHA)

MHA projects the input hidden state H into three distinct spaces using learned weight matrices:

$$Q = HW^{Q}, \quad K = HW^{K}, \quad V = HW^{V}, \tag{1}$$

where  $H \in \mathbb{R}^{l \times d}$  is the input sequence representation with l being sequence length and d internal hidden state dimension, and  $W^Q, W^K, W^V \in \mathbb{R}^{d \times n_h d_h}$  are the learnable projection matrices where  $n_h$  is the number of attention heads, and  $d_h$  is the head dimension. The attention scores and final outputs are computed as:

$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right), \quad O = AVW^O$$
 (2)

<span id="page-2-0"></span>where  $W^O \in \mathbb{R}^{d \times d}$  is the output transformation matrix. During inference, MHA requires caching K and V for all past tokens, leading to a storage requirement of  $2n_hd_hl$ .

### <span id="page-2-1"></span>3.2 Multi-Head Latent Attention (MLA)

MLA introduces a low-rank joint compression strategy for keys and values, reducing the KV cache size. Instead of caching K and V, MLA compresses them into a lower-dimensional latent representa-

tion *C KV*:

$$C^{KV} = HW^{DKV}, (3)$$

where *WDKV* ∈ **R***d*×*rkv* is the down-projection matrix, and *rkv* ≪ *dhn<sup>h</sup>* is the compressed dimension for keys and values. The keys and values are then reconstructed from *C KV* using:

$$K^C = C^{KV} W^{UK}, \quad V^C = C^{KV} W^{UV}, \tag{4}$$

where *WUK* ∈ **R** *<sup>r</sup>kv*×*nhdqk* and *WUV* ∈ **R***rkv*×*nhd<sup>h</sup>* are the up-projection matrix for keys and values. Notice that in this paper we consider a more flexible setting where the queries and keys could have different dimensionality *dqk* other than *d<sup>h</sup>* . During inference, the learned matrices can be absorbed into the existing projection layers: *WUK* and *WUV* can be merged into *W<sup>Q</sup>* and *W<sup>O</sup>* respectively.

In the meantime, such low-rank compression can be also applied to the queries to reduce the memory usage while training:

$$C^{Q} = HW^{DQ}, \quad Q^{C} = C^{Q}W^{UQ}, \tag{5}$$

where *C <sup>Q</sup>* ∈ **R***l*×*r<sup>q</sup>* represents the compressed latent vector for queries, *r<sup>q</sup>* denotes and query compression dimension, *WDQ* ∈ **R***d*×*r<sup>q</sup>* is the down-projection matrix and *WUQ* ∈ **R** *rq*×*nhdqk* denotes the up-projection matrix.

However, such low-rank KV compression is not compatible with Rotary Position Embedding (RoPE) as it breaks the matrix absorbing mechanism. As a result, the authors in [Liu et al.](#page-11-0) [\(2024a\)](#page-11-0) propose decoupled Rotary Position Embeddning where additional multi-head queries *Q<sup>R</sup>* and a shared key *K <sup>R</sup>* are applied to carry RoPE, which can be expressed as:

$$Q^{R} = \text{RoPE}(C^{Q}W^{QR}), \quad K^{R} = \text{RoPE}(HW^{KR}), \tag{6}$$

where *WQR* ∈ **R***rq*×*nhd<sup>r</sup>* and *WKR* ∈ **R***rq*×*d<sup>r</sup>* represent the matrices to produce the decoupled keys and queries. Then, the RoPE embeddings *QR*, *K <sup>R</sup>* and Non-RoPE (NoPE) embeddings *QC*, *K <sup>C</sup>* are concatenated to perform the attention operation:

$$Q = [Q^C; Q^R], \quad K = [K^C; repeat(K^R)], \tag{7}$$

where repeat(.) denotes duplicating *K <sup>R</sup>* for each head. After concatenation, the same attention operation is applied as in Equation [2.](#page-2-0) During inference, MLA requires caching only *C KV* and *K R*, reducing the storage requirement to (*rkv* + *dr*)*l* which is significantly smaller than the standard MHA cache size.


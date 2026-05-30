# 2 Related Work

## 2.1 Linear Complexity Language Models

Traditional transformer-based language models primarily rely on self-attention mechanisms with quadratic complexity, resulting in significant computational demands—especially when processing long sequences. In response, recent research has focused on alternative architectures that achieve linear complexity while maintaining competitive performance. Notable examples include State-Space Models [\(Gu and Dao,](#page-8-1) [2023\)](#page-8-1), Linear Attention models [\(Katharopoulos et al.,](#page-8-0) [2020\)](#page-8-0), and Linear RNNs [\(Orvieto et al.,](#page-8-7) [2023\)](#page-8-7).

The RWKV model[\(Peng et al.,](#page-8-2) [2023\)](#page-8-2) represents a noteworthy advancement, integrating characteristics of both transformers and RNNs. By maintaining a recurrent state representation similar to RNNs, while preserving the parallelizability of transformers during training, RWKV achieves linear complexity without compromising performance. The foundational RWKV-4 [\(Peng et al.,](#page-8-2) [2023\)](#page-8-2) architecture demonstrated that an RNN-like model could deliver competitive performance in language modeling. RWKV-5 [\(Peng et al.,](#page-8-4) [2024\)](#page-8-4) introduced matrix-valued states and dynamic recurrence mechanisms, further enhancing efficiency and stability. RWKV-6 [\(Peng et al.,](#page-8-4) [2024\)](#page-8-4) improved training stability, facilitating greater scalability, while RWKV-7 [\(Peng et al.,](#page-8-3) [2025\)](#page-8-3) incorporated dynamic state evolution, surpassing traditional attention-based models and enabling more flexible in-context learning. These advancements position the RWKV model as an effective and efficient linear complexity language model. As a result, it offers a promising alternative to transformers, maintaining robust language modeling capabilities while being computationally efficient.

#### 2.2 Hybrid Language Models

Recent advances have shown that hybrid models outperform traditional architectures in both accuracy and efficiency. For example, Mamba [\(Gu and](#page-8-1) [Dao,](#page-8-1) [2023\)](#page-8-1) integrates retrieval-based and generative components, allowing models to leverage both pre-trained knowledge and external information

sources. Building on this paradigm, recent models such as Jamba [\(Lieber et al.,](#page-8-5) [2024\)](#page-8-5), Zamba [\(Glo](#page-8-6)[rioso et al.,](#page-8-6) [2024\)](#page-8-6), and MiniMax [\(MiniMax et al.,](#page-8-8) [2025\)](#page-8-8) further enhance hybrid architectures.

Jamba improves multilingual understanding in low-resource settings by refining the interaction between retrieval and generation. Zamba introduces a compact SSM-transformer hybrid design that balances attention capabilities with parameter efficiency. MiniMax combines mixture-of-experts (MoE) with advanced sparse attention mechanisms to scale to longer contexts.

While these hybrid approaches bring notable improvements, they still rely on full attention mechanisms, and thus inherently preserving the O(N<sup>2</sup> ) complexity, which limits their scalability to truly long-context scenarios.

#### 2.3 Sparse Attention

Native Sparse Attention [\(Yuan et al.,](#page-9-1) [2025\)](#page-9-1) reduces token interactions by structuring keys and values into temporal blocks and processing them through three distinct attention paths: compressed coarsegrained tokens, selectively retained fine-grained tokens, and sliding windows for local contextual information. This approach dynamically selects the most relevant tokens, optimizing the balance between global and local context while significantly reducing computational overhead. Similarly, Seer-Attention [\(Gao et al.,](#page-8-9) [2025b\)](#page-8-9), inspired by the gating mechanism in Mixture-of-Experts (MoE) models, enhances efficiency by introducing learnable gating units within the attention mechanism.

Moreover, Mixture of Block Attention (MoBA) [\(Lu et al.,](#page-8-10) [2025\)](#page-8-10) addresses the inefficiencies of traditional attention mechanisms by partitioning the input context into blocks and employing a gating mechanism to selectively route query tokens to the most relevant blocks. This design not only improves computational efficiency but also enables seamless transitions between full and sparse attention modes. However, during autoregressive decoding, MoBA suffers from increasing memory usage as the KV cache grows with the sequence length, leading to a linear space complexity. As a result, it cannot guarantee constant memory consumption during inference, which limits its scalability for long-context generation.


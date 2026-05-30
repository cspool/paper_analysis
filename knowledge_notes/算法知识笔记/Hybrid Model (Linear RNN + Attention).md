## Hybrid Model (Linear RNN + Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Model 指在同一语言模型中混合使用不同 token mixer 层的架构策略。在 Gated DeltaNet 中，指将线性 RNN 层（Mamba2/GatedDeltaNet）与 Sliding Window Attention (SWA) 层混合。动机：线性 RNN 在局部模式建模和长程检索上有固有局限（state size 固定导致 memory collision），SWA 提供精确窗口内 attention 弥补局部模式缺陷；同时线性 RNN 提供 O(1) 推理效率。类似架构包括 Griffin（RG-LRU + local attention）、Samba（Mamba + SWA + SwiGLU MLP）、Hymba（hybrid-head attention+SSM）。SAMBA 论文是首个证明混合线性复杂度模型在大规模（3.8B）上能显著优于 SOTA Transformer 架构的工作。

SAMBA 中的 Hybrid 设计：层排列为 [Mamba → MLP (for Mamba) → SWA → MLP (for SWA)] 的 4 层 block 重复 N/4 次。关键设计：(1) Mamba 和 SWA 各有独立的 SwiGLU MLP，分别处理不同类型的信息——Mamba 的 MLP 处理压缩后的递归语义，SWA 的 MLP 处理精确检索信号；(2) SWA 窗口 2048，使用 RoPE（base=10,000）和 FlashAttention 2；(3) 训练序列长度 4096 = 窗口大小 × 2；(4) 注意力熵分析（Figure 5a）显示 Samba 中 SWA 层的注意力熵方差更大——中间层熵低（专注精确检索），顶层/底层熵高（整合全局信息），呈现专业化分工；(5) Mamba 的选择熵在混合架构的中间层更高（Figure 5b），说明有了 SWA 负责召回，Mamba 可以更专注于递归结构建模。消融研究（Table 5）表明：即使仅 1 层全注意力也无法外推至超训练长度（16K 时 perplexity 从 10.29 升到 13.66），而 SWA 可外推至 1M 且 perplexity 持续改善。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GatedDeltaNet-H1: [GatedDeltaNet, SWA, GatedDeltaNet, SWA, ...] 交替
GatedDeltaNet-H2: [Mamba2, GatedDeltaNet, SWA, Mamba2, ...] 三层最优顺序（消融验证）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SWA 层使用 FlashAttention-2 kernel，swa window 通常 2048。Hybrid 总训练吞吐量因 SWA 的高效 kernel 反而高于纯线性 RNN。层分配策略为均匀交替，attention 层比例通常 5-15%。适用于需要训练效率和长程性能最佳 balance 的大规模 LM。

GoldFinch采用不同的hybrid策略：前2/3层为Finch-C2 RNN层（O(1) per token），后1/3层为GOLD Transformer层（full MHA over compressed key cache）。关键创新：Finch-C2最终层输出被压缩为全局shared key cache（仅D/16 per token），所有GOLD层共享同一cache。这使得KV-cache从per-layer 2·d_model·n_layer降至(1+d_model/16)，约756-2550×压缩。Pre-fill仅需运行Finch-C2部分（O(1) per token），decoding时GOLD attention O(N)但仅在生成新token时运行（通常很短）。GoldFinch 1.45B在lambada ppl 48.2远优于Finch 81.9和Llama 71.7。

M1采用不同的hybrid策略：28层中6层保留为interleaved standard attention（~21%），22层替换为Mamba SSM层。动机：(1) 少量attention层提供关键的长程信息路由能力——完全去除attention会导致reasoning性能崩溃；(2) 22层Mamba提供O(1)推理效率和大batch吞吐量优势；(3) 通过MambaInLlama权重初始化+reverse KL蒸馏+分阶段SFT+GRPO RL实现跨架构推理能力迁移。M1-3B在数学推理benchmark上匹配DeepSeek-R1-Distill-Qwen-1.5B的性能，同时提供3x inference throughput（vLLM, H100, batch=512）。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

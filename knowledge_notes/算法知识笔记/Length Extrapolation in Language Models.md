## Length Extrapolation in Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Length Extrapolation（长度外推）指语言模型在超过预训练序列长度的上下文上仍能保持或改善性能的能力。传统 Transformer（全注意力）在超训练长度时 perplexity 爆炸（Table 3: Llama-2 438M 在 16K 时 perplexity 从 11.14→249.03）。SAMBA 通过 SWA 的平移不变性 + Mamba 的递归压缩实现高效外推：仅用 4K 训练长度，零样本外推到 1M（256× 外推率）时 Proof-Pile perplexity 持续改善（Figure 2a）。Passkey Retrieval 上：Samba 1.7B 仅用 4K 长度 500 步微调即可外推到 256K（64× 外推率）且准确率完美（Figure 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
核心原理——SWA 的长度外推能力来自 RoPE 的相对位置编码 + 固定窗口的平移不变性：训练时（seqlen=4096, window=2048），对任意位置 i，SWA 只关注 [max(0,i-2048), i]，RoPE 编码的是相对距离而非绝对位置 → 天然平移不变；推理时（seqlen=1M）计算模式与训练时完全相同 → 无分布外问题。Mamba 递归状态累积全部历史信息 → 绑定短期精确 + 长期压缩。关键前提：RoPE 对 SWA 长度外推至关重要——Samba-NoPE 在 16K 时 perplexity 爆炸至 314.78（Table 3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流外推方法：(1) PI（线性缩放位置索引）——需要微调；(2) NTK-aware——修改 RoPE base frequency，零样本可用；(3) SelfExtend——用 group attention + neighbor window，零样本但增加延迟；(4) SWA 从零训练——最干净但需重新预训练。SAMBA 的方法属于 (4)：从零预训练就包含 SWA，使模型"原生"支持外推。评估方法：(a) perplexity 外推——Proof-Pile 测试集 sliding window 评估；(b) 检索外推——Passkey Retrieval 和 Phonebook；(c) 长文本任务——GovReport/SQuALITY 摘要。

涉及论文标题：
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

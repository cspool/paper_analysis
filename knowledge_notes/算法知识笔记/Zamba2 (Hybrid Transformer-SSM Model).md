## Zamba2 (Hybrid Transformer-SSM Model)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zamba2（Glorioso et al., 2024a）是Zyphra公司开发的混合架构语言模型，将Mamba SSM层与Transformer attention层结合。Zamba2-1.2B在LongMamba实验中作为hybrid代表被评测——训练长度4k tokens。vanilla Zamba2-1.2B在LongBench-E上avg 11.43%高于纯SSM（8.21-8.37%），LongMamba后提升至17.82%（+6.39%）。使用θ=10^{-5}、C=5配置。纯SSM在coding任务更优，hybrid在few-shot learning更优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Zamba2是shared-parameter hybrid架构，结合Mamba层和shared attention层。其Mamba层包含标准SSM机制（Δ_t, Ā_t, B̄_t等），因此LongMamba的通道分类和token filtering同样适用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
模型权重在HuggingFace发布。LongMamba直接加载官方预训练checkpoint，无微调。实验显示hybrid和纯SSM各有所长，LongMamba能缩小两者差距。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

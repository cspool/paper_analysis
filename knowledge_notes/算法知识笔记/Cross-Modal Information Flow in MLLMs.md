## Cross-Modal Information Flow in MLLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Modal Information Flow（跨模态信息流）指MLLM的Transformer层中不同模态token（visual和text）之间的attention交互模式和强度。在MLLM的每一层attention计算中，QK^T attention矩阵包含四个block：visual→visual、visual→text、text→visual、text→text。Cross-modal attention scores A_{v→t}（visual→text）和A_{t→v}（text→visual）共同构成跨模态信息流。FlowMM发现这一信息流在MLLM不同层中存在显著分化：浅层以intra-modal交互为主（cross-modal attention比例低），负责低层单模态特征提取；深层跨模态交互显著增强（cross-modal attention比例高），负责跨模态融合和高层语义抽象。此pattern在ALFRED/MMCoQA/TextNeedle三个不同任务上一致。这一发现是FlowMM层自适应合并策略的理论基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Cross-Modal Information Flow量化（FlowMM公式6-7）：
for head h in 1..H:
    A_{v→t}^{l,h} = Σ_{v∈V} Σ_{t∈T} α_{v→t}^{l,h}   # visual→text attention
    A_{t→v}^{l,h} = Σ_{t∈T} Σ_{v∈V} α_{t→v}^{l,h}   # text→visual attention

ρ^l = (1/H) · Σ_{h=1}^{H} (A_{v→t}^{l,h} + A_{t→v}^{l,h}) / A^{l,h}

# ρ^l ∈ [0, 1]: 
#   → 0: 几乎纯intra-modal交互
#   → 1: 几乎纯cross-modal交互
# FlowMM在Qwen2.5-VL-7B上的发现:
#   浅层(layers 1-12): ρ^l < 0.2, intra-modal主导
#   深层(layers 13-28): ρ^l > 0.2, cross-modal显著增加
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Cross-Modal Information Flow分析通过一次校准前向传播即可完成：在校准样本上执行inference，从每层attention矩阵中提取visual↔text的attention scores，计算ρ^l。通常需少量（数十到数百个）校准样本取平均以得到稳定pattern。FlowMM将此用于指导KV cache合并策略——若ρ^l ≥ θ（阈值最优值0.2-0.3），执行跨模态合并；若ρ^l < θ，执行模态内合并。阈值过低（<0.1）导致浅层过早跨模态合并→模态信息混淆（modal confusion）；过高（>0.4）限制深层跨模态融合→跨模态语义理解不足。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

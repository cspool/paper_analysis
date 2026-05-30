## Visual Token Pruning in Multi-modal LLM (多模态 LLM 视觉 Token 剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Visual Token Pruning in MLLM 是一种在多模态 LLM 推理过程中，基于重要性度量从 LLM 各层中逐步移除视觉 token，以减少 Attention 和 FFN 计算量的训练无关技术。与纯文本 Decoder 的 KV Cache token pruning（为节省显存）不同，MLLM 的视觉 token pruning 主要目标是减少计算量——因为视觉 token 数量（数千个）远超文本 token。AIM 的核心发现：LLM 早期层做跨模态融合需要 visual tokens，后期层专注文本推理，可以大幅剪除 visual tokens。基于此，AIM 设计分层剪枝策略：早期层全保留，中期层线性递减，后期层全部移除，同时文本 token 始终不剪。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**AIM Visual Token Pruning 伪代码**：

```
// 超参数：l1（开始剪枝层）, l2（完全移除层）, L（总层数）
// 保留率调度：r^l = 1 if l<l1; 1 - (l-l1)/(l2-l1) if l1≤l≤l2; 0 if l>l2

for l in 1..L:
    // Step 1: 标准 Self-Attention（当前层输入含 visual + text tokens）
    x = [v^{l-1}; t]  // visual + text tokens
    A = softmax(Q @ K^T / sqrt(d_k))  // Attention 权重矩阵
    
    // Step 2: PageRank 重要性计算
    s = ones(N_v + N_t) / (N_v + N_t)  // 均匀初始化
    for _ in range(num_iterations):
        s = (A @ s) / (N_v + N_t)  // PageRank 迭代（公式 1）
    
    // Step 3: 仅对 visual tokens 按 s 排序剪枝
    v_scores = s[:N_v]  // 仅取 visual token 分数
    k = int(N_v * r^l)  // 当前层保留数量
    keep_idx = top_k(v_scores, k)
    
    // Step 4: 移除被剪枝的 visual tokens
    v^l = v^{l-1}[keep_idx]  // 保留的 visual tokens
    // text tokens 始终全部保留
```

**与文本 Decoder Token Pruning 的关键差异**：
| 维度 | 文本 Decoder (H2O/A2SF) | MLLM (AIM) |
|------|------------------------|-----------|
| 剪枝对象 | KV Cache 中的 token | 当前层输入中的 visual token |
| 目标 | 减少显存 + 计算 | 减少计算（FLOPs） |
| 重要性度量 | A2S/A2SF（累积 Attention Score） | PageRank（Attention 图稳态分布） |
| 文本 token | 可剪（选择性保留关键 token） | 不剪（文本始终全保留） |
| 层级策略 | 每层独立决策 | 全局 Scheduler 控制逐层保留率 |

术语一般如何实现？如何使用？

AIM 实现中，Token Pruning 以 hook 方式插入到 Qwen2/Vicuna LLM 的每层 Attention 后。默认配置：video（Qwen2-7B, 28 layers）l₁=14, l₂=22；image（Vicuna-1.5-7B, 32 layers）l₁=13, l₂=21。额外开销极小：video 场景 Token Pruning 仅 4.18 GFLOPs（<0.03% LLM FLOPs）。代码开源：https://github.com/LaVi-Lab/AIM。

注意：AIM 的 Token Pruning 与 FlashAttention 不兼容（需要显式 Attention 矩阵计算 PageRank），但与量化（quantization）和稀疏注意力（sparse attention）兼容。Dynamic-LLaVA 的 Vision-Language Context Sparsification 与 AIM 的关键差异：Dynamic-LLaVA 同时稀疏化 vision 和 language 上下文（而非仅 vision），使用可学习 predictor（而非 PageRank 启发式），且适用于 decoding w/ KV cache 的在线 KV 压缩场景。

**TransPrune 的 Token Pruning 方法**：TransPrune 从与 attention/similarity 完全不同的视角出发——利用 token 表征在 LLM 层间传播时的变化（Token Transition）来反映 token 重要性。核心包括：(1) **TTV（Token Transition Variation）**：测量每个 token 在 self-attention 和 FFN 模块中表征的幅度变化（L2 norm 比率）和方向变化（cosine similarity），仅依赖 token 自身的输入→输出变化（无需 inter-token 依赖），天然避免 attention 位置偏差；(2) **IGA（Instruction-Guided Attention）**：计算 instruction tokens 对 image tokens 的单向 attention 权重，引入任务相关语义监督；(3) **Accumulation**：跨中间层（7-12）累积 TTV 值，稳定剪枝决策。TransPrune 的 TTV 计算仅需模块输入/输出 tensor（兼容 FlashAttention），IGA 仅计算 instruction→image 的单向 attention（非完整 N×N attention map）。在 LLaVA-v1.5-7B 上降低 TFLOPs 至 40.8% 时性能几乎无损。代码将开源于 https://github.com/liaolea/TransPrune。与 AIM 关键差异：TransPrune 为 training-free（无需学习），使用 token 自身 transition 信号（非 attention 图结构），且与 FlashAttention 完全兼容。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification
- TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

---

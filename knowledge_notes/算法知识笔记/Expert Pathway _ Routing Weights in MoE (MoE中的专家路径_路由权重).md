## Expert Pathway / Routing Weights in MoE (MoE中的专家路径/路由权重)

术语解释
Expert Pathway（专家路径）是 MoE LLM 中一个样本在逐层经过所有 MoE 层时，每层被选中的 experts 及其对应权重的序列。形式上，对于 L 层、每层 E 个 experts 的 MoE，pathway 是一个矩阵 ω ∈ R^{L×E}，其中 ω_{l,e} 表示第 l 层第 e 个 expert 的路由权重。Pathway 决定了每个 token 在各层中由哪些 expert 处理以及各 expert 的贡献比例。

术语是什么？
在标准 MoE 架构中，每层的 router（gate）计算：

$$s_{l,e} = x \cdot W_{\mathrm{gate}}[e] \quad \text{(affinity score)}$$

$$w_{l,e} = \frac{\exp(s_{l,e})}{\sum_{j \in \mathrm{TopK}(s_l)} \exp(s_{l,j})} \quad \text{(routing weight after TopK softmax)}$$

最终 MoE 层输出：

$$h_l = x + \sum_{e \in \mathrm{TopK}(s_l)} w_{l,e} \cdot \mathrm{Expert}_e(x)$$

Pathway 矩阵 ω 收集所有层的 {w_{l,e}}。Router 在 pretraining 阶段与模型参数端到端训练，推理时冻结。

C3PO 的发现：预训练的 router 产生的 pathway 存在严重次优性。在 OLMoE 上，base model 的 pathway 与 Oracle pathway 之间存在 15.3% accuracy gap（69.9% vs 85.2%）。

从算法pipeline角度拆解术语：
Pathway 在 MoE 推理中的流转：

```
输入 token x → Layer 1:
  gate_logits = x @ W_gate[1]            # [E] = [64]
  topk_weights, topk_idx = topk(softmax(gate_logits), k=8)
  h_1 = x + Σ_{j in topk_idx} topk_weights[j] * Expert_{1,j}(x)
  → 记录 ω[1, topk_idx] = topk_weights

→ Layer 2: ... (重复)
→ Layer L: 记录 ω[L, topk_idx]

最终 pathway ω ∈ R^{L×E}，其中非 top-k 位置为 0
```

C3PO 对 pathway 的操作：
- **读取**: 从模型 forward pass 中 hook 各层的 gate_logits 或 softmax 后的 routing weights
- **修改**: 在 gate_logits 层面加偏移 Δω（直接修改 softmax 前的 logits），或直接替换 routing weights
- **裁剪**: 只保留 Critical Layers（最后 5 层）和 Core Experts（top-20）的 routing weights

术语一般如何实现？如何使用？
- **获取 pathway**: HuggingFace 模型的 forward hook 机制，在 MoE 层的 router 输出处注册 hook 捕获 routing weights
- **注入 pathway**: 替换 `olmoe_modeling.py` 中的 MoE 层实现，在 forward 时接受外部 routing weights 参数覆盖内部 router 输出
- **优化粒度**: C3PO 实验表明优化单个 last token 的 pathway 效果最好（vs 多个 token），因为 last token 承载了最丰富的任务决策信号

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Contextual Sparsification (Contextual Activation Sparsity)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Contextual Sparsification（上下文化稀疏化）是一种 training-free 的推理时激活稀疏化技术。在不重新训练模型的前提下，对每个输入 token 的激活值按**上下文相关**的幅值阈值进行剪枝：对给定输入 x 和投影矩阵 W 产生的激活向量 a(x) = x·W，仅保留 |a_i| ≥ t 的元素，将其余置零（S_t(a_i) = a_i if |a_i| ≥ t else 0）。阈值 t 不是全局常量，而是根据目标稀疏率 k（如 90%）从采样数据集的激活幅值经验 CDF 反向确定：t = min{t': F(t') ≥ k}。与全局固定阈值（如 ReLU 的 max(0,x)）或结构化的通道级剪枝不同，contextual sparsification 的剪枝模式**每个输入 token 都不同**——对当前 token 不重要的激活通道被完全跳过。

FloE 将 contextual sparsification 应用于 MoE expert 内部：基于 up projection 的输出激活 a_up = x·W_up 的幅值决定剪枝 mask，然后用该 mask 同时剪枝 W_gate 的对应列和 W_down 的对应行（转置后为列），实现计算量和传输量的双重减少。关键理论贡献：证明了在相同稀疏率下，三种激活（a_down, a_up, a_gate）剪枝的 L2 恢复误差满足 L_down ≤ L_up < L_gate，即剪枝 a_up 的误差严格小于剪枝 SiLU(gate) 输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Contextual Sparsification 的阈值确定（offline, per-expert）
// 输入: calibration dataset (e.g., C4), target sparsity k
// 输出: threshold t for each expert

for each expert E_ij in model:
    activations = []
    for each sample in calibration_data:
        h = model_forward_to_layer_i(sample)
        a_up = h @ W_up_ij                    // up projection 输出激活
        activations.extend(|a_up|.flatten())  // 收集幅值
    // 经验 CDF 反函数
    activations.sort()
    idx = int(k * len(activations))           // k=0.9 → 90% 稀疏
    t_ij = activations[idx]                   // 阈值

// 推理时的 sparse forward pass (Algorithm 1):
v = x @ W_up                                  // 全精度 up projection
mask = (|v| >= t_ij)                          // bool mask, ~10% True at 90% sparsity
x_prime = SiLU(x @ W_gate[mask]) ⊙ v[mask]    // 仅加载被选中的 gate 列
y = (W_down^T[mask] @ x_prime)^T              // 仅加载被选中的 down 列
```

FloE Figure 3(a) 的稀疏化敏感度对比（WikiText-2 perplexity, Mixtral-8×7B）：
| 稀疏率 | Down input pruning | Up output pruning | SiLU(gate) output pruning |
|--------|-------------------|-------------------|--------------------------|
| 50% | PPL +0.5% | PPL +3% | PPL +12% |
| 70% | PPL +3% | PPL +16% | PPL +40% |
| 90% | PPL +27% | PPL +77% | PPL +259% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 相关系统：CATS (Context-Aware Thresholding for Sparsity, Lee et al. 2024a) 是首个 training-free contextual sparsification 方法，应用于 dense LLM；TEAL (Liu et al. 2024) 扩展了训练无关激活稀疏化
- FloE 的差异化贡献：(1) 将 contextual sparsification 首次应用于 MoE expert 内部（而非 dense FFN），(2) 仅对 up projection 输出做剪枝并联动剪除 gate/down 对应通道，(3) 理论证明了 L_down ≤ L_up < L_gate 的误差排序
- 阈值由 calibration dataset 离线确定，推理时无额外计算开销
- 与量化技术正交：FloE 将 contextual sparsification（gate/down）与 INT2 量化（up）结合形成 hybrid compression
- 局限：极端稀疏率（>90%）下精度退化显著，且对 SiLU(gate) 输出的剪枝不可行（因误差太大）

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

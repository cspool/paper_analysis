## Stacking Computer (Batched Gating for MoE Expert Prefetching)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Stacking Computer 是 HOBBIT / MoE-APEX 中用于加速 MoE expert 预取预测的批量 gating 计算技术。在逐层预测后续层所需 expert 时，naive 方法需要逐层执行 gating 计算（线性增长开销）。Stacking Computer 利用 gating 权重矩阵的一维为 expert 数量（通常很小：8/16/64），将所有后续层的 gating 权重堆叠成 [N, M, E] 张量（N=层数, M=d_model, E=experts），与 hidden state x ∈ R^M 做一次批量矩阵乘 matmul(x, W_stacked)，结合 top-k 选择，利用 GPU 并行性实现接近单层 gating 的计算速度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Stacking Computer: 批量 gating 计算
// 输入: x [1, M] (当前 token hidden state)
//       W_gate[l] [M, E] (第 l 层 gating 权重, l=0..L-1)
// 输出: pred_experts[l] [K] (第 l 层预测的 top-K experts)

// Step 1: 堆叠所有后续层 gating 权重
// naive: for l in next_layers: gate_logits[l] = x @ W_gate[l]
// stacking: 一次批量计算

W_stacked = stack([W_gate[l] for l in range(cur_layer+1, L)])  // [N, M, E]
                                 // N = L - cur_layer - 1

// Step 2: 批量矩阵乘 (GPU 高度并行)
x_expanded = x.unsqueeze(0)               // [1, 1, M]
gate_logits_all = x_expanded @ W_stacked  // [1, N, E]
gate_probs_all = softmax(gate_logits_all, dim=-1)  // [1, N, E]

// Step 3: 批量 top-k 选择
pred_experts = topk(gate_probs_all, k=K, dim=-1)  // [1, N, K]

// Step 4: 自适应层数选择
// 从最近层开始，若所有 pred experts 已在 cache 则继续下一层
for l_idx in range(N):
    needed = pred_experts[0, l_idx]
    if not all_in_cache(needed):
        prefetch(needed)  // 发起预取
        break             // 或继续检查（取决于配置）
```

堆叠计算的关键：
- W_stacked 维度 [N, M, E]，M 是 hidden dimension（如 4096），E 很小（8/16/64）
- 批量 matmul 的 FLOPS 与单层类似（N 层 × E 小 → 总计算量 ≈ 单层大矩阵乘）
- GPU 上 N×E 并行执行，latency 接近常数（不随 N 线性增长）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现方式：在 Llama.cpp 中，将所有后续层的 gating Linear 层权重在初始化时预堆叠为扁平张量。推理时用 CUDA batched GEMV 或 GEMM（M=1 时为 GEMV batch）。
- 堆叠范围：建议 1-3 层 ahead（HOBBIT 推荐 p=1~3）。更深层预测准确率下降但仍有 ~90%，收益递减。
- 开销：堆叠操作的 overhead 在 Mixtral-8x7B 上 <0.1ms，相比 expert 加载 (~10ms) 可忽略。
- 与混合精度预取的配合：预取时同时加载低精度 expert，即使预测错误也仅浪费 1/4 带宽。

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading

## GroupedGEMM (Grouped General Matrix Multiplication)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GroupedGEMM（Grouped General Matrix Multiplication）是一种将多个不同尺寸、不同转置方式和不同缩放因子的矩阵乘法操作合并为单次 kernel launch 的批量 GEMM 操作。在 fine-grained MoE 推理中，GroupedGEMM 是 expert 层的核心计算原语：每个 expert 需要对分配给它的 token 执行独立的 FFN 矩阵乘法（W_up, W_gate, W_down），由于各 expert 接收的 token 数量不同（门控路由不均衡），这些 GEMM 操作具有不同的 M 维度（token count × d_model），但共享 K 和 N 维度（expert 权重矩阵维度）。

与普通批量 GEMM（所有子任务形状相同）相比，GroupedGEMM 的关键特性是支持**异构子任务形状**——每个 expert 的 token 数量可能不同，甚至该 expert 可能没有收到任何 token。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fine-grained MoE 中 GroupedGEMM 的计算过程（以 Cutlass 实现为例）：

```
# GroupedGEMM for MoE expert computation
# 输入: hidden_states [total_tokens, d_model], token_to_expert mapping
# 输出: expert_outputs [total_tokens, d_model]

# Step 1: Token Routing & Grouping
for token i in range(total_tokens):
    topk_experts[i] = Router(hidden_states[i], k=6)

# Step 2: Build GroupedGEMM problem for FC1 (W_up)
groups = []
for expert_j in range(num_experts):
    tokens_for_j = [i for i where expert_j in topk_experts[i]]
    if len(tokens_for_j) > 0:
        groups.append({
            'A': hidden_states[tokens_for_j],  # [n_j, d_model]
            'B': W_up[j],                       # [d_model, d_ff]
            'C': output_up[tokens_for_j]        # [n_j, d_ff]
        })

# Step 3: Single kernel launch - all groups in parallel
cutlass_grouped_gemm(groups, alpha=1.0, beta=0.0)

# Step 4: Activation + FC2 similarly via GroupedGEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

IFMoE 论文讨论了三种 GroupedGEMM 实现：
1. **Triton GroupedGEMM**：Triton 语言编写，灵活性最高但性能可能稍逊
2. **Cutlass GroupedGEMM**：NVIDIA Cutlass 库实现，IFMoE 的实际选择（因为 PyTorch 与 CUDA 12.5 版本冲突，无法使用 cuBLAS 版本）
3. **cuBLAS GroupedGEMM**：CUDA 12.5 新增的 GroupedGEMM API，预期性能最优但受限于 PyTorch/CUDA 兼容性

IFMoE 中的性能瓶颈分析：GroupedGEMM 是 memory-bound 操作——单个 expert 的 memory footprint 较小，但当 batch size 增大（激活 expert 数线性增长），总 memory pressure 上升。同时，MoE 动态路由使 Torch Compile 和 CUDA Graph 无法优化，进一步加剧延迟。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE

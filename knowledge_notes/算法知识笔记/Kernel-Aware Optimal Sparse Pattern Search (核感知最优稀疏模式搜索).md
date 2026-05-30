## Kernel-Aware Optimal Sparse Pattern Search (核感知最优稀疏模式搜索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Kernel-Aware Optimal Sparse Pattern Search 是 MInference 论文提出的离线（offline）搜索方法，用于为每个 attention head 确定最优的稀疏模式（A-shape / Vertical-Slash / Block-Sparse 之一）及其具体参数配置（如 VS 的 k_v 和 k_s 数量、BS 的 k_b 数量）。它是 MInference 三步 pipeline 的第一步。

"Kernel-Aware"（核感知）的含义是：搜索空间中的 FLOPs 使用**真实 GPU kernel 中的 FLOPs**（而非概念上的稀疏 token 数），确保搜索出的最优配置在实际 GPU 执行时确实能达到预期的加速效果。例如，一个 $64 \times 64$ 的 block 块计算在 GPU 上的实际 FLOPs 与 $1 \times 64$ 的 column 不同，虽然它们覆盖的 token 数量相同。

搜索优化目标：最小化稀疏 attention 输出与 dense attention 输出的差异（$\argmin |y_i - y|$），而非仅最小化 attention score 的差异。这使用 FlashAttention 进行计算（降低 GPU 内存），并包含了 V 矩阵的信息，实现了 end-to-end 的最优模式选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Algorithm 1: Kernel-Aware Sparse Pattern Search
输入: Q, K, V ∈ R^{S×d_h}, patterns p ∈ {A-shape, VS, BS}, target FLOPs t

# Phase 1: 构建 Kernel-Aware 搜索空间
ρ = ∅  # 候选配置集合
for each pattern candidate σ_i:
    t_i = FLOPs_in_kernel(σ_i)     # GPU kernel 真实 FLOPs
    while |t_i - t| > ε:           # 调整参数使 FLOPs 逼近目标
        σ_i = ChangeSpace(σ_i, p_i)  # 微调参数（step=50）
        t_i = FLOPs_in_kernel(σ_i)
    ρ = ρ ∪ {σ_i}                  # 加入候选集

# Phase 2: 基于 Recall 的最优模式选择
y = FlashAttention(Q, K, V)       # Dense attention 输出作为 ground truth
for each candidate σ_i in ρ:
    y_i = SparseAttention(Q, K, V, σ_i)  # 候选配置的稀疏输出
p_best = argmin(|y_i - y|)        # 选择输出差异最小的配置
```

**搜索空间配置（MInference 论文）**：
| Pattern | Search Space |
|---------|-------------|
| A-shape | {(1024, 4096)} — 1K global + 4K local |
| Vertical-Slash | {(30, 2048), (100, 1800), (500, 1500), (3000, 200)} |
| Block-Sparse | {100} — top-100 blocks |

搜索使用一条 30K tokens 的 KV retrieval 合成样本，约 15 分钟在单 A100 上完成。同一模型的不同 context 长度版本（如 262K 和 1M）可复用相同的最优配置，展示了搜索结果的泛化性。

术语一般如何实现？如何使用？

实现步骤：
1. 选取一条代表性的 reference sample（不需要与下游任务完全一致，论文验证了 KV retrieval 合成数据的泛化性）
2. 运行 FlashAttention 获取 dense attention output（所有 query 的 attention output 作为 ground truth）
3. 对每个候选模式配置，执行对应的稀疏 attention 计算
4. 计算 $|y_i - y|$（L2 distance）并选择最小差异的配置
5. 记录最优配置到配置文件中，推理时直接读取

使用时注意事项：
- 需要确保 target FLOPs 与目标加速比匹配——更高 target FLOPs 意味着更高的准确率但更低加速比
- search space 的 ChangeSpace step 太小会导致搜索时间过长，太大可能跳过最优配置
- 搜索结果可以在模型的不同 context 版本间转移（论文验证了从 262K 模型迁移到 1M 模型的有效性）

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

## Wanda（Activation-aware Weight Pruning，激活感知权重剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（ICLR 2024，Sun et al.）是免重训练的 one-shot 非结构化权重剪枝方法：重要性分数 S_ij = |W_ij| · ‖X_j‖₂，其中 ‖X_j‖₂ 是用校准集算出的第 j 个输入通道激活的 L2 范数，与权重逐元素相乘后按输出行 top-k 保留（如 50% 稀疏度）。直觉：LLM 存在 emergent large-magnitude features——激活在固定通道持续大值，这些通道对应的权重更关键，单纯幅度剪枝（如 SparseGPT 无修正）会误剪。校准成本极低：约 128 个样本、单次前向，无梯度、无重训练（区别于 SparseGPT 的权重更新）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
X = forward_calib(model, calib_set)        # 约 128 样本，单次前向
for layer in linear_layers:
    a_j = sqrt(sum(X_j^2))                 # 输入通道激活 L2 范数（per-channel）
    S_ij = |W_ij| * a_j                    # 逐元素重要性
    mask = topk_per_row(S_ij, k=(1-p)*N)   # 每输出行保留 k 个
    W_pruned = W ⊙ mask
```
本文用法：Cassandra 用 Wanda 选择权重的 speculation 组（默认 40% 剪枝）；被剪掉的权重不丢弃，进入 verification 组供 target 前向使用——把"损失压缩"变成"无损投机"的关键。对照实验：Wanda 单独作损失压缩在推理 LLM 上精度崩塌（Deepseek-R1-Distillated-Llama3-8B：GPQA 16.0、Math-500 33.0、AIME2025 0.0 vs BF16 49.0/87.0/26.7），而 Cassandra-1（Wanda 草稿 + 全量验证）与 BF16 逐项相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：locality0/wanda（PyTorch 实现，支持 Llama/Vicuna/Bloom）；HuggingFace 有社区 Wanda 剪枝权重。使用：LLM 压缩（50–85% 稀疏）、量化前置（SqueezeLLM 结合）；Cassandra 式用法（草稿构造）与 vLLM 2:4/Wanda 稀疏集成。局限：非结构化稀疏在稠密 GEMM 上无直接延迟收益（需 2:4 结构化或专用稀疏 kernel/硬件）；离线校准存在 domain shift（μ-MoE 等指出）。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

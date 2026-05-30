## AQLM (Additive Quantization of Language Models，语言模型加性量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AQLM (Additive Quantization of Language Models) 是 Egiazarian et al. (2024) 提出的面向 LLM 的极端压缩后训练量化方法。核心思想：使用加性量化（Additive Quantization）——将每个权重向量表示为 M 个 codebook 向量之和（w ≈ Σ_{m=1}^{M} c_m[i_m]），而非传统均匀量化的单一离散值。每个 codebook 包含 2^B 个码字，总 bit 数为 M × B。例如 M=2, B=8 时，每个权重向量用 2×8=16 bits 表示，但通过 256+256 个码字的组合可实现 256×256=65536 种可能的量化值，远超均匀量化的表达能力。AQLM 通过 beam search 或 iterative optimization 为每组权重找到最优的 codebook 索引组合。该方法在 2-bit 量化下能保持 LLM 的推理能力，是 PTQ 方法中压缩率的 SOTA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# AQLM 量化（简化，实际使用 beam search / iterative descent）
# 对权重矩阵 W 按 group_size=d 分组，每组使用 M 个 codebook

# Codebook 学习（offline, per layer）
for each group g:
    # W_g ∈ R^{d}, codebooks C_m ∈ R^{d × 2^B}, m=1..M
    # 目标: min Σ_{i in group} |W_i - Σ_m C_m[i_m]|^2

    # beam search 或交替优化
    for iter in range(max_iters):
        # Fix codebooks, update assignments
        for each weight i:
            best_indices = beam_search(W_i, {C_m})
        # Fix assignments, update codebooks via k-means
        for m in range(M):
            C_m = update_codebook(W_g, assignments)

    # 存储: 每组存储 M 个 codebook (M × d × 2^B × FP16 bytes) + 每个权重的 M 个索引 (M × B bits)

# 推理时 dequant
for each group g:
    w_hat = sum(C_m[code_idx[i][m]] for m in range(M))
    # 从 codebook 中查表并求和
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/Vahe1994/AQLM。基于 PyTorch，支持 HuggingFace Transformers 模型。使用方式：加载模型 → 用校准数据（如 WikiText-2）逐层量化 → 保存量化权重 + codebooks。推理需要使用 AQLM 特定的 CUDA kernel 进行高效的 codebook 查表和累加操作。AQLM 属于 PTQ w/ FT 类别——量化后通过 fine-tuning 进一步优化 codebook 以恢复精度。在 Q-resafe 的安全评估中，AQLM 在 benign 数据集（Risk-I）上 INT4 ASR=18.5%，但在直接有害数据集（Risk-III）上飙升至 77.4%，显示校准数据集的选取对安全至关重要。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

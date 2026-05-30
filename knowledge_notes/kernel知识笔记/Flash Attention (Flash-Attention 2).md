## Flash Attention (Flash-Attention 2)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flash Attention 是 Dao et al. (2022) 提出的 IO-aware 注意力算法，通过 kernel fusion 和 tiling 策略避免将完整的 N×N attention matrix 写入 GPU HBM。核心机制：(1) Tiling——将 Q/K/V 分块，逐块计算 softmax 并增量更新（online softmax），保持 SRAM 内所有计算；(2) Recomputation——反向传播时从 SRAM 中的 Q/K/V 重计算 attention matrix 而非从 HBM 读取（avoiding O(N²) HBM access）；(3) Flash Attention 2 (Dao 2023) 进一步优化了 parallelism strategy（在 sequence length 而非 batch/head 维度做并行）和 warp 级别 work partition。

从kernel调度角度拆解术语：
```
// Flash Attention 2 forward pass (simplified per attention head):
// Q,K,V ∈ R^{N×d}, block sizes B_r (rows), B_c (columns)

parfor i from 0 to ceil(N/B_r)-1:              // outer loop over Q blocks
    Q_i = Q[i*B_r : (i+1)*B_r, :]              // [B_r, d] in SRAM
    O_i = zeros(B_r, d); l_i = -inf; m_i = zeros(B_r)

    for j from 0 to ceil(N/B_c)-1:              // inner loop over K,V blocks
        K_j = K[j*B_c : (j+1)*B_c, :]; V_j = V[j*B_c : (j+1)*B_c, :]
        S_ij = Q_i @ K_j.T                      // [B_r, B_c]

        // Online softmax rescaling (update running statistics):
        m_new = max(m_i, row_max(S_ij))
        l_new = exp(m_i - m_new) * l_i + row_sum(exp(S_ij - m_new))
        O_i = diag(exp(m_i - m_new)) @ O_i + exp(S_ij - m_new) @ V_j
        m_i = m_new; l_i = l_new

    O_i = diag(1/l_i) @ O_i                     // final normalization
    store O_i to HBM
```
Flash Attention 2 改进：
- 减少非 matmul FLOPs（从 inner loop 消除 rescaling 中的 division）
- Sequence length 维度并行化（而非 batch/head），使 block 间更独立
- Warp 调度优化减少 shared memory bank conflict

术语一般如何实现？如何使用？
CUDA 实现（https://github.com/Dao-AILab/flash-attention），通过 `flash_attn_func(q, k, v, causal=True)` 调用。支持 BF16/FP16。Mordal 在 VLM alignment 训练中使用 Flash Attention-2 加速 attention computation（`vlm_kwargs` 中通过底层框架配置）。训练时 memory 节省 O(N²)→O(N)，使更大 batch/sentence length 训练可行。H100 上支持 FP8 版本（Flash Attention-3, 2024）。PyTorch 2.0+ 通过 `torch.nn.functional.scaled_dot_product_attention` 集成（自动 dispatch 到 Flash Attention backend）。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

---

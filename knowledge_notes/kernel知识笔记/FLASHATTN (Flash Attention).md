## FLASHATTN (Flash Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FLASHATTN (Dao et al., 2022/2024) 是一种硬件感知的精确注意力计算 kernel，通过矩阵分块（tiling）和 online softmax 将注意力计算在 GPU SRAM 中完成，避免将中间 N×N 注意力矩阵写入 HBM。核心思想：(1) 将 Q、K、V 分成多个 block，每次只将一个 Q block 和一个 K/V block 加载到 SRAM 中计算局部 softmax；(2) 用 online softmax（running max + running sum）增量更新最终结果，避免存储完整 attention matrix；(3) 通过 kernel fusion 将 softmax、masking、dropout 融合到单个 CUDA kernel 中。

FLASHATTN-2 (Dao, 2024) 进一步优化了 work partitioning：将 Q 沿 sequence length 维度并行化，减少 warp 间通信。FLASHATTN-3 (Shah et al., 2024) 利用 H100 的 FP8 和异步指令进一步提升性能。

在 APB 中，FLASHATTN 被修改为支持自定义 attention mask M'，以处理 [anchor block, passing block, local context block] 三部分联合注意力计算。

从kernel调度角度拆解术语。

**FLASHATTN 的 Tiling 计算流程（简化伪代码）**：

```
// 输入：Q[N, d], K[N, d], V[N, d] 在 HBM 中
// 输出：O[N, d] 在 HBM 中
// 分块大小：B_r 行 for Q/O, B_c 行 for K/V

// 外层循环：遍历 Q 的 block
for i in 0..ceil(N/B_r):
    Q_i = load_HBM_to_SRAM(Q[i*B_r : (i+1)*B_r])     // [B_r, d]
    O_i = zeros(B_r, d)
    l_i = zeros(B_r, 1)                               // running sum
    m_i = -inf * ones(B_r, 1)                         // running max

    // 内层循环：遍历 K, V 的 block
    for j in 0..ceil(N/B_c):
        K_j = load_HBM_to_SRAM(K[j*B_c : (j+1)*B_c])  // [B_c, d]
        V_j = load_HBM_to_SRAM(V[j*B_c : (j+1)*B_c])

        // Step 1: 计算局部 attention scores
        S_ij = Q_i @ K_j^T / sqrt(d)                  // [B_r, B_c]

        // Step 2: 应用 mask（causal 或自定义）
        S_ij = apply_mask(S_ij, mask[i*B_r:(i+1)*B_r, j*B_c:(j+1)*B_c])

        // Step 3: Online softmax 更新
        m_new = max(m_i, row_max(S_ij))
        P_ij = exp(S_ij - m_new)                      // [B_r, B_c]
        l_new = exp(m_i - m_new) * l_i + row_sum(P_ij)

        // Step 4: 增量更新输出
        O_i = diag(exp(m_i - m_new)) @ O_i + P_ij @ V_j

        m_i = m_new
        l_i = l_new

    // Step 5: 最终归一化
    O_i = diag(1/l_i) @ O_i

    // Step 6: 写回 HBM
    store_SRAM_to_HBM(O[i*B_r:(i+1)*B_r], O_i)
```

**APB 中 FLASHATTN 的修改**：
APB 仅修改 attention mask 部分（Step 2），将 M' 传入以支持 [A, P_h, B_h] 三部分间的因果/跨块遮罩。核心计算流程不变。

术语一般如何实现？如何使用？

FLASHATTN 是最广泛使用的 attention kernel，在 PyTorch/HuggingFace Transformers 中通过 `pip install flash-attn` 安装。APB 基于 FLASHATTN-2 修改 mask 逻辑，通过 Python/CUDA 扩展集成。开源：https://github.com/Dao-AILab/flash-attention。APB 定制版：https://github.com/thunlp/APB。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---

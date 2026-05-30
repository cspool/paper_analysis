## State-Space Duality (SSD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

State-Space Duality (SSD) 是 Mamba-2 中提出的硬件高效 SSM 并行处理算法。其核心洞察：SSM 的计算可以表示为**半可分矩阵**（semiseparable matrix），因而可同时进行两种计算：(i) **线性（递归）计算**——逐时间步更新 hidden state，$O(L)$ 复杂度，适合推理；(ii) **二次（注意力式）计算**——通过 MatMul 并行处理整个序列，利用 GPU/Tensor Core 的 MatMul 算力。SSD 使用 block decomposition 策略将序列分段：diagonal blocks 内独立并行计算局部 SSM 输出（通过 MatMul），off-diagonal 部分分解为 right factor（block 内状态汇总）、center factor（block 间累积乘法，1-semiseparable multiplication，传递全局状态信息）、left factor（将累积全局状态投影到每个 block 的输出），最终 $Y = Y_{Diag} + Y_{Off}$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SSD 的 block decomposition 伪代码（来自 Mamba-2/HLX 论文 Fig. 6）：

```
Input: dt:[b,n,l], A:[n], B:[b,s,l], C:[b,s,l], x:[b,h,l]
Output: state_Final:[b,n,h,s], Y_Final:[b,n,h,l]

# 0. Block decomposition: l → [c, cl] (c chunks of size cl)
# 1. Chunk Cumsum kernel
sdt = softplus(dt + dt_bias)                    # [b, n, c, cl]
dA_CS = cumsum(sdt × A)                          # cumulative decay

# 2. Chunk State kernel (right factor)
decay_states = exp(dA_CS[:,:,:,-1:] - dA_CS)   # time decay within block
states = einsum(B, decay_states, sdt, x)         # (right factor) [b,n,h,s,c]

# 3. State Passing kernel (center factor)
dA_chunkCS = cumsum(zero_padding(dA_CS[:,:,:,-1]))  # inter-chunk decay
decay_chunk = causal_mask(exp(dA_chunkCS[:,:,:,None] - dA_chunkCS[:,:,None,:]))
states_int = einsum(decay_chunk, states)         # (propagated states)

# 4. BMM Chunk kernel
CB_T = einsum(C, B^T)                            # [b, c, cl, cl]

# 5. Chunk Scan kernel
L = causal_mask(exp(dA_CS[:,:,:,:,None] - dA_CS[:,:,:,None,:]))
Y_Diag = einsum(CB_T, L, sdt, x)                # diagonal output
state_decay_out = exp(dA_CS)
Y_Off = einsum(C, states_int × state_decay_out)  # off-diagonal output
Y_Final = Y_Diag + Y_Off
```

其中 `h` = head dim, `n` = num heads, `s` = state dim, `l` = seq len, `c` = num chunks, `cl` = chunk len。

对比：SSD 相比 Mamba-1 的关键改进是增加了 MatMul 操作的可并行性——通过 tile 分解将递归操作转为可并行的 MatMul blocks。但 SSD 仍有大量 element-wise 操作和 Einsum 多维张量运算，导致 memory-bound 特征和低 compute utilization（GPU 上约 26.9% on A100, 38% on H100）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPU 实现：SSD 在 GPU 上分为 5 个 CUDA kernel 执行（chunk cumsum, chunk state, state passing, BMM chunk, chunk scan），每个 kernel 之间中间数据通过 DRAM 传递。PyTorch 参考实现位于 mamba 仓库 (https://github.com/state-spaces/mamba)。由于 5 kernel 分离执行导致大量 DRAM 流量和低数据重用，HLX 提出 PipeSSD 将其融合为单 kernel 三阶段流水线（详见 PipeSSD 条目）。Fused SSD 虽然最大化数据重用，但在 GPU 上不可行——中间数据 642KB/block 超过 SM 寄存器+共享内存容量（A100: 256KB RF + 164KB SMEM, H100: 256KB RF + 224KB SMEM），导致 register spilling 和 occupancy 下降，延迟反而恶化 1.74×。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

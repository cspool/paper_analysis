## Grouped GEMM (Varlen-M / Varlen-K) for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Grouped GEMM 是将多个独立 GEMM（矩阵乘法）作为一个 batch 在 GPU 上并行执行的技术，特别适合 MoE 场景。MoE 的每个 expert 独立处理不同数量的 token（T_e 可能不同），因此需要 varlen-M Grouped GEMM（M 维度=token 数可变，N/K 维度=权重矩阵固定）或 varlen-K Grouped GEMM（K 维度=token 数可变）。具体：(1) Varlen-M Grouped GEMM：用于 forward up-proj/down-proj 和 backward activation gradient，A∈R^{T_e×d}, B∈R^{d×2n}, T_e 各 expert 不同；(2) Varlen-K Grouped GEMM：用于 backward weight gradient，A^T∈R^{d×T_e}, B∈R^{T_e×2n}，在 K 维度 reduction，T_e 不同。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SonicMoE 中 varlen-M Grouped GEMM kernel（H100, per thread block）：

```
// Persistent tile scheduler：所有 expert 的 tiles 进入统一 work queue
while (tile = atomicAdd(&work_counter, 1) < total_tiles):
    expert_id, m_tile, n_tile = decode_work_tile(tile)
    Te = expert_token_counts[expert_id]
    
    // Prologue: TMA load weight + cp.async gather input
    update_tensor_map_for_Te(desc, Te)
    cp.async.bulk.tensor.load(W_smem, W_desc[expert_id])
    cp.async.gather.load(X_smem, X, routing_idx[expert_id], m_tile)
    
    // Mainloop over K dimension
    for k in [0..ceil(d/Ktile)):
        wgmma(acc, X_smem[k], W_smem[k])
    
    // Epilogue: activation or store
    tma_store(output[expert_id], acc)
```

DeepGEMM 的 varlen-M Grouped GEMM 要求每个 expert 的 token 数必须是 M_tile 的倍数（不支持运行时 tensor descriptor 更新），SonicMoE 通过在线 descriptor 更新解决了此限制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：CUTLASS 3.x Grouped GEMM (CuTe-DSL)、DeepGEMM、SonicMoE。在 PyTorch 层面通过自定义 CUDA extension 调用。SonicMoE 提供 PyTorch nn.Module 封装，直接替换 MoE layer。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

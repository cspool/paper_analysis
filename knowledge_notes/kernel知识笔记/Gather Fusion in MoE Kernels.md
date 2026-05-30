## Gather Fusion in MoE Kernels

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gather Fusion 是将 MoE 中 token gather 操作（根据 routing mask π 将 token 从不同原始位置按 expert 收集）与 GEMM 的 GMEM-to-SMEM load 融合的技术。传统方式需先 launch gather kernel（X → X_e 连续 buffer），再在 GEMM 中加载。Gather fusion 直接在 GEMM prologue 使用 cp.async 指令按 routing index 从分散位置读到 SMEM，消除 X_e 物化（saves 2TKd bytes IO）和额外 kernel launch。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 传统（含 gather fusion 的 GEMM 需单独 gather kernel）
// Step 1: gather(X, routing_idx) → X_e  // 2TKd bytes HBM traffic
// Step 2: GEMM(X_e, W)                  // 需读 X_e

// SonicMoE gather fusion
for each expert e:
    for m_tile in expert_e_tiles:
        // cp.async 直接从原始 X 分散加载
        for t in m_tile:
            src = routing_idx[e][t]
            cp.async.load(SMEM[t], X[src])
        cp.async.wait()
        wgmma(acc, SMEM, W[expert_id])
    tma_store(output, acc)
// 总 HBM 访问相比无 gather fusion 节省 2TKd bytes
```

ScatterMoE/MoMoE 仅在 forward varlen-M 实现 gather fusion（backward 仍单独 gather）。SonicMoE 在 forward varlen-M 和 backward varlen-K (dW1/dW2) 均实现 gather fusion。Blackwell 2-CTA cluster 的特殊处理：CTA 1 的 relay warp 接收 cp.async completion → 用 mbarrier cluster-scope 转发给 CTA 0。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA PTX `cp.async.ca.shared.global` 指令（Ampere+）。ScatterMoE/MoMoE 仅 forward varlen-M fusion；SonicMoE forward + backward 全覆盖。当输入已 contiguous-packed 时无需 gather fusion。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

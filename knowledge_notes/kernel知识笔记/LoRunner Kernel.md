## LoRunner Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRunner Kernel 是 SVDQuant (Li et al., NeurIPS 2024) 提出的用于扩散模型低比特量化的融合 CUDA kernel。其核心动机：在使用 LoRA-like 低秩分支补偿量化误差时，额外分支引入的显存访问主导了推理开销瓶颈（而非计算本身）。LoRunner Kernel 通过两个融合操作消除冗余显存访问：(1) 将低秩分支的 down projection（X → Δ = X·α）与激活量化 kernel 融合——两者共享已加载的激活张量 X；(2) 将低秩分支的 up projection（Δ·β^T → 输出）与 INT GEMM 计算 kernel 融合。这样 kernel 调用次数减半，低秩分支几乎无额外内存访问开销。在 SVDQuant 中 rank=16 时额外延迟仅 5%，在 Q-VDiT 中 rank=1 时开销更低。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LoRunner Kernel 的执行流程（以 Q-VDiT rank=1 TQE 为例）：
```
// 传统非融合执行 (3次kernel调用 + 2轮显存读写):
Kernel 1: 量化激活 = quantize(load(X), s_x, z_x)       // 读写X
Kernel 2: Δ = (M ⊙ 量化激活) @ α                         // 读X, 写Δ
Kernel 3: Y = INT_GEMM(量化激活, Q_W) + Δ @ β^T         // 读X+Δ, 写Y
// 总显存流量: 3*read(X) + write(X_q) + write(Δ) + write(Y)
// 瓶颈: 激活X被读取3次

// LoRunner 融合执行 (1次kernel调用 + 1轮显存读写):
Fused Kernel:
  __global__ void lora_quant_gemm(X, W_q, s_x, z_x, α, β, M, ...):
    // 共享显存中的 tile 加载
    __shared__ float X_tile[TILE_M][TILE_K]
    __shared__ float W_tile[TILE_K][TILE_N]

    // Step 1: 加载+量化激活 tile (down projection 融合)
    X_tile = load(X)
    X_q_tile = quantize_tile(X_tile, s_x, z_x)    // 就地量化

    // Step 2: 计算 Δ = (M ⊙ X_q) @ α (rank=1)
    for each frame i:                               // 在shared mem中完成
        Δ_local[threadIdx] += (M[i] * X_q_local) * α[threadIdx]

    // Step 3: INT GEMM + low-rank up projection
    accum = 0
    for k in range(0, K, TILE_K):
        accum += X_q_tile @ W_q_tile[k]              // 量化矩阵乘
    accum += Δ_local @ β^T                            // 低秩输出 (up proj fused)

    Y = accum
    store(Y)
// 总显存流量: 1*read(X) + write(Y) (shared mem 消去中间RD/WR)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LoRunner Kernel 的实现内置于 SVDQuant 的量化框架中：https://github.com/mit-han-lab/nunchaku。SVDQuant 论文通过 CUDA kernel 实现融合，支持可变 rank（默认 16，可降至 1）。Q-VDiT 通过调用相同的 LoRunner kernel 接口，将其应用于 Video DiT 的 TQE module (rank=1)。使用方式：(1) 在 PyTorch 模型中用 `LoraLinear` wrapper 替换标准 Linear 层；(2) 设置 rank=1 和低秩参数 α, β；(3) 推理时自动调用 fused kernel。在 Q-VDiT W4A8 Open-SORA 模型中，LoRunner 融合使得 TQE 模块在实现 2.40× 显存节省和 1.35× 推理加速的同时，引入的额外延迟可忽略（<5% vs 非 TQE 的量化推理）。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---

## Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 Comet 在 **Megatron-LM** 中集成 fine-grained communication-computation fused kernel 替代默认 MoE layer 实现。Comet 提供 Python API 无缝接入 Megatron-LM 的 forward pass 流程——用户仅需将原有 MoE layer 的 expert 计算和 all-to-all 通信替换为 Comet 的 fused kernel 调用。Comet 支持所有 Megatron-LM 的混合并行策略（EP + TP + DP），并与 Megatron-LM 的 pipeline 并行兼容。

  实验比较（Figure 9，端到端 MoE 模型延迟）：
  - **Models**: Mixtral 8x7B (E=8, topk=2, N=4096, K=14336), Qwen2-MoE-2.7B (E=64, topk=4, N=2048, K=1408), Phi-3.5-MoE (E=16, topk=2, N=4096, K=6400)
  - **Baselines**: Megatron-Cutlass, Megatron-TE (Transformer Engine), FasterMoE, Tutel
  - **各种 parallelism (EP×TP)**: Mixtral: 8×1/4×2/2×4; Qwen2: 8×1/4×2/8×2/8×4 (不同 M); Phi: 8×1/8×2/4×4/8×4 (不同 M)
  - **End-to-end latency reduction**: Comet vs Megatron-Cutlass -34.1%, vs Megatron-TE -42.6%, vs FasterMoE -44.4%, vs Tutel -31.8%
  - **生产部署**: 已部署到超过万卡 GPU 的生产集群，累计节省数百万 GPU 小时

- 硬件平台是什么，配置是什么。
  **H800 集群**: 8× NVIDIA H800 GPU (80GB HBM)，NVLink 互联。CUDA 12.3, NVSHMEM 2.11, PyTorch 2.4.0。
  **L20 集群**: 8× NVIDIA L20 GPU (46GB)，PCIe 桥互联，GPU-to-GPU 带宽约 25 GB/s。

- 开源Serving框架是什么。修改了什么。
  **开源框架**: **Megatron-LM** (git-hash 6dbe4c)，用于实现 expert parallelism + tensor parallelism + data parallelism 的分布式 MoE 训练/推理。

  **Comet 修改内容（~12k lines C++/CUDA + 2k lines Python）**:
  
  1. **MoE Layer 替换**: 将 Megatron-LM 中默认的 MoE forward 流程（Router → All-to-All dispatch → Expert FFN → All-to-All combine）替换为 Comet 的 fused kernel。Comet 提供 Python API:
     ```python
     # Megatron-LM original MoE layer:
     #   token_permutation → alltoall_dispatch → expert_gemm → alltoall_combine → token_unpermutation
     
     # Comet replacement:
     #   Layer0: NVSHMEM receive (fine-grained) + GroupGEMM (tile-rescheduled) in fused kernel
     #   Layer1: GroupGEMM (column-wise) + topk-reduce + NVSHMEM send in fused kernel
     ```
  
  2. **Shared Tensor Buffer 管理**: 在每个 GPU 上通过 NVSHMEM 分配 shared memory buffer（大小 = 2×M×N bytes for BF16/FP16），作为 layer0 和 layer1 的共享缓冲区。该 buffer 跨所有 MoE layers 和 experts 全局复用，内存开销可忽略（M=4096 时 Mixtral 仅 32MB, Qwen2 仅 16MB）。
  
  3. **Parallelism 兼容**: 
     - Expert Parallelism (EP): expert 分布在不同 GPU → NVSHMEM 跨 GPU fine-grained 读写 token
     - Tensor Parallelism (TP): expert 权重沿 hidden 维度分片 → GroupGEMM tile reschedule 消除 weight switching overhead
     - TP < W 时 Megatron-LM 对非 MoE 层启用 Data Parallelism → Comet 仅修改 MoE 层，与 attention 层的 DP 兼容

  4. **Adaptive Kernel Selection**: 预编译内核库含多个 (n^c, n^p) 变体 → 部署前 profile 最优配置 → 运行时根据 M 和 parallelism 查表选择 kernel。

  5. **Production Integration**: 在生产环境 Megatron-LM 中，Comet 替换了 MoE 层的 forward/backward 实现，支持 training 和 inference。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源**: https://github.com/bytedance/flux（Project Page）。Comet 将开源。

  **Comet + Megatron-LM 端到端 MoE 执行全过程（以 Mixtral 8x7B, EP=8, TP=1, DP=1, M=4096, 一个 MoE layer 完整 forward 为例）**:

  **=== 框架层面执行流程 ===**
  
  1. **输入阶段**: batch tokens [M, N] = [4096, 4096] → 8 GPU DP=1 复制（各 GPU 持有 M/W = 512 tokens）→ Embedding → 32 Transformer layers，其中 MoE layers（alternate layers）使用 Comet

  2. **Self-Attention**（各 GPU 独立，标准 Megatron-LM）: FlashAttention on all 512 tokens → 输出 hidden states [512, 4096]

  3. **MoE Router**（各 GPU 独立）: hidden states → Gate Linear W_g[4096, 8] → Softmax → TopK(k=2) → routing_map (每个 token → top-2 experts)

  4. **MoE Layer0 - Comet Fused Kernel (NVSHMEM Receive + GroupGEMM)**:
     - Comet Python API 调用: `comet.moe_layer0_forward(hidden_states, routing_map, expert_weights)`
     - **Shared tensor 分配**: NVSHMEM buffer [M×topk, N] = [8192, 4096]，全局复用
     - **Dependency resolving**: shared tensor 沿 M 维度分解，tokens 按 source rank 排序
     - **Fused kernel launch**: 同时包含通信 TB（NVSHMEM get 拉取 remote tokens）和计算 TB（CUTLASS GroupGEMM 处理已就绪的 tiles）
     - **Tile 调度**: local token tiles 优先计算 → remote token tiles 延后（等待 NVSHMEM 完成）
     - 输出: expert FFN layer0 结果 [M×topk, K] = [8192, 14336]

  5. **MoE Layer1 - Comet Fused Kernel (Column-wise GroupGEMM + Reduce + NVSHMEM Send)**:
     - Comet Python API 调用: `comet.moe_layer1_forward(layer0_output, routing_map, expert_weights)`
     - **Shared tensor 分配**: 复用 layer0 的 NVSHMEM buffer
     - **Column-wise GEMM**: 所有 expert 并行计算第 col_block 列 → T^N 列完成后立即 top-K reduce → NVSHMEM write 回 source rank
     - 后续 col_blocks 计算与 reduce/通信重叠
     - 输出: 返回各 token source rank 的 MoE 输出 [M, N] = [4096, 4096]

  6. **Residual Add + Next Layer**: MoE 输出 + attention 输出 → LayerNorm → 下一 transformer layer

  **=== 与 Megatron-LM baseline 的关键差异 ===**
  | 阶段 | Megatron-LM Baseline | Comet |
  |------|---------------------|-------|
  | Token dispatch | NCCL all-to-all (coarse, 完整大 tensor) | NVSHMEM get (fine, token-level, fused in kernel) |
  | Expert FFN | Sequential per-expert GEMM kernel launches | Fused GroupGEMM + tile-rescheduled |
  | Token combine | NCCL all-to-all | NVSHMEM write fused with column-wise GEMM |
  | 通信-计算重叠 | 无（顺序执行） | Fine-grained overlap (hide 86.5% comm) |
  | Host scheduling | 每步多次 kernel launch from CPU | 单 fused kernel launch, kernel 内调度 |

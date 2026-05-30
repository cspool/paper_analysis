## Memory-Bound Regime (in LLM Inference)

术语解释
Memory-Bound Regime（内存受限状态）是 GPU 推理工作负载的一种执行状态，指 GPU runtime 受限于 HBM（高带宽内存）的带宽而非 Tensor Core 的计算能力。在此状态下，增加计算量（如处理更多 token）不会显著增加延迟，但增加需要从内存加载的数据量（如加载更多 expert weight）会线性增加延迟。

术语是什么？
GPU kernel 执行可分为两种类型：(1) **Compute-Bound**: Tensor Core 计算吞吐是瓶颈，内存带宽充足；GPU runtime ≈ 计算量 / 计算吞吐；(2) **Memory-Bound**: HBM 带宽是瓶颈，Tensor Core 空闲等待数据；GPU runtime ≈ 数据传输量 / 内存带宽。区分标准为 Operational Intensity（算术操作数 / 内存访问字节数）与 GPU FLOPs/Byte ratio（硬件计算能力 / 内存带宽）的比较。当 Operational Intensity < FLOPs/Byte ratio 时，kernel 处于 memory-bound 状态。

在 LLM 推理中，decode 阶段因每 token 仅处理少量计算（自回归每次 1 token/batch）且需要加载大量模型 weights（每个 MoE expert 的 FFN 权重），典型的 Operational Intensity 远低于 GPU 硬件 FLOPs/Byte ratio。METRO 论文分析（Fig. 3）：DeepSeek-V3 和 Qwen3-30B 在 decode batch size < 64 时的 Operational Intensity 比 H100/B200 的 FLOPs/Byte ratio 低两个数量级；即使 batch size 达 1024，仍低 47%–3.0x。实际 Operational Intensity 因 memory hierarchy buffer capacity 约束通常更低。

从硬件架构角度拆解术语：
Memory-bound decode 的物理过程（以 MoE expert FFN layer 为例）：

```
=== 单层 MoE FFN Decode Step（8 GPU EP, 32 tokens/GPU）===

GPU0 硬件状态:
  HBM (40GB): 存储分配到本 GPU 的 expert weights（~4 experts × ~200MB = 800MB）
  L2 Cache (40MB): 可能缓存部分 weight tile
  SMEM (164KB/SM): token activations
  Tensor Cores: 等待数据

Step 1 - Load Expert Weights from HBM → L2 → Registers:
  每个 activated expert 的 FFN weight 需从 HBM 加载
  若激活 4 个 experts（EPLB routing 下）:
    加载 4 * (gate_proj + up_proj + down_proj) weights ≈ 4 * 200MB = 800MB
    内存带宽: 600 GB/s (A100 80GB) → 加载耗时 ≈ 800MB / 600GB/s ≈ 1.33ms
    
  若激活 2 个 experts（METRO routing 下）:
    加载 2 * 200MB = 400MB → 耗时 ≈ 0.67ms
    
  → 节省的 0.66ms 即 METRO 的 FFN time reduction

  注意: token activations 的内存流量（32 tokens × hidden_dim × fp16）< 0.6% expert weight 流量
  → 这就是为何 memory-bound 下 minimizing "tokens" 无意义，minimizing "activated experts" 才有效

Step 2 - Tensor Core 计算（compute time << memory time）:
  计算: tokens @ weight → 输出
  Tensor Core 计算量: 2 * 32 * hidden_dim * intermediate_dim * num_experts ≈ 小量
  → Tensor Core 大部分时间空闲，等待 weight 加载
  
Step 3 - Output write to HBM:
  输出 activation 写回 HBM（可忽略）
```

术语一般如何实现？如何使用？
- Memory-bound vs compute-bound 的判定：计算 kernel 的 Operational Intensity (Op/s) = FLOPs / Bytes_transferred，与 GPU 硬件的 FLOPs/byte ratio 比较
- GPU 硬件趋势：B200 的计算能力增长快于内存带宽增长 → FLOPs/byte ratio 升高 → 更多 kernel 进入 memory-bound 状态
- 在 memory-bound regime 下的优化策略：(a) **减少 weight 加载量**（如 pruning, quantization, METRO's expert minimization）；(b) **增加 batch size**（摊销 weight 加载开销）；(c) **提高内存带宽利用率**（如 kernel fusion, 减少内存访问次数）；(d) **使用更快的 HBM**（如 HBM3e）
- METRO 的核心洞察：在 memory-bound decode 阶段，应最小化 activated experts 数量而非平衡 token 数，因为 token activation 的内存流量微不足道（<0.6% of weight traffic）

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

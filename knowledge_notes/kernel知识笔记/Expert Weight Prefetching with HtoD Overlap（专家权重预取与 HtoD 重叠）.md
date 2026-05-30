## Expert Weight Prefetching with HtoD Overlap（专家权重预取与 HtoD 重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Weight Prefetching with HtoD Overlap 是指在 MoE offloading 场景中，利用 GPU 计算当前 expert 的时间窗口，通过 HtoD engine 异步将下一个 expert 的权重从 host memory 预取到 GPU 的 expert buffer（$S_{Expert}$）中，使 GPU 计算和 PCIe 数据传输完全重叠，消除 GPU idle time。关键前提是 expert batch size 足够大（图 3 Right：需要 $>2^{11}$ tokens per expert），使得 GPU 计算时间 ≥ HtoD copy 时间，才能完全隐藏传输延迟。

从kernel调度角度拆解术语：

```
Expert 顺序执行的 kernel 调度时间线 (gantt):

section Expert 1
Compute Expert 1 :a1, 0, 5ms
section Expert 2  
HtoD Prefetch Expert 2 :a2, 1ms, 4ms
Compute Expert 2 :a3, 5ms, 5ms
section Expert 3
HtoD Prefetch Expert 3 :a4, 6ms, 4ms
Compute Expert 3 :a5, 10ms, 5ms

HtoD Prefetch Expert 2 与 Compute Expert 1 重叠 (时间 1-5ms)
HtoD Prefetch Expert 3 与 Compute Expert 2 重叠 (时间 6-10ms)

GPU 无 idle time —— 计算和传输完全 overlap
```

调度伪代码（MoE layer, expert 阶段）：
```
function execute_experts_sequential(B_tokens, experts[E]):
    // B_tokens: accumulated batch after router
    // S_Expert: GPU buffer for one expert's weights
    token_map = router.dispatch(B_tokens)  // token → expert mapping
    
    // Prefetch first expert
    async_htod_copy(experts[0].weights, S_Expert)
    
    for e in 0..E-1:
        wait_htod_copy()           // 等待当前 expert weights 传输完成
        tokens_e = token_map[e]    // 分配给 expert e 的 tokens
        
        if e < E-1:
            // 在计算当前 expert 时，同时预取下一个
            async_htod_copy(experts[e+1].weights, S_Expert2)  // double buffer
        
        // GPU compute: expert e forward pass
        gate_weight = experts[e].gate  // router 输出的 weight
        output[tokens_e] += gate_weight * experts[e].ffn(input[tokens_e])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **Double buffering**：使用两个 expert buffer（$S_{Expert}$ 和 $S_{Expert2}$），当前 expert 用 buffer A，HtoD 预取下一个 expert 到 buffer B，交替使用。
- **Buffer size trade-off**：$S_{Expert}$ 越大，可预取更多 expert weights（减少 on-demand wait），但挤压 attention batch size 和 KV-cache buffer。MoE-GEN 的 DAG scheduler 搜索最优 $S_{Expert}$。
- **与 Dense module 的区别**：Dense module（attention weights, shared expert）token 数固定，buffer 只需大小为单层 dense module。MoE-GEN 实证验证更大的 dense buffer 不会增加 throughput（Section 4.2）。
- **其他系统的预取策略**：
  - **MoE-Infinity**：使用 activation-aware predictor 预测将被激活的 expert，提前预取。
  - **MoE-Lightning**：profiling-based 的 memory movement schedule，通过 CUDA stream 实现 overlap。
  - **Pre-gated MoE**：在 gating 之前使用轻量级 predictor 提前预取 expert weights。
- **大 batch 下的简化**：大 batch（$B$ 大）下 token 均匀分配到各 expert，MoE-Gen 无需 predictor，直接顺序执行 experts 并预取下一个。这避免了 prediction miss 带来的 on-demand fetch 延迟。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

从kernel调度角度拆解术语：
以 MoDES 中 Group GEMM 执行一个 token 的 multi-expert 计算为例：

```
# 输入
x: token hidden state, shape [1, d]                     # 单 token (decoding)
kept_experts: 保留的 expert indices, 数量 k' <= k
W_experts: shape [k', d, d_ff]                          # k' 个 expert 权重矩阵

# === Group GEMM kernel (单次 launch) ===
grid_dim = num_SMs                                    # 或按 workload 调优
block_dim = (tile_M, tile_N) 的 thread block

for each SM in parallel:
    # 每个 SM 处理一组 sub-tile
    expert_idx = assigned_expert_range[sm_id]
    token_range = assigned_token_range[sm_id]

    for tile_m in token_range step tile_M:
        for tile_n in [0..d_ff] step tile_N:
            for tile_k in [0..d] step tile_K:
                # K 维度 reduce
                acc[tile_m][tile_n] +=
                    x[tile_m, :] @ W_experts[expert_idx][:, tile_n:tile_n+tile_N]

# 输出: y = sum(pi[i] * expert_out_i for each expert i)
# shape [1, d]
```

与 naive 逐 expert 执行的对比：
- **Naive (k 次 kernel launch)**: for each expert → launch GEMM(expert_i, x) → sync → 总耗时 = k × (kernel_time + launch_overhead)
- **Group GEMM (1 次 launch)**: 所有 k' 个 expert 在一个 kernel 内以 group sub-task 形式并发执行 → 总耗时 = max(kernel_time) + 1 × launch_overhead

MoDES 的关键实现细节：
1. **离线 profiling**：对不同代表性 activation pattern（不同 token 数 × 不同保留 expert 数组合），做 grid search 确定最优 tile size（tile_M/N/K），保证动态 workload 下的最高计算吞吐。
2. **Skipped expert 过滤**：在 dispatch 阶段，哨兵 ID（M+1）的 expert 被过滤，不进入 Group GEMM 的 sub-tasks 列表，减少无效计算。
3. **Decoding 阶段限制**：Decoding 仅处理 text token 且为 memory-bound（auto-regressive, seq_len=1），加速比低于 prefilling（batch_size=8, seq_len>1, compute-bound）。

术语一般如何实现？如何使用？
- **DeepGEMM** (DeepSeek)：CUDA JIT 编译的 FP8 Group GEMM 库，~300 行核心 kernel。支持 contiguous layout（训练/prefill，M 维分组）和 masked layout（decoding，CUDA Graph 兼容）。H100 上可达 1346 TFLOPS。
- **Triton persistent kernel** (PyTorch 官方博客)：grid 设为 SM 数量实现 persistent kernel（避免 wave quantization），配合 TMA（Hopper）和 grouped launch ordering 优化 L2 cache reuse。比手动 PyTorch loop 快 2.62×。
- **PyTorch `torch.nn.functional.grouped_mm`**：官方 API，支持 jagged token counts per expert，CUDA SM≥80 (BF16)。
- **MoDES 用法**：Group GEMM 与 GMLG/DMT 正交——GMLG/DMT 决定跳过哪些 expert，Group GEMM 加速保留 expert 的执行。离线 profiling 确定 tile size，在线推理无额外开销。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

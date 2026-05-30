## MoE Adaptive Parallelism Switching（MoE 自适应并行切换 / Zero-Cost Parallelism Switch）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Adaptive Parallelism Switching 是 TUTEL 的核心机制，允许 MoE 训练在 DP（数据并行）和 EP+DP+MP（专家+数据+模型并行混合）之间**零成本**运行时切换。传统系统中切换并行策略需要：(1) 不同的张量分片布局；(2) 参数/优化器状态迁移（Figure 4）；(3) 框架接口变更。TUTEL 通过单一统一张量布局（基于 ZeRO-DP Stage-3 风格的分片）消除了这些开销——DP 和 EP+DP+MP 共享相同的 weight slicing 格式和 data layout，仅通过控制参数 r 来切换执行流。

从kernel调度角度拆解：

Adaptive Parallelism Switching 的 tensor layout 和调度逻辑：

```
# === 统一张量布局（Single Layout） ===
# 所有 parallel 策略共享此布局:
# - Weights: ZeRO-DP Stage-3 style sharding (每个 GPU 持有 1/W 的权重分片)
# - Data: 每个 GPU 仅持有本地 tokens
# - Optimizer states: 分片存储，与 weight sharding 对齐

# 控制参数 r 的含义:
# r = 0: DP (纯数据并行)
# r = 1: EP+DP (EP+DP, 等价于 EP+DP without MP)
# 1 < r < ceil(W/E): EP+DP+MP (混合)
# r = ceil(W/E): EP+MP (等价, group_size=1 消除 DP all-gather)

# === Switchable DP (r=0) Execution Flow ===
def moe_forward_dp(input, gate_output):
    # All-gather: 收集 W 个 GPU 的完整权重分片
    W_full = all_gather(W_local, group=range(W))  # comm: O(P)
    # 用完整权重计算
    expert_output = expert_ffn(W_full, local_tokens)  # 每个GPU用自己token算所有专家
    # Backward: Reduce-scatter gradients
    grad_W = reduce_scatter(grad_W_local, group=range(W))
    return expert_output

# === Switchable EP+DP+MP (r in [1, ceil(W/E)]) Execution Flow ===
def moe_forward_epdpmp(input, gate_output, r):
    group_size = ceil(W/E) / r  # DP group size
    groups = split_gpus(W, group_size)
    
    # Step 1: LOCAL_REPEAT — 复制 gating 结果 r 份
    gate_replicated = repeat(gate_output, r)  # (T*r, ...)
    
    # Step 2: 基于 replicated gating 进行 All-to-All Dispatch
    dispatched = all_to_all(encode(input, gate_replicated))
    
    # Step 3: Expert FFN —— MP 风格: 各 GPU 计算专家的一部分
    # 仅需要 DP group 内的 all-gather 来获取权重分片
    if group_size > 1:
        W_local_group = all_gather(W_local_shard, group=groups[my_group])
    expert_out = expert_ffn_partial(dispatched, W_local_group)
    
    # Step 4: All-to-All Combine
    combined = all_to_all(expert_out)
    
    # Step 5: LOCAL_SUM — 对 r 份 replica 求和
    output = reduce_sum(combined.reshape(r, T, ...), dim=0)
    
    return output
```

通信复杂度分析（Table 4）：DP 为 O(P)，EP+DP+MP 在 r ≥ W/E 时为 O(C_g · W/E)，在 r < W/E 时为 O(C_g · r + P/E/r)。通过 Ternary Search 在 r ∈ [1, ⌈W/E⌉-1] 中找到凸函数最小值，加上边界 case r=0 和 r=⌈W/E⌉。

术语一般如何实现？如何使用？

实现基于 PyTorch 分布式通信原语（all_gather, reduce_scatter），通过控制 r 值和 group 划分实现不同并行策略的切换。预构建字典 `⌊c/R⌋ → {r*, d*, a*}` 在训练前 profiling 完成（约 (log_{1.5}⌈W/E⌉ + 2) × 4 × 2 次 trial per key）。运行时每 iteration O(1) 查表确定最优 r。TUTEL 用户通过 MoE 层 API 的 `adaptive_r` 参数控制，或设为自适应模式自动选择。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale

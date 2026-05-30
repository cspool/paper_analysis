## Memory Spilling (NKI Kernel / Trainium)

术语是什么？
Memory Spilling 在 NKI kernel 上下文中指：当 kernel 在 SBUF（State Buffer）中分配的 tile 总大小超过 SBUF 物理容量时，Neuron Compiler 自动将超出部分暂存到 HBM（High Bandwidth Memory），在需要时再加载回来。具体表现为：中间计算结果 tensor 无法全部驻留在 SBUF 中，编译器插入 spill_save（SBUF → HBM）和 spill_reload（HBM → SBUF）操作。Neuron Profile 中的指标 `spill_save_bytes` 和 `spill_reload_bytes` 直接量化 spill 的严重程度。Spilling 严重影响性能——因为 HBM 访问带宽远低于 SBUF，且 spill 引入额外的 DMA 延迟，可能导致原本 compute-bound 的 kernel 变成 memory-bound。

从kernel调度角度拆解术语：
Memory Spilling 的成因和优化伪代码：

```
// 示例: Spilling 的成因 (AccelOpt 图 8 scenario)
// Problem: tile v 和 p 需要跨越 i1 和 i2 两个嵌套循环存活
//         但 SBUF 容量不足以同时容纳两个循环的所有 tile

// Baseline kernel (有 spilling):
for i1 in affine_range(256):
    v = load(input[i1])          // v tile 在 i1 循环开始分配
    for i2 in affine_range(128):
        p = compute(v, weight[i2]) // p tile 在 i2 循环内分配
        // v 和 p 同时存活 → SBUF 不够 → v 被 spill 到 HBM
        for i2_inner in affine_range(64):
            result = matmul(p, other[i2, i2_inner])
            // 仅当 i2_inner 需要 v 时 reload → spill_reload
        store(result)
    // 若 p 需要跨 i2 迭代存活也可能 spill

// 优化后 kernel (消除 spilling via recomputation):
for i1 in affine_range(256):
    for i2 in affine_range(128):
        v_prime = load_and_recompute(input[i1], weight[i2])
        // v_prime 仅在本 i2 迭代存活 → 无需 spill
        // v' 不再需要跨循环存活
        for i2_inner in affine_range(64):
            result = matmul(v_prime, other[i2, i2_inner])
        store(result)
```

优化策略优先级：(1) 调整 tile size 减少存活张量数 → 若硬件 optimal tile (128×128+128×512) 本身已经最小化 spill，则 (2) 重组 loop ordering 改变数据生命周期（如 AccelOpt 的 Loop Invariant Code Motion），(3) recomputation trade-off（用额外计算换取 spilling 消除，需要判断计算开销是否 < spill 内存开销）。

术语一般如何实现？如何使用？
Spilling 由 Neuron Compiler 自动管理，开发者通过查看 Neuron Profile 中的 `spill_save_bytes` 和 `spill_reload_bytes` 得知 spill 情况。Spilling 触发条件是编译器判断 SBUF allocation 超出硬件限制（每 partition 192KB），此时编译器自动插入 spill 操作（无需开发者手工处理）。优化的关键在于 kernel 源码层面控制 tile 的生命周期和分配顺序——这是 AccelOpt agent 通过分析 profile 数据（spill bytes + HFU + memory write bytes）识别 spill 瓶颈并提出优化的依据。实验观察到 spilling 消除可将 kernel 性能显著提升（如 BatchMatmul+Softmax 案例中，通过先 recompute 消除 spill 再进一步优化 loop ordering）。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

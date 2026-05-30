## Hardware-Aware NAS

术语解释
Hardware-Aware NAS（硬件感知神经架构搜索）是在 NAS 搜索过程中将目标部署硬件的实际性能指标（如 latency、energy、memory）作为约束条件或优化目标，而非仅使用与硬件无关的代理指标（如 FLOPs、参数数量），确保搜索到的架构在目标硬件上真正高效。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
传统 NAS 通常使用 FLOPs 或参数量作为效率指标，但这些指标与实际硬件延迟存在 gap——不同操作在硬件上的执行时间可能差异巨大（如 memory-bound vs compute-bound 操作）。Hardware-Aware NAS 在搜索循环中直接测量每个候选架构在目标设备上的实际 latency。

AutoMoE 的实现：
- 每个候选架构在 Intel Xeon CPU 上实测推理 latency（前向传播 + beam search 翻译）
- 测量方法：batch translation 重复 100 次（partial gold）或 300 次（gold），去除 top/bottom 10% 异常值，取 truncated mean
- Latency constraint 作为硬约束：只有满足 latency ≤ threshold（如 600ms）的架构才会被纳入 population
- 也可使用 FLOPs constraint（但 latency constraint 提供更严格的硬件控制）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Hardware-Aware NAS: Latency 测量与约束

# Latency 测量（在目标设备上实测）
def measure_latency(arch_config, device, num_passes=100):
    model = build_model(arch_config)
    times = []
    for _ in range(num_passes):
        t_start = time()
        # 模拟推理：source sentence → target sentence
        output = model.translate(source_sent, beam=5, max_len=30)
        t_end = time()
        times.append(t_end - t_start)
    # truncated mean: 去除 top/bottom 10%
    times_sorted = sorted(times)
    trim = int(num_passes * 0.1)
    return mean(times_sorted[trim:-trim])

# 在演化搜索中使用 latency constraint
def filter_by_latency(architectures, constraint_ms=600):
    valid = []
    for arch in architectures:
        if arch.latency <= constraint_ms:
            valid.append(arch)
    return valid

# Latency vs FLOPs constraint 对比（Table 6 in AutoMoE）
# Latency constraint ≤ 200ms (GPU): BLEU 41.23, FLOPs 2.9G, Latency 176ms
# FLOPs constraint ≤ 3G:          BLEU 41.09, FLOPs 3.0G, Latency 216ms
# 结论: Latency constraint → 更严格控制，FLOPs 和 latency 均更优
#        FLOPs constraint → 模型"用完"FLOPs budget 但 latency 偏高
```

术语一般如何实现？如何使用？
- 需要在目标硬件上进行实际测量（profiling），而非使用代理模型预测
- AutoMoE 在搜索中使用 partially gold latency (100 passes) 加速，最终报告用 gold latency (300 passes)
- HAT (Wang et al., ACL 2020) 首次将 hardware-aware NAS 应用于 NLP Transformer
- Look-up table (LUT) 是另一种常用方法：预先测量每个基础操作在目标硬件上的 latency，然后求和估计架构总 latency
- SCAN-Edge (ICLR 2024), PEL-NAS, MicroNAS 等后续工作在 edge device 上进一步推进了 hardware-aware NAS

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---

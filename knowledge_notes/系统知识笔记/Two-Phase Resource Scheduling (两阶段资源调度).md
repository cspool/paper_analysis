## Two-Phase Resource Scheduling (两阶段资源调度)

术语解释
Two-Phase Resource Scheduling 是 Lina 提出的 Inference 端动态 expert-device 调度策略。Phase 1 基于 token-level expert selection pattern 的估算预先分配 expert-device 映射（overlap with computation），Phase 2 在 gate routing 结果与估算偏差较大时做微调（blocking but rare, ~23% cases）。

术语是什么？
Lina 的 Two-Phase Scheduling 解决 inference 中 expert popularity 倾斜且无法提前获知的问题：
- **Phase 1 (Pre-scheduling)**: 在 gate 执行前，利用 profiled expert selection patterns 和当前 batch 的 token sample paths 估算下一层 expert popularity → 按比例分配设备 → piggyback 在 all-to-all 上通信（无额外 latency）
- **Phase 2 (Fine-tuning)**: gate 执行后对比实际 routing vs 估算 → 若 top-2k expert 列表不一致 → Scheduler 重算 mapping → broadcast 新映射 → 模型 blocked 直到收到命令

从系统架构角度拆解术语。
```
# Two-Phase Scheduling Flow per MoE Layer
def two_phase_schedule(batch_tokens, current_layer, profiled_patterns):
    # Phase 1: Pre-scheduling (overlapped with computation)
    sample_paths = get_sample_paths(batch_tokens, current_layer - l, current_layer)
    estimated_popularity = estimate_from_patterns(sample_paths, profiled_patterns)
    # n_e = N * sum_t P(e) / N_t
    mapping = first_fit_decreasing(estimated_popularity)
    # Piggyback on current layer's all-to-all for communication
    piggyback_on_alltoall(mapping)  # async, no extra latency
    
    # Wait for gate execution
    actual_routing = execute_gate(batch_tokens)
    
    # Phase 2: Fine-tuning (only when estimation deviates)
    if top_2k(estimated_popularity) != top_2k(actual_routing):
        # Recompute mapping with actual popularity
        actual_mapping = first_fit_decreasing(actual_routing)
        broadcast(actual_mapping)  # blocks model computation
        # Overhead: ~6.2ms (occurs in ~23% cases)
    else:
        broadcast(resume_signal)  # overhead: ~1.45ms
```

关键参数：
- Path length l=3 (accuracy/computation tradeoff: l=1→accuracy 31.6%, l=3→60.4%, l=6→71.4%)
- Phase 2 triggering rate: ~23% (Transformer-XL), ~27-32% (BERT-Large)
- Phase 1 scheduling overhead: ~6.2ms (piggybacked on all-to-all → largely hidden)
- Phase 2 re-scheduling overhead: ~6.2ms (blocking, but infrequent)

术语一般如何实现？如何使用？
- Scheduler 在 device 0 上运行独立线程
- Phase 1 通信 piggyback: 在第一次 all-to-all 中附加 popularity 估算 → 在第二次 all-to-all 中返回 mapping
- Phase 2 通信: 独立 NCCL send/recv（tiny transfer size）
- Expert swap: 各 device 从 host DRAM 加载新 expert 权重（GPU memory 紧张时逐层加载）
- 使用 unequal split all-to-all 避免多次 process group 初始化

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---

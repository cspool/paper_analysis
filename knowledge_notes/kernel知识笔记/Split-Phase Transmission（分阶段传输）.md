## Split-Phase Transmission（分阶段传输）

术语是什么？
Split-Phase Transmission 是 PROBE 中用于在 dual-track 架构下管理 P2P expert weight prefetch 带宽的技术。核心思想是将 expert 传输拆分为两个阶段：(1) Phase 1 — 在当前层 L 的 MoE Compute 期间启动 P2P 传输，利用计算 kernel 不消耗网络带宽的特性；(2) 在 All-to-All Combine 之前暂停传输，释放 NVLink/NVSwitch 带宽给关键通信路径；(3) Phase 2 — Combine 完成后恢复传输，利用下一层 L+1 的 Attention 计算窗口完成剩余传输。

从kernel调度角度拆解术语：
```
Timeline of Split-Phase Transmission on Rank r:

  Main Stream:           |   Aux Track (Prefetch):
                         |
  MoE Compute █████████  |   Prefetch Phase 1 ░░░░░░  (P2P put expert weights)
                         |   ↑ 启动传输
  ── barrier ──          |
  All-to-All Combine ███ |   (PREFETCH SUSPENDED)     ← 释放带宽
                         |   ↑ 暂停传输
  ── barrier ──          |
  Layer L+1 Attention ██ |   Prefetch Phase 2 ░░░░░░  (resume & complete)
                         |   ↑ 恢复传输
```
Split-phase 的关键约束：(1) 传输必须可暂停/恢复——PROBE 将 expert weight 分 chunk，通过 CUDA event 机制控制传输窗口；(2) 暂停时机必须精确对齐 All-to-All Combine 的开始时间——通过 CUDA stream callback 或 pre-recorded event 实现。

术语一般如何实现？如何使用？
PROBE 通过 CUDA stream 管理和 custom Triton P2P kernel 实现。传输在独立 CUDA stream 上进行，主 stream 的 All-to-All launch 前通过 cudaStreamWaitEvent 确保 prefetch stream 的所有 in-flight 传输已完成。适用场景：任何需要在通信密集期（All-to-All）前后进行 bulk 数据传输的 MoE 系统，特别是高带宽 NVSwitch 环境下的 online expert replication。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

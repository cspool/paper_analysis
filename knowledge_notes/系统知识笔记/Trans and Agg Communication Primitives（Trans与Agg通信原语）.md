## Trans and Agg Communication Primitives（Trans与Agg通信原语）

术语是什么？
Trans 和 Agg 是 Pro-Prophet 定义的两个轻量级通信原语，用于 Lightweight Expert Placement 中的 model states 传输。Trans（Transfer）在 Forward Pass 中执行，将 expert 的 parameters 从原始 device 传输到需要该 expert 的其他 devices；Agg（Aggregation）在 Backward Pass 中执行，将该 expert 在各 device 上产生的 gradients 聚合回其原始 device。两者均仅传输 parameters/gradients（不含 optimizer states），且仅在持有该 expert input 的 device 子集内通信。

从系统架构角度拆解术语：
Trans 和 Agg 在 MoE block 中的执行时序：

```
// Forward Pass
Plan (search placement) → Trans:
  对每个需迁移的 expert e:
    send(e.params, src_device → dst_devices)
  // Trans 与 FEC/FNEC 并行（block-wise scheduling）

// Backward Pass
各 device 独立计算 e 的 gradients → Agg:
  对每个 expert e:
    reduce_scatter(e.grads, src_devices → original_device)
  // Agg 与 BEC/BNEC 并行（block-wise scheduling）
```

Performance model 中的建模：
$$T_{\mathrm{Trans}}(s, n) = \frac{s \cdot (D - n) \cdot \mathrm{size}(e_j.\mathrm{params})}{D \cdot \overline{B}}$$
$$T_{\mathrm{Agg}}(s, n) = \frac{s \cdot (D - n) \cdot \mathrm{size}(e_j.\mathrm{grads})}{D \cdot \overline{B}}$$

对比 FasterMoE 的全局传输（所有 devices 参与），Trans/Agg 仅在必要的 device 子集内通信（D-n 而非 D），通信量随 n 增大而减少。

术语一般如何实现？如何使用？
Pro-Prophet 在 PyTorch 中实现，Trans 使用 P2P send/recv 原语，Agg 使用 Reduce 或 AllReduce 原语。通信量取决于：(1) s=需迁移的 expert 数（由 planner 通过 greedy algorithm 决定）；(2) n=该 expert 无 input 的 device 数（越大越好，因为排除的 device 不参与通信）。在 Pro-Prophet scheduler 中，Trans/Agg 被拆分为 sub-operators 与 computation 重叠，重叠后的实际暴露时间 T_PTrans 和 T_PAgg 被 performance model 用于指导 planner 做更优 placement。

涉及论文标题：
- Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

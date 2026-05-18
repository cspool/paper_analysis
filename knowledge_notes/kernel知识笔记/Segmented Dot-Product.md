## Segmented Dot-Product

术语是什么？

Segmented Dot-Product 是 Uni-STC 的 SDPU 硬件执行单元采用的核心计算原语，将多个短向量点积（1×1×4 T4 tasks）按"相同 C 写入目标"的原则拼接成一个连续的执行 segment，在局部累积 partial products 后在段尾统一写回 C。与传统的"一个 task→一次 multiply-accumulate→一次 write C"模式不同，segmented dot-product 允许最多四个相邻 T4 task 的 partial products 在 SDPU 内部的累加器中被预合并（merge-forward），然后再写入 C tile 的 accumulator buffer，从而减少 C 网络上的写回流量和数据 conflict。

从kernel调度角度拆解术语：

Segmented dot-product 的执行伪代码：
```
// SDPU 执行一个 segment（包含多个连续性 T4 task）
// T4 task = {A_idx, B_idx, C_target, K_mask(4-bit)}

SDPU_execute_segment(tasks[T4_start..T4_end]):
    acc_cur = {target: -1, partial: 0}   // 当前段的部分和
    for t4 in tasks[T4_start..T4_end]:
        if t4.C_target != acc_cur.target and acc_cur.target != -1:
            // 段边界：预合并完成，写回 accumulator
            accumulator[acc_cur.target] += acc_cur.partial
            acc_cur = {target: t4.C_target, partial: 0}
        
        // 执行 1×1×4 dot-product: 对 K 维度的 4 个位置做乘加
        for k in 0..3:
            if t4.K_mask & (1 << k):
                acc_cur.partial += A[t4.A_idx + k] × B[t4.B_idx + k]
    
    // 写回最后一段
    if acc_cur.target != -1:
        accumulator[acc_cur.target] += acc_cur.partial

    // merge-forward: 最多合并 4 个不同 C_target 的 partial products 后
    // 通过 C network 写回 C tile
```

与传统 dot-product 的对比：
- **传统方法（RM-STC/DS-STC 类）**：每个 outer-product 或 row-row 任务独立计算→独立通过大 network 搬运中间乘积→独立写 C，带宽浪费大
- **Segmented dot-product（Uni-STC）**：T4 code 编码 C_target 信息→SDPU 在 pop T4 时识别段边界→同 C_target 的 T4 在内部累加→仅段结束时才触发 C 写入，网络带宽需求降低且 conflict 减少

术语一般如何实现？如何使用？

Segmented dot-product 由 SDPU 硬件实现，其内部包含：1KB accumulator buffer（存 C tile 的部分和）、merge-forward 逻辑（比较相邻 T4 的 C_target，最多合并 4 个 partial products 后写回）、Benes/MUX network（将 1-4 个 partial products 路由到正确 accumulator 槽位）。Z-shaped T4 task ordering 由 DPG 在写入 Dot-product queue 时选择，目标是将共享 A 或 B tile 的 T4 聚集以减少 A/B buffer 的重复读取。SDPU 的 merge-forward 深度（4）是 hardware cost vs. write traffic reduction 的 trade-off——更深可进一步减少 C traffic 但增加 merge-forward logic 面积和 critical path。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core


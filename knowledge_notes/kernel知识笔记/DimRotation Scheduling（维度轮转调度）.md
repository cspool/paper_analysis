## DimRotation Scheduling（维度轮转调度）

术语是什么？
DimRotation 是针对 N-D torus 网络上 All-to-All 集合通信的多维调度方法。将数据分为恰好 N 个 chunk（N 为拓扑维度数），第 i 个 chunk 按维度 i → i+1 → ... 的循环顺序执行单维 All-to-All。与 Pipeline 调度（所有 chunk 使用相同维度顺序 X→Y→Z）相比，DimRotation 实现零气泡（bubble-free）的全维度链路利用率。

从kernel调度角度拆解术语：
DimRotation 的 chunk 调度逻辑（3D torus 为例）：

```
Procedure DimRotation_Scheduler(S, N):
    Chunk_Num = N              // chunk 数 = 维度数
    Chunk_Size = S / N
    for chunk = 0 to Chunk_Num - 1:
        Schedule[chunk] = []   // 该 chunk 的维度遍历顺序
        for phase = 0 to N - 1:
            dim = (chunk + phase) % N
            Schedule[chunk].append(dim)
    // 3D torus: chunk0→[X,Y,Z], chunk1→[Y,Z,X], chunk2→[Z,X,Y]
    // 3 个 chunk 在 3 个维度上形成完美全覆盖
```

调度时间线优势（以 3D torus 为例）：
- Pipeline（6 chunks, X-Y-Z 顺序）：各 chunk 在不同维度上流水线执行，但固定顺序产生气泡——当 chunk1 在 X-dim 完成需等待 chunk0 释放 Y-dim 链路
- DimRotation（3 chunks, 轮转顺序）：每个时刻恰好有 1 个 chunk 在 X-dim、1 个在 Y-dim、1 个在 Z-dim，链路利用率恒为 100%，零调度气泡

术语一般如何实现？如何使用？
DimRotation 与单维算法（HalfRing/FoldedRing）配合使用——Scheduler 过程（Algorithm 1）先确定每个 chunk 的维度遍历顺序，在每个 phase 调用对应的单维算法生成器（HalfRing_Generator）生成该维度上逐跳传输时间表。对于异构带宽或 mixed-radix torus（如某维度无 wrap-around 链路），总时间受限于性能最差维度的完整数据传输时间。Chunk 数固定为 N（最小充分数量），避免 Pipeline 中 chunk size 选择的困境。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

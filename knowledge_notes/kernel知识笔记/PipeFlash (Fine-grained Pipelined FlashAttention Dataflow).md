## PipeFlash (Fine-grained Pipelined FlashAttention Dataflow)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PipeFlash 是 HLX 论文提出的细粒度流水线 FlashAttention-2 数据流。其核心创新是将 FA-2 的块级同步计算改为**更细粒度的行级流水线执行**——每次处理 Q block 中的两行而非整个 block，使 attention 的四个步骤（$QK^T$、local softmax、PV、update O）以流水线方式并发执行。关键效果：(i) 非 MatMul 操作（softmax, update O）的延迟被 MatMul 操作（$QK^T$, PV）的计算时间完全隐藏；(ii) 中间数据量大幅减少——score 矩阵从 FA-2 的 128KB（全 block 尺寸）降至 1KB（仅 2 行 Q），probability 矩阵同理，总计减少 $4.8\times$。PipeFlash 的 compute utilization 达到 97.5%@128K seqlen，而 FA-2 在 A100 仅约 61%，FA-3 on H100 也仅约 61%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PipeFlash 的数据流伪代码：

```
# PipeFlash: Fine-grained pipelined FA-2
# 每次流水线级处理 2 rows of Q (row granularity)
for K_block in K_V_blocks:          # iterate over KV tiles
    load K_block, V_block into GS
    for Q_block in Q_blocks:
        load Q_block
        # Pipeline: 4 stages running concurrently
        for i in range(0, block_size, 2):  # 2 rows at a time
            STAGE0 (DPE#0): Q[i:i+2] @ K_block^T  → score_2row
            STAGE1 (RVPE):  local_softmax(score_2row_prev)  → prob_2row, rescale
            STAGE2 (DPE#1): prob_2row @ V_block  → PV_2row
            STAGE3 (UpE):   rescale(O_prev) + PV_2row  → O_updated
        # After all rows processed for this Q_block
        final_rescale(O_final) → write to DRAM
```

数据流映射到硬件引擎：DPE#0 执行 $QK^T$ → 结果转发至 RVPE 执行 local softmax → 结果转发至 DPE#1 执行 PV → 结果转发至 UpE 执行 update O。四个引擎形成四级流水线，每个 cycle 同时处理不同行。与 FA-2 的关键区别：FA-2 对整个 Q block 先完成全部 QK^T，再 softmax，再 PV，再 update O——四步骤串行，非 MatMul 延迟无法隐藏。

流水线阶段平衡策略（来自 HLX Fig. 13）：当 $QK^T$ 和 PV 的 FLOPs 相同时，第一阶段（DPE#0，QK^T）和第三阶段（DPE#1，PV）的行数比例由 $\lceil block_{size} / d_{head} \rceil$ 决定。DPU 计算周期公式：$\lceil d_{reduction} / DPU_{size} \rceil \times \lceil (d_{in} \times d_{out}) / DPE_{size} \rceil$。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PipeFlash 当前仅在 HLX 自研 cycle-level simulator 中实现（论文未开源，2026年5月检索无公开仓库）。其硬件依赖 URSC 的四个引擎间直接数据转发（DPE#0→RVPE→DPE#1→UpE via NoC），无需经过 DRAM 或大容量 SRAM 中转。GPU 上难以实现 PipeFlash 的原因：(i) FA-2 已为 block 级融合，要进一步细粒度流水线需要在不同 warp 间协调异构操作；(ii) GPU 的 SIMT 执行模型假设统一 warp 执行，warp-specialized pipeline 的异构性导致调度开销；(iii) H100 TMA 适合粗粒度 tile 移动，对 PipeFlash 的细粒度 streaming/gather 访问模式支持不足。因此 PipeFlash 天然适合专用硬件加速器实现。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

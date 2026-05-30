## DualPipe Pipeline Parallelism (DualPipe 流水线并行)

术语解释
DualPipe 是 DeepSeek-V3 提出的双向流水线并行调度算法，用于 MoE 大模型分布式训练。核心创新：(1) 将每个 forward/backward chunk 拆分为 attention、all-to-all dispatch、MLP、all-to-all combine 四个组件（backward 进一步拆分为 backward for input 和 backward for weights）；(2) 通过手动调整 GPU SM 比例实现 all-to-all 和 PP 通信与计算的完全重叠；(3) 双向调度：micro-batches 从 pipeline 两端同时注入，减少 pipeline bubble。比 1F1B 和 ZB1P 的 bubble 更小，仅需 PP stages 和 micro-batches 可被 2 整除。

术语是什么？
DualPipe 的关键设计：(1) **Chunk 分解**：forward chunk = [Attention | Dispatch | MLP | Combine | PP_Comm]，backward chunk = [Attn_BW_Input | Attn_BW_Weight | Disp_BW | MLP_BW_Input | MLP_BW_Weight | Comb_BW | PP_Comm]；(2) **Overlap 策略**：一对 forward+backward chunk 中，通信（all-to-all + PP）完全与计算重叠；(3) **Bubble formula**：Bubble = (PP-1)/(PP) * (F&B-3W)/(F+B-W)，小于 ZB1P 和 1F1B；(4) **内存**：峰值激活内存 = PP/(PP+1) * (2× normal)，需保留两份模型参数，但大 EP size 下参数显存占比小，总体可接受。

从kernel调度角度拆解术语：
```
=== DualPipe 调度时间线 (8 PP stages, 双向) ===

Time ──────────────────────────────────────────────────────────────►

正向 micro-batch (forward direction):
  Stage0: [F0_Attn][F0_Disp][F0_MLP][F0_Comb]  [F1_Attn][F1_Disp]...
  Stage1:          [F0_Attn][F0_Disp][F0_MLP][F0_Comb]  [F1_Attn]...
  ...

反向 micro-batch (reverse direction, symmetric):
  Stage7: [Fr0_Attn][Fr0_Disp][Fr0_MLP][Fr0_Comb]  [Fr1_Attn]...
  ...

一对 Forward+Backward Chunk 的重叠细节:
  Forward:  ┌─Attn─┬─Disp─┬──MLP──┬─Comb─┐
  Backward: │Attn_BW_In│Attn_BW_W│Disp_BW│MLP_BW_In│MLP_BW_W│Comb_BW│
  Overlap:  │██████████│        │███████│         │        │███████│
            ↑ All-to-all communication hidden ↑   ↑ PP comm hidden ↑

SM 分区策略:
  - 计算 SMs: 112/132 (attention + MLP forward/backward)
  - 通信 SMs: 20/132 (dispatch + combine, warp specialization)
  - 动态调整: 根据实际 workload 在通信 channel 间分配 warp 数
```

术语一般如何实现？如何使用？
DeepSeek-V3 训练使用 16-way PP，配合 64-way EP（跨 8 nodes）和 ZeRO-1 DP。DualPipe 要求：(1) pipeline stages 和 micro-batches 可被 2 整除（无需 micro-batches 被 stages 整除，比 Chimera 更灵活）；(2) bubble 和激活内存不随 micro-batch 数增加而增长。与 Chimera 的对比：Chimera 要求 micro-batches 被 PP stages 整除，DualPipe 仅要求可被 2 整除。DualPipe 的关键优化：随着模型 scale up，只要维持恒定的计算-通信比，cross-node fine-grained experts 的 all-to-all 通信开销可近零。

涉及论文标题：
- DeepSeek-V3 Technical Report

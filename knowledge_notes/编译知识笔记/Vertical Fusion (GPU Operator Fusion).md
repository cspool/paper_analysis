## Vertical Fusion (GPU Operator Fusion)

术语是什么？
Vertical Fusion（垂直融合）是将 DL 计算图中多个连续算子融合为单个 CUDA "mega kernel" 的编译优化技术。融合后的 kernel 在单个 CTA 内 **时间复用**（temporal multiplexing）各算子的部分执行——CTA 先执行算子的一个 tile，然后切换到下一个算子处理同一 tile（而非完成整个算子），使 intermediate tile 数据可保持在 shared memory 或 register file 中复用，避免 round-trip DRAM。已在 TensorRT（商业）和 Welder、AStitch、Chimera 等学术工作中实现。

从编译框架角度拆解术语：
Vertical fusion 的编译流程：

```
输入: DL计算图子图 (如 Linear→ReLU→Linear)
  ↓
Step 1 - 子图选择: 识别可融合的连续算子链
  (排除: 需要全局index的gather/embedding算子, batch-norm等)
  ↓
Step 2 - Tiling分析: 确定各算子的tile size
  约束: 所有tile的shared memory需求之和 ≤ SM shared memory capacity
  例: A100 shared memory = 192 KB/SM
     Linear1 tile (128×768 BF16) = 196 KB → 超过192 KB → 无法垂直融合！
     即使仅128×768一个tile就超出capacity，导致spill to DRAM
  ↓
Step 3 - 代码生成: 生成单个mega kernel
  内部结构: for tile in tiles:
              partial_Linear1(tile)  // 仅计算该tile的部分结果
              partial_ReLU(tile)     // 在处理下一tile前立即消费
              partial_Linear2(tile)
  CTA内部通过shared memory/register传递tile data
  ↓
输出: 融合后的CUDA kernel
```

垂直融合的三大局限（Kitsune 分析）：
1. **Temporal multiplexing → 资源闲置**：任一时刻仅 TensorCore 或 SIMT Core 之一活跃。论文实测即使 TensorRT 优化后，inference 中仍存在大量 "low utilization" 时间。
2. **Shared memory capacity → 大intermediate溢写**：MLP hidden dim ≥ 768（A100 192KB 约束）时 intermediate tile 超容量，必须 spill 到 DRAM（A100 latency ≈ 409ns/572 cycles）。多 CTA/SM 分摊会进一步分割 shared memory。
3. **不支持 back-propagation**：学术工作和 TensorRT 均未展示 training 执行。Training 场景中间激活需保存、gradient reduction 等 pattern 无法通过垂直融合处理。

术语一般如何实现？如何使用？
TensorRT 通过 layer fusion + kernel auto-tuning 实现，Welder 使用 tile-graph scheduling，AStitch 使用 anchor-and-propagate 方案处理 streaming compatibility。Kitsune 将垂直融合视为 baseline 对比方案，其 spatial dataflow 在三个维度上突破垂直融合局限。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---

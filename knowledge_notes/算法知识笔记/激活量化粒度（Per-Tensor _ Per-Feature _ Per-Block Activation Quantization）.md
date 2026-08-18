## 激活量化粒度（Per-Tensor / Per-Feature / Per-Block Activation Quantization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 激活量化粒度指对激活张量施加 affine 量化（scale+zero-point，[85] AffineQuant）时共享量化参数的维度范围：per-tensor（整个张量一个 scale/zero-point）、per-feature（激活张量每个 hidden dimension 一个）、per-block（一个块共享，论文取 14 batch × 74 hidden dim，为 Qwen2.5-7B 上困惑度最优的块大小）。粒度越细越能适配通道间动态范围差异（尤其 LLM 的 outlier 激活），但元数据开销越大。论文用它评估 Qwen2.5-7B-instruct-AWQ 部署到 SiPh 加速器时的激活量化损失。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 流程（论文 Sec-III-C/IV-E）：①基线 Qwen2.5-7B-instruct-AWQ：int4 权重（AWQ）+ fp16 激活，Wikitext-2 困惑度 6.79；②激活进一步量化到 int8~int4，按三种粒度：每 hidden dim（per-feature，如 hidden=3584 维每维一个 scale）、或每 (14,74) 块（per-block）。伪代码：
  ```
  # 激活量化（per-block 例，块 (B=14, H=74)）
  for b, h in block_indexes:
      s = (max(act[b,h]) - min(act[b,h])) / (2^bits - 1)   # 每块 scale
      z = round(-min(act[b,h]) / s)                        # zero-point
      act_q[b,h] = clamp(round(act[b,h]/s) + z, 0, 2^bits-1)
  # 困惑度评估（Table-II）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：per-tensor 最简单但 outlier 破坏动态范围；per-feature 对每 hidden dim 用独立 scale；per-block 在 batch×hidden 子块共享 scale（论文块尺寸经搜索确定，更大块困惑度变差）。结果（Table-II）：int8 时 per-block 6.82 ≈ 基线 6.79，per-tensor 17.71；int5 时 per-block 182（好于 per-tensor 83150 与 per-feature 182441）但仍远差于基线；int4 全线崩溃（per-tensor 120 万）。结论：LLM outlier 激活需要细粒度量化与高精度累加（数字加速器用 FP8/24-bit 累加），SiPh 加速器无法用 ADC 量化位补回丢失动态范围，因此低比特 LLM 部署 SiPh 需进一步算法/器件改进。

涉及论文标题：
- Shining Light on Silicon Photonic DNN Accelerators

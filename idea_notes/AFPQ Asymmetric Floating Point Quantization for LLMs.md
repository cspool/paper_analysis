## AFPQ Asymmetric Floating Point Quantization for LLMs

- baseline方法是什么？
  基线方法是 **对称 FP 量化**（FP-sym），对每个 weight group 使用单一 scale 进行缩放。具体而言：`scale = max(w_max, |w_min|) / (range/2)`，所有正值和负值共享同一个 scale，即 FP 候选值的覆盖范围关于零对称。此方法无法适配 LLM 权重 tensor 中普遍存在的非对称分布——论文在 LLaMA2-7B 上随机抽样 group-size=128 的 weight groups，发现超过 50% 的组呈现最大最小值不关于零对称的特征。这导致：scale 由绝对值较大的一侧决定，另一侧的表达范围被浪费，部分 FP 候选值落在原始权重范围之外，量化精度下降。此问题在 group size 较小和 sub-4-bit 时尤为严重。另外，也尝试了仿照 INT-asym 的"一个 scale + 一个 zero_point"方法直接套用到 FP 量化，但这会使 FP 的密集表示区域从零偏移，丧失 FP 格式的核心优势。

  Baseline 全栈执行例子（LLaMA2-7B FP4-sym RTN group-size=128）：
  - 算法pipeline：weight group 内计算 `scale = max(w_max, |w_min|) / 7` → `w_4bit = round(w / scale)` → `w_deq = scale * w_4bit` → FP16 激活 × 反量化 FP16 权重 → Layer output。高级方法 GPTQ/AWQ 中使用 INT 量化（INT-asym），即 `scale = (w_max - w_min) / 15`、`zero_point = -w_min / scale`、`w_4bit = round(w / scale) + zero_point`。
  - 系统框架：基于 AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）执行量化。GPTQ 使用二阶 Hessian 信息逐列补偿量化误差，AWQ 在量化前对 salient channels 的权重乘以 per-channel scaling factor。
  - 编译框架：论文未明确说明。
  - kernel调度：FasterTransformer 中的 INT4/FP16 混合精度 GEMM kernel，INT4 权重通过 scale+zero_point 反量化到 FP16 后与 FP16 激活做矩阵乘。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AFPQ（Asymmetric Floating Point Quantization）**，核心设计：为 weight group 内的正值和负值分别设置独立 scale——`scale_pos = w_max / (range/2)` 用于正值，`scale_neg = -w_min / (range/2)` 用于负值。这解决了三个关键问题：
  (1) **适配非对称分布**：正负两侧各自缩放，FP 值的覆盖范围与原始权重范围精确匹配，不浪费表达空间。
  (2) **保留 FP 优势**：与"scale + zero_point"方法不同，AFPQ 不移动 zero point，FP 格式在零附近密集分布的优势得以完整保留，因为在 LLM 权重中大部分值集中在零附近。
  (3) **无额外存储开销**：每组存储 scale_pos 和 scale_neg 两个参数，与 INT-asym 存储 scale 和 zero_point 两个参数的开销完全相同。
  AFPQ 还作为即插即用的底层量化格式，无缝替换 GPTQ 和 AWQ 中的 INT 量化步骤，保持高层算法的二阶补偿/显著性缩放逻辑不变。

  论文方法全栈执行例子（LLaMA2-70B NF3-asym GPTQ group-size=128）：
  - 算法pipeline：加载 FP16 LLaMA2-70B 权重 → GPTQ 框架以 group_size=128 分组 → 对每组计算 `scale_pos = w_max / 3.5`、`scale_neg = -w_min / 3.5`（NF3 range/2 = 3.5）→ `w_3bit_pos = round(w_pos / scale_pos)`、`w_3bit_neg = round(w_neg / scale_neg)` → GPTQ 的二阶 Hessian 补偿：OBS 式逐列更新未量化权重以补偿当前列量化误差 → 最终每组存储 scale_pos、scale_neg 和 packed 3-bit NF3 权重 → 推理时 GPU kernel：LUT 将 NF3 索引映射为 FP16 → `w_deq = scale_pos * w_nf3_pos + scale_neg * w_nf3_neg` → FP16 GEMM。结果：WikiText-2 ppl 从 GPTQ-INT3 的 3.77 降至 3.66，MMLU 从 67.25% 升至 68.05%。
  - 系统框架：基于 AutoGPTQ 进行量化，在 FasterTransformer 中部署 NF3-asym dequantization kernel 用于推理。量化时使用 AutoGPTQ 的 GPTQ/AWQ 实现，仅修改底层 Quant/Dequant 函数。
  - 编译框架：论文未明确说明。
  - kernel调度：在 FasterTransformer 中自定义 NF-asym dequantization kernel——packed byte 解包 → LUT NF→FP16 映射 → 按正负通道分别乘 scale_pos/scale_neg → FP16 GEMM。在 A6000 GPU 上，NF4-asym LLaMA2-13B 推理延迟 485.42ms（FP16 baseline 788.01ms，1.62x speedup）。
  - 硬件架构：论文未明确说明。

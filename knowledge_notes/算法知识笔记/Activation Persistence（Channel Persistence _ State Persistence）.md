## Activation Persistence（Channel Persistence / State Persistence）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation persistence 是 Quamba2 在 Mamba2 SSM 激活中发现的两种持久化现象：(1) **Channel persistence**：SSD 输入 x 在各 channel 上的激活幅度（最大值）在不同输入样本间保持一致——即若 channel c 在 calibration 样本中是高激活 channel，则在新样本中它仍然是高激活 channel。这使得 offline calibration 得到的 channel order 在 online 推理时依然有效。(2) **State persistence**：$B_t$ 和 $C_t$ 的各 state group 的激活模式（哪些 group 数值大、哪些 group 数值小）在时间步和输入样本间保持一致——例如 group 6 在 B 中持续高激活，group 7 在 B 和 C 中持续低变化。这两种 persistence 是 sort-and-cluster 和 per-state-group quantization 技术的事实基础：若 patterns 不 persistent，offline 校准的 scale 会在 online 推理中失效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文在 Mamba2-8B 的最后一个 block 中验证了这些 properties（图 3）：(a) x 按 calibrated channel max 排序后，对任意输入样本保持大致有序（Spearman 秩相关高）；(b) 排序后的 x 输出 y 也保持顺序；(c-d) B/C 的 state group 激活在时间步上一致；(e-f) B/C 的 state group 激活在不同输入样本间一致。这些 properties 的物理直觉：SSM 的参数 $\Delta_t, B_t, C_t$ 由输入投影生成，而投影权重是固定的，因此相似的输入产生相似的参数分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文仅在 Mamba2 中验证了这些 properties，Mamba1 中的 persistence 程度未明确说明。使用方式：(1) offline calibration 收集 channel/state group 统计信息；(2) 基于 persistent patterns 设计量化分组；(3) 由于 patterns 是 persistent 的，offline 确定的 groups 和 scales 在 online 推理中直接使用，无需动态重新校准。注意这是**经验性发现**而非理论保证，论文未在不同架构/任务上验证其普遍性。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

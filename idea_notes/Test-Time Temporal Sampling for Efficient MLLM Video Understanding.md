## Test-Time Temporal Sampling for Efficient MLLM Video Understanding

- baseline方法是什么？
  Baseline 是标准 MLLM 视频推理 pipeline：从视频 F 帧中均匀或规则采样子集 N 帧，每帧经视觉编码器 E_v 编码为 M 个 patch token，共 L = N×M 个视觉 token。可选地经压缩器 C 缩短后，与文本 token 拼接送入 MLLM 做自回归解码。全栈执行例子：
  - 算法层：规则帧采样（如 Qwen2.5-VL 的均匀采样），全量 token 送入 self-attention，每个 token 与所有 L 个 token 计算 attention（O(L²)）。关键帧可能因采样稀疏被遗漏，且相邻帧的冗余 patch 浪费 attention 计算。
  - 系统框架层：VLMEvalKit 评估工具包，单次前向传播处理完整长序列。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件层：单 GPU 推理，长序列 self-attention 导致 GPU 显存和延迟瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 T3S，通过随机多试次采样和聚合，将单长序列替换为多个短且多样化的子序列，利用视频时空冗余在降低 attention 成本的同时保持或提升覆盖率。全栈执行例子：
  - 算法层：对视频随机采样 m 个独立的 N 帧子集并分别做 token 子采样（保留率 αᵢ），m 个短子序列打包在一个前向传播中处理（块对角线 attention mask），最后对各试次 logit 进行聚合（均值/置信度加权/双试次交叉验证）。Attention 复杂度从 O(L²) 降为 O(∑αᵢ²L²)，m=2、α₁=0.5、α₂=0.3 时理论降为 0.34L²。随机采样的无偏性保证多试次统计上覆盖关键时间片段，弥补单试次可能的遗漏。
  - 系统框架层：VLMEvalKit + 自定义 inference wrapper，序列打包（packing）实现多子序列单次前向传播。对 Qwen2.5-VL-7B 在 LongVideoBench 上准确率提升 3.1%，首次 token 延迟降低 2.04×。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件层：单 GPU 推理，由于每个 attention 块更小且打包后总序列长度更短（0.8L vs L），显著减少 GPU 计算和显存占用。论文 4.5 节指出单 GPU 上各 chunk 并行计算已使硬件饱和，多 GPU 可将各试次分配到独立设备进一步加速。

  **Baseline 缺陷 → 论文设计对策**：
  1. 缺陷：规则采样无语义感知 → 对策：随机多试次采样，统计上无偏覆盖时间轴，避免遗漏关键帧。
  2. 缺陷：全量 token 的 O(L²) attention 开销 → 对策：token 子采样 + 多短序列打包，每个 attention 块更小，总复杂度降低。
  3. 缺陷：需要额外训练或模型修改（如 learned selector、memory summarization）→ 对策：完全训练无关，即插即用于任何预训练 MLLM。
  4. 缺陷：学习型选择器在推理时仍需先处理所有帧再选择 → 对策：随机采样在前，无需预处理全量帧。

## AdaptVision__Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition

- baseline方法是什么？
  Baseline 方法是使用固定压缩比的 passive visual token 压缩方法，主要包括：(1) FastV —— 在 layer 2 之后按 attention score 固定剪枝 50% visual tokens；(2) SparseVLM —— 基于跨模态 relevance 选择语义相关 visual tokens，固定 50%/70% retention；(3) VisionZip —— 保留语义重要的 visual tokens，固定 50%/70% retention；(4) VisionThink —— 使用 RL 在低分辨率（25% token）和高分辨率（100% token）之间二选一，但限于 coarse-grained 决策；(5) Down-Sample baseline —— 固定 1/4 分辨率 25% tokens 直接回答。所有方法都是被动、固定比例压缩，无法自适应不同任务复杂度所需的 token 数量。

  Baseline（FastV 50%, Qwen2.5-VL-7B-Instruct）全栈执行例子：
  - 算法层：用户上传一张高分辨率图表（2048×1024） + 问题 "What is the value at Q3?" → Vision Encoder (ViT) 编码为 2678 个 visual tokens → Projector 对齐 → 与 system prompt 和 question tokens 拼接 → LLM decoder layer 1-2: 全量 visual tokens prefill → layer 2 之后: 按累积 attention score 排序，固定剪枝 50%（保留 1339 tokens） → 其余 token 的 KV cache 被丢弃 → layer 3-28: 在 1339 个 visual tokens + text tokens 上继续 prefill + decoding → 生成答案
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：使用标准 FlashAttention
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **被动固定压缩比无法适应任务难度差异**：简单任务（如 POPE 物体存在性判断）仅需极少 visual tokens 即可正确回答，但 FastV 仍保留 50% tokens 造成浪费；复杂任务（如 MathVerse 数学推理、ChartQA 图表问答）50% 压缩损失关键细节导致精度下降。
  2. **全局固定压缩忽略图像空间局部性**：视觉问答通常仅依赖图像中少数关键区域（如表格中的特定单元格、文档中的特定段落），但 uniform pruning 在整个图像上均匀丢弃 token，无法聚焦关键区域。
  3. **VisionThink 的二选一粗粒度决策**：只能整体切换低/高分辨率，无法精细定位需要高分辨率的具体区域，导致高分辨率模式下仍消耗 100% tokens。
  4. **缺乏 human-like active vision 的 coarse-to-fine 处理**：人类视觉系统先获取场景 gist（低空间频率），再选择性关注 salient 区域（高空间频率），现有方法未建模这种自适应信息获取过程。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：AdaptVision 通过 RL 训练 VLM 实现自适应 coarse-to-fine visual token 获取：
  (1) **Coarse-to-fine 框架**：始终以 1/4 低分辨率图像（25% tokens）起步，模型自主决策是直接回答还是调用 bounding box tool 裁剪高分辨率关键区域。最大化在简单样本上节省 token，在困难样本上精准获取关键高分辨率信息。
  (2) **Decoupled Turn Policy Optimization (DTPO)**：解耦策略损失为 Tool Token 和 Answer Token 分别归一化（解决 imbalanced optimization），并分别计算 outcome advantage 和 tool advantage（解决 ambiguous credit assignment），使 RL 训练稳定收敛到自适应 tool-use 策略。
  (3) **精细奖励设计**：Outcome Reward（准确度 + 格式 + 平衡惩罚防止 tool 过度使用或 lazy guessing）+ Tool Reward（裁剪区域正确性 - 面积惩罚鼓励最小化 crop 区域）。
  (4) **Fine-grained visual acquisition**：通过 bbox tool 精准裁剪关键区域，而非整体切换分辨率，使 token 消耗更具针对性。

  对比 baseline 的全栈执行例子（AdaptVision + Qwen2.5-VL-7B-Instruct，同一图表问答）：
  - 算法层：用户上传同一张高分辨率图表（2048×1024） + 问题 "What is the value at Q3?" → I_low = resize(2048×1024 → 512×256) → Vision Encoder 编码 ≈670 tokens → Projector 映射 → 拼接 x_sys + V_low + q → LLM 首轮自回归生成 → 模型推理：`<think> 需要在图表中定位 Q3 对应的数值，低分辨率下无法辨认细节...</think> <tool call>{"name":"request_local_region","arguments":{"bbox_2d":[420,180,680,320]}}</tool call>` → 从 I_high 裁剪 bbox 区域（260×140 pixels） → Vision Encoder + Projector 得到 ≈170 个 crop visual tokens → 拼接续推 → `<think> Q3 对应数值为 47.2</think> <answer>47.2</answer>` → 总 visual tokens ≈ 670 + 170 = 840（仅 31.4% of baseline 2678），远少于 FastV 50% 的 1339 tokens，且保留了关键高分辨率信息
  - 系统框架层：veRL 框架进行 RL 训练（4 节点 × 8 H20 GPU），vLLM 框架推理（temperature=0）
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention
  - 硬件架构层：论文未明确说明

  对应解决：
  | Baseline 缺陷 | AdaptVision 解决方案 |
  |---|---|
  | 固定压缩比无视任务难度 | RL 学习自适应策略：POPE 直接回答（25% token），ChartQA 频繁 tool call（~33% avg token） |
  | 全局压缩忽略空间局部性 | Bbox tool 精准裁剪关键区域，面积惩罚确保 crop 最小化 |
  | VisionThink 粗粒度二选一 | Fine-grained bbox 定位：不切换全局分辨率，仅获取必要区域的高分辨率信息 |
  | 缺乏 human-like active vision | Coarse-to-fine 框架 + RL 训练：先 gist（低分辨率）→ 选择性 attention（crop 关键区域）= 模拟人类视觉 |

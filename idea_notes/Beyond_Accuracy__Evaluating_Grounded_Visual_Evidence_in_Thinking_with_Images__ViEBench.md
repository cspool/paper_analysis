## Beyond_Accuracy__Evaluating_Grounded_Visual_Evidence_in_Thinking_with_Images__ViEBench

- baseline方法是什么？
  Baseline 方法是现有的 outcome-oriented 多模态 benchmark 评估范式，代表性 benchmark 包括：(1) V* Bench —— 191 个 QA pairs，仅做 perception 评估、无 reasoning 任务、无 BBox 标注、无过程评估；(2) HRBench —— 1600 个 QA pairs，仅做 perception 评估；(3) InfoVQA —— 2801 个 QA pairs，仅做 perception 评估；(4) VisualProbe —— 515 个 QA pairs，仅做 perception 评估。所有现有 benchmark 依赖单一最终答案 accuracy 作为唯一指标，将模型视为"黑盒"，无法诊断性能退化来源于 grounding 失败还是 reasoning 不足。此外，现有 benchmark 的任务设计以 fine-grained recognition 为主，不要求多步逻辑推理。

  Baseline（现有 accuracy-only 评估范式，以 V* Bench 为代表）全栈执行例子：
  - 算法层：给定高分辨率图像 + 问题 "What brand is the coffee machine on the third shelf?" → 模型（如 Qwen3-VL-32B）处理图像 → 生成裁剪/回答 → evaluator 仅比较最终答案与标准答案 → 输出 accuracy score。在此范式下，若模型聚焦于完全无关的区域（如天花板）但凭文本先验猜对品牌，仍被判为"正确"——accuracy 无法区分 faithful reasoning 与 lucky guessing
  - 系统框架层：VLMEvalKit (Duan et al., 2024) 作为统一评估框架，论文未涉及 Serving 框架修改
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **仅衡量最终答案、无过程级诊断**：模型通过 irrelevant visual region 猜对答案（Ungrounded Correct Answer）与基于正确视觉证据推理得到答案（Valid Grounded Reasoning）在 accuracy 指标下无法区分——传统 benchmark 系统性高估了模型的可靠性。
  2. **无视觉证据标注（BBox）**：现有 benchmark 不提供 expert-annotated 黄金 BBox，无法定量验证模型的视觉操作（zooming/cropping）是否聚焦于正确的图像区域。
  3. **任务类型单一（仅 perception）**：现有 benchmark 以 OCR、object counting、attribute recognition 等纯感知任务为主，不评估模型整合局部视觉线索与先验知识进行多步推理的能力——这与真实应用（如工业检测中判断设备故障、城市导航中识别违规行为）的巨大差距。
  4. **缺乏高空间稀疏性设计**：现有 benchmark 未刻意控制关键视觉证据的空间占比，许多问题可通过全局视图直接回答，无法强制模型执行 zooming/cropping 操作。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：ViEBench 通过以下设计实现从 outcome-oriented 到 process-verifiable 的评估范式转变：
  (1) **Expert-annotated Gold BBox**：为每个样本标注"最小不可或缺证据"的精确 BBox，作为 IoA 计算的 ground truth，使视觉操作的准确性可量化。
  (2) **Dual-Axis Capability Matrix**：基于 IoA 构建 Grounding 轴（G⁺/G⁻）× Answer 轴（A⁺/A⁻）的二维诊断矩阵，将模型表现分解为四个象限：G⁺·A⁺（faithful reasoning）、G⁺·A⁻（定位成功但推理失败）、G⁻·A⁺（无根据的正确答案——hallucinatory reasoning）、G⁻·A⁻（双重失败）。
  (3) **Reasoning + Perception 双任务设计**：引入 reasoning 任务要求模型在定位视觉线索后整合先验知识进行多步逻辑推理，暴露 accuracy-only 下不可见的 capability collapse。
  (4) **Extreme Spatial Sparsity**：关键证据平均仅占 0.32%-0.63% 图像面积，在全局视图下 sub-perceptual，强制模型执行精确本地 zooming 操作。

  对比 baseline 的全栈执行例子（ViEBench 评估, Qwen3-VL-32B-Instruct 在 reasoning 任务上）：
  - 算法层：给定 2048×1536 工业场景图像 + reasoning 问题 "Is the pressure gauge reading within the safe operating range?" → 模型自主 zooming 到 pressure gauge 区域 → B_pred 由模型生成 → evaluator 计算 IoA(B_pred, B_gt)：B_gt 为专家标注的 pressure gauge 精确 BBox（仅占 ~0.6% 图像面积） → IoA = max(coverage, concentration) → G⁺ if IoA>0.5 else G⁻ → 同时判断最终答案正确性 → 分配到四象限之一 → 汇总指标：Acc=74%, GS=68%, G⁺·A⁺=56%, G⁺·A⁻=13%, G⁻·A⁺=17%, G⁻·A⁻=15% → 诊断：13% 的样本模型成功定位了 pressure gauge 但推理错误（可能误读刻度），17% 的样本模型在错误区域获得了正确答案（可能靠文本先验"/pressure gauge 一般在安全范围内"猜测）
  - 系统框架层：各 agentic model 官方仓库的评估 pipeline，End-to-end models 使用 VLMEvalKit；论文未涉及 Serving 框架修改
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  | Baseline 缺陷 | ViEBench 解决方案 |
  |---|---|
  | 仅衡量最终答案、无过程诊断 | Dual-Axis Capability Matrix：G⁺·A⁻ 直接揭示"定位成功但推理失败"的 semantic reasoning bottleneck（Mini-o3 reasoning 高达 28%）；G⁻·A⁺ 暴露"无根据正确答案"的 superficial correctness（DeepEyes reasoning 33%） |
  | 无视觉证据标注 | Expert-annotated Gold BBox + IoA：双向 IoA 容忍 expansive coverage 和 tight focus 两种策略，Fig.5 显示 Qwen3-VL 系列使用 expansive coverage（高 IoA(B_pred,B_gt)），DeepEyes 使用 tight focus（高 IoA(B_gt,B_pred)） |
  | 任务类型单一（仅 perception） | ViEBench-R (reasoning)：要求多步逻辑推理整合视觉线索与先验知识，Mini-o3 perception Acc=73% → reasoning Acc=58%（-15%），而 GS 保持一致（78%），证明瓶颈在 reasoning 而非 perception |
  | 缺乏高空间稀疏性 | Gold BBox 平均占 0.32%-0.63%：Qwen3-VL-32B 在 perception 上 TR=93%（频繁 tool call）、reasoning 上 TR=95%，验证了稀疏性设计成功迫使模型执行 zooming |

## TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

- baseline方法是什么？
  Baseline 是使用 Qwen2.5-VL-7B（或 Qwen3-VL-8B）基础 MLLM 直接做 Video Temporal Grounding（VTG），未经专门的 VTG 后训练。模型使用 MRoPE（Multimodal Rotary Position Embedding）将视频帧的空间和时间维度编码到 position embedding 中，通过 SFT 在通用多模态数据上训练，未针对时间定位任务进行专门优化。

  Baseline 在模型推理算法-系统框架全栈的执行例子：

  - **算法/模型推理层**：给定视频 v 和文本查询 q（如 "When does the person turn off the light?"），Qwen2.5-VL-7B 按 2 FPS 采样视频帧，vision encoder 将相邻两帧合并为一个 patch embedding（每两个连续帧 merge），通过 MRoPE 注入 frame 的时间位置信息。LLM 接收 interleaved visual-text 序列，autoregressive 生成时间片段 `(t_start, t_end)`。由于 MRoPE 需要对 LLM 的 RoPE 机制进行底层修改，且未在大规模 VTG 数据上专门训练，模型缺乏精确的时间感知能力。
  - **系统框架层**：论文未明确说明。推断使用标准的 HuggingFace Transformers / vLLM 推理框架，无专门的 VTG 优化。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **TimeLens**，通过 **Data Curation + Algorithmic Design** 两大维度解决 Baseline 的三个核心缺陷：

  **缺陷 1**：Baseline 缺乏高质量 VTG 训练/评估数据 → 时间定位不准。
  **解决**：手动审查并重标注三大 benchmark（Charades-STA、ActivityNet Captions、QVHighlights）产出 TimeLens-Bench，自动化重标注训练数据产出 TimeLens-100K。原始 benchmark 中 Charades-STA 有 20.6% 样本违反 query 唯一性、34.9% 存在标注精度问题；TimeLens-100K 替代 noisy 训练数据后，mIoU 从 35.6 提升至 48.3（Charades-TimeLens）。

  **缺陷 2**：MRoPE position embedding 需要底层修改 LLM 的 RoPE 机制，难以在大规模重训中实用化，且时间感知精度不足。
  **解决**：采用 **Interleaved Textual Timestamp Encoding** — 将每帧的原始时间戳（如 "10.2s"）通过 LLM text tokenizer 转为文本 token，交错插入到对应帧的 visual tokens 之前。这无需修改 LLM 底层结构，利用 MLLM 现有的文本理解能力直接感知时间。实验中 Interleaved Textual + raw timestamp 在所有 encoding 方案中效果最优（mIoU: 48.3 vs MRoPE 36.6 on Charades-TimeLens）。

  **缺陷 3**：Baseline 的 SFT 训练范式在 VTG 任务上效率低，且 thinking-based RLVR 的显式推理对感知主导型任务无益。
  **解决**：采用 **Thinking-free RLVR (GRPO)** — 模型直接输出 `(t_start, t_end)` 而非 "think-then-answer" 格式。奖励函数简化为单一的 `r(y) = IoU(Ŝ, S*)`，无 format reward。训练效率 1.0×（约 4h10m on 8×H20），而 thinking-based RLVR 需要 1.9× 训练时间且性能更差。原因是 VTG 本质上是感知任务（perception-driven），显式推理过程被模型学成 bypass 的空操作（论文观察到 thinking 长度随训练收敛至简单内容）。

  TimeLens 方法在模型推理算法-系统框架全栈的执行例子：

  - **算法/模型推理层**：给定视频 v 和 query q，(1) 视频按 1 FPS 采样帧，每帧复制为两份以绕过 Qwen2.5-VL 的 frame merge 机制（同时让计算量等同 2 FPS）；(2) 每帧前插入文本时间戳 token（如 "10.2s"），使用 LLM text tokenizer 编码；(3) vision encoder 对每帧独立提取 visual tokens（frozen）；(4) 形成 interleaved 序列：`[prompt_tokens, timestamp_0, visual_0, timestamp_1, visual_1, ..., timestamp_T, visual_T]`；(5) LLM 在 GRPO 训练后直接 autoregressive 输出 `"The event happens in 5.2 - 12.7 seconds"`。推理时无需 thinking 过程，latency 低于 thinking-based 方法。
  
  - **训练/RLVR 层（系统框架层）**：(1) 离线阶段：用待训练模型对 TimeLens-100K 做 offline inference，计算每个样本的 difficulty `d_i = 1 - IoU(Ŝ_i, S*_i)`；(2) 按高斯分布 g(d; μ=0.05, σ=0.2) 进行 density-corrected 采样，获得约 12K 困难样本；(3) GRPO 训练：per prompt 采样 G=8 个 responses，对每个 response 计算 IoU reward，以 group 内 relative advantage `A^(g) = r^(g) - mean(r)` 更新策略；(4) 追踪 temporal IoU reward 和 group reward std，当两者 plateau 时 early stop（约 310 steps）。

  - **编译框架层**：论文未明确说明。使用 HuggingFace Transformers 标准训练流程（DeepSpeed ZeRO 或 FSDP 推断）。
  
  - **kernel调度层**：论文未明确说明。GRPO 的 8× roll-out 采样可能受益于 batch 推理加速，但论文未深入讨论。
  
  - **硬件架构层**：8 × NVIDIA H20 GPU 训练，vision encoder frozen 降低显存需求。推理时视频帧采样和 tokenization 在 CPU 预处理。

  对比总结（Baseline vs TimeLens on Qwen2.5-VL-7B, Charades-TimeLens）：
  | 维度 | Baseline (Qwen2.5-VL-7B) | TimeLens-7B |
  |---|---|---|
  | Timestamp Encoding | MRoPE (position embedding) | Interleaved Textual Prefix + Raw Timestamps |
  | 训练范式 | SFT (多任务通用) | Thinking-free RLVR (GRPO) |
  | 训练数据 | 通用多模态 SFT 数据 | TimeLens-100K (高质量 VTG 专用) |
  | 数据采样 | 随机 | Difficulty-based Gaussian Sampling |
  | 停止策略 | 固定 epoch | Early Stopping (reward plateau) |
  | mIoU | 39.3 | **48.8** (+9.5) |
  | R1@0.5 | 37.8 | **55.6** (+17.8) |

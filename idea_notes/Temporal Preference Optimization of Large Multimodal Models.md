## Temporal Preference Optimization of Large Multimodal Models

- baseline方法是什么？
  Baseline 是 video-LMM 的 **标准 Supervised Fine-tuning (SFT)** 范式：video-LMM（如 LongVA-7B、LLaVA-Video-7B）通过弱监督学习隐式获取时序定位能力——训练时依赖视频帧与文本回答之间弱对应关系的 next-token prediction loss，缺乏显式的时序对齐信号。全栈执行例子：
  - 模型推理算法层：video-LMM 将视频帧均匀采样 F=64 或 128 帧（LongVA）或 96 帧（LLaVA-Video）拼接为视觉 token 序列，与文本问题 token concatenate 后送入 LLM backbone 自回归生成回答。没有任何偏好信号告诉模型"什么样的回答是时序上更准确/更相关/更完整的"。训练数据来自视频 caption 合成的 Q&A 对（如 ShareGPT4Video），缺乏对模型时序定位能力的直接监督。
  - 系统框架层：标准 PyTorch + Transformers 训练框架，8 × A100 80GB，DeepSpeed/FSDP（论文未明确说明具体分布式框架，但 standard full fine-tuning）。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。标准 attention 推理，无特殊 kernel 优化。
  - 硬件架构层：论文未明确说明。

  核心缺陷：(1) **缺乏显式时序对齐**：SFT 的 next-token prediction 只优化文本 token 匹配概率，不区分回答是否实际参考了正确的视频时间片段，模型可能"蒙对"文字但并未真正定位到正确帧；(2) **无法区分时序答案质量**：当模型给出两个不同的回答时，SFT 无法区分哪个在时序上更准确——训练信号仅来自 ground truth 文本而非视频-回答的时序对齐程度；(3) **数据标注成本高**：若要提供显式时序标注（如 temporal grounding 标签），成本极高且难以扩展到大规模训练集。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Temporal Preference Optimization (TPO)** 通过操纵视频输入自动生成对比偏好数据，在 post-training 阶段使用 DPO 注入时序偏好先验。核心洞察：不需要人工标注时序标签——只需改变视频输入（完整/不完整/不相关帧）的"可见证据量"，就能让模型自己产生质量有差异的回答，从而自动构建时序偏好对。

  对应解决三个缺陷：
  **(1) 操纵视频输入替代人工时序标注**：TPO 不依赖人工标注时序标签，而是通过操纵视频帧本身来制造时序信息差——preferred response 使用完整相关帧（充分时序证据），dis-preferred response 使用不完整帧（部分证据）或不相关帧（无证据）。这自动确保 preferred > dis-preferred 的时序质量，无需人工判断"哪个回答时序上更好"。
  **(2) DPO 注入时序偏好信号**：将自动生成的偏好对 (V, Q, r⁺, r⁻) 送入 DPO 训练，lose 函数 L_DPO = -log σ(β(log π_θ(r⁺)/π_ref(r⁺) - log π_θ(r⁻)/π_ref(r⁻))) 直接教模型区分"好回答"和"差回答"（从时序定位角度），同时配合 SFT loss L_SFT = -log π_θ(r⁺) 保持基础生成能力。
  **(3) LLM-based Post-Filtering 保证数据质量**：GPT-4o-mini 自动过滤三类噪声——dis-preferred 偶然优于 preferred、preferred 事实错误、问题模糊——确保偏好信号的可靠性，提供可扩展的数据清洗方案。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：
    Baseline → LongVA/LLaVA-Video 采样 64/96/128 帧拼接送入 LLM，逐 token 生成回答，仅依赖 SFT 阶段的弱时序对应。
    TPO → 第一阶段（数据生成）：对每个视频 V，CogVLM2 首先生成逐帧 caption → GPT-4o-mini 根据 caption 生成问题 Q → video-LMM 用 Q + 完整帧 F 生成 preferred r⁺ → video-LMM 用 Q + 不完整帧或不相关帧生成 dis-preferred r⁻ → GPT-4o-mini 评估 r⁺ vs r⁻ 的质量并过滤。第二阶段（训练）：用 (V, Q, r⁺, r⁻) 四元组进行 DPO + SFT 联合优化，β 控制 KL 散度约束，α 控制 SFT 权重。
    关键差别：TPO 不改变推理时的模型架构，仅通过 post-training 阶段的偏好信号让模型学会在推理时更好地利用时序信息。消融实验显示 TPO 在 128 帧输入下优于 baseline 在 64 帧输入下的表现，且 TPO 随帧数增长性能持续提升而 baseline 在 >64 帧后退化。
  - 系统框架层：论文未明确说明。标准 PyTorch full fine-tuning，8 × A100 80GB，batch size 64，约 4 小时训练。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。训练在 8 × NVIDIA A100 80GB 上。

## Adaptive_Keyframe_Sampling_for_Long_Video_Understanding

- baseline方法是什么？
  Baseline 方法是 uniform keyframe sampling（UNI），即当前主流 MLLM（Qwen2VL、LLaVA-OV、LLaVA-Video）在长视频理解中的默认策略：从输入视频中以固定帧率均匀采样 M 个帧（如 32 或 64 帧），不区分每帧与 prompt 的相关性，不考虑帧在时间轴上的覆盖分布。这是一种使用 dummy VL scorer 的退化 BIN sampling 策略（$s(\mathbf{Q}, \mathbf{F}_t)$ 为常数，所有帧平等对待）。

  Baseline（LLaVA-Video-7B, 64 frames, uniform sampling）全栈执行例子：
  - 算法层：用户上传一段长视频 + 文本 prompt "What is the male protagonist's 5th outfit?" → 视频帧采样器以等间隔从视频中抽取 64 帧（如 600s 视频每 ~9.4s 抽一帧）→ SigLIP Visual Encoder 逐帧编码 → Projector 对齐到 LLM embedding space → 64 组 visual tokens 与 text question tokens 拼接 → LLM (Qwen2-7B) decoder prefill → autoregressive 生成答案 → 随机猜答错误（Figure 1 示例）
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 attention 计算，论文未明确说明具体 kernel
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **完全忽略 prompt-frame 相关性**：uniform sampling 的 $KS_M(\cdot)$ 函数不使用 $\mathbf{Q}$ 和 $\mathbf{F}$，无法区分哪些帧包含回答问题所需的信息。如图 1 所示，uniform sampling 可能抽到完全无关的帧（如风景空镜），导致 MLLM 只能随机猜测。
  2. **忽略时间轴上的信息覆盖分布**：当关键信息集中在某个时间段时（如 LongVideoBench 中大量问题聚焦于单一时钟），uniform sampling 在非关键时段浪费宝贵的 keyframe 配额；当问题需要从多个时刻收集信息时（如 VideoMME 中统计某事件发生次数），uniform sampling 可能恰好遗漏那些关键时刻。
  3. **单一策略无法适应不同问题类型的需求**：LongVideoBench 的问题倾向于 single-moment focus（"在某时间点在做什么？"），需要集中高质量帧于关键时段；VideoMME 的问题倾向于 multi-moment comprehension（"发生了多少次？"），需要在时间轴上均匀分布但仍有选择性的帧。Uniform sampling 对所有问题一视同仁。
  4. **缺乏对相邻帧冗余的感知**：均匀采样可能选中多张内容几乎相同的相邻帧，造成视觉 token 冗余，挤占其他时段有用信息的空间。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Adaptive Keyframe Sampling (AKS) 将 keyframe 选择形式化为 relevance + coverage 双目标优化，并通过 ADA（Adaptive Sampling）算法自适应平衡两者：
  (1) **Relevance 计算**：使用轻量 VL 模型（默认 BLIP ITM）计算每个候选帧与 prompt 的匹配分数 $s(\mathbf{Q}, \mathbf{F}_t)$，量化每帧对回答问题有用信息的含量。
  (2) **Coverage 估计**：基于 Ripley's K-function 的递归 binning 机制——将时间轴 [0, T) 递归二分为 bin，通过统计每个 bin 内 keyframe 数量分布的不均匀程度定义 coverage penalty，防止冗余相邻帧挤占其他时间段。
  (3) **ADA 分层优化算法**：在每层递归中计算 $s_{\text{top}} - s_{\text{all}}$（Top-M 帧与全帧平均分的差距），若差距超过阈值 $s_{\text{thr}}$，认为帧间区分度足够高 → 直接选 Top-M 高分帧（TOP 模式，最大化 relevance）；否则拆分当前 bin 并均分 keyframe 配额（BIN 模式，最大化 coverage）。这种自适应折中使得 AKS 在 single-moment 问题上表现为 TOP（集中资源于高相关性时刻），在 multi-moment 问题上表现为 BIN（在时间轴上分布覆盖）。

  对比 baseline 的全栈执行例子（AKS + LLaVA-Video-7B, 64 keyframes, 同一 long video from VideoMME）：
  - 算法层：用户上传同一段长视频 + prompt → 以 1 fps 采样全部候选帧 → 逐帧经 BLIP ITM 计算 $s(\mathbf{Q}, \mathbf{F}_t)$（预计算存储）→ ADA(level=0, L=5, s_thr=0.8, M=64): Level-0: s_top-s_all < s_thr → 拆分 [0,T/2)/[T/2,T)，各 32 帧 → Level-1: 对 [T/2,T) 内 s_top-s_all > s_thr → 直接选 Top-32 高分帧（该段包含主人公换装场景）；对 [0,T/2) 继续拆分 → ... → 最终选中 64 帧覆盖视频中的 5 次换装关键时刻 → SigLIP 编码 → Projector → 64 组 visual tokens → LLM 推理 → 生成正确答案 "The male protagonist changed 5 outfits in total"
  - 系统框架层：基于 HuggingFace Transformers，BLIP ITM 预计算 relevance score 在 MLLM 推理之前完成
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（忽略 prompt-frame 相关性）→ Relevance 计算：BLIP ITM 计算每个候选帧与 prompt 的匹配分数，确保高分帧被优先选中。消融实验：TOP sampling（仅用 relevance，λ=0）在 LongVideoBench 上从 58.9%（UNI）提升至 62.4%（+3.5%），证明 relevance 信号至关重要。
  - Baseline 缺陷 2（忽略 coverage 分布）→ Coverage 估计 + ADA 分配：递归 binning 确保 keyframes 不能全部集中在狭窄时间段。BIN sampling（仅用 coverage，λ→∞）在 VideoMME 上从 64.4%（UNI）提升至 65.2%（+0.8%），证明了 coverage 约束在多时刻问题上的价值。
  - Baseline 缺陷 3（单一策略无法适应不同问题类型）→ ADA 自适应切换：通过 $s_{\text{top}} - s_{\text{all}}$ 和阈值 $s_{\text{thr}}$ 自动在 TOP 和 BIN 模式间切换。在 LongVideoBench 上 ADA=62.7%（> TOP 62.4%，> BIN 60.2%），在 VideoMME 上 ADA=65.3%（> TOP 63.7%，> BIN 65.2%），证明自适应折中在两个互补 benchmark 上同时达到最优。
  - Baseline 缺陷 4（冗余相邻帧）→ Coverage penalty 设计：递归 binning 的 $|m_1 - m_2|$ penalty 隐含地惩罚了相邻帧的过度集中——若大量 keyframe 落入同一个 bin，penalty 增大，促使算法将配额分散到其他 bin。
  - 额外优势（prompt-adaptive）：如图 6 所示，同一段视频针对不同问题，AKS 会选出不同的 keyframe 集合——这使固定的 MLLM 能灵活适应不同场景，而 uniform sampling 对所有问题返回相同的帧集合。

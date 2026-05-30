## OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

- baseline方法是什么？
  Baseline是**Token Concatenation**方案：将视觉嵌入序列 $\mathbf{E}_v \in \mathbb{R}^{N_v \times C}$ 和音频嵌入序列 $\mathbf{E}_a \in \mathbb{R}^{N_a \times C}$ 简单拼接为 $[\mathbf{E}_v, \mathbf{E}_a]$ 送入LLM backbone。这种方案在三个层面存在缺陷：(1) **语义对齐缺失**：视觉和音频嵌入来自各自独立的projector，缺乏显式的跨模态对齐机制，导致LLM难以利用video-audio互补信息；(2) **时序关系丢失**：拼接序列中视觉和音频token的相对位置无法反映其真实时间戳关系，LLM的position embedding仅编码序列位置而非绝对时间；(3) **omni-modal数据稀缺**：缺乏高质量的视频-音频联合标注数据，现有video QA数据仅利用视觉信息，忽视了同步音频轨中的监督信号。

  全栈执行例子（Baseline - Token Concatenation处理一个带音频的视频问答请求）：
  - 算法层：ViT提取视频帧特征 → 2-layer MLP project → $\mathbf{E}_v$；AF-Whisper提取音频特征 → 2-layer MLP project → $\mathbf{E}_a$；直接拼接 $[\mathbf{E}_v, \mathbf{E}_a]$ 送Qwen2.5-7B LLM，无跨模态对齐loss，无时间编码
  - 系统框架层：论文未明确说明training framework具体名称，使用标准PyTorch分布式训练
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明，标准Transformer compute kernels在H100 GPU上运行
  - 硬件架构层：NVIDIA DGX H100集群

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文从**架构对齐**和**数据增强**两个维度解决baseline缺陷：

  **(1) 架构对齐三件套：**
  - **OmniAlignNet**：在视觉和音频projector之上增加一个共享omni-modal潜在空间，通过可学习query $\mathbf{Q}_v, \mathbf{Q}_a$ 将变长嵌入投影为固定维度 $1 \times C$，再经3层self-attention + L2归一化得到 $\mathbf{V}, \mathbf{A} \in \mathbb{R}^{K \times C}$，施加对称CLIP contrastive loss $\mathcal{L}_{\text{o-align}} = \frac{1}{2}(\mathcal{L}_{v \to a} + \mathcal{L}_{a \to v})$。这直接解决了语义对齐缺失问题——同一视频的视觉和音频嵌入被拉近，不同视频的被推远。
  - **Temporal Embedding Grouping (TEG)**：按固定时间窗口 $T_G$ 将视觉和音频嵌入分组，根据时间戳排序重组为 $[G_v^1, G_a^1, G_v^2, G_a^2, ...]$ 的omni-modal序列。这解决了相对时序关系丢失问题——LLM通过序列中embedding的位置即可感知跨模态的时间对应关系。
  - **Constrained Rotary Time Embedding (CRTE)**：基于几何级数频率 $\omega_i = 2\pi/(T_{\max}\theta^{i/C})$ 对嵌入向量进行元素级旋转变换，高频维度捕获细粒度时间差，低频维度编码粗粒度长时间关系。这解决了绝对时间戳编码问题——相比Learned Time Embedding和RoTE，CRTE的多尺度频率设计同时兼顾局部和全局时序。

  **(2) 数据增强：**
  - **Implicit Omni-Modal Learning**：利用现有video QA数据中自带的音频轨进行隐式omni-modal监督（先前的video LLM仅用视觉信息，浪费了同步音频中的监督信号）。
  - **Explicit Omni-Modal Learning (Data Engine)**：视觉captioning模型 + 音频captioning模型独立生成标注 → LLM进行跨模态纠错和总结（解决"modality-specific hallucination"：仅凭视觉误判场景，仅凭音频误判内容）→ Reasoning LLM合成QA对。生成3.6M omni-modal对话数据。

  **(3) GRPO Omni-Modal Reasoning Post-Training：**
  将GRPO的输入空间扩展为 $\{q_t, q_v, q_a\}$（文本+视觉+音频），18K omni-modal MCQ数据，rollout=8，证明了audio input对RL训练的boost效果（accuracy reward收敛+0.1高于video-only）。

  全栈执行例子（论文方法 - OmniVinci处理同一个带音频的视频问答请求）：
  - 算法层：ViT提取视频帧 → MLP project → $\mathbf{E}_v$；AF-Whisper提取音频(16kHz STFT→Conv+Transformer→750 tokens/30s)→Max Pooling压缩(375 tokens/30s)→MLP project → $\mathbf{E}_a$；**OmniAlignNet**对比学习对齐 $\mathbf{V}, \mathbf{A}$；**TEG**按 $T_G$ 窗口分组重排token序列；**CRTE**对每个embedding施加时间戳旋转编码；送入Qwen2.5-7B LLM autoregressive生成文本响应；可选TTS模块生成语音输出
  - 系统框架层：7阶段渐进式训练（Vision: 5 stages → Audio: 2 stages → Omni-Modal Joint Training: 200B tokens），Long-RL framework for GRPO post-training
  - 编译框架层：论文未明确说明。部署时使用AWQ (W4A16 LLM + W8A8 vision/audio towers) + TinyChat engine
  - kernel调度层：论文未明确说明。RTX 4090/A100/L40s上运行，TTFT ~160ms for 16-frame video+audio
  - 硬件架构层：NVIDIA DGX H100训练集群；部署支持RTX 4090 (24GB)、A100、L40s

  关键结果：OmniVinci用0.2T tokens（Qwen2.5-Omni的1/6）取得Dailyomni +19.05, MMAR +1.7, Video-MME +3.9的显著提升。Ablation验证：Token Concatenation Baseline平均45.51 → +TEG 47.72(+2.21) → +CRTE 50.25(+4.74) → +OmniAlignNet 52.59(+7.08)，每项技术均有显著增益。

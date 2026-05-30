## PEARL__Personalized_Streaming_Video_Understanding_Model

- baseline方法是什么？
  Baseline 是现有的**个性化图像/视频理解方法**，可分为两类：
  1. **离线模型**（LLaVA-OV-7B, Qwen2-VL-7B, InternVL3.5-8B, Qwen3-VL-8B）：均匀采样 64 帧（frame-level）或 64 秒窗口 1fps（video-level）处理视频，无显式记忆机制。每次查询独立处理当前窗口，无法跨时间步维护概念和视觉证据。
  2. **在线模型**（ReKV, StreamForest-7B, TimeChat-Online-7B）：支持流式视频输入和多轮对话，但将历史信息压缩为固定大小的状态表示，缺乏概念级（concept-grounded）检索能力，不能精确检索与用户定义概念相关的历史视觉证据。

  Baseline 全栈执行例子（以离线 LLaVA-OV-7B 64帧均匀采样为例）：
  - **算法层**：长视频 → 均匀采样 64 帧（丢失关键时刻） → ViT 编码所有帧 → LLM 自回归生成答案。所有查询独立处理，无概念记忆，无历史检索。概念定义信息在下一个查询中消失，必须重复提供。问题：(a) 64 帧限制无法保留长范围历史证据用于 Past-Time QA；(b) 无概念存储机制使 Real-Time QA 无法可靠地将个性化名字链接到视觉实体；(c) 无检索机制意味着历史视觉证据完全不可访问。
  - **系统框架层**：PyTorch + HuggingFace Transformers。视频帧通过 OpenCV/decord 解码，按 instruction template 与文本 token 拼接送入 VLM。无流式视频管理、无外部记忆模块。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 推理。
  - **kernel调度层**：标准 Transformer attention kernel（Flash Attention 或 PyTorch native SDPA）。无自定义 kernel。
  - **硬件架构层**：NVIDIA H200 GPU。

  在线模型（如 ReKV）虽然支持流式输入，但其 KV-cache 压缩将历史信息丢失到固定大小的压缩状态中，无法为 Past-Time QA 提供精确的检索式历史证据访问。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PEARL 通过 **Dual-grained Memory System + Concept-aware Retrieval Algorithm** 解决 baseline 的三大核心缺陷：

  **(a) 缺陷1：无概念记忆 → Concept Memory（概念记忆）**
  Baseline 在每次查询中都不知道用户定义了哪些个性化概念。PEARL 的 Concept Memory 在 Concept-Definition QA 触发时，从当前 clip 提取视觉证据（frame-level 取最后一帧，video-level 取 clip），用 VLM 生成聚焦永久/稳定特征的紧凑文本描述（frame-level：性别/面部/发型/体型；video-level：核心运动学/动作序列），将概念名、视觉证据和文本描述结构化存储。后续查询通过概念名匹配快速检索 Csub，使模型在任何时间点都知道"Adaliz 是一个年轻女性，长黑发"。消融实验：加 Concept Memory 使 Real-Time 准确率从 15.84% 飙升至 51.41%（+35.57%）。

  **(b) 缺陷2：无法访问历史视觉证据 → Streaming Memory + Concept-aware Retrieval**
  Baseline 的 64 帧窗口无法覆盖 Past-Time QA 所需的历史 clip（例如查询"Adaliz 做饭时穿什么颜色"需要检索 30 分钟前的 cooking 场景）。PEARL 的 Streaming Memory 将视频流增量归档为 (clip, embedding) 对，Concept-aware Retrieval 通过 Query Rewriting 将概念名替换为视觉描述后编码为嵌入，与 Streaming Memory 中所有历史 clip 嵌入做余弦相似度匹配，精确检索 Top-K 最相关历史 clip。消融实验：加 Streaming Memory 使 Past-Time 准确率从 25.43% 提升至 45.69%（+20.26%）。

  **(c) 缺陷3：个性化名称无法被通用嵌入模型理解 → Query Rewriting（查询重写）**
  通用多模态嵌入模型未见过用户定义的个性化名称（如"Adaliz"）。PEARL 在检索前将查询中出现的概念名替换为对应的视觉描述文本（如"a young female with long black hair"），使重写后的查询能被嵌入模型有效编码，从而与 clip 嵌入进行语义匹配。消融实验：加 Query Rewriting 进一步提升 Avg 准确率 4.28%。

  对比 baseline 的全栈执行例子（PEARL + Qwen3-VL-8B, 1fps）：
  - **算法层**：流式视频 → PySceneDetect 检测场景边界（HSV delta threshold=27.0, min 1s/max 8s clip） → 每个新 clip 经 Qwen3-VL-Embedding-2B 编码为 embedding → 存入 Streaming Memory。用户定义概念时：从当前 clip 提取视觉证据 → VLM 生成概念描述 → 存入 ConceptMemory{(name, evidence, desc)}。用户查询时：(1) 从 Q 中提取概念名 → 检索 ConceptMemory 获取 Csub；(2) VLM 重写 Q → Q̃（替换概念名为描述）；(3) Qwen3-VL-Embedding-2B 编码 Q̃ → e^Q → 与 StreamingMemory 中所有 ei 计算 cosine similarity → Top-K=4 clips + N=1 邻接扩展 → Vcontext；(4) Csub + Vcontext + X^tq + Q → VLM decoder → 生成答案 A。全程 training-free，不更新任何模型参数。
  - **系统框架层**：PyTorch + Qwen3-VL-Embedding-2B + PySceneDetect。修改点：(a) 在 VLM 推理 pipeline 外挂 Dual-grained Memory System（StreamingMemory 增量归档线程 + ConceptMemory 注册/检索接口）；(b) 在 VLM 推理前插入 Concept-aware Retrieval 预处理步骤（Query Rewriting + embedding-based clip retrieval）；(c) 多 GPU 评估 pipeline（server/ 启动 VLM 和 embedding server，scripts/ 协调多 GPU 并行推理 → eval.py 聚合评估指标）。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 推理。
  - **kernel调度层**：标准 Transformer attention kernel。PEARL 的额外计算来自：(a) embedding 编码（Qwen3-VL-Embedding-2B 前向，~constant cost）；(b) cosine similarity 矩阵计算（clip 数 × d_embed 维向量，线性于 StreamingMemory 规模）；(c) VLM Query Rewriting（一次额外 VLM 推理，仅处理文本长度级别的 token）。延迟分解（Fig.5）显示 PEARL 核心模块（Concept Retrieval + Query Rewriting + Streaming Memory Retrieval）的延迟极低且跨模型恒定，主要瓶颈仍是 LLM 推理。
  - **硬件架构层**：NVIDIA H200 GPU。PEARL 与 baseline 共享相同硬件，额外的检索和重写模块仅引入可忽略的计算开销。LLaVA-OV-7B+PEARL 端到端延迟 775ms（vs 670ms baseline），以 105ms 额外延迟换取 8.55% 平均准确率提升。

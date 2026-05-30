## WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

- baseline方法是什么？
  Baseline是**基于文本摘要的固定尺度记忆检索方法**，代表工作包括EgoRAG、HippoMM、M3-Agent等。这些方法的通用设计是：
  (1) 将长视频按固定时间粒度（如30s）分段，每段用Video LLM生成caption/text摘要；
  (2) 构建单一文本形式的外部记忆（层级摘要、知识图谱或实体关系图）；
  (3) 检索时按固定策略返回预定数量的文本片段（如3个30s片段），用这些文本片段作为LLM推理上下文。
  
  **全栈执行例子（以M3-Agent/EgoRAG为代表性baseline）**：
  - **算法pipeline层**: 视频V（周级→44.3h）按固定30s粒度划分为~5,316段，每段采样0.5fps→768帧上限，VideoLLM输出caption文本。对caption按层级聚合生成event摘要（EgoRAG三层层级：moment→event→activity），构建纯文本知识图谱（如HippoRAG PPR图）。用户query q到来时，用固定k=3检索文本片段→LLM基于3个30s片段文本生成答案。缺点：(a) **纯文本丢失视觉细节**——同一段caption"[I stand and walk to dining table, discuss AC temperature with Shure]"无法传达场景中正在"吃火锅(hot pot)"的视觉证据；(b) **固定时间粒度无法适配不同问题**——"Where did I leave my glasses?"只需几秒，"What happened in the soccer match second half?"需要数十分钟——固定30s粒度要么信息不足要么冗余噪声；(c) **固定检索策略**对所有问题统一返回固定数量片段，无法按需动态扩展。
  - **系统框架层**: 论文未明确说明具体Serving框架。baseline方法通常将记忆构建作为离线预处理（caption生成→图构建），在线推理时用LLM API（如GPT-5/Gemini）进行检索+RAG生成。预处理开销大（如M3-Agent需要实体识别），但检索时延较低。
  - **编译框架层**: 论文未明确说明。
  - **kernel调度层**: 论文未明确说明。检索基于图算法（PPR等）和向量相似度计算，使用标准CPU/GPU运算。
  - **硬件架构层**: 论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  WorldMM通过三大设计解决上述三个缺陷：
  
  **(A) 多模态记忆替代纯文本记忆** — 解决缺陷(a):
  WorldMM显式构建三类互补记忆：(i) Episodic Memory（文本KG，事实事件），(ii) Semantic Memory（文本KG，长期关系/习惯），(iii) Visual Memory（特征向量 + 原始帧索引）。与baseline纯文本相比，Visual Memory保留了baseline caption丢失的视觉细节（如场景外观、物体状态、空间关系）。当问题需要视觉推理（如"eating hot pot"需要识别桌上火锅）时，Retrieval Agent能自主从Visual Memory获取帧级视觉证据，而非仅依赖可能缺失视觉信息的文本caption。
  
  **(B) 多时间尺度记忆替代固定粒度** — 解决缺陷(b):
  Episodic Memory构建多尺度知识图谱 M_e = {G_{30s}, G_{3min}, G_{10min}, G_{1h}}（以EgoLifeQA为例）。检索时采用coarse-to-fine策略：先从各尺度用PPR分别检索top-k候选，再由LLM cross-scale reranker联合评估所有尺度的候选，选择最相关的时间范围并输出top-m结果。相比baseline只能返回30s固定片段，WorldMM能动态组合小时级摘要（如"下午的会议讨论了什么"）和秒级细节（如"19:30我把眼镜放在哪儿"）。
  
  **(C) 自适应多轮检索替代固定策略** — 解决缺陷(c):
  Retrieval Agent以LLM驱动多轮迭代检索。每轮Agent决定：(i) 用哪个记忆类型（episodic/semantic/visual），(ii) 用什么搜索关键词，(iii) 是否已收集足够信息（输出STOP）。通过迭代式策略，WorldMM能根据query复杂度自适应扩展检索范围：简单问题1-2轮即够（STOP早），复杂问题多轮深化（最多5轮）。EgoLifeQA上5轮vs1轮提升9.3%。特别地，Retrieval Agent可以跨记忆类型混合检索——先用episodic memory定位时间戳("DAY2 18:34:01")，再用visual memory按时间戳获取对应帧——实现文本+视觉的跨模态推理。

  **全栈执行例子（WorldMM）**：
  - **算法pipeline层**: 视频V（44.3h，EgoLifeQA）→ 离线构建：(i) M_e = {G_{30s}, G_{3min}, G_{10min}, G_{1h}} [四尺度的(entity, action, entity)三元组KG]，(ii) M_s [Consolidation增量更新的语义关系KG]，(iii) M_v = M_v^f ∪ M_v^I [VLM2Vec-V2编码特征 + 时间戳帧索引]。在线推理：query q="What were we doing last time we discussed the air conditioning temperature?" → Round1: Search/Memory:Episodic/Query:"discussing AC temperature"，检索到[DAY2 13:36-13:39]文本证明讨论了AC但未说明活动 → Round2: Search/Memory:Episodic/Query:"air conditioning"（更泛化），检索到[DAY2 18:34:01-18:34:29]描述"Shure set AC to 26°, eating..."，文本指向食物但未明确具体种类 → Round3: Search/Memory:Visual/Query:"DAY2 18:34:01-18:34:29"，获取对应帧图像（画面中桌上火锅+投影屏幕）→ Round4: Decision:Answer，基于文本+视觉综合证据选(A) Eating hot pot。整个过程是文本→文本+视觉的跨模态证据链构建。
  - **系统框架层**: 论文未明确说明具体Serving框架。WorldMM的预处理与baseline类似（caption→triplet extraction→graph construction→semantic consolidation），但论文指出支持在线操作——记忆每10s固定间隔更新，每段预处理可在间隔窗口内完成，consolidation增量合并无需重建。LLM推理使用GPT-5 API（闭源）或Qwen3-VL-8B本地部署。
  - **编译框架层**: 论文未明确说明。
  - **kernel调度层**: 论文未明确说明。PPR图检索、cosine向量检索等使用标准库实现，无自定义kernel。
  - **硬件架构层**: 论文未明确说明。

  设计思路核心：**用多模态+多尺度+自适应迭代三元设计替代单一的文本固定检索范式**。关键在于将"记忆构建(what to remember)"和"记忆检索(how to retrieve)"解耦并对每个维度进行专门化设计——记忆维度上分离episodic/semantic/visual三类互补表示，时间维度上构建多粒度KG层级，检索维度上用LLM agent实现跨类型/跨尺度的自适应多轮调度。这使得模型能根据query特性按需组合不同记忆类型和时间粒度，避免了baseline"一刀切"策略带来的信息缺失或噪声干扰。

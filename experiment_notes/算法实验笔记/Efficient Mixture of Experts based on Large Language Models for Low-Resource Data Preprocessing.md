## Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 MELD（Mixture of Experts on Large Language Models for Data Preprocessing），一个基于 MoE 架构的通用低资源数据预处理（DP）求解器。核心贡献包括：(1) 增强型 RAG 系统用于跨域检索与自标注；(2) 启发式 meta-path 搜索用于数据增强；(3) 基于信息瓶颈理论的 expert 精炼；(4) 独立 router network 基于对比学习训练实现 top-k expert 调度。实验在 19 个数据集、10 个 DP 任务上与 non-LLM baseline、LLM baseline 和 Mixtral 8×7B 进行 few-shot 性能比较。

- 硬件平台是什么，配置是什么。
  单机：256GB RAM、32 处理器 Intel Xeon Gold 5320 CPU @2.20GHz、4× NVIDIA GeForce RTX 3090 GPU（24GB VRAM）。训练与推理均在 consumer-level GPU 上完成。

- 模型是什么。数据集和bench分别是什么。
  **Backbone 模型**：RAG 系统使用 bge-large-en（Sentence-BERT），Expert 模型使用 Mistral-7B。LoRA（Low-Rank Adaptation）进行参数高效微调。对比 baseline 包括：
  (1) Non-LLM 方法：Raha(ED)、IPM(DI)、DeepBlocker(Blocking)、Ditto/PromptEM(EM)、Baran/Garf(DC)、RECA(CTA)、TURL(RE/EL)、CONSchema/SMAT(SM)、MAVE(AVE)；
  (2) LLM 方法：JellyFish(13B)、TableLLaMa(7B)、ExtractGPT；
  (3) MoE 模型：Mixtral 8×7B。
  **19 个数据集**覆盖 10 个 DP 任务：EM(Amazon-Google, Walmart-Amazon, WDC-All, Ant-Buy)、Blocking(Semi-Text-Watch, Semi-Text-Computer)、DC(Hospital, Rayyan, Beer)、ED(Hospital, Rayyan, Beer)、CTA(SemTab19, WebTables)、RE(WikiGS-RE)、EL(WikiGS-EL)、SM(CMS, Synthea)、DI(Walmart, Amazon, Restaurant)、AVE(OA-mine)。
  **指标**：F1 score(EM/ED/DC/SM)、accuracy(DI/AVE)、top-1 accuracy(EL)、top-1 recall(Blocking)、micro-F1(CTA/RE)。Few-shot 设置为 ≤10% 标注数据。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/authurlord/MELD。使用 LLaMA-Factory 进行 expert 训练，Punica + vLLM 进行多 LoRA 推理。

  **算法 Pipeline 伪代码**（MELD 训练流程）：
  ```
  Input:  Tasks T = {T_1,...,T_n}, few-shot labeled data X = {X_1,...,X_n}
  Output: Expert set E^{aug}, Router network N

  // Step 1: Enhanced RAG for Cross-domain Retrieval
  for each task T_i:
    初始化 sentence-bert 模型 M_RAG
    for each query q in X_i:
      搜索正例集 P_q（对齐的 entries），负例集 N_q（未对齐的 entries）
    Fine-tune M_RAG with contrastive loss:
      min Σ_{p∈P_q} -log( exp(<emb_q, emb_p>/τ) / Σ_{p'∈P_q∪N_q} exp(<emb_q, emb_{p'}>/τ) )
    用 fine-tuned M_RAG 自标注未标注数据 X̃_i
    通过 query 变换跨任务扩充标注数据 X_i → X_i (enlarged)
    初始化 expert e_i: 用 X_i 对 Mistral-7B 做 LoRA fine-tune

  // Step 2: Heuristic Meta-path Search
  for each task T_i:
    贪心搜索 meta-path E_i = {e_{j1}, ..., e_{jm}}
      目标: argmax_{E_i} Eval(e_i, X_i^{E_i})
      约束: 用户定义的 sub-optimal paths 缩减搜索空间
    X_i^{aug} ← 沿 meta-path E_i 依次查询 experts 进行数据增强

  // Step 3: Expert Refinement (Information Bottleneck)
  for each expert e_i:
    迭代 σ 次:
      (a) min_{θ_M_RAG} I(M_G(X_i); M_G(RAG(X_i)))
          // 通过控制 RAG 采样参数和 meta-path 添加多样化训练数据 ΔX_i
      (b) max_{θ_M_G} I(M_G(X_i); Y_i)
          // 用 X_i ∪ ΔX_i 继续 LoRA fine-tune M_G
  Output: E^{aug} = {e_1^{aug}, ..., e_n^{aug}}

  // Step 4: Router Network Training
  初始化 transformer-based router N (共享 M_RAG 编码层)
  for each labeled query q_u:
    N(q_u) → top-k experts from E^{aug}
    优化目标:
      max Σ_{e_i∈N(q_u)} I(e_i(q_u^i); l_u^i)   // 专家相关
      min Σ_{e_i≠e_j∈N(q_u)} I(e_i(q_u^i); e_j(q_u^j))  // 专家多样
    近似为对比学习损失训练 N
  ```

  **张量计算示例**（以 EM 任务 meta-path 增强为例）：
  - 给定 EM query q = (t_1, t_2)，meta-path E_EM = {e_blocking, e_DI, e_AVE, e_EM}
  - Step 1: e_blocking(q) → 判断 t_1, t_2 是否可能匹配（候选对筛选）
  - Step 2: e_DI(t_1) → 对 t_1 缺失属性做数据填补，输出增强 t_1'
  - Step 3: e_AVE(t_1') → 提取 t_1' 的关键属性值，附加为新特征
  - Step 4: e_EM(t_1', t_2) → 最终匹配判断

  **Router 推理张量计算**：
  - Input: query embedding emb_q ∈ R^{d}（从 M_RAG 编码器获取）
  - N(emb_q) → softmax(W_N · emb_q) ∈ R^{n}（n 个 expert 的权重分布）
  - Top-k 选择: indices = argsort(weights)[-k:]
  - Final output: y = Σ_{i∈top-k} g_i · e_i^{aug}(q)，其中 g_i = softmax(weights_i)
  - 实际部署时每个 query 只激活 k=3 个 experts，其余 expert 的 LoRA 权重不加载

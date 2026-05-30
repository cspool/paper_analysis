## Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

- baseline方法是什么？
  Baseline 分为三类：
  (1) **Non-LLM 方法**：针对每个 DP 任务单独训练的专用模型或规则系统，如 Raha(ED)、Ditto/DeepBlocker(EM/Blocking)、Baran(DC)、RECA(CTA)、TURL(RE/EL)、IPM(DI)、MAVE(AVE) 等。这些方法依赖手工特征工程或任务特定架构，无法跨任务泛化。
  (2) **LLM-based 方法**：JellyFish(13B) 和 TableLLaMa(7B) 采用 MTL 范式，用一个大而密的 LLM 同时处理所有 DP 任务。需要在海量 task-specific 语料上预训练（数千 GPU hours），且单一大模型难以同时学好多个离散的 DP 任务子空间。
  (3) **MoE 模型（Mixtral 8×7B）**：内置 MoE layer，router 和 experts 联合训练。但缺乏独立 expert 微调的灵活性，且 load balancing 不均匀，在简单/封闭域任务（如 EM、DC）上性能差。

  **Baseline 全栈执行例子**（以 JellyFish 13B 处理 EM 任务为例）：
  - **算法pipeline层**: JellyFish 使用单一 dense 13B LLM，通过 MTL 在所有 DP 任务上联合训练。给定 query q=(t1,t2)，instruction prompt + few-shot demonstrations 输入 LLM，输出 match/mismatch 判断。由于所有任务共享同一参数空间，EM 任务的学习会受其他任务（如 RE、CTA）的梯度干扰，且 13B 模型需要 tensor parallelism 跨 GPU 部署。
  - **系统框架层**: 训练需数千 GPU hours 预训练（如 TableLLaMa 在百万 Wikipedia webtables 上预训练）。推理时 JellyFish 需 4-bit 量化才能在单 3090 上运行，量化带来性能损失。推理框架为标准 transformers/HuggingFace pipeline。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 论文未明确说明。使用标准 PyTorch GEMM kernel 执行 dense LLM forward pass。
  - **硬件架构层**: 训练需多 GPU（A100 级别或以上），推理时 JellyFish 13B 在单 RTX 3090 上需 4-bit 量化（GPTQ），Mixtral 56B 即使 4-bit 量化也无法部署在单 3090 上（OOM）。1×3090 上 JellyFish 4-bit 推理吞吐约 1.3× MELD，但 model process time 慢 10×。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MELD 通过四个关键设计解决 Baseline 缺陷：

  **(1) 增强型 RAG + 自标注 → 解决 few-shot 数据稀缺（针对 Non-LLM 和 LLM baseline 的过拟合问题）**：
  Baseline 在 few-shot 场景下易过拟合（Theorem 2 证明 single expert error bound 受样本数 N 限制）。MELD 用 fine-tuned sentence-bert 进行跨域相似 entry 检索，用 contrastive learning 训练 RAG 模型后进行自标注（self-annotation），将少量标注数据扩增为大规模自标注训练集。RAG 模型还能通过 query 变换实现跨任务数据增广（如将 EM query 转换为 DI query）。

  **(2) Meta-path 数据增强 → 解决跨域泛化能力不足（针对 Multi-task learning 的任务子空间离散问题）**：
  Baseline MTL 方法难以处理离散且远离的 DP 任务子空间。MELD 用启发式贪心搜索找到每个 task T_i 的 meta-path（如 EM_BLK → DI → AVE → EM），沿 meta-path 调用多个 experts 进行数据增强，为原始 task 补充结构化信息和跨域特征。这对半结构化数据和低质量数据（如 semi-text-watch, amazon-google）尤为重要（ablation 显示 w/o meta-path 性能下降 10-30%）。

  **(3) 信息瓶颈引导的 Expert 精炼 → 解决 "多 expert 训练不收敛/不均衡" 问题**：
  Mixtral 的内置 MoE layer 中 expert 训练不均衡（load imbalance），且不能独立 fine-tune 单个 expert。MELD 采用 Min-Max 优化目标（基于 Information Bottleneck）：min I(X; θ) 确保训练数据多样性以防止过拟合，max I(Y; θ) 确保 expert 与标签的相关性以防止欠拟合。通过迭代控制 RAG 数据增强 + LoRA fine-tune 实现该目标，每个 expert 既能保持对自身 domain 的高性能，又对跨域 query 具有鲁棒性。实验显示 w/o MoE（单 expert per task）在所有数据集上性能下降。

  **(4) 独立 Router Network + 动态 LoRA 切换 → 解决 MoE 推理开销大和部署限制（针对 Mixtral 的 load imbalance 和部署问题）**：
  Mixtral 的 router 与 experts 联合训练，不可独立调整，且 56B total params 无法在单 3090 部署。MELD 的 router 是独立的轻量 transformer（共享 M_RAG 编码层），用对比学习训练，为每个 query 选择 top-k diverse 且 relevant 的 experts。推理时通过 Punica + vLLM 实现 dynamic LoRA switch，单 3090 可同时 serving 200 个 LoRA experts，无需 merge 操作，model process time 比 Mixtral 快 30×。4×3090 吞吐量为 Mixtral 的 5.6×（MELD 用 data parallelism，而 Mixtral 需 tensor parallelism 的跨 GPU 通信）。

  **MELD 方法全栈执行例子**（以 EM 任务、query q = (t1="Apple iPhone 13", t2="iPhone 13 by Apple")、k=3 experts 为例）：
  - **算法pipeline层**: Query q 经 serializer 序列化为 dict{instruction: "Entity Matching", tuples: [{t1}, {t2}], meta: {table: "Products", columns: ["name","brand"]}}。M_RAG 编码 q 为 emb_q，检索跨域相似 entries 作为 demonstrations D_EM（来自 Walmart-Amazon、Ant-Buy 等域）。若 meta-path E_EM = {e_BLK, e_DI, e_EM}，则先由 e_BLK 判断候选对、e_DI 填补缺失属性、e_EM 最终判断。精炼后的 e_EM^{aug} 不仅学会 EM 分类边界，还融合了其他 experts 的知识特征。
  - **系统框架层（Serving调度）**: Router N 计算 top-3 experts: e_EM^{aug}, e_DI^{aug}, e_CTA^{aug}（权重 [0.5, 0.3, 0.2]）。Punica + vLLM 在单 3090 上动态加载对应的 3 个 LoRA adapter 到 base Mistral-7B。每个 expert 独立推理后加权融合输出。vLLM 的 PagedAttention 管理 KV cache，continuous batching 处理并发 queries。Load balancing 机制将同类型 queries 路由到同一 GPU 以提高 cache 命中率。
  - **编译框架层**: 论文未明确说明。LoRA adapter 的前向计算通过标准 PyTorch linear + low-rank matrix multiplication 实现。
  - **Kernel调度层**: 论文未明确说明。LoRA forward: y = W_0·x + (B·A)·x，其中 B∈R^{d×r}, A∈R^{r×d}（r=rank，远小于 d）。Punica 优化了多个 LoRA 的 GEMM 融合，减少 kernel launch overhead。
  - **硬件架构层**: RTX 3090 (24GB VRAM)。Base Mistral-7B 约 14GB (FP16)，每个 LoRA adapter 约 10-50MB（取决于 rank），200 个 LoRA 约 2-10GB 额外显存。3 个 experts 的推理延迟 ≈ 1 个 expert 的延迟 + LoRA switch 开销（Punica 报告接近零开销）。相比 Mixtral 需要 >48GB 显存（无法在单 3090 部署）或 JellyFish 需 4-bit 量化，MELD 在 consumer GPU 上实现 full-precision MoE 推理。

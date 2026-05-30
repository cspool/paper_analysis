## Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

- baseline方法是什么？
  Baseline 方法包含两类：(1) **Dense Merging**（BTM 风格）：将多个域独立训练的 dense expert 和 seed model 做等权平均合并为一个 dense Transformer；(2) **MoE with Linear Router**（BTX 风格）：将 dense expert 的 FFN 分别初始化为 MoE 各 expert，使用标准线性路由器 W_r ∈ R^{h×n}（s_i = softmax(W_r^T·x)）从零训练。

  核心缺陷：
  (1) **线性路由器缺乏域归纳偏置**：标准 router 是一个随机初始化的线性层，训练时需要 load balancing loss 来防止 expert 崩溃，导致学到的 expert assignment 与域边界不相关，无法保持 expert 的域专业化。
  (2) **扩展新 expert 时 router 需要重新训练**：扩展新 expert 时，线性路由器 W_r 的维度 h×n 变为 h×(n+1)，新列需随机初始化或全量重新训练。在有限微调数据（如 1B tokens）下难以收敛。
  (3) **Dense Merging 存在跨任务干扰**：多个域 expert 的权重平均会导致不同域之间的知识冲突，下游性能甚至不如原始 seed model。

  全栈执行例子（以 2.8B MoE with Linear Router baseline 为例）：
  - 算法 Pipeline：5 个 dense expert（ArXiv/Books/C4/StackExchange/Wikipedia 域）+ seed model → 合并 FFN 为 8-expert MoE 层 → 随机初始化线性路由器 W_r → 在所有域数据 mix 上训练 MoE 40B tokens，router 通过 load balancing loss + LM loss 联合学习 → 问题：router 的 W_r 列与域含义无关，expert 选择基于 token 的表征相似度而非域归属 → 扩展新 expert 时需扩大 W_r 为新维度，新列随机初始化 → Code expert 的 routing 在 1B tokens finetuning 下不收敛。
  - 系统框架：论文未明确说明（推测为 PyTorch + 标准 Transformer 训练框架）。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Nexus**，核心思路：**将路由器从"学习输入→expert 选择"重构为"学习域嵌入→专家嵌入投影 + 相似度路由"**。

  **对应缺陷 1（线性路由器缺乏域归纳偏置）→ 基于域嵌入的自适应 Router**
  - Router 本身不再是线性层 W_r，而是一个投影函数 P_r（2-layer SwiGLU MLP），输入是预计算的域嵌入 d_i（代表第 i 个域的整体语义），输出是专家嵌入 e_i = P_r(d_i)。
  - 路由概率 s_i = softmax(x · e_i)，即 token x 与每个域的专家嵌入做点积——高相似度意味着 token 在语义上接近该域。
  - 因为每个 e_i 是从对应域的语义嵌入投影而来，即使 load balancing loss 很低（如 0.0005），expert embedding 的内在域语义也能保持稳定的 token 分配（Ablation 验证：load balancing loss factor 降到 0.0005 时 Nexus 性能稳定而 Linear Router 下降 2%）。
  - 域嵌入通过 Cohere Embed v3 编码数据获得（也可通过无监督聚类 centroids 获取），可在保持域间相对关系（如 Books-C4、GitHub-StackExchange 的高 cosine similarity）的同时，将专家嵌入推得更远以避免 expert 使用重叠。

  **对应缺陷 2（扩展新 expert 需重训 router）→ 投影函数的零样本泛化**
  - 新域到来时：仅需计算新域嵌入 d_new → P_r(d_new) = e_new，无需修改 W_r 的维度也不需要重新训练 router。因为 P_r 是在初始 upcycling 阶段学到的通用"域→专家嵌入"映射函数，对未见域有泛化能力。
  - 新 expert 的 FFN 直接 append 到已有 FFN 数组，非 FFN 参数用加权平均 merge，全程无需 router 随机初始化或维度变化。
  - 实验结果：Nexus 在 1B token finetuning 时 Code 性能相对 gain 18.8%（vs MoE Linear Router），且新 expert 的域路由精度高达 69.1%（Code token 被路由到 Code expert 的比例）。

  **对应缺陷 3（Dense Merging 跨任务干扰）→ 稀疏激活 + 共享专家**
  - 与所有 token 激活全部参数的 dense merging 不同，Nexus 稀疏激活仅 2 个 expert/token（1 shared + 1 routed），避免了不同域知识的直接参数冲突。
  - Shared expert（seed model FFN 始终激活）保留了 seed model 的通用语言能力作为基础，routed expert 按域选择性地注入专业知识。
  - 实验：Nexus 比 dense merging 在平均指标上高 8.5%（470M scale），在 2.8B scale 上也显著优于 dense merging。

  全栈执行例子（Nexus 方法）：
  - 算法 Pipeline：5 个 dense expert 训练完成 → 用 Cohere Embed v3 编码每个域数据得 d_arXiv, d_Books, d_C4, d_SE, d_Wiki → 合并 FFN 为 MoE 层（seed FFN 为 shared）→ 初始化 P_r（2-layer SwiGLU）→ 训练 MoE：forward 时 P_r(d_i) → e_i，s_i = softmax(x·e_i)，top-1 gate 选 expert + shared expert → 所有域数据 mix 上训练 → 扩展阶段：训练 Code dense expert → d_code = Embed(code_data) → e_code = P_r(d_code) → append FFN_code → 加权平均非 FFN 参数 → 1B tokens 轻量微调 → Code token 69.1% 被路由到 Code expert。
  - 系统框架：论文未明确说明。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明。
  - 硬件架构：论文未明确说明。

  - 算法 Pipeline：同 baseline GPT-2 backbone MoE 模型，gating + routing → tokens 按 routing 分发到 expert GPU → experts FFN → **Expert Residual Inlining: output = input + expert_output（此时 token 仍在 expert GPU 上，非原 GPU）** → gather 使用优化后 SmpDev 将 token 发往新位置而非原位置。
  - 系统框架（训练框架）：PyTorch + 自定义 C++/CUDA。**新增模块**：(a) num 矩阵计算（GPU kernel, Eq. 2）；(b) c/c' 通信代价矩阵计算（Eq. 8）；(c) KM 求解器（CPU, background thread）；(d) 样本放置策略注入 gather 通信。**关键时序**：scatter → expert compute || KM solver（CPU）→ inlining → gather with optimized SmpDev。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明（使用 NCCL All-to-All，无自定义通信 kernel）。
  - 硬件架构：同 baseline。经过 NetMoE 优化后，2 nodes/16 GPUs 下 MoE-GPT-S inter-node 通信量从 191MB 降至 116MB（\downarrow 39.1%），约 91.4% 的 samples 被调整位置，端到端加速 1.67x vs FastMoE。

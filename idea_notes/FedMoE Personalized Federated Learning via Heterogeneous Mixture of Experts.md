## FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

- baseline方法是什么？
  - **Baseline 1: FedProx**：联邦优化算法，在本地更新时引入正则化项 $\frac{\mu}{2} \|w - w^t\|^2$ 来约束本地模型不偏离全局模型太远，缓解数据异构带来的 client-drift。每轮所有客户端收到相同的全局 dense 模型（Switch Transformers, 8 experts/layer），本地微调后上传，服务器 FedAvg 聚合。缺点是所有客户端共享相同参数，无法针对不同任务/数据分布定制模型；正则化参数 μ 敏感且难以推广到跨任务复杂场景。
  - **Baseline 2: SCAFFOLD**：使用控制变量（control variates）$c$ 和 $c_k$ 来修正本地更新方向 $w_k \leftarrow w_k - \eta(g_k - c_k + c)$，克服异构数据导致的 client-drift。每轮除模型参数外还需传输控制变量，额外通信和内存开销随训练累积。同样使用 dense 模型所有客户端共享参数，缺乏个性化能力。
  - **Baseline 3: randomMoE**：从全局 MoE（32 experts/layer）中为每个客户端随机选择 expert 子集构建个性化边缘模型，保证一定程度的信息隔离。但由于 expert 选择是随机的，可能选到对客户端任务无关或次优的 expert，无法利用 MoE 的稀疏激活特性来针对性适配数据分布。
  - **全栈执行例子（以 FedProx 在 Standard-Hetero-T 设置、Switch Transformers 8 experts、30 客户端为例）**：
    - **模型推理算法层**：Switch Transformers dense 模型，每层 8 experts，top-1 routing。所有 30 个客户端共享相同的模型架构和参数。每轮随机选 5 个客户端，各客户端在本地 AG News/SQuAD/XSum 数据上执行: forward（gate 选 top-1 expert）→ compute loss（cross-entropy + load balance loss + proximal term $\frac{\mu}{2}\|w-w^t\|^2$）→ backward → 上传模型 → 服务器 FedAvg 聚合。所有客户端用统一模型处理分类、阅读理解、摘要三种不同任务，不同任务的梯度更新方向可能冲突。
    - **系统框架层**：基于 PyTorch + HuggingFace Transformers 实现 FL 模拟。服务器-客户端通信模式：broadcast 全局模型 → 客户端本地训练 → 上传更新 → FedAvg 聚合。每轮传输完整模型参数（约 24.7GB 内存 + 2.30GB 通信量）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel，每个 MoE 层 gate softmax top-1 selection + 选中 expert GEMM 计算。
    - **硬件架构层**：模拟边缘设备 18-24GB 内存（高端智能手机/边缘计算平台级别），云服务器执行聚合。18-24GB 内存限制下全局模型只能用 8 experts/layer，无法充分发挥 MoE 大容量优势。
  - **Baseline 痛点**：
    1. **统一模型无法适应异构任务**（核心痛点）：FedProx/SCAFFOLD 等传统 FL 方法让所有客户端共享相同模型参数，但不同客户端有不同的数据分布和任务类型（分类/阅读理解/摘要），统一模型要么牺牲个性化性能，要么不同任务的梯度方向相互冲突导致收敛缓慢或不稳定。
    2. **资源受限与模型容量的矛盾**：边缘设备内存仅 18-24GB，限制全局模型只能使用 8 experts/layer（FedProx/SCAFFOLD），而 MoE 的优势在于大量 expert 提供丰富的知识库。设备能力限制了模型容量上限。同时 FedProx 的 proximal term 和 SCAFFOLD 的控制变量引入额外内存和通信开销（FedProx: 24.71GB 内存/2.30GB 通信，SCAFFOLD: 17.29GB 内存/4.61GB 通信）。
    3. **randomMoE 的盲目性**：随机选择 expert 构建子模型虽然保证了个性化（不同客户端不同 expert），但无法利用数据特性选择最优 expert，导致性能次优（Standard-Hetero-T 设置下 randomMoE TC 仅 91.63, TS 仅 14.51）。
    4. **缺乏结构-性能协同优化**：现有 PFL 方法要么固定模型结构只用 loss 约束（FedProx/SCAFFOLD），要么静态剪枝/蒸馏后不再调整（knowledge distillation 类方法），无法在训练过程中根据实际反馈动态调整模型结构。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FedMoE 方法**：以 MoE 架构天然的解耦 expert 参数空间为基础，构建两阶段个性化 FL 框架：
    1. **Stage One: Coarse-grained Submodel Initialization**（解决痛点 3——randomMoE 盲目性）：通过短轮次 PEFT + LoRA 收集各客户端 expert 激活模式，以激活概率 $p_{i,j}=n_{i,j}/N$ 衡量 expert 对特定数据的重要性。基于此执行启发式二分搜索——在每层保留 expert 组合概率 ≥ θ 的约束下，寻找满足内存限制的最大 θ，构建"高性价比"初始子模型。这替代了 randomMoE 的随机选择，确保每个客户端获得与其数据分布最相关的 expert 子集。
    2. **Stage Two: Modular Aggregation**（解决痛点 1——统一模型无法个性化 + 痛点 4——缺乏结构-性能协同）：突破 FedAvg 的"一刀切"聚合——dense 层保持 FedAvg；sparse 层按 expert 粒度的共享情况分别处理（未激活不变、单客户端直接更新、多客户端加权聚合）。这使得相关客户端在共享 expert 上协作学习，不相关客户端互不干扰，实现"知识共享 + 负迁移隔离"。
    3. **Expert Recommendation**（解决痛点 4——缺乏结构动态调整）：利用其他客户端作为"全局视角"的参考——基于 expert 激活概率的 cosine similarity 找到 top-K 最相似客户端，通过加权平均估算所有 expert（包括子模型外的）的预期激活概率。若参考组平均 expert 数更多则推荐引入高效 expert；否则推荐裁剪低效 expert。调整具有探索性（性能不改善则回退），在训练过程中持续优化个性化结构。
    4. **资源效率**（解决痛点 2）：通过子模型 sub-sampling 使每个客户端仅持有最优 expert 子集（平均 65-78 experts/layer 从 32 中选出），大幅降低内存（13.44GB vs FedProx 24.71GB，−45.6%）和通信量（1.76GB vs FedProx 2.30GB，−23.5%）。Stage One 的一次性开销约 7.46GB 通信和 13.06GB 内存且仅持续数轮。

  - **全栈执行例子（FedMoE 在 Standard-Hetero-T 设置、Switch Transformers 32 experts global/子模型平均 65 experts、30 客户端）**：
    - **模型推理算法层**：全局模型 Switch Transformers 32 experts/layer，top-1 routing。两阶段流程——Stage 1: 5 轮 PEFT+LoRA 微调，收集各客户端 expert 激活概率 → 二分搜索构建初始子模型（每层子集 experts 满足概率阈值且不超内存）；Stage 2: 联邦训练，每轮 subsample 子模型 → 本地训练（cross-entropy + load balance loss，无 proximal term）→ 上传（仅上传子模型所含参数而非全量）→ Modular Aggregation（expert 粒度差异化更新）→ Expert Recommendation（相似客户端参考调整结构）。关键差异 vs baseline：不同客户端持有不同 expert 子集（个性化参数空间），expert 选择由激活数据驱动而非随机，聚合策略按 expert 粒度差异化。
    - **系统框架层**：基于 PyTorch + HuggingFace Transformers 的自建 FL 模拟框架。与 baseline 的关键差异：服务器维护 client-expert map（每客户端每层保留哪些 expert），subsample 逻辑在服务器端执行（从 32 experts 中按 map 提取参数），上传/下发仅涉及子模型参数。Stage 1 增加一轮激活概率收集通信（one-time 约 7.46GB）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行。关键差异在每个 MoE 层 gate 的 top-1 选择范围从全量 32 experts 缩小为子模型内的 experts 数量，单个 token 的 expert GEMM 计算不变但总参数量因加载更少 expert 而降低。
    - **硬件架构层**：与 baseline 相同（边缘 18-24GB，云端聚合服务器）。结果：Standard-Hetero-T 下 TC 94.76（+2.0% vs best baseline），TS 16.92（+16.7% vs best baseline 的 14.51），通信 1.76GB（−23.5%），内存 13.44GB（−45.6%），收敛加速 1.35×–2.92×。

    **关键性能对比**：
    - Standard-Hetero-T: TC 94.76 (FedProx 92.92), RC 86.64 (FedProx 87.99), TS 16.92 (randomMoE 14.51)
    - Standard-Hetero-TD: TC 88.44 (FedProx 85.09), TS 16.63 (randomMoE 13.51)
    - Enforced-Hetero-T: TC 94.85 (FedProx 92.51), TS high (baselines significantly lower)
    - 消融: w/o stage1 → TS 降至 14.50 (vs 16.92), expert 数不减反增 (96→104); w/o stage2 → expert 数不变 (78), 丧失动态优化能力

    **核心设计洞察**：FedMoE 的核心创新在于利用 MoE 架构的"expert 并行 + 稀疏激活"天然属性来实现个性化 FL——不再强迫所有客户端共享相同参数，而是让每个客户端从全局 expert 池中"挑选"最相关的 expert 构建个性化子模型。两阶段设计（先粗后细）巧妙平衡了搜索效率（Stage 1 快速收敛到近优解）和优化精度（Stage 2 动态调整），Modular Aggregation 在 expert 粒度实现"合作但不干扰"，Expert Recommendation 利用群体智慧指导个体结构调整。这是一种"数据驱动的结构个性化"思路，区别于传统 loss 约束或静态剪枝方法。

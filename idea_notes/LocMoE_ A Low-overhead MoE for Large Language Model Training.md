## LocMoE: A Low-overhead MoE for Large Language Model Training

- baseline方法是什么？
  - **HashMoE / SwitchMoE 的经典 MoE 路由策略**：
    - HashMoE：采用平衡哈希函数将 token 均匀分配到各 expert，不使用可学习参数。优点是绝对负载均衡，缺点是缺乏语义区分能力（token 与 expert 匹配无学习过程），导致收敛速度虽快但推理精度可能不足。
    - SwitchMoE：使用 Dense 层门控网络计算 gating scores，选 Top-1 expert 并通过 softmax 加权。辅助 loss (L_aux) 鼓励负载均衡。但路由策略无局部性感知，token 可能被路由到远程节点的 expert，导致频繁的跨节点 All-to-All 通信。且存在 "winner-take-all" 问题——少数 expert 接收大部分 token，约 40% expert 几乎不被使用。
    - 两者均未优化 expert capacity：使用经验性的 capacity factor c_f 设定 expert capacity = ceil(b_s·c_f/(ep·n))，未从理论上分析容量下界，导致冗余计算。
  - 全栈执行例子（Baseline SwitchMoE 在 PanGu-Σ 128 Ascend 910A NPU 上的一个 training step）：
    - **训练算法层**：Dense 层计算门控值 G(x) = ReLU(ω·x + ε) → Softmax → Top-1 argmax 选 expert → Token dispatch via All-to-All → Expert FFN (GeLU 激活) → Token combine via All-to-All。Load balance 仅依赖 L_aux = α·n·Σ f_i·P_i。
    - **系统框架层**：MindSpore 2.0.0 框架，PanGu-Σ 模型。RRE 两级路由：第一级按领域分组，第二级随机哈希（无学习参数）。16 experts 通过 expert parallelism=16 分布在 Ascend NPU 上。All-to-All 通信由 HCCL 执行。
    - **编译框架层**：论文未明确说明（MindSpore 提供图编译优化，但非本文修改目标）。
    - **kernel 调度层**：HCCL All-to-All 原语（算法带宽见图 2，随节点数增加而递减）。通信与计算串行执行——All-to-All 完成前无法开始 FFN 计算。All-to-All 占总训练时间 18.10%（128N）~28.74%（256N）。
    - **硬件架构层**：Ascend 910A NPU × 128，32 AI Cores/芯片，HCCS intra-node 高带宽互联，Fat-tree + RoCE inter-node 网络。跨节点 All-to-All 因小数据量频繁传输导致带宽利用率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LocMoE 方法**：通过三个核心设计解决 baseline 缺陷：
    1. **GrAP 正交门控（解决语义区分 + 降低开销）**：用固定正交权重矩阵替代可学习 Dense 层计算门控值。正交性使不同领域 token 被路由到不同 expert（降低 cosine similarity），增强语义区分能力，同时避免 Dense 层参数量和计算开销。满足 Lemma 2 的均匀分配概率假设，为理论分析奠定基础。
    2. **局部性正则化（Locality Loss）（解决跨节点通信开销）**：在辅助 loss 基础上增加 KL 散度项 L_loc = μ·KL(D_c||D_l)，促使 token 优先路由到同节点本地 expert。将部分跨节点 All-to-All 转为节点内高带宽通信（HCCS），直接降低 All-to-All 时间 5.13%。同时 locality 的软约束避免 "winner-take-all"——更多 expert 参与早期训练（图 12 显示 LocMoE 的 expert 分配比 SwitchMoE 更均衡）。
    3. **理论下界指导 expert capacity 缩减（解决冗余计算）**：基于高维球面几何 + 正交门控假设，推导 NLP 领域 expert capacity 临界值 ec_min。证明当 token 与 gating weight 夹角余弦 δ 较大时，class-discriminative token 概率 p_δ 极小（≈0.3 at δ=Θ(1/√d)），说明仅少量 token 需要特定 expert 处理。据此可安全降低 expert capacity 不损失精度（pMoE 在 CV 领域得到类似结论，LocMoE 首次推广到 NLP 并结合网络结构分析）。
  - 对应解决 Baseline 缺陷：
    - **Dense 门控开销大 + 缺乏语义正交性** → GrAP 固定正交权重，O(1) 计算（仅 mean pooling），无参数量，自然正交
    - **跨节点 All-to-All 占比高（18%~29%）** → Locality loss 鼓励本地路由，减少跨节点通信量，配合 MindSpore Group-wise All-to-All + FFN 重叠
    - **Expert capacity 经验设定冗余** → 理论推导下界，提供安全的 capacity 缩减指导
    - **SwitchMoE 的 winner-take-all（~40% expert 闲置）** → Locality 软约束 + auxiliary loss 双重正则化，使 expert 分配更均衡
  - 全栈执行例子（LocMoE 在 PanGu-Σ 128 Ascend 910A NPU 上的一个 training step）：
    - **训练算法层**：GrAP 分组平均池化计算门控值 → Softmax → Top-1 argmax → KL(L_aux + L_loc) 双约束路由决策 → 按理论下界设定 expert capacity → Token dispatch via Group-wise All-to-All（TP/EP 域拆分）→ Expert FFN 与 All-to-All 切片重叠执行 → Token combine → L_task = L_aux + L_loc + L_cross 反向传播
    - **系统框架层**：MindSpore 2.0.0，PanGu-Σ 的 RRE 第二级路由被 LocMoE 改写（从随机哈希变为可学习的 locality-aware routing）。Group-wise All-to-All 利用 TP 域高速带宽分担 EP 域通信压力。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：HCCL Group-wise All-to-All + All-Gather（TP 域同步）+ FFN kernel 与 All-to-All 切片重叠。通信时间下降 5.13%，64N 下每 epoch 总时间减少 12.68%~22.24%。
    - **硬件架构层**：Ascend 910A NPU × 128，HCCS intra-node + Fat-tree inter-node。Locality loss 使更多通信利用 HCCS 高带宽（256GB/s），减少 RoCE inter-node 通信。但 256N 下因部分节点无本地 expert 导致 locality 策略失效，性能不如 HashMoE。

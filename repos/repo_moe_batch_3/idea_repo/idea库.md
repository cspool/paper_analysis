## mHC Manifold-Constrained Hyper-Connections

- baseline方法是什么？
  - Baseline 方法包括：
    - **标准残差连接（Residual Connection）**：$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$，identity mapping 保证 shallower layer 信号直接映射到 deeper layer，训练稳定。核心缺陷：残差流宽度固定为 C，信息容量受限于层输入维度，无法在不增加 FLOPs 的情况下扩展残差流的表达能力。
    - **Hyper-Connections (HC, Zhu et al. 2024)**：将残差流宽度扩展 n 倍（$\mathbf{x}_l \in \mathbb{R}^{n \times C}$），引入三个可学习映射——$\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$ 聚合 n-stream 为 C 维层输入，$\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$ 将层输出映射回 n-stream，$\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ 混合残差流内特征。不增加 FLOPs 前提下提升了模型性能，但 **核心缺陷**：(1) $\mathcal{H}_l^{\text{res}}$ 无约束导致复合映射 $\prod \mathcal{H}_l^{\text{res}}$ 偏离 identity mapping，信号可能爆炸或消失（Amax Gain Magnitude 可达 ~3000）；(2) 显存 I/O 开销约为标准残差连接的 $(5n+1)C$ 倍读和 $(3n+1)C$ 倍写；(3) pipeline parallelism 通信量增加 n 倍。
  - 全栈执行例子（以 HC 训练 27B MoE 模型为例）：
    - **算法层**：输入 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ → RMSNorm → 线性投影计算动态映射 $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}, \mathcal{H}^{\text{res}}$（含 tanh 激活）→ 加上学习 bias 静态映射 → $\mathcal{H}^{\text{pre}} \mathbf{x}_l$ 聚合为 C-dim → 标准 Attention/FFN 计算 → $\mathbf{x}_{l+1} = \mathcal{H}^{\text{res}} \mathbf{x}_l + \mathcal{H}^{\text{post}^\top} \mathcal{F}$ 更新 n-stream。因 $\mathcal{H}^{\text{res}}$ 无约束，跨 30 层（60 个 sublayer）后的复合映射行列和可达 3000×，梯度爆炸导致 12k step 处训练崩溃。
    - **系统框架层**：DeepSeek-V3 训练框架，使用 DualPipe pipeline parallelism。n-stream 残差在 stage 边界需传输 n×C 元素（vs 标准残差连接的 C），通信量增加 n 倍。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准矩阵乘法和逐元素操作，无融合。每个 $\mathcal{H}$ 的计算涉及独立 kernel launch（matmul + bias add + tanh + 应用映射），读/写量 = $(5n+1)C + n^2 + 2n$ / $(3n+1)C + n^2 + 2n$。中间激活需完整保存用于反向（nC + n×C 的 stream + 小系数）。
    - **硬件架构层**：在 NVIDIA GPU 集群上训练，具体型号论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **mHC 的核心设计**：将 HC 的 $\mathcal{H}_l^{\text{res}}$ 通过 Sinkhorn-Knopp 算法投影到双随机矩阵流形（Birkhoff polytope），同时约束 $\mathcal{H}_l^{\text{pre}}/\mathcal{H}_l^{\text{post}}$ 为非负（Sigmoid），恢复 identity mapping 的稳定性，并通过基础设施优化解决 I/O 瓶颈。
  - **解决 HC 的三个缺陷**：
    1. **训练不稳定性 → 双随机约束**：$\mathcal{H}_l^{\text{res}}$ 限制为双随机矩阵（行和=列和=1，元素≥0），(a) 谱范数 ≤ 1 防止梯度爆炸；(b) 双随机矩阵乘法封闭性保证跨任意深度的复合映射仍为双随机，Amax Gain Magnitude 从 ~3000 降至 ~1.6（降低 3 个数量级）；(c) Birkhoff polytope = 置换矩阵凸包，残差映射解释为"凸组合置换"，一致地单调增加特征混合而非发散。$\mathcal{H}_l^{\text{pre}}/\mathcal{H}_l^{\text{post}}$ 施加 Sigmoid 非负约束，防止正负系数抵消导致的信号衰减。
    2. **显存 I/O 开销 → Kernel Fusion + Recomputing**：5 个融合 kernel 消除冗余内存访问；RMSNorm 重排序优化；Sinkhorn-Knopp 单 kernel 实现（含定制反向）；Post+Res 映射应用与 residual merge 融合（读取从 $(3n+1)C$ 降至 $(n+1)C$）；选择性重计算策略（$L_r^*$ 最优块大小与 pipeline stage 对齐）使 n=4 时总开销仅 6.7%。
    3. **Pipeline 通信膨胀 → DualPipe 通信重叠**：扩展 DualPipe schedule，MLP 层 $\mathcal{F}_{post,res}$ kernel 在专用高优先级 compute stream 上运行以允许被通信抢占；attention 层避免 persistent kernel；重计算与通信解耦（首层激活已在本地缓存）。
  - 全栈执行例子（以 mHC 训练 27B MoE 模型为例）：
    - **算法层**：输入 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ → **Kernel 1**：flatten + RMSNorm + 线性投影获得 $\tilde{\mathcal{H}}^{\text{pre}}, \tilde{\mathcal{H}}^{\text{post}}, \tilde{\mathcal{H}}^{\text{res}}$ → **Kernel 2**：乘以 gating factor α、加 bias、RMSNorm 归一化 → **Kernel 3**：$\tilde{\mathcal{H}}^{\text{pre}}$ 经 $\sigma(\cdot)$、$\tilde{\mathcal{H}}^{\text{post}}$ 经 $2\sigma(\cdot)$（非负约束）、$\tilde{\mathcal{H}}^{\text{res}}$ 经 Sinkhorn-Knopp 20 次迭代（双随机约束）→ **Kernel 4**：$\mathcal{H}^{\text{pre}} \mathbf{x}_l$ 聚合为 C-dim → 标准 Attention/FFN → **Kernel 5**：$\mathcal{H}^{\text{res}} \mathbf{x}_l + \mathcal{H}^{\text{post}^\top} \mathcal{F}$ 更新 n-stream（融合 residual merge）。信号经 60 个 sublayer 后 Amax Gain Magnitude 仅 ~1.6，梯度稳定，无 loss spike。
    - **系统框架层**：DeepSeek-V3 + DualPipe，pipeline stage 边界通信与 mHC kernel 重计算被重叠。重计算块边界与 pipeline stage 对齐（$L_r^* \approx \sqrt{nL/(n+2)}$）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：5 个定制融合 kernel + 选择性重计算。前向仅持久化 $\mathbf{x}_{l_0}$ 和 $\mathcal{F}$ 的输出（每层），中间 stream 和映射系数均在反向重计算。TileLang 用于实现复杂 kernel，混合精度（bfloat16 输入 → tfloat32 权重 → float32 计算）。
    - **硬件架构层**：在 NVIDIA GPU 集群上训练，具体型号论文未明确说明。mHC 在 n=4 时额外时间开销仅 6.7%，意味着相对于 Baseline 的吞吐量损失非常小。

## Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

- baseline方法是什么？
  - Baseline 方法包括：
    - **Dense LSTM 模型**：Jozefowicz et al. (2016) 的 stacked LSTM（LSTM-512-512, LSTM-1024-512, LSTM-2048-512, 2xLSTM-8192-1024），所有参数对每样本均激活，模型容量受限于计算资源。参数从 2M 到 151M 不等。
    - **GNMT (Wu et al., 2016)**：9 层 LSTM encoder + 8 层 LSTM decoder 的机器翻译模型（278M 参数，214M ops/timestep）。所有参数对每样本均激活。
    - **计算匹配的密集 baseline（本文内部）**：MoE-1-Wide（单 FFN 4096 hidden）、MoE-1-Deep（4 层 1024 FFN）、4xLSTM-512、MoE-4（无稀疏，4 个 expert 始终激活）。
    - **GNMT-Multi (Johnson et al., 2016)**：单一 GNMT 模型处理 12 个语言对的翻译，但因容量有限，结果不如 12 个单独的单语言对模型。
  - 全栈执行例子（以 LSTM-2048-512 Baseline 为例，1 Billion Word LM Benchmark）：
    - **算法层**：输入句子 `x[1:T]` → Word Embedding (d_model=512) → 经过 2048 单元 LSTM 层（全激活）→ 输出投影到 512 → Softmax 层（重要性采样）预测下一个词。所有计算对所有 token 和所有时间步全激活。容量 = 参数数（~151M），计算开销 ≈ O(d_model × n_params) 随参数数线性增长。
    - **系统框架层**：数据并行：每个 GPU 处理独立的 batch，参数服务器同步梯度。Batch size 受激活值存储限制。GPU 间通信主要传输梯度。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：TensorFlow 框架 (Abadi et al., 2016) 在 Tesla K40 GPU 上执行标准 LSTM kernel（matmul, sigmoid, tanh），计算效率约 1.07-1.29 TFLOPS/GPU。
    - **硬件架构层**：Tesla K40 GPU 集群（16-32 卡），内存限制每 GPU 能容纳的参数规模。带宽需求与参数数成正比。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：
    - **Sparsely-Gated MoE Layer**：引入一个可训练的门控网络 G(x)，对每个输入选择稀疏的 k 个 expert 子集进行条件计算（而非全部）。输出 y = Σ_i G(x)_i · E_i(x)，其中 G(x) 是稀疏 n 维向量。每个 expert 是独立参数的 FFN。
    - **Noisy Top-K Gating**：在 Softmax 前加入可调高斯噪声（噪声幅度由 W_noise 控制），然后 KeepTopK 保留最大的 k 个值。噪声项使负载均衡 loss 可微（通过 Φ(CDF) 估计 P(x,i)）。
    - **两级辅助损失函数**：L_importance = w·CV(Importance)²（防止门控塌缩到少数 expert）和 L_load = w·CV(Load)²（防止 expert 负载不均，Load 为平滑估计器）。
    - **混合数据并行与模型并行**：标准层和门控网络用数据并行，各 expert 只保留一份共享副本（模型并行）。同一设备既做数据并行副本又做模型并行分片。所有数据并行 input batch 中的相关样本组合后送给每个 expert，使 expert batch size 放大 d 倍（d=设备数），解决 shrinking batch 问题。
    - **卷积式 MoE 应用**：等前一层对所有时间步完成后，将 MoE 卷积式应用于所有时间步，将 seq_len 折叠入 batch dim，进一步增大 expert batch size 至 b × T × kd / n。
    - **Hierarchical MoE**：主门控选择次级组，次级门控在组内选择 expert。第一级分支因子 = GPU 数，第二级在单 GPU 内执行，消除跨设备通信。
  - 对比 Baseline 的全栈执行改进（以 MoE-4096-h, k=4, 8M ops/timestep 为例）：
    - **算法层**：输入 x → Word Embedding (512) → LSTM (512) → 门控网络 G(x) 计算稀疏 top-k=4（从 4096 个 expert 中选 4 个，稀疏度 99.9%）→ 仅 4 个 expert 执行 FFN(1024 ReLU → 512) → 加权 sum → LSTM (512) → Softmax。模型参数从 ~151M 增至 ~4.3B（28×），但计算量仅 ~8M ops/timestep（baseline ~151M 的 5.3%）。Perplexity 从 34.7 降至 34.1（仅需 6% 的计算）。
    - **系统框架层**：16 GPU 同步训练。标准层 + 门控网络在每 GPU 上全复制（数据并行），4096 个 expert 分布到 16 GPU（每 GPU 256 个 expert，模型并行）。每个 expert 接收来自所有 16 GPU 的 input batch 中选中该 expert 的样本。branching factor=16（第一级 16 个次级组，每 GPU 一个），第二级 256 选 2。总 batch size ~300K words，expert batch ≈ kb×d/n ≈ 4×300K×16/4096 ≈ 4690 words/expert。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：TensorFlow 在 16-32 K40 GPU 上执行。Expert 计算占 37%-46% 的总浮点运算。计算效率 0.74-0.90 TFLOPS/GPU（低计算模型），最高 1.56 TFLOPS/GPU（高计算模型 MoE-143M），均为 Tesla K40 理论峰值 4.29 TFLOPS/GPU 的显著比例。关键瓶颈从"计算所有参数"变为"网络带宽传输 expert 输入/输出"——通过增大 expert hidden layer（1024/2048/8192）提高 compute-to-IO ratio。
    - **硬件架构层**：Tesla K40 GPU 集群。每 GPU 内存需求恒定（不随 expert 总数增加），因为每个 expert 参数固定为 ~1M，每 GPU 只需存储其托管的 256 个 expert。模型容量（#expert × params/expert）随设备数线性扩展。内存优化：不存储 expert hidden layer 激活（reverse 时重算），Adam 二阶矩使用分解近似（row-wise × col-wise outer product ÷ mean），降低优化器内存从 3× 至 ~2×。

## Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling

- baseline方法是什么？
  - Baseline 方法包括：
    - **通信调度**：RCS（Random Communication Scheduling，随机通信调度）和 SJF（Shortest Job First，最短作业优先调度）。
    - **GPU 分配**：RGA（Random GPU Assignment，随机 GPU 分配）。
    - **专家共置**：Lina（将同模型的最热和 coldest expert 共置在同一 GPU），REC（Random Expert Colocation，随机跨模型专家共置）。
  - 全栈执行例子（以 Colocating+Heterogeneous 的 Lina baseline 为例，2 个 MoE 模型各 8 expert，异构 GPU 集群）：
    - **算法层**：MoE 模型（B/16, B/32，来自 Google LiMoE，每模型 8 expert，4 层），Gate 网络使用 top-k routing 为每个 token 选择 expert。FFN 处理分配的 token，Aggregation 合并输出。
    - **系统框架层**：Lina 将同模型的最热门和最冷门 expert 打包在同一 GPU。因同模型 expert 受到同步 all-to-all 通信约束——所有 GPU 必须等通信完成后才能开始计算——即使同一 GPU 上有两个 expert，一个 expert 计算完毕也必须等待另一个的通信完成。通信使用 NCCL 集体通信库，采用同步 all-to-all 实现。GPU 分配为随机。两个模型的 all-to-all 通信时间直接叠加（无交错）。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch eager mode）。
    - **Kernel层**：论文未明确说明（使用标准 GPU kernel，通信通过 NCCL）。
    - **硬件架构层**：异构 GPU 集群通过 big switch 无阻塞网络互联。同构集群带宽统一 100 Gbps；异构集群包含 100/80/50/40 Gbps 四种带宽 GPU。SJF 调度优先发送流量最小的 token，但在 all-to-all 场景下无法减少带宽竞争（与 RCS 效果相当）。
  - Baseline 缺陷：
    1. **同步 all-to-all 通信导致 GPU 空闲**：同模型 expert 共置时，计算（FFN/Aggregation）必须等待所有 GPU 的通信完成，无法交错利用 GPU 资源，GPU 利用率极低（低于 20%）。
    2. **通信时间次优**：随机或 SJF 的 token 传输顺序未考虑接收端带宽竞争——两个 GPU 同时向同一目标 GPU 发送 token 会导致带宽争用，延长通信时间。
    3. **GPU 分配不考虑性能差异**：在异构集群中随机分配 expert，冷门 expert 可能占用高性能 GPU，而热门 expert 运行在低性能 GPU 上。
    4. **缺乏理论指导**：现有方法依赖经验方法，没有理论推导出最小推理时间的下界。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Aurora 系统，通过联合优化模型部署和 all-to-all 通信调度，在所有四种 GPU 集群场景下最小化 MoE 推理时间：
    (1) **通信调度（§4.2）**：Theorem 4.2 证明通过避免接收端带宽竞争的 token 传输顺序，可将通信时间压缩至由最大发送/接收 GPU 流量决定的理论下界 b_max。Alg. 1 提供构建最优传输顺序的多项式算法。
    (2) **跨模型专家共置（§6）**：将来自不同模型的专家放在同一 GPU，打破同步 all-to-all 约束——两个模型的计算和通信可完全交错（Model a 计算时 Model b 通信，反之亦然）。通过 bottleneck matching 找到最优 expert 配对（Case I: Theorem 6.2 交替热门-冷门；Case II: bottleneck matching + Hopcroft-Karp）。
    (3) **性能排序 GPU 分配（§5.1）**：Theorem 5.1 证明按 expert token 负载降序分配给性能降序 GPU 是最优的。
    (4) **完整理论分析**：覆盖四种场景，前三种给出最优解，第四种（Colocating+Heterogeneous, NP-hard）给出仅偏离 1.07× 的次优多项式解。
  - 全栈执行例子（以 Aurora Colocating+Heterogeneous 为例）：
    - **算法层**：同 baseline，MoE 模型本身不变。Aurora 在算法之下做系统级优化。
    - **系统框架层**：Aurora 接收 traffic matrix D_N、D_C 和计算时间作为输入。(a) 先解耦 3D 匹配为两个 bottleneck matching：先确定 Model a 和 Model b 的 expert 配对（最小化聚合通信时间的最大列/行和），再将配对后的 expert 通过 bottleneck matching 分配给 GPU（按性能排序）。(b) 通信阶段：对每个 all-to-all，Aurora 通过 Alg. 1 确定 token 传输顺序——以瓶颈 GPU 为起点，剩余 GPU 按流量降序排列，避免向同一目标 GPU 同时发送。(c) 执行时：同一 GPU 上的两个 expert（来自不同模型）交替使用计算和通信资源，完全消除同步等待——Model a 的第一个 all-to-all 与 Model b 的第二个 all-to-all 可在时间上重叠。
    - **编译框架层**：论文未明确说明。通信调度可通过在计算操作的 buffer 层调用 NCCL 集体库按指定顺序执行实现。
    - **Kernel层**：论文未明确说明（标准 GPU kernel + NCCL）。
    - **硬件架构层**：同 baseline 的 big switch 网络。Aurora 的最优调度将通信时间压缩至 b_max——由最大列/行和除以带宽决定的理论下界。在 Colocating+Homo 场景下推理时间加速达 2.38×，Colocating+Hetero 达 3.54×。

- baseline方法是什么？
  - Baseline 是在 CPU 集群上部署 MoE 模型进行推理（如 2×64-core AMD EPYC + 512GB DRAM），所有 expert 均分同一集群资源，按固定周期（月/小时）计费，空闲资源仍需付费。
  - 全栈执行例子（GPT2 MoE，10240 tokens，text generation）：
    - **算法层**：MoE 模型（GPT2 改造，12 MoE layers × 4 experts），gating network 使用 top-k routing 将每个 token 路由到最相关的 expert。token ID 是唯一直接可用的 token 特征。
    - **系统框架层**：使用 expert parallelism，每个 expert 分配到一个 CPU 设备。所有 expert 同时执行，通过 all-to-all 通信完成 scatter（gating→experts）和 gather（experts→next layer）。非 MoE 层通过 model parallelism 分配。资源按固定粗粒度周期计费，空闲资源（如冷门 expert 所在设备）仍产生成本。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch eager mode）。
    - **Kernel层**：论文未明确说明（使用标准 PyTorch CPU kernel）。
    - **硬件架构层**：CPU cluster 节点间通过数据中心网络通信。所有 expert 的模型参数常驻内存（512GB DRAM），无冷启动问题，但每个设备无论负载高低均占用固定资源。
  - Baseline 缺陷：
    1. **计费模式不灵活**：CPU 集群按月/小时计费，即使 expert 在推理间隔期间空闲也产生成本。
    2. **无法差异化资源配置**：所有 expert 分配相同资源，冷门 expert 资源浪费，热门 expert 可能资源不足。
    3. **无 serverless 通信优化**：scatter-gather 通信在 CPU 集群上的 pipelining 设计（如 PipeMoE、MpipeMoE）依赖 GPU 集群硬件架构，无法直接迁移到 serverless 平台。
    4. **缺乏专家热度预测**：CPU 集群可在推理过程中动态调整资源分配，但在 serverless 平台上函数部署需要数分钟，无法动态调整；需要预先知道 expert popularity 来配置函数内存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：在 AWS Lambda serverless 平台上构建端到端 MoE 推理系统，包含三个核心模块：
    (1) **贝叶斯专家选择预测器**：使用 token ID + position ID + attention ID 三维特征，通过重新设计的后验概率计算方法（Bayes 定理 + 两重积分引入 P'(f2) 和 P'(f3)），在推理前预测每个 expert 的 token 负载。
    (2) **三种 serverless scatter-gather 通信方法**：pipelined indirect transfer（通过 S3，minibatch pipeline 重叠下载/上传/计算）、non-pipelined indirect transfer、direct function invocation。支持按 MoE 层混合选择，在 payload size 限制下灵活权衡。
    (3) **BO + ODS 全局优化框架**：将 MoE 部署形式化为 MIQCP 问题，ODS 算法（O(|E|) 时间）联合决策 communication method、memory config、replica count；多维度 ε-GS 迭代调整 key-value table 优化预测准确性，在 billed cost 反馈下收敛到近优部署。
  - 对比 baseline 全栈执行例子（GPT2 MoE，10240 tokens，AWS Lambda）：
    - **算法层**：相同 MoE 模型结构，gating network 不做修改。但增加了**前置专家预测**：从 profiled data 中统计 (token_id, position_id, attention_id) → expert 的映射频次；对新 token，用 token ID 查表，position ID 假设均匀分布，attention ID 用最高 attention score token ID 近似，通过两重积分计算 P(N_{e,i}|f1') = ∫∫ P*(N_{e,i}|f1',f2,f3) · P*(f1',f2,f3)·P'(f3)/P*(f1',f2) · P*(f1',f2)·P'(f2)/P*(f1') df3 df2，取 argmax 得预测专家。
    - **系统框架层**：
      - 每个 expert 部署为一个独立的 AWS Lambda 函数，内存按预测 popularity 差异化配置（热门 expert 3008MB，冷门 128MB）。
      - 热门 expert 可复制多份（最多 8 副本），每副本处理一部分 token。
      - Scatter-gather 通信由 ODS 算法逐层选择最优方法：
        - 小 batch（256 tokens）：direct transfer（a^e=3），gating 函数直接调用 expert 函数传输数据。
        - 大 batch（2560 tokens）：pipelined indirect transfer（a^e=1），gating 函数将 minibatch 写入 S3 → expert 函数从 S3 下载+计算，同时 S3 上传前一批处理结果 → 非 MoE 层从 S3 下载聚合。
      - 按毫秒粒度按量计费（GBs），无请求时零成本。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：论文未明确说明（CPU serverless 函数使用 PyTorch 标准 kernel）。
    - **硬件架构层**：AWS Lambda 自动分配 vCPU（内存越大 vCPU 越多），每个函数独立执行。S3 bucket 作为中间存储 relay 数据。函数冷启动延迟通过预热（warm start）缓解。
  - **关键设计应对 Baseline 缺陷**：
    - 缺陷1（粗粒度计费）→ serverless per-GBs 计费，按需调用，无空闲成本。billed cost 降低至少 75.67% vs CPU cluster。
    - 缺陷2（均分资源）→ 基于预测 popularity 的差异化内存配置 + 专家副本机制。热门 expert 多资源，冷门省资源。
    - 缺陷3（通信无优化）→ 三种 serverless 专用通信方法（含 pipeline 设计），按 MoE 层 + token 量灵活选择，兼顾 payload size 约束。
    - 缺陷4（无预先预测）→ 三维 token 特征的贝叶斯预测 + BO 框架迭代优化。比 Lina（仅用 token ID）预测更准确，比 LSTM 方法更轻量。

## Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

- baseline方法是什么？
  - Baseline 是标准 top-k MoE 路由（如 Qwen3 默认的 top-8 out of 128 experts），在 decode 阶段每个 token 独立激活 k=8 个 router 得分最高的专家。
  - 全栈执行例子（Qwen3-30B，B=16，decode 阶段）：
    - **算法层**：Router R 对每 token 输出 N=128 维 softmax 分布，取 top-8 索引。batch 中 16 个 token 各选 8 个专家，理论上 T（唯一激活专家数）在 8~128 之间，期望值约 82 个（公式 N(1-(1-k/N)^B)）。
    - **Serving框架层**：SGLang 将 16 个 token 的 decode batch 送入 MoE 层。Router 评分后，每个 token 的 top-8 专家索引确定。SGLang 聚合所有被选中的专家权重，调用 Grouped GEMM kernel 加载权重到 SRAM 并计算。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：Grouped GEMM（cuBLAS）批量执行不同专家的 (2048×768) 矩阵乘法。对于 memory-bound 的 decode，每个专家权重加载（b 项）主导总延迟（b·T + a·Bk），T≈82 时延迟 ~175μs。
    - **硬件架构层**：H100 HBM→SRAM 带宽是瓶颈。每个专家 3 个权重矩阵（SwiGLU 的 3 个 2048×768 GEMM），加载 82 个专家的权重耗时远大于计算时间。
  - Baseline 缺陷：在中等 batch size（如 16）下，MoE 层处于 memory-bound 状态。因为每个 token 仅激活 k=8/N=128 个专家（稀疏因子 16×），平均每专家负载仅 B·k/N=1 token，远低于 compute-bound 所需的大量 tokens。延迟被"加载所有被激活专家的权重"主导（T 项），而非计算量（Bk 项）。T 随 batch size 快速增长（batch=1 时 T=8，batch=16 时期望 T=82），导致 decode 延迟恶化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：OEA（Opportunistic Expert Activation），一种无需重新训练的 batch-aware 动态路由算法，通过两阶段策略最小化 batch 内唯一激活专家数 T。
    - **Phase 1（Baseline Expert Selection）**：每 token 仅激活 top-k0（k0 < k）个最关键的专家，构成 S_base，保证每 token 的独立质量底线。关键洞察：top 排名的专家对输出质量最为关键（Gupta et al., 2024）。
    - **Phase 2（Opportunistic Piggybacking）**：每 token 从 S_base（Phase 1 中所有 token 已激活的专家集合）中寻找其 k0+1 到 k 位的"次优先"专家，若能找到则免费附加上去（因为这些专家权重已在 SRAM 中）。这保持了 T = |S_base| 不变，仅增加计算量（可忽略，因 memory-bound）。
  - **对比 baseline 全栈执行例子**：
    - **算法层**：OEA 替换 top-8 为 top-k0 + piggybacking（k0=5 时）。Phase 1 每 token 只保证 5 个基线专家，Phase 2 有机会用已在 S_base 的专家补到 8 个。每 token 最终仍激活约 8 个专家（质量不降），但 T = |S_base| ≈ 35（vs. baseline 约 48）——仅 Phase 1 控制 T 的规模。
    - **Serving框架层**：在 SGLang 的 MoE decode 路径中插入 OEA 路由器。OEA 先统计 S_base 再分配 piggybacking。路由仅在 decode 阶段使用（prefill 已足够 compute-bound）。额外修改：捕获 CUDA Graph 到 batch size 16 以避免 SGLang padding 引入无用专家。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：Grouped GEMM 加载 T≈35 而非 T≈48 个专家的权重。b 主导延迟，T 降低 27% 意味着 ~23% 的延迟降低（k0=5 时 175.7→136.0μs）。
    - **硬件架构层**：HBM 带宽压力降低 ~27%。更少的权重加载意味着每个 decode step 更快完成，latency 从 175.7μs 降至 136.0μs（k0=5, 23% reduction）和 106.8μs（k0=3, 39% reduction），且准确率无统计显著退化。

## No Need to Talk: Asynchronous Mixture of Language Models

- baseline方法是什么？
  Baseline 是标准同步分布式数据并行训练的 dense LLM（Standard Synchronous Distributed Data Parallel Training）：
  - 算法层：单个 dense Transformer decoder 模型，所有参数同时参与训练和推理。
  - 系统框架层：分布式训练使用 PyTorch/JAX 的数据并行，每个 training step 后进行 all-reduce 梯度同步。对于 1.3B 参数模型，每步每节点需传输约 10.4GB 梯度数据（float32）。推理时整个模型需全部加载到 GPU 内存。
  - 编译框架层：论文未明确说明（使用标准 PyTorch eager mode / JAX XLA 编译）。
  - Kernel调度层：论文未明确说明（使用框架默认 kernel 实现，如 cuBLAS、FlashAttention 等标准算子）。
  - 硬件架构层：依赖高带宽互联（如 NVLink、InfiniBand）进行梯度同步，8-128 GPUs 集群。
  Baseline 的核心痛点：(1) **训练通信瓶颈**——每次迭代需要 all-reduce 同步梯度，数据量巨大（1.3B 模型每步 ~10.4GB/节点），严重依赖高速互联硬件；(2) **推理参数冗余**——推理时所有参数均需驻留在 GPU 内存中并参与计算，无法稀疏激活；(3) **异步训练方案性能退化**——已有的异步 SGD、Local SGD 等减少同步频率的方法会导致 perplexity 显著低于每步同步的 baseline；(4) **先前 MoE 方案依赖 token 级路由**——Switch Transformer 等 MoE 方法虽能稀疏激活参数，但路由决策在每个 token 上做出，要求所有 expert 常驻 RAM 且仍需高通信开销进行梯度同步。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 SMALLTALK LM——一种序列级硬混合专家（Hard MoE）方法，通过小型路由 LM 实现数据分区和独立 expert 训练：
  - 算法层：将训练分解为两个解耦阶段。**Stage 1——Router 训练**：使用 E 个极小的语言模型（4.4M 参数，仅为 expert 的 1.3%）作为 router，通过 EM 算法（交替优化 router 的 NLL 和 balanced assignment）学会将数据按 prefix（256 token）分配到不同 expert。关键设计是 **balanced assignments**——按 min log-likelihood 排序后贪心分配，确保每个 expert 获得等量数据。**Stage 2——Expert 训练**：每个 expert 在自己的数据子集上完全独立训练，无需任何梯度同步。推理时，router 对输入 prefix 评分并选择单个 expert 执行自回归生成。
  - 系统框架层：训练框架从同步分布式变为**完全独立并行**——router 训练用 PyTorch（仅需少量 all-gather 通信，每次 <6MB，总计约 100 次），expert 训练用 JAX 独立运行，零通信。推理时仅需加载被选中的单个 expert（总参数的 1/E），其他 expert 可以 offload 或驻留在不同节点。对比 baseline：1.3B × 32 experts 模型，训练通信从 ~10.4GB/步 降至 router 训练期间约 100 次 <6MB 通信，推理参数从全量 1.3B 降至单 expert 1.3B（相同的 inference FLOPs）。
  - 编译框架层：论文未明确说明（使用标准 PyTorch eager mode / JAX 编译）。
  - Kernel调度层：论文未明确说明（使用框架默认 kernel）。
  - 硬件架构层：由于 expert 训练和推理完全独立，**不再需要高带宽互联**——每个 expert 可在独立的低带宽节点上训练，甚至可以在不同时间、不同地理位置训练。对比 baseline 需要紧密互联的 GPU 集群，SMALLTALK LM 可在松耦合的异构节点上运行。

  核心设计如何解决 baseline 痛点：
  1. **训练通信瓶颈** → Router 完成数据分区后，expert 训练**零梯度同步**，仅 router 训练期间有少量 loss 值通信（<100 次，每次 <6MB）。
  2. **推理参数冗余** → 序列级路由使推理时仅激活 **1 个 expert**，参数量与 dense baseline 相同，但混合模型总容量为 E 倍。
  3. **异步训练性能退化** → 不同于 Local SGD 的方案（因梯度延迟导致性能下降），本方法不降级是因为每个 expert 在自己的不相交数据子集上做**标准同步训练**（子集内），不存在跨 expert 的梯度 staleness 问题。最终 1.3B × 32 experts 的 perplexity 比 dense baseline 低 17.56%。
  4. **Token 级 MoE 的高通信/内存需求** → 序列级路由使 expert 间完全解耦，无需在推理时为每个 token 做跨 expert 路由决策，也无需所有 expert 常驻 RAM。

- baseline方法是什么？
  Baseline 是 LLaVA-1.5 的全参数微调方法（以及对 MoE 模型做全参数微调的 MoExtend-Full / MoE-LLaVA）。核心缺陷：
  (1) **Catastrophic Forgetting（灾难性遗忘）**：全参数微调让 LLM 在学会视觉理解的同时遗忘原有的文本知识。Mixtral 8x7B 全参数微调后（MoExtend-Full），在纯文本 benchmark 上平均下降 3.30 分（Avg. drop），而 MoE 架构对全参数微调尤为敏感——MoE-LLaVA 的 Avg. drop 高达 7.86 分。
  (2) **高昂的训练成本**：全参数微调 Mixtral 8x7B 需要 ~200 小时的 instruction tuning（8×A800），而随着模型规模增长，这一成本愈发不可承受。
  (3) **模态间隙未有效弥合**：使用少量线性投影层或 LoRA 等参数高效方法虽然减少遗忘，但无法让 LLM 充分理解新模态，限制了多模态能力。

  全栈执行例子：一张图片 + 一个问题文本输入 LLaVA-1.5 → CLIP ViT 编码 visual token (P×D) → project 层映射到 LLM hidden space → 与 text token 拼接为 (N+P)×D → 进入 Vicuna-13B（dense LLM）逐层 attention + FFN 前向 → 全参数微调时所有 13B 参数参与梯度更新 → 问题：原有 MMLU/GSM8K 等文本知识在 FFN 权重中被覆盖 → 推理文本任务时使用被覆盖的 FFN 权重，性能下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoExtend**，一种专为 MoE 模型设计的新模态扩展方法。核心思路：**冻结所有原有 MoE 参数，只在关键层新增 expert，仅训练新增部分**。

  **对应缺陷 1（Catastrophic Forgetting）→ 冻结原有参数 + Calibration Module**
  - 不修改原有 MoE 模型的任何参数（expert FFN、router、attention 全部冻结），原有知识完全保留。
  - 新增 expert 后 softmax 概率分布会变化：s(x)_j' = e^{f(x)_j} / (Σ e^{f(x)_h} + e^{f(x)_{m+1}}) ≤ s(x)_j，原有 expert 被选概率下降，forward 输出分布漂移。Calibration Module 通过 s_c(x) 修正每个 expert 的输出权重：MoE(x) = Σ s(x)_j · [1+s_c(x)] · FFN(x)_j，s_c 初始零输出来保证初始一致性。

  **对应缺陷 2（高昂训练成本）→ 仅训练新增参数**
  - 新增 expert 仅添加到 50% 的 MoE 层（由 Extender 自动决定），每层仅加 1 个 expert。对于 Mixtral 8x7B（32 层，8 experts/层），仅训练 16 个新 expert + 对应 router 列 + 轻量 Calibration modules。
  - 激活训练参数量仅 ~3B，训练时间 ~30 小时（Alignment ~15h + Fine-tuning ~30h），对比全参数微调 ~200 小时，加速约 6 倍。

  **对应缺陷 3（模态间隙）→ 新增专用 expert 而非投影层微调**
  - 新增的 expert 是完整的 FFN 层（而非 LoRA adapter 或投影层），拥有足够容量学习新模态的特征变换。
  - 新 expert 初始化策略：复制该层对视觉数据响应最活跃的原有 expert 权重，使得新 expert 从"最接近视觉理解"的参数空间出发训练，加速收敛并保证选中概率。

  全栈执行例子：一张图片 + 问题文本输入 MoExtend → CLIP ViT 编码 visual token → MLP project 对齐 → 拼接为 (N+P)×D → 进入 Mixtral 8x7B，逐层 attention + MoE：在未扩展层（16层），router 从 8 个原有 expert 中选 top-2 计算（与原始模型完全一致，无遗忘风险）；在扩展层（16层），router 从 9 个 expert（8 个冻结原有 + 1 个新增可训练）中选 top-2 → 若新增 expert 被选中，其 forward 输出经过 Calibration s_c(x) 修正后加权求和 → 新增 expert 专门处理视觉 token，原有 expert 处理文本 token → 仅新增 expert 的 FFN 参数 + v_new router 列 + s_c 参与梯度更新（~3B params vs 46.7B 全参数）→ 推理时与原始 MoE 流程一致，无额外开销。

## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- baseline方法是什么？
  Baseline 是 DeepSpeed-MoE 的训练和推理系统。DeepSpeed 使用 ZeRO 策略 + 参数预取实现 MoE 训练，但存在三个核心缺陷：
  (1) **存储管理粗粒度**：DeepSpeed 的 Zero-Infinity 将参数统一 prefetch，不区分 MoE 中 sparse parameter（expert FFN，选择性激活且占存储大头）和 dense parameter（attention，始终激活）的异构特性，导致 SSD 寿命损耗和性能下降（SSD 满容量时性能衰减）；
  (2) **通信效率受限**：DeepSpeed 的 AlltoAll 设计主要通过层间 tensor fusion 将小 packet 合并为大 packet 通信，解决了 per-port 通信量小的问题，但未针对实际集群网络拓扑（intra-node NVSwitch vs inter-node switch hierarchy）做优化，跨 rank 通信经过 spine switch 造成路由冲突和带宽浪费；
  (3) **负载均衡盲点**：在 multi-task MoE 训练（如 UFO）中，不同 task 的 batch size 差异导致"木桶效应"——重 task 节点处理时间远长于轻 task 节点，轻 task 节点计算完毕后空闲等待（bubble），整体 FLOPS 利用率低。
  全栈执行例子：训练一个 100B+ MoE 模型 → 每层 MoE 含 attention (dense) + MoE FFN (sparse, 64 experts) → DeepSpeed 通过 Zero stage 3 对所有参数做统一 partition 和 prefetch → "Forward: 从 SSD/CPU 预取所有参数 → GPU compute → AlltoAll exchange expert hidden states → Backward → AlltoAll sync gradients → Optimizer update" → 问题 1：AlltoAll 通信中 GPU0 of Node1 (cluster A, rank0) 与 GPU7 of Node2 (cluster B, rank7) 间的数据经过路径 NIC1→LE1→SPq→LE1→NICn，spine switch 成为瓶颈且与其它 GPU pair 的通信竞争，形成阻塞。问题 2：SSD 频繁写入导致擦除次数高，且不区分 sparse/dense 使得大量 dense 参数预取也走 PCIe，浪费 NVLink 高带宽。问题 3：UFO 多任务训练时 task1（batch 512）的 GPU 需要 2× 的 batch 数据，处理时间远长于 task3（batch 128），其余 GPU 等待。编译框架/Kernel调度：论文未明确说明 baseline 的 kernel 细节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoESys**，一个基于 PaddlePaddle/PaddleFleetX 的 MoE 训练与推理系统，通过四项核心设计解决 DeepSpeed 的三个缺陷：

  **对应缺陷 1（存储管理粗粒度）→ Hierarchical Storage + 2D Prefetch Scheduling**
  - 将 MoE 参数按激活特性分为 sparse（expert FFN）和 dense（attention）：dense 参数始终在 GPU HBM；sparse 参数存 SSD，通过 CPU memory 做 LFU 缓存。
  - 2D Prefetch：水平维度利用 NVLink（高带宽）预取 dense 参数 → AllGather 获取所有 shard；垂直维度利用 PCIe 从 CPU cache 或 SSD 预取 sparse 参数。两个维度并行，与当前层计算重叠。
  - 引入 Intel Optane PMem (AppDirect + DAX) 替代传统 SSD，提供字节级寻址 + DRAM-like 延迟 + SSD-like 持久性，解决传统 SSD 的擦除寿命和延迟问题。
  - GPU-Node/CPU-Node/SSD-Node 的容量约束公式确保各存储层不溢出。

  **对应缺陷 2（通信效率受限）→ Hierarchical AlltoAll (Resource-Aware Communication)**
  - 利用网络拓扑层次：阶段一 intra-node AlltoAll via NVSwitch → 将数据搬移到同节点内对应 rank 的 GPU；阶段二 inter-node AlltoAll via NIC grouped by rank → 同 rank GPU 直连同一 leaf switch，不经过 spine switch。
  - 效果：peer-to-peer 通信效率提升 p 倍（p=单节点 GPU 数），inter-node 带宽利用率最大化。80.7B model / 4 nodes 32 GPUs 下通信阶段 speedup 15.5%，整体训练提升 10.3%。

  **对应缺陷 3（负载均衡盲点）→ Elastic MoE Training**
  - 动态调整节点数：轻量 task 合并节点（2 task→1 node），重量 task 拆分节点（1 task→多 node + data parallelism）。
  - 成本感知的 scale up/down 策略：upscaling 提升整体 throughput，downscaling 在资源受限时控制成本。
  - UFO model / 4 tasks 下 per-GPU throughput 提升 18.2%；VIMER-UFO 2.0 上 throughput 提升 64%，memory 降低 18%。

  **额外优化：Embedding Partition in Data Parallelism + Ring Memory Offloading**
  - Embedding Partition：沿 hidden_size 列切分（非 vocab 维度），3 次 AlltoAll 替代 AllReduce，大幅降低大 vocabulary 场景下的 GPU memory（如 400M param 配置从 15.81 GB 降至 8.63 GB）。
  - Ring Memory Offloading (inference)：CPU-GPU 环形内存流水线——GPU 缓存 K 份 expert 参数，计算第 i 层时释放 Pi 并异步加载第 (K+i) 层，多个 CUDA stream 实现 compute 与 data movement 重叠，GPU memory 节省 ≥30%。

  全栈执行例子（对比 baseline）：训练 104.1B MoE model on 64 A100 GPUs → 
  - **算法pipeline层**：参数分类后 dense 16D=16×1B≈16GB per device 常驻 GPU HBM；sparse 12S≈12×103B≈1236GB 存 SSD，α=0.02 激活概率下 GPU 仅需 4αS/L≈0.7GB 的 sparse 参数空间。
  - **系统框架层**：PaddleFleetX 分布式训练 → data parallelism (dense) + expert parallelism (sparse) → Gate 网络 AlltoAll 收集路由结果 → 2D prefetch 同时从 NVLink 预取下一层 dense 参数 + 从 CPU/SSD 预取下一层 sparse 参数（LFU cache 命中检查 → 未命中则 SSD→CPU→GPU）。
  - **编译框架层**：PaddlePaddle JIT 转静态图 → graph fusion 消除冗余 → kernel fusion (fused MHA) 减少 kernel launch 开销。
  - **Kernel调度层**：Hierarchical AlltoAll 替代标准 AlltoAll：intra-node 900GB/s NVSwitch → inter-node 100G NIC 同 rank 分组的 leaf switch 直连 → 通信耗时减少 15.5%。Custom H2D/D2H kernel：cudaHostAlloc pinned memory + cudaMemcpyAsync 异步传输与计算重叠。
  - **硬件架构层**：A100 GPU (108 SM, 80GB HBM2e, 2TB/s) + NVSwitch (900GB/s) + Mellanox 100G NIC + leaf/spine 交换机拓扑。论文未涉及 RTL/芯片/模拟器。

  效果：training throughput 209970 tokens/s vs DeepSpeed 157728 tokens/s（+33%），memory 54.4GB vs 66.3GB（-18%）。编译框架/芯片设计：论文未明确说明。

- baseline方法是什么？
  Baseline 是传统的 Speculative Decoding（SD）应用于 MoE 模型。学术界普遍认为 SD 对 MoE 无效，原因有二（1）对于小 batch：验证阶段多 draft token 激活更多 expert，导致参数加载量显著增加，T_T(B,γ) 远大于 T_T(B,1)，SD speedup 很低甚至 <1.0；（2）对于大 batch：系统进入 compute-bound 状态，验证时间随 token 数线性增长 T_T(B,γ)/T_T(B,1)→γ，SD 同样失去加速效果。现有 SD 研究主要关注提升 acceptance rate α（算法指标），忽视了目标模型架构和 workload 等系统因素对 speedup 的影响。
  全栈执行例子：1 个 request → Draft model 自回归生成 4 个 draft tokens → Target MoE model（如 Mixtral-8x7B，K=2）以 batch=1, γ=4 验证 4 个 tokens：4×2=8 次 expert 激活，可能激活 5-8 个不同 experts → 加载 5-8 个 expert 的权重（每个约数百 MB）→ 由于 batch=1 时仅需激活 2 个 expert 做单 token 解码，验证的参数加载量变为原来的 2.5-4× → T_T(B,γ) >> T_T(B,1) → SD speedup 远低于 dense 模型。Serving 调度层：vLLM continuous batching 下请求数少时无法隐藏 expert 加载延迟。编译框架/Kernel调度/硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法推翻"SD 对 MoE 无效"的固有认知，证明在**中等 batch size**（所有 expert 已激活但 GPU FLOPs 未充分利用）下，SD 对 MoE 的加速效果甚至优于 dense 模型。核心设计：
  1. **中等 batch size 消除额外 expert 加载**：当 batch size B 足够大使 N(B)≈E（所有 expert 已被单步解码激活），验证 B×γ 个 tokens 不再激活新 expert，仅增加计算量 → 系统处于 memory-bound 时计算增量几乎免费 → T_T(B,γ)≈T_T(B,1)。
  2. **更稀疏的 MoE 放大加速窗口**：Texp(t;ρ) = ρt/(1-(1-ρ)^t)，ρ 越小 → 每个 expert 处理 token 越少 → 系统更 memory-bound → 延迟 compute-bound 转型 → SD 有效加速的 batch size 范围更广。
  3. **Target Efficiency（新系统指标）**：= T_T(B,1)/T_T(B,γ)，解耦系统瓶颈与 acceptance rate 等算法指标。即使 acceptance rate 相同，target efficiency 也能解释不同模型/工作负载下 SD speedup 的巨大差异。
  4. **性能模型（Algorithm 1）**：融合 roofline model（G(t)）、激活专家数 N(t)、expert load Texp 三个因子，通过最小二乘拟合确定参数，预测任意 workload 的 SD speedup。
  全栈执行例子（对比 baseline）：B=32 个 requests → Draft model 生成 γ=4 个 draft tokens → Target MoE model（Qwen2-57B-A14B，K=8，E=14）：N(32)≈14=满激活 → 验证 32×4=128 tokens 时仍仅激活 14 个 expert → T_T(32,4)≈T_T(32,1)（计算增量在 memory-bound 下近乎免费）→ speedup ≈ 2.29×。对比 baseline 的 B=1：N(1)≈8，验证时 N(4)≈13.5，多加载 5+ 个 expert → T_T(1,4)≈1.7×T_T(1,1)。
  关键设计决策：
  - 通过理论推导 N(t) 和 Texp(t;ρ) 精确刻画了 batch size B 与 sparsity ρ 如何共同决定 SD 有效加速窗口。
  - 提出 target efficiency 指标使研究者能独立评估系统因素对 SD speedup 的影响，补充了仅关注 acceptance rate 的不足。
  - 性能建模以 roofline model 为核心，引入可解释的拟合参数（bias=kernel launch overhead，k1/k3=roofline 强度，k2=单 expert 加载时间），使端到端加速透明可解释。
  - 对 private serving、latency-critical、memory-constrained 等实际场景的适用性分析，证明理论发现在实际部署中的价值。
  - 编译框架/Kernel调度/硬件架构/芯片设计：论文未明确说明。

## MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

- baseline方法是什么？
  Baseline 是现有 LLM 后训练剪枝方法 SparseGPT 和 Wanda。SparseGPT 的剪枝度量 S = [|W|^2 / diag(H^{-1})]，需要估计逆 Hessian 并更新剩余权重；Wanda 简化度量 S = |W_ij| * ||X_j||，仅需校准数据计算输入激活列范数，不更新权重。两种方法都**未考虑 MoE router 的路由信息**，对所有 FFN 层（包括 MoE expert 层）使用统一的剪枝度量。
  全栈执行例子：一个 token 进入 MoE layer → Router 计算 top-2 gating 并选择 Expert_i 和 Expert_j → 每个 expert 执行 SwiGLU FFN：x → W_gate·x 与 W_up·x → SiLU(W_gate·x) ⊙ (W_up·x) → W_down 输出 → 加权求和。Wanda 剪枝时：对 expert 内的 W_gate/W_up/W_down，用校准数据前向得到 X → 计算 S = |W| * ||X||（所有 expert 共用同一度量，router 选择的差异化信息被丢弃） → 每个输出神经元保留 top-(1-p%) 重要性权值。问题：Router 权重 Gate 本身反映了"这个 expert 对这个 token 的重要性"——若 Gate_i ≈ 0（该 expert 对此类 token 几乎不被激活），其权值即使 magnitude 大也应被优先剪除，而 Wanda 无法捕获此信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-Pruner，核心是将 router 权重显式纳入剪枝度量：**S = |W_ij| * ||X_j * Gate_j||**。Gate_j 是 router softmax 输出中对当前 expert 的归一化权重（广播到所有输入维度后与 X_j 逐元素乘），使重要性计算包含"这个 expert 对这一批输入 token 有多重要"的信息。此外提出 expert-wise knowledge distillation 做剪枝后性能恢复：以未剪枝 pretrained model 为 teacher，逐 expert 计算 MSE loss 叠加 CE loss 蒸馏 student。
  全栈执行例子（对比 baseline）：同一 token → Router 计算 top-2 gating 同时输出所有 expert 的 Gate 权重向量 [g_0, g_1, ..., g_7] → Expert_i 执行前向得到 X' 和 Gate[:,i] → **MoE-Pruner 度量计算**：X_gated = X ⊙ broadcast(Gate[:,i])，S = |W| * ||X_gated||（此时被 Gate 放大的激活维度对应权值重要性更高，被 Gate 压制的维度权值重要性更低） → 剪枝后保留的权值集中在"高 router 权重 token 的活跃激活路径"上。关键区别：（1）Wanda 对所有 expert 平等对待，MoE-Pruner 利用 router 告诉它"哪些 expert 对这个 token 更重要"，从而更精确地保留关键权值、剪除非关键权值；（2）剪枝后通过 expert-wise KD 蒸馏：对每个 MoE layer 的每个 expert，L_expert = MSE(E_teacher, E_student)，保证每个 expert 输出分布逼近 teacher，而非仅靠全局 CE loss。编译框架/Kernel调度/硬件架构：论文未明确说明。

<｜｜DSML｜｜parameter name="replace_all" string="false">false

- baseline方法是什么？
  Baseline 是标准的 MoE 架构（如 Mixtral-8x7B、DeepSeek-V2-Lite），其中每个 MoE layer 包含 N 个 monolithic expert（每个 expert 是一个完整 FFN），router 通过 top-k 选择机制激活固定数量的 expert（如 k=2 或 k=6），最终输出为激活 expert 输出的加权求和。
  全栈执行例子：一个请求 token 进入→Router（线性层）计算所有 N 个 expert 的 logits→top-k 选择 Expert 3 和 Expert 7→GPU 加载 Expert 3 和 Expert 7 的完整权重矩阵（W_gate, W_up, W_down）→依次计算两个 expert 的 SwiGLU FFN→加权求和得到输出。若降低 k 从 6 到 5，模型质量会因"Quality Cliff"而出现不成比例的大幅下降（因为 monolithic expert 内部冗余性未被利用，丢弃任何一个完整 expert 都会丢失其中被训练为协作的关键 neuron）。云端调度：FIFO 或 FullBatch 调度器使用批次内最高 k_min 作为全局 k_active，粗粒度配置导致小 batch 时资源浪费或大 batch 时无法灵活降级。Offloading 场景：即使只需 expert 中 25% 的 neuron 计算，仍需从 CPU 加载整个 multi-GB monolithic expert 到 GPU VRAM，I/O 浪费严重。编译框架/Kernel调度/硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法分为 Offline Refactoring Engine（离线模型重构）和 Online Scheduling Engine（在线调度引擎）两阶段。
  全栈执行例子：
  - **算法Pipeline层**：Neuron Activation Profiler 在 Wikitext-2-raw-v1 上运行原模型，从每个 expert 的 SwiGLU FFN 中收集激活矩阵 M(B×C numpy)→Partitioning Optimization Solver 用 Simulated Annealing（T₀=100, α=0.995, 100K 迭代）将每个 expert 的 C 个 neuron 划分为 N=4 个子 expert 的分区 P*，优化目标为最小化所有 batch 上被 deactivated sub-experts 的 L1 norm 之和→Gating Mechanism Reconstructor 构建共激活矩阵 C_co=B^T·B（B 为 top-k_a 激活二值化矩阵），从每个子 expert 选择 centrality 最高的 r=4 个 gate neurons 作为代理评分器。可选：在 SlimPajama 上仅微调 linear router（<0.1% 参数），curriculum training 逐步增加 k。
  - **Serving调度层**：部署前 benchmark 构建 C(k_active) 性能模型（sub-expert 数量→延迟/内存的 lookup table）。云端场景：请求到达→按 k_min 加入所有符合条件的虚拟队列→对 M 个虚拟队列并行计算效用 U_m=Σtokens/C(|Q_m|,m)→发射最高效用批次（或触发 Batch Full/Timeout 硬触发器）→修改版 vLLM 0.9.1 执行 fine-grained sub-expert 推理，仅激活必要的子 expert。Offloading 场景：VRAM Cache Manager 用 LRU 管理 sub-expert 缓存→解码循环中 router 通过 gate neurons 估算每个子 expert 的 L1 norm→对 miss 子 expert 异步 CPU→GPU 传输→GPU 计算→LRU 更新。
  - **编译框架/Kernel调度/硬件架构/芯片设计**：论文未明确说明。
  核心设计如何解决 Baseline 痛点：
  1. **Quality Cliff → Smooth Trade-off Curve**：通过将 1 个 monolithic expert 分解为 4 个子 expert，一个 k=6 的模型变为 k_active=24 的细粒度配置空间，提供 4 倍以上的可区分稳定操作点。例如原模型只能选择 k=2 或 3，MoE-Prism 可选择 k=9（相当于原模型 2.25 个 expert），精确匹配 SLO 需求。
  2. **粗粒度 Offloading I/O → 精准按需传输**：传统方式必须加载整个 expert（即使只需其中部分 neuron），MoE-Prism 只传输 S_req(t) 中命中的子 expert，16GB 配置下 cache hit ratio 从 0.4375 提升到 0.4453，且可通过加载 17 个子 expert（等效 4.25 experts）满足"需要 4.2 experts"的 SLO，避免被迫加载 5 个完整 expert 的浪费。
  3. **固定 k 调度僵局 → 效用驱动的动态多队列调度**：打破"批次组成依赖 k_active，k_active 选择又依赖批次组成"的循环依赖，通过维护 M 个虚拟队列并行评估所有可能的 (k_active, batch) 组合，选择最高瞬时吞吐效用的配置，实现吞吐提升 19.9%（Deepseek）和 14.9%（OLMoE）。

## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- baseline方法是什么？
  Baseline 是标准 Hugging Face Transformers 的 device_map offloading 和 Mixtral-Offloading。两者均采用反应式策略管理 expert 参数——GPU 计算到某个 MoE layer 时，同步触发 PCIe H2D 传输所需的 experts，GPU compute units 在此期间停滞等待。
  全栈执行例子（以 Phi-MoE 在 A100-40G 上解码一个 token 为例）：
  - **算法层**：Standard autoregressive decoding。token hidden state h → Self-Attention (FP16, on-GPU) → MoE layer: Router(W_gate * h) → softmax → top-2 selection (say Expert 5, Expert 7) → **I/O Stall Begins**: GPU 发起从 host DRAM 加载 Expert 5 和 Expert 7 的 FP16 权重（每个 expert ~6400×4096×3 ≈ 150MB FP16）→ PCIe 4.0 x16 传输耗时 ~9.4ms → GPU compute: gate_proj + up_proj + SiLU + down_proj (~0.2ms) → token output。Figure 4 profile 显示 Mixtral-8x7B 中 Memory 操作（主要是 PCIe 传输）占总时间 98.9%，GPU compute < 15%。
  - **系统框架层（Serving）**：HuggingFace Transformers device_map 将 MoE expert 权重静态映射到 CPU RAM，GPU 端无 expert cache 或预取逻辑，每 token 每层触发同步 PCIe 传输。Mixtral-Offloading 引入 per-expert on-demand swapping 和 LRU cache，但 cache 策略是反应式的（LRU 响应已发生的访问模式而非预测未来），在高 entropy expert activation（图 5: Qwen-1.5MoE 每层激活熵接近理论最大值）下命中率仅 29.2%（16GB 配置, expert capacity=6）。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch + CUDA。
  - **Kernel调度层**：标准 PyTorch FP16 GEMM kernels (cuBLAS)，fine-grained MoE 中单个 expert 的 GEMM 矩阵太小（Qwen2-MoE: inter_dim=1408），GPU SM utilization 很低。Marlin 量化后端在 MoE 场景同样面临 kernel launch overhead 问题，甚至慢于 PyTorch FP16 baseline（图 11）。
  - **硬件架构层**：NVIDIA A100-40GB GPU + CPU DRAM + PCIe 4.0 x16（32GB/s 双向理论带宽）。Baseline 的核心瓶颈：数据依赖（expert selection 必须在 token 的 attention hidden state 产生后才能确定）→ I/O 在关键路径上 → GPU compute unit 闲置率 > 85%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE-SpeQ 提出 **quantized speculative decoding × expert offloading co-design**——用 INT4 量化 MoE 模型（GPTQ）作为高速 draft model，在 I/O latency 期间做 useful computation（生成 draft tokens + 预测 expert activation patterns），将预测转换为 lookahead-driven prefetching 以隐藏 PCIe 传输延迟。
  全栈执行例子（同一 Phi-MoE token 在 MoE-SpeQ 下的执行路径）：
  - **算法层（Speculative Decoding with Quantized Draft）**：前一步 target verify 完成后共享 KV cache → Draft model (INT4, fuseMoE kernel, on-GPU) 自回归生成 k 个候选 token。每 draft token 的 MoE forward：Router(FP16, on-GPU) → softmax → top-2 → ELB entry = (expert_id, gating_score)。关键设计：(a) **Hybrid Precision**——Router/Attention/Shared Experts 保持 FP16 以保证 routing 保真度（router 量化误差通过 softmax 放大导致误路由→ELB 污染→cache miss）；MLP expert 主体 INT4 激进压缩速度和内存（43% VRAM 节省: 13.40GB→7.68GB）。(b) **KV Cache Sharing**——draft 在 target 的高精度 KV cache 上运行而非独立 cache，直接提升 token 接受率（>90% vs Eagle 的 80%）。(c) **Quantized Draft as Expert Predictor**——INT4 量化模型预测 target 的 top-4 expert selection 达 90.9% 准确率，优于专用 one-layer-ahead predictor (84.7%)，且单次 forward 预测所有层。
    **解决 Baseline 缺陷(1)**：将不可预测的 expert selection 变为可预测——量化 draft 的 90.9% fidelity 使系统获得 k 步 future token 的 expert 需求 lookahead，打破"必须先算 attention 才知道要哪些 experts"的串行依赖。
  - **Serving调度层（Expert Scheduler + Speculative Governor）**：
    - Expert Scheduler 的三阶段流水线：Phase I 利用 cache hits（locality-aware priming）→ Phase II 对 ELB 中部高 confidence 条目选择性预取（adaptive bandwidth-guided）→ Phase III 对尾部全部缺失 experts aggressive prefetch（cache saturation）。lookahead-aware eviction 替换最不可能被后续使用的 expert。
    - Speculative Governor 的 Amortization Roofline Model: 定义两个 Roof——Compute Roof（horizontal, Θ_max when I/O perfectly hidden）和 I/O Roof（sloped, Θ = B_PCIe × I_amort）。在线搜索 argmax_k Θ(k) = (Σ∏p_j) / [max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1)]，受离线 SLO 约束 k_SLO。k 的选择权衡：k 大 → amortization 效果好但 expert union 大（VRAM 压力 + 若草稿频繁被 rejected 则浪费计算）；k 小 → overhead 低但 I/O hiding 不足。
    **解决 Baseline 缺陷(2)**：将"等待 I/O → 传输 → 计算"的串行执行变为"草稿生成（与初始 I/O overlap）→ 预取（与草稿 overlap）→ 验证（无 I/O stall）"的流水线执行。Figure 13 显示 Phi-MoE 上 TPOT 从 536.7ms (Mixtral-Offloading-SC) 降至 163.1ms (3.3× speedup)。
  - **Kernel调度层（fuseMoE CUDA Kernel）**：细粒度 MoE 中每个 expert 的 GEMM 维度太小（K=1408, N=2048），单独 launch 无法占满 GPU SM。fuseMoE 将 per-layer 所有 expert 的 gate_proj + up_proj + SiLU + gate×up + down_proj 融合为单次 kernel launch，batch 不同 expert 的 token hidden states 使有效矩阵维度增大 → GPU occupancy 提升 → kernel launch overhead 减少。消融显示 fused kernel 贡献 31.8% speedup (Fig. 消融: w/o fused kernel = 68.2% of full speed)。
    **解决 Baseline 缺陷(3)**：解决细粒度 MoE 下量化推理 kernel 利用率低的问题，使 draft 阶段足够快（开销 < I/O latency），从而整个 speculative pipeline 有意义。
  - **编译框架/硬件架构/芯片设计**：论文未明确说明。

## MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

- baseline方法是什么？
  Baseline 是 **MoE-Lightning** [9]，state-of-the-art 资源受限 MoE LLM 推理系统，基于 Hierarchical Roofline Model（HRM）指导系统设计。以 Mixtral8x7B（32 layers, 8 experts/layer, k=2, 94GB BF16）在 A40 GPU（16GB effective）+ CPU（750GB DRAM）上的离线批量推理执行路径为例：
  - **算法层（MoE Routing）**：标准 top-k gating，Router 对每层 self-attention 输出计算 logits → Softmax → SelectTopK(k=2)。每个 token 分配给 2 个 expert。**缺陷(1)**：HRM 仅建模 arithmetic intensity 和 IO bandwidth，忽略了 CPU memory capacity 对并行 token 数量的约束——Table 1 显示 MoE-Lightning 的 CPU memory utilization 仅 35%-56%，大量 KV cache 容量未被利用，导致 GPU 端并发 token 数不足。
  - **系统框架层**：MoE-Lightning 使用 HRM 指导下的 CPU-GPU hybrid 执行——decode attention offload 到 CPU，避免 KV cache 传输。prefill 和 decode 作为两个独立阶段串行执行。**缺陷(2)**：prefill/decode 分离导致资源利用不均衡——prefill 阶段 IO 仅 23.9% 活跃、decode 阶段 GPU 仅 16.5% 利用率（图 1）。**缺陷(3)**：独立阶段执行使 KV cache 的峰值内存占用为 $p+g$（最大序列长度），而非 prefill/decode 重叠下的平均占用 $p+g/2$，浪费了 CPU memory capacity 的有效利用率。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch + CUDA。
  - **kernel调度层**：MoE-Lightning 使用标准 PyTorch/CBLAS 进行 CPU attention 计算，未针对 CPU 向量单元优化。**缺陷(4)**：auto-vectorized CPU attention 无法充分利用向量单元，达不到系统所需的 attention throughput。
  - **硬件架构层**：NVIDIA A40 GPU + CPU DRAM + PCIe 互连。IO bandwidth $B_{IO} \approx 19.5-32$ GB/s。MoE-Lightning 的 pipeline 中 weight transfer API 调用嵌入在执行流水线内，未与 PyTorch 操作和 attention 同步数据做隔离，导致 head-of-line blocking。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE-Lens**，核心是 holistic two-stage performance model + model-informed system design（Resource-Aware Scheduler + VSLPipe + Contiguous Data Mover + hand-optimized CPU attention kernel）。以 Mixtral8x7B 在 A40（16GB effective）+ CPU（750GB）上运行 MTBench（p=98, g=32, 70GB KV cache）的执行路径为例：
  - **算法层（MoE Routing 不变）**：Router 逻辑不变，但通过 system-level optimization 使 GPU 端并发 token 数大幅提升。PME（Equation 3）量化了不同 prompt/generation length 下的 memory efficiency——g=32 时 PME 较高，GPU utilization 可达 ~90%；g=256 时 PME 显著下降，GPU utilization 受限。**解决缺陷(1)**：通过 Stage 1 model 的 CPU memory capacity 分析（Equation 2, Table 2），识别出饱和 GPU compute 所需的 KV cache 大小，指导 system 配置 KV cache capacity 以最大化并发 token 数。
  - **系统框架层（MoE-Lens System）**：
    1. **Stage 2 performance model** (§5.5)：将 bounded batch size K、paged KV cache（block size b）、prefill/decode overlapping 调度策略纳入模型，预测真实 throughput（94% accuracy）。**解决缺陷(2)+(3)**：prefill/decode overlapping 调度（Equation 7-13）不仅平衡了 prefill 和 decode 阶段的资源利用（GPU 利用率从 MoE-Lightning 的 16.5% 提升到 ~90%），还通过重叠执行使 KV cache 有效容量从 $C_{KV}$ 扩展为 $\frac{p+g}{p+g/2}C_{KV}$（Equation 7），降低了峰值内存占用。
    2. **Resource-Aware Scheduler** (§6.2)：Normal Inference Mode 下 Prefill Scheduler 和 Decode Scheduler 并行调度——Decode Scheduler 先调度所有 decode sequences → Prefill Scheduler 根据 Pipeline Profiler 的 $n_{real}$ 阈值补充 prefill tokens。KV cache 不足时进入 Preemption Mode 抢占部分 decode sequence、回收 KV cache、重新注入 pipeline。**解决缺陷(2)**：重叠调度最大化 GPU 利用率，preemption 机制保障 KV cache constrained 场景下的鲁棒性。
    3. **VSLPipe 执行引擎** (§6.4)：将每层计算图重组为 GA→C→GB，跨 layer 合并成 execution stage（CPU-only phase → H2D/D2H → GPU-only phase）。$\alpha$/$\beta$ 两组交替执行，CPU attention 与 GPU GEMM 完全重叠。每个 stage 开始时 Contiguous Data Mover 预取下一 stage weights。**解决缺陷(3)+(5)**：Contiguous Data Mover 将 weight transfer 从执行流水线中解耦，独立线程以 100MB packet size 分批传输，避免 head-of-line blocking 和与 PyTorch compute transfer 的竞争。
    4. **Hand-optimized CPU Decode Attention** (§6.6)：AVX512 intrinsics + loop unrolling + data prefetching，单线程 4.7× auto-vectorized baseline。**解决缺陷(4)**：CPU attention 达到系统所需 throughput（满足 KV cache = 2× model size 时的 attention 计算需求），使 CPU 不成为 bottleneck。
  - **编译框架层**：论文未明确说明。使用 C++ PyTorch extension（Contiguous Data Mover）和手写 SIMD kernel。
  - **kernel调度层**：Contiguous Data Mover 作为独立线程调度 CPU→GPU weight transfer packets，100MB packet size 平衡带宽和竞争。CPU attention kernel 在 VSLPipe CPU phase 执行，与 GPU GEMM 并行。
  - **硬件架构层**：同 baseline（A40 GPU + CPU + PCIe），但利用率大幅提升：GPU utilization 从 16.5% → ~90%（70GB KV cache, g=32），平均 throughput 4.6× MoE-Lightning（up to 25.5× on RAG）。

## MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

- baseline方法是什么？
  Baseline 是 **model-based batching**，即在整个 MoE 模型的 forward pass 中统一使用一个全局 batch size。以 Mixtral-8x7B（32 layers, 8 experts/layer, k=2）在 NVIDIA A5000 24GB + 512GB Host Memory 上的离线推理执行路径为例：
  - **算法层（MoE Routing）**：标准 top-k gating。Router 对每层 self-attention 输出计算 logits → Softmax → SelectTopK(k=2)。每个 token 分配给 2 个 expert。在 model-based batching 下，若全局 batch size=16（受限于 attention peak memory），prefill 阶段每个 expert 平均收到 $16 \times 512 \times 2 / 8 = 2048$ tokens，解码阶段每个 expert 平均收到 $16 \times 1 \times 2 / 8 = 4$ tokens。**缺陷(1)**：解码阶段 expert batch size 极小（4 tokens），远低于充分 GPU 利用所需的最小 $2^{10}$ tokens（图 3 Left），GPU FLOPs 利用率仅 0.1%（表 1）。**缺陷(2)**：batch size 受 attention 模块 peak memory 限制，而 expert 模块需要的 batch size 未被独立考虑。
  - **系统框架层**：FlexGen/DeepSpeed-Inference/MoE-Lightning。均使用 model-based batching。以 DeepSpeed-Inference 为例：将整个 MoE layer 视为 dense MLP，一个统一 batch 从 input → attention（QKV proj → self-attn → output proj）→ MoE layer（router → 逐 expert 加载权重 → expert 计算 → weighted sum）→ next layer。CPU 到 GPU 的 expert weight 传输按需触发（on-demand fetch），KV-cache 部分 offload。**缺陷(3)**：每个 forward pass 中同一 expert 可能只被少量 token 激活，但 expert weights 仍需完整传输（反复 HtoD copy），PCIe 带宽浪费严重。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch 和 CUDA kernel。
  - **kernel调度层**：论文未明确说明。FlexGen/MoE-Lightning 支持 CPU attention（使用 PyTorch/CBLAS），但未针对 MoE 解码场景优化 cache 性能。
  - **硬件架构层**：NVIDIA A5000/A6000 GPU（单卡）+ CPU DRAM（host memory），PCIe 4.0 互连。Model-based batching 在解码阶段 GPU 几乎完全 idle，等待少量的 token 完成计算后立即等待下一个 batch。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE-GEN**，核心是 **module-based batching**——将 MoE 模型分解为 attention 和 expert 两个计算密集型模块，分别为其分配不同的微批次大小，累计 token 形成大 batch 才在 GPU 执行。以 Mixtral-8x7B 在 A5000 24GB + 512GB Host Memory 上的执行路径为例（解码阶段，$B=3640, b_a=75, b_e=150, \omega=0.6$）：
  - **算法层（MoE Routing 不变）**：Router 逻辑不变（top-k selection），但 batch 规模因 module-based batching 大幅提升。解码阶段累计 batch $B=3640$ 个 token 进入 MoE layer 时，每个 expert 平均收到 $3640 \times 2 / 8 \approx 910$ tokens（vs baseline 的 4 tokens），GPU 利用率提升至 41%（表 1）。**解决缺陷(1)**：通过多轮 attention 小批次累计，最终在 expert 模块形成大 batch，使 GPU FLOPs 接近饱和。
  - **系统框架层（MoE-GEN Engine）**：
    1. **Module-based batching**：attention 模块以 $b_a=75$ 为微批次循环执行约 $B/b_a \approx 49$ 轮。每轮：Pre-Attention → (CPU: 60% tokens self-attn via AVX kernel，GPU: 40% tokens self-attn + KV-cache HtoD copy) → Post-Attention。所有 49 轮的 output tokens 累计后在 expert 阶段一次性处理。**解决缺陷(2)**：attention batch 由 peak memory 决定（small），expert batch 由累计决定（large），两者解耦。
    2. **DAG-based scheduling + search**：将整个 layer 的执行建模为 DAG（图 6），每个节点为 computation 或 memory copy，边为依赖关系。Scheduler 在 search space（$B, b_a, b_e, \omega, S_{Expert}, S_{Params}$）中枚举候选，通过 DP 计算 critical path 选择最短执行时间的配置。**解决缺陷(3)**：通过最优 $S_{Expert}$ buffer 实现 expert weight 预取与 GPU 计算 overlap，消除 expert weight 反复传输的带宽浪费。大 batch 下 expert 顺序执行，每个 expert weight 只需加载一次处理大量 token。
    3. **Full KV-cache offloading**：KV-cache 全部保留在 host memory，GPU 仅保留当前需要的 KV-cache 窗口，减少 expert weight fetching 流量达 20×（图 4）。DtoH engine 异步更新 KV-cache。
    4. **CPU attention offloading**：60% attention 计算在 CPU 执行（$\omega=0.6$），CPU 直接访问 host memory 中的 KV-cache 无需 HtoD copy，节省的 PCIe 带宽用于 expert weight 预取。**解决缺陷(3) 的带宽瓶颈**：attention 阶段的 KV-cache HtoD 传输与 expert 阶段的 weight 预取竞争 PCIe 带宽，CPU attention 直接消除这部分竞争。
  - **编译框架层**：论文未明确说明。使用标准 C++/CUDA 编译工具链。
  - **kernel调度层（CPU Attention Kernel）**：
    - 基于 AVX intrinsics 的 Grouped Query Attention（BF16 格式），FP32 累加，每次点积累加后按 BF16 舍入规则舍入。设计类似 FlashAttention CPU 版本的 cache 优化策略。**解决缺陷**：PyTorch/CBLAS 的 CPU attention 实现未针对 MoE 解码场景的 GEMV 中等算术强度优化 cache，MoE-GEN 的 AVX kernel 使 CPU 处理 self-attention 速率达到与 PCIe4.0 传输 KV-cache + GPU 计算时间可比，使得 CPU offloading 有 net throughput gain。
  - **硬件架构层**：同 baseline（单 GPU + CPU + Host Memory, PCIe 4.0）。区别在于 module-based batching 下 GPU 利用率显著更高：expert 模块 GPU 计算与下一个 expert weight HtoD 预取完全 overlap（图 3 Right 的 $>2^{11}$ tokens 区域），GPU idle 时间降至接近零。

- baseline方法是什么？
  Baseline 为 Top-K routing with quantization + LRU caching（基于 `dvmazur/mixtral-offloading`），即在资源受限的 GPU 上运行 Mixtral-8x7B 时，部分 expert 参数 offload 到 CPU DRAM，使用量化压缩和 LRU 缓存来缓解 offloading 带来的延迟。以 Mixtral-8x7B（32 layers, 8 experts/layer, k=2）在 H100 + CPU DRAM 上的执行路径为例：
  - **算法层（MoE Routing）**：Gating network 接收 self-attention 输出 H_i，计算 Logits=H_i·W_exp → Softmax → SelectTopK(k=2)。纯粹基于模型 logits 选择 top-2 experts，完全无视 expert 的物理位置（HBM vs CPU）。**缺陷**：(1) 若 Top-2 中任一 expert 在 CPU DRAM 中，必须等待 PCIe 传输完成才能继续——图 2 显示 CPU read time 比 GPU read time 高数个数量级；(2) gating 的选择无记忆性，可能连续选中冷门 off-chip expert，导致频繁的 CPU↔GPU swap；(3) 当 offload 比例增加（更少 VRAM）时，性能劣化加剧，因为更多 expert 不在 HBM 上。
  - **系统框架层**：`dvmazur/mixtral-offloading` 框架。包含 expert 量化（压缩权重减少传输量）、LRU caching（保留最近使用的 expert 在 HBM）、expert offloading manager。每次解码步骤：gating → Top-K → 检查 expert residency → 若缺失则触发 CPU→GPU load → 可能触发 LRU eviction → expert 计算。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。CPU↔GPU 传输通过 PCIe，无自定义 kernel。
  - **硬件架构层**：NVIDIA H100 GPU + CPU DRAM host memory。GPU HBM 带宽远高于 PCIe 带宽（图 2：CPU read 延迟 >> GPU read 延迟）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-ERAS——在 gating 阶段引入 expert residency awareness，通过 thresholding 和 biasing 两种技术修改路由决策，使模型倾向选择已驻留在 HBM 的 expert，从而减少 costly 的 CPU→GPU 传输。以 Mixtral-8x7B 在 H100 + CPU DRAM（3 experts offloaded/layer）上的执行路径为例：
  - **算法层（Residency-Aware Routing）**：
    1. **Thresholding**：Weights=Softmax(Logits) 后，对 HBM 中 expert 加 α bias → SelectTopK。当 off-chip expert 与 on-chip expert 的 logits 接近时（"无绝对赢家"场景），α 使 on-chip expert 胜出。**解决 baseline 缺陷(1)**：避免为微小 logit 优势触发 costly CPU→GPU 传输。α=0.15 时在 3 experts offloaded 场景下减少 10-13% 解码延迟。
    2. **Biasing**：Logits 中 off-chip expert 减去 β(1-freq(E_i)) 惩罚 → Softmax → SelectTopK。freq(E_i) 是从 profiling（500k tokens）收集的归一化激活频率。**解决 baseline 缺陷(2)**：冷门 off-chip expert 惩罚大（避免"加载-立即换出"的双重 swap），热门 off-chip expert 惩罚小（值得加载因为后续 token 会复用）。
    3. 两者均通过超参数（α 或 β）提供 controllable speedup-quality trade-off。**解决 baseline 缺陷(3)**：offload 越多、α 越大，speedup 越显著——在极端 offload 下减少 21.2% 延迟。
  - **系统框架层**：在 `dvmazur/mixtral-offloading` 基础上增加：(1) residency table 维护模块（每层 expert 的 HBM/CPU 状态）；(2) profiling 模块（收集 expert activation frequency 用于 biasing）；(3) residency-aware routing 模块（在 gating→TopK 之间插入 thresholding/biasing 逻辑）。与 quantization、LRU caching、prefetching 正交，可叠加使用。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。方法在路由层面操作，不涉及 kernel 修改。
  - **硬件架构层**：NVIDIA H100 GPU + CPU DRAM。MoE-ERAS 的效果随硬件不对称性（GPU HBM bandwidth >> PCIe bandwidth）加剧而更显著——资源越受限，residency-aware selection 越有价值。

## MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- baseline方法是什么？
  Baseline 为 MoE 推理时不使用任何 prediction 的标准 Expert Parallelism (EP) 方案。以 Mixtral 8×7B（32 layers, 8 experts/layer, Top-K=2）在 4×A100 NVLink 上的 prefill 推理（batch=1, seq_len=512）为例说明全栈执行路径：
  - **算法层（MoE Routing）**：Gating network 接收 self-attention 输出 → Linear projection → Softmax → SelectTopK(k=2)。路由结果固定（inference 时不可修改 token-to-expert 映射），导致 skewed distribution：如 Expert 1 承接 75% tokens（skewness=3.0）。**缺陷**：(1) FFN compute imbalance——GPU 1（hosts Expert 1）成为 compute bottleneck，延迟被最慢 GPU 主导，放大倍数为 skewness；(2) All-to-All communication imbalance——GPU 1 接收最多 token → 通信 bottleneck 也被 skewness 放大：$(N-1)·skewness/N^2$ vs 平衡时的 $(N-1)/N^2$；(3) 无动态调整机制，skewness 随 workload 变化而无法适应。
  - **系统框架层**：标准 EP 推理 pipeline：Attention（TP, 含 Ring All-Reduce）→ Gating → All-to-All Scatter → Expert FFN → All-to-All Gather → 下一层。无 predictor，无 expert duplication 机制，expert 在每个 GPU 上静态驻留。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch/NCCL 通信 kernel（All-to-All, All-Reduce），无自定义 kernel。
  - **硬件架构层**：4×A100 NVLink 3.0 (2TB/s) fully connected。skewness 导致的 compute + communication imbalance 与硬件 topology 无关——即使 NVLink bandwidth 充足，bottleneck GPU 的 compute delay 仍构成硬上限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-GPS——一个系统性能模拟框架，用于量化 MoE 推理中不同 expert prediction 策略的 runtime trade-off，指导选择最优 predictor 设计。核心方法包括两种预测策略（Distribution-Only 和 Token-to-Expert）配合 dynamic expert duplication（Algorithm 1）。
  - **算法层（Expert Prediction Strategies）**：
    1. **Distribution-Only Prediction**：使用 Multinomial Distribution + MLE 对每层 expert 激活概率建模（$\hat{p}_i^l = n_i^l/N$），仅预测 coarse-grained token 分布比例。配合 Algorithm 1 将热门 expert 复制到 underloaded GPU → FFN compute 均衡化。**解决 baseline 缺陷(1)**：通过 expert duplication 打破单 GPU 上的 expert compute bottleneck，skewness 越高收益越大。**代价**：不减少通信（仍做 All-to-All Scatter）。**优势**：zero predictor overhead（offline MLE 估计）。
    2. **Token-to-Expert Prediction**：将 expert selection 建模为分类问题（Probability / Conditional Probability / FFN / LSTM 四类 predictor），预测每个 token 的目标 expert → Direct Routing 跳过 Scatter 阶段。**解决 baseline 缺陷(1)+(2) 同时**：expert duplication 均衡 compute + 跳过 All-to-All Scatter 节省通信。**代价**：predictor inference overhead，accuracy 越高通常 overhead 越大（U-shape trade-off）。
    3. 两种策略的选择由 MoE-GPS simulator 在给定 hardware + model + workload 下自动决策。**解决 baseline 缺陷(3)**：系统可根据实时条件切换策略。
  - **系统框架层（MoE-GPS Simulation Framework）**：以 LLMCompass（ISCA 2024 block-level simulator, silicon-validated）为基础，增强：(1) MoE + EP 模块（custom EP communication + FFN workload 建模）；(2) Mixtral 架构支持（GQA, SwiGLU, Sliding Window）；(3) Prediction strategy 建模（tunable accuracy + overhead）。以 Mixtral 8×7B, 4×A100 NVLink, skewness=1.4, batch=1, seq_len=512 为例：
    - **Distribution-Only 路径**：Offline MLE → 预测 expert 分布 → Algorithm 1 决定 duplication → Expert copy over NVLink (~0.1ms, hidden by Attention) → Attention (TP, ~12ms) → Scatter → Balanced FFN Compute → Gather → 23% speedup vs Token-to-Expert best config。
    - **Token-to-Expert 路径**：Predictor inference (overhead) → Algorithm 1 duplication → Attention → Direct Route (skip Scatter) → Balanced FFN Compute → Gather。Overhead trade-off 导致 U-shape 性能曲线（Figure 6b, 6d）。
    - **决策准则**（Figure 7）：
      - Distribution-Only 更优：low skewness OR high-bandwidth interconnect (NVLink)——因为通信不是瓶颈，predictor overhead 不值得。
      - Token-to-Expert 更优：high skewness（预测更容易，accuracy/overhead 比更优）AND low-bandwidth interconnect（PCIe）——因为节省的通信远大于 predictor overhead。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。使用 LLMCompass 模拟的 GEMM + communication + element-wise 操作，不涉及自定义 kernel。
  - **硬件架构层**：4×A100 NVLink 3.0 (2TB/s) 和 PCIe 4.0 (32GB/s) 两种配置。核心 insight：interconnect bandwidth 直接决定两种策略的盈亏平衡点——高带宽下 Distribution-Only 几乎始终更优，低带宽+高 skewness 下 Token-to-Expert 有机会反超（Figure 7 的 32GB/s PCIe 场景）。论文还在 Appendix C 验证了 LLaMA-MoE 和 Switch Transformer 上的一致性趋势。

## MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

- baseline方法是什么？
  Baseline 为 Full-Parameter MoE Training，即传统的完整参数 MoE 训练方式。以 Qwen1.5-MoE-2.7B（E=4 experts）在 C4 数据集上训练为例说明全栈执行路径：
  - **算法层（MoE Training）**：完整的 MoE 模型（含共享 backbone + E 个 expert + gating network）在整个训练过程中全部加载到 GPU 内存中。每个 forward pass 经过 embedding → attention → LayerNorm → gating 选择 Top-K experts → 对应 expert MLP 计算 → 输出。Backpropagation 必须遍历所有 expert 路径（即使 inference 时只激活 Top-K），所有 expert 参数同时更新。**缺陷**：(1) 内存和计算开销随 expert 数量线性增长，导致无法在低内存 GPU（如 RTX 4090 ≤ 24GB）上训练；(2) 整个训练过程必须在昂贵的高带宽 GPU（A100, $2.28/GPU·h）上完成，训练成本极高——Qwen on C4 需 $22.50，Llama on OpenWebText 需 $32.13；(3) 大规模多 GPU 训练时，梯度同步和 activation 传输等通信开销导致 per-GPU MFU 随 GPU 数量增加而下降。
  - **系统框架层**：标准 PyTorch 分布式训练，使用 AdamW optimizer (LR=3e-4)、Cosine LR scheduler、warmup_ratio=0.03、weight_decay=0.01、batch_size=16、bf16、seq_len=1024。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。使用标准 PyTorch kernel，无自定义 kernel 优化。
  - **硬件架构层**：NVIDIA A100 80GB × 1 进行全参数训练，GPU 租赁 $2.28/GPU·h。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-DisCo——基于 BCD + SimulParallel SGD 的分阶段 MoE 训练框架，将 MoE 训练分解为"低成本并行子模型训练 + 短时高成本全局微调"。以 Qwen1.5-MoE-2.7B（E=4）在 C4 上的执行为例说明全栈执行路径：
  - **算法层（分阶段 MoE Training）**：
    1. **Model Decoupling**：完整 MoE 参数 Θ = (θ_shared, θ_1, ..., θ_E) 分解为 E 个子模型 Θ_k = (θ_shared^(k), θ_k)。每个子模型为 dense 模型，参数量大幅减小（仅 1/E 的 expert 参数），可直接放入 RTX 4090 的 24GB 显存。**解决 baseline 缺陷(1)**：将 O(E) 的内存需求降为 O(1)。
    2. **Data Decoupling**：用预训练 embedding 提取句子向量 → K-Means 聚类 → E 个语义区分的子数据集，分配给不同 expert。最大化子数据集间的分布差异以促进 expert 专业化。**解决 baseline 缺陷(1)延伸**：通过数据-模型联合解耦确保每个 expert 学到互补表征而非冗余知识。
    3. **Independent Parallel Training (S-phase)**：E=4 个子模型在 4 块 RTX 4090 上完全并行训练（最慢子模型 4200 steps/2.09h），零通信开销。S-phase 成本仅 $2.93。**解决 baseline 缺陷(3)**：完全消除跨设备通信开销，MFU 不随训练规模下降。**解决 baseline 缺陷(2)**：将大部分训练从 A100（$2.28/h）移至 RTX 4090（$0.35/h），成本降低约 6.5 倍。
    4. **Reintegration + Global Fine-Tune (F-phase)**：Expert 参数直接拼接，共享参数按 WP-SGD 加权平均。组装完整 MoE 后在单块 A100 上短时微调（1730 steps/0.76h），成本 $1.55。总成本 $6.87 vs baseline $22.50，节省 69.5%。
  - **系统框架层**：S-phase 使用标准 PyTorch + RTX 4090 × 4（完全独立，无分布式框架），F-phase 在单块 A100 上使用标准 PyTorch + AdamW。超参数：S-phase 用 AdamW (LR=1e-4, constant scheduler)，F-phase 用 AdamW (LR=3e-4, Cosine scheduler)。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。使用标准 PyTorch kernel。
  - **硬件架构层**：S-phase 使用 4 块 NVIDIA RTX 4090（$0.35/GPU·h），F-phase 使用 1 块 NVIDIA A100 80GB（$2.28/GPU·h）。训练时间从 baseline 9.87h 降至 3.82h。

## MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

- baseline方法是什么？
  Baseline 为 Soft MoE [6]（from scratch training），即从随机初始化训练 MoE 模型，不使用任何预训练 dense checkpoint。以 ViT-T 架构上的 Soft MoE 为例说明全栈执行路径：
  - **算法层（MoE Routing）**：输入 token X ∈ R^{m×d} → 通过 learnable parameters Φ ∈ R^{d×(e·s)} 将 m 个 token 映射到 e×s 个 slot：X̃ = softmax(XΦ)^T X → e 个 expert（MLP）各自处理 s 个 slot → 输出 Ỹ → 通过 softmax(XΦ) 重组为 m×d token 输出。**缺陷**：(1) MoE 模型从随机初始化训练，缺乏预训练知识的加速，在小数据集上收敛极慢（如 STL-10 仅 67.7% accuracy）；(2) Soft MoE routing 使用 dense Φ 矩阵对所有 token 做 soft assignment，未专门设计以适配从 dense checkpoint 继承的权重分布，导致优化困难和 expert 过度特化（over-specialization）；(3) 所有 expert 大小一致，对重要性不同的 token 分配等量计算资源，计算冗余。
  - **系统框架层**：使用 MMPretrain（OpenMMLab 预训练工具箱）实现，标准 PyTorch 训练循环。训练使用 AdamW optimizer、cosine decay LR schedule、RandAugment/Mixup/CutMix 等数据增强。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：传统 MoE 用 for-loop 逐个 expert 处理 token。论文指出 vanilla MoE 设计不提供运行时加速，需要额外并行策略。论文在 Appendix C 提供了高效的并行 expert 前向实现（合并所有 expert 权重为单个大矩阵，单次 einsum 替代多个逐 expert 操作）。
  - **硬件架构层**：NVIDIA RTX 4090 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 MoE Jetpack 通过两个核心技术逐层解决 Baseline 缺陷：
  1. **Checkpoint Recycling（解决从零训练的收敛慢和资源消耗）**：不随机初始化 MoE expert，而是从预训练 dense checkpoint（ImageNet-21K 预训练的 ViT-S/ConvNeXt-T）中通过 Importance-Based Weight Sampling 提取权重来初始化 MoE expert。具体地：跑一批图像通过 predecessor dense 模型获取每个 channel 和 hidden neuron 的 activation 值，channel 层按跨层平均 activation 排序选 top-d'，hidden neuron 按 activation 概率分布采样分配给不同 expert 以保证 diversity。这使 MoE 模型继承了 dense 模型的预训练知识，大幅加速收敛（ImageNet 上 2× 加速，CIFAR-100 上 8× 加速）并有显著精度提升。相比 Sparse Upcycling [16]（仅复制 MLP），checkpoint recycling 能利用更大/不同的 dense checkpoint 构造不同大小的 expert，灵活性更高。
  2. **SpheroMoE Layer（解决优化困难和 expert 过度特化）**：
     - SpheroMoE Routing：用 cross-attention 替代 Soft MoE 的线性 Φ 分配。随机初始化的 Q 经 L2 normalize 投影到超球面，与从 input token（经继承自 dense checkpoint 的 LayerNorm 处理后）投影得到的 K 计算 cosine similarity，解决了随机初始化 Q 导致的数值不稳定和与 dense checkpoint 分布不一致的问题。
     - Expert Regularization：learnable softmax temperature T（早期大→分散注意力，逐步减小→专精）+ Gaussian noise 到相似度 logits + stochastic expert dropout（概率 p），三者共同防止 expert 对特定 token 的过度聚焦和对特定 expert 的过度依赖。
     - Adaptive Dual-path MoE：利用 checkpoint recycling 赋予的 dense 先验知识区分重要/非重要 token。Core experts（数量少，占总数 1/3，每个完整 hidden dim 4d'）处理高重要性 token；Universal experts（数量多，每个 hidden dim ≈ d'，约 1/4 参数）处理低重要性 token。在保持 FLOPs 不变的前提下提升性能。

  MoE Jetpack 全栈执行路径（与 baseline 同框架对应）：
  - **算法层（Dense→MoE Fine-tuning）**：ImageNet-21K 预训练 ViT-S dense checkpoint → Checkpoint Recycling（Importance-Based Weight Sampling）→ 初始化 V-JetMoE-T 的 expert 权重 → 前 N/2 层保留 dense ViT 结构（继承全部 dense 权重）→ 后 N/2 层为 SpheroMoE 层：input token X → 继承的 LayerNorm → Q L2-norm 超球面投影 → cross-attention 计算 dispatch/combine logits S → Adaptive Dual-path 分离 core/universal token → 并行 expert 前向（合并权重矩阵单次 einsum）→ softmax 重组 token。
  - **系统框架层**：PyTorch 2.1.0 + MMCV 2.1.0 + MMPretrain。训练配置：AdamW（lr=4e-3, weight_decay=0.05, β=(0.9,0.999)）、batch_size=4096（ImageNet）/512（其他）、300 epochs、cosine decay、50 warmup epochs。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文提供了并行 expert 前向实现（Appendix C, Algorithm 1），将所有 expert 的 weight_1 合并为 e×d2×d1 大矩阵，单次 einsum("b e s d1, e d2 d1 -> b e s d2") 平行处理所有 b×e×s 个 slot，替代传统 for-loop。FLOPs 与原始 dense 模型相当（V-JetMoE-T: 1.1G vs ViT-T: 1.1G）。
  - **硬件架构层**：NVIDIA RTX 4090。训练 V-JetMoE-T 在 ImageNet-1K 上需要 120 GPU hours，与原始 dense ViT-T 训练时间几乎相同（论文称 "nearly equivalent training times"）。

## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- baseline方法是什么？
  Baseline 为 LLM 领域的 expert skipping 方法（NAEE [42]、MC-MoE [22]、DiEP [6]），它们仅依赖 **intra-layer routing probabilities** 决定跳过哪些 expert，且为 **unimodal LLM** 设计。以 NAEE 为例说明全栈执行路径：
  - **算法层（Expert Skipping）**：输入 token 经 router 得 routing probs $\pi_1,...,\pi_M$ 和 top-k indices $\mathcal{S}^{(l)}$。NAEE 判断：若累积尾部概率 $\sum_{u=i}^k \pi_{\text{top-}u}^{(l)} < \beta^{(l)} \cdot \sum_{v=1}^k \pi_{\text{top-}v}^{(l)}$，则跳过 top-i 到 top-k 的 expert。**缺陷**：(1) 忽略了 layer-level 的全局贡献差异——浅层 expert 的错误会经后续层放大，应保守跳过，深层可激进跳过，但 NAEE 对所有层同等对待；(2) 忽略了 modality gap——vision token 的 expert 冗余度远高于 text token（FFN 对 vision token 的更新幅度小），应更激进跳过 vision expert，但 NAEE 对所有 token 一视同仁。直接应用这些方法到 MoE MLLM 在 83% skipping ratio 下导致 >10% 平均精度下降。
  - **系统框架层**：使用 HuggingFace Transformers 加载模型，标准 PyTorch 推理。baseline（DiEP 等）通过离线校准选择超参数 $\beta^{(l)}$，推理时无额外开销。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch MoE kernel 实现。论文提到 DiEP 等 baseline 在相同 skipping ratio 下的 inference speedup 与 MoDES 类似（差别 <1%），因其推理开销可忽略。
  - **硬件架构层**：单张 H200 GPU 推理。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 MoDES 方法通过三个组件逐层解决 Baseline 缺陷：
  1. **GMLG（解决全局贡献忽略）**：通过离线校准计算每层的全局重要性因子 $\alpha^{(l)}$（跳过该层所有 expert 后的 KL 散度），推理时 $s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$。浅层 $\alpha^{(l)}$ 大 → $s_i^{(l)}$ 大 → 更难被跳过 → 保护关键浅层 expert。深层 $\alpha^{(l)}$ 小 → $s_i^{(l)}$ 小 → 更容易被跳过 → 激进去除深层冗余。
  2. **DMT（解决 modality gap 忽略）**：设置独立阈值 $\tau_t$（text）和 $\tau_v$（vision）。基于发现 vision token 与 FFN 权重的夹角更接近 $90^\circ$（更新幅度小），设置 $\tau_v > \tau_t$ 使 vision expert 被更激进跳过。如图 8 所示，实际 skipping ratio 在 vision token 上远高于 text token。
  3. **Frontier Search（解决阈值搜索效率）**：利用 $f$（KL 散度）和 $g$（skipping ratio）对阈值单调递增的性质，设计 $\mathcal{O}(ND)$ 算法替代 $\mathcal{O}(ND^2)$ exhaustive search，搜索时间从 >2 天降至 <2 小时（30B 模型）。

  MoDES 全栈执行路径（与 baseline 同框架对应）：
  - **算法层（Expert Skipping）**：输入 token x 进入第 l 个 MoE FFN → Router 输出 routing probs $\pi_i^{(l)}$ 和 top-k set $\mathcal{S}^{(l)}$ → **GMLG** 计算 $s_i^{(l)} = \widetilde{\alpha}^{(l)} \cdot \pi_i^{(l)}$（$\widetilde{\alpha}^{(l)}$ 离线预计算，推理无开销）→ **DMT** 根据 token modality 选择 $\tau_t$ 或 $\tau_v$，跳过 $s_i^{(l)} < \tau$ 的 expert → 仅保留的 expert 参与加权聚合 $\mathbf{y}^{(l+1)} = \sum_{i \in \text{kept}} \pi_i^{(l)} \cdot \text{Expert}_i^{(l)}(\mathbf{x}^{(l)})$。
  - **系统框架层**：基于 HuggingFace Transformers。离线阶段：用 GQA 1024 样本 calibration → 计算 $\alpha^{(l)}$ → Frontier Search 得 $(\tau_t^*, \tau_v^*)$。在线阶段：加载预计算的 $\widetilde{\alpha}^{(l)}$ 和阈值 pair，推理时动态跳过。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：编写自定义 CUDA kernel：(a) Router kernel 内嵌 DMT branch-free comparison + sentinel ID assignment（无额外 kernel launch）；(b) Dispatch/Gather 过滤 sentinel entries；(c) Group GEMM 统一 kernel launch 并发执行保留 expert，离线 profiling 确定最优 tile size。实现 prefilling 2.16× 加速，decoding 1.26× 加速（Qwen3-VL-MoE-30B-A3B-Instruct, 88% skip）。
  - **硬件架构层**：单张 H200 GPU。MoDES 的 overhead 仅来自 top-k 列表上的 element-wise 操作（batch-free comparison），对 warp divergence 影响极小。

- baseline方法是什么？
  Baseline 为 LoRA-MoE（如 MoLORA），m 个独立的 LoRA expert（每个 expert = A^i B^{iT}），router $\mathcal{R}(\mathbf{x})$ 对 m 个 expert 加权。以 MoLORA 16×4（4 experts, rank=4 per expert）为例说明全栈执行路径：
  - **算法层（PEFT 微调）**：预训练 LLM（Gemma 2B）冻结权重 W0，每层注入 LoRA-MoE 模块。输入 x → 每层的 m 个 expert 各自执行 $\mathbf{x}\mathbf{A}^i\mathbf{B}^{iT}$（各 expert 独立拥有一对 A^i, B^i 矩阵）→ router softmax ($\mathbf{x}\mathbf{W_R}$) 输出 m 维权重 → 加权求和 → 与 frozen output 相加。**缺陷**：每个 expert 拥有独立的 down-projection 矩阵 A^i，造成参数冗余——PCA 分析表明不同任务的 down-projection 向量高度聚类（task-agnostic），而 up-projection 向量分散（task-specific）。
  - **系统框架层**：标准 HuggingFace Transformers 或类似 LLM 训练框架加载预训练 Gemma 2B 权重，注入 PEFT adapter（仅训练 adapter 部分）。论文未明确说明具体框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。Gemma 2B 规模较小（2B），通常使用标准 PyTorch kernel 即可。
  - **硬件架构层**：论文未明确说明硬件平台。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 MoDE 方法通过两项创新解决 LoRA-MoE 的冗余问题：
  1. **共享 down-projection 矩阵 A**：观察发现不同任务/experts 的 down-projection 向量在 PCA 空间中聚类（Figure 3），说明 down-projection 是 task-agnostic 的，无需每个 expert 独立学习。MoDE 让所有 expert 共享一个 A，将参数从 $m \cdot P \cdot r$ 降至 $P \cdot r$。仅此改进（LoRA-MoE-SD / MoLORA-SD）就用 36% 的参数实现了 0.88% ROUGE-L 提升。
  2. **原子 rank-one adapter + fine-grained routing**：LoRA-MoE-SD 虽省参数，但 router 只能提供 m 种选择（所有 r 个 rank 维度绑定在一起路由）。MoDE 将 LoRA 更新分解为 dyadic sum $\Delta\mathbf{W} = \sum_{j=1}^r (\mathbf{a}_j \otimes \mathbf{b}_j)$，对每个 rank j 独立设置 m 个 expert $\{\mathbf{b}_j^1, ..., \mathbf{b}_j^m\}$ 并通过 router $\mathcal{R}_j$ 独立选择，共 $m \times r$ 个 rank-one expert，可表达 $m^r$ 种组合（vs baseline 的 m 种）。这使 MoDE 能动态组合出针对不同输入的专用 up-projection 矩阵。

  MoDE 全栈执行路径（与 baseline 同框架对应）：
  - **算法层（PEFT 微调）**：输入 x 进入 Transformer layer → frozen 前向 $\mathbf{xW_0}$ → 共享 A down-project 得 $\mathbf{h} = \mathbf{xA} \in \mathbb{R}^{1\times r}$ → 对每个 rank j ∈ {1..r}，独立 router softmax(x·W_{R;j}) 产生 m 维权重 → 第 j 个 rank 的 dyadic 贡献为 $\mathbf{h}_j \sum_{i=1}^m \mathcal{R}_j^i(\mathbf{x}) \mathbf{b}_j^{iT}$（标量 h_j 乘以加权 up-projection 向量）→ r 个 rank 求和得到总 adapter 输出 → 与 frozen 输出相加。与 baseline 的关键差异：baseline 中整个 r 维的 up-projection B^i 被 router 绑定为一个整体选择；MoDE 允许"B 的第 1 列用 expert 1，第 2 列用 expert 3，第 3 列用 expert 2，第 4 列用 expert 1"这样的细粒度组合。
  - **系统框架层**：与 baseline 相同，标准 LLM 训练框架。论文未明确说明具体框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。MoDE 的 rank-one 操作可视为矩阵-向量乘，标准 PyTorch 即可高效执行。
  - **硬件架构层**：论文未明确说明硬件平台。

  **实验结果证明**：MoDE 16×4 达到 60.00 ROUGE-L（vs LoRA 64: 56.11, MoLORA 16×4: 57.77, MoLORA-SD 16×4: 58.28）。Task-level win rate 分析：MoDE vs LoRA 78%, vs MoLORA 73%, vs MoLORA-SD 68%。Iso-parametric 下最佳配置为 MoDE 12×16×8（ROUGE-L 60.94）。

## MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

- baseline方法是什么？
  Baseline 为 vLLM（TP+PP 和 DP+EP 两种配置）和 Tutel（TP+EP），均为现有 MoE 模型推理服务系统。以 vLLM DP+EP（DeepSeek-R1, 4-node Ascend 910B, TP=8+DP=4, EP=32）为例说明全栈执行路径：
  - **算法层（MoE 推理）**：每层 Decoder 执行 Attention（QKV projection + attention score + output projection）→ Gating（top-K routing）→ Expert FFN（MLP）。Attention block 使用 TP（intra-node AR 同步），MoE block 使用 EP（inter-node A2A dispatch/combine）。AR 通信量 O(bs·h/d)，每层需 RS+AG 两阶段各 1 轮（Broadcast algorithm）。A2A 通信量 O(bs/d·hk)，需 d-1 轮（Pairwise algorithm）。
  - **系统框架层**：vLLM 使用 PagedAttention + continuous batching 管理 KV cache 和请求调度。TP 限于 intra-node（利用 NVLink/HCCS 高带宽），EP 跨 node（利用 InfiniBand/RoCE）。Tutel 额外支持 hybrid TP+EP（TP=4+DP=4, TP=4+EP=4），但仅限 intra-node TP + inter-node EP 的固定组合。两者的并行策略由用户手动指定（基于经验直觉），无自动策略选择机制。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA kernel）。
  - **kernel 调度层**：NCCL/HCCL collective communication library 处理 AR 和 A2A 原语。通信算子同步执行——AR（RS+AG）和 A2A（Dispatch+Combine）各自串行，互不重叠。inter-node A2A 的低带宽（RoCE 200 Gbps vs intra-node HCCS 480 Gbps）成为瓶颈。无 intra-node 和 inter-node 通信间的重叠设计。
  - **硬件架构层**：NVIDIA H20（96 GB, NVLink 4.0 900 GB/s intra-node, InfiniBand 400 Gbps inter-node）/ Ascend 910B（64 GB, HCCS 480 Gbps intra-node, RoCE 200 Gbps inter-node）。Intra-node 带宽显著高于 inter-node（2×-∞），但现有策略将所有通信统一处理，未利用带宽层次差异。
  - Baseline 核心缺陷：
    1. **缺乏系统性理论分析**：并行策略选择基于经验直觉和实践，未考虑模型超参数、网络拓扑和硬件资源配置间的复杂交互。TP degree、DP degree、EP degree 的组合空间巨大，人工枚举不可行。
    2. **无法有效利用通信带宽层次**：AR-based TP 在 inter-node 场景效率低（Fig. 3：d>8 时通信开销激增），A2A-based EP 存在负载不均衡（尤其高并行度时）。现有策略将所有通信统一处理，错失了利用 intra-node（高带宽）和 inter-node（低带宽）差异优化性能的机会。
    3. **Intra-node 和 inter-node 通信串行执行**：NCCL/HCCL 中 AR 和 A2A 作为独立 collective 算子串行执行，intra-node 通信期间 inter-node 链路 idle，inter-node 通信期间 intra-node 链路 idle，网络带宽利用率低。
    4. **并行度固定且无法自适应**：用户需手动指定 TP/DP/EP degree，无法根据 cluster 配置和 workload 特征自动调整。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MixServe 通过 **Automatic Analyzer（理论通信建模 + 自动策略选择）+ Hybrid TP-EP Partitioner（解耦重组 AR/A2A）+ Fused AR-A2A Communication Algorithm（intra/inter-node 通信异步重叠）** 三层设计解决上述缺陷。全栈执行路径（以 DeepSeek-R1 在 4-node Ascend 910B, TP=8+DP=4, TP=8+EP=4 为例）：
  - **算法层（理论通信建模）**：
    1. 形式化定义并行策略（§III-B1 context-free grammar）：为每层 Decoder 的 Attention block 和 MoE block 独立定义 intra-node 和 inter-node 并行策略。Attention 支持 TP+DP，MoE 支持 TP+EP。PP 跨 Decoder layer 正交叠加。
    2. 细粒度 AR 和 A2A 通信开销建模（式 1-3）：AR(size,degree) = RS(size/degree,degree) + AG(size/degree,degree) ∝ size/degree；A2A(size,degree) ∝ (size/degree) × (degree-1)。量化 DP vs EP 的三种 trade-off case（d_DP = / > / < d_EP，Fig. 6）。
    3. Token 生成延迟模型（式 4-7）：τ(computation) ∝ Ψ/(d_TP·d_EP) · b/d_DP · sh；λ(communication) = 2×AR + 2×A2A（含 d_DP < d_EP 时的 hidden states 冗余修正）；Δtsvc = l[τ+λ] + (d_PP-1)·P2P；M/M/1 排队模型预测 queuing delay Wq。
    4. 内存约束（式 8）：Ψ_Attn/d_TP + Ψ_MoE/(d_EP·d_TP) + KV cache < M。
    5. 理论性能指标（式 9-11）：TTFT = Wq + Δtsvc|s=Lin；ITL = Δtsvc|s=1；Throughput = (Lin+Lout)/(Wq + TTFT + Lout·ITL)。
  - **系统框架层（Hybrid TP-EP Partitioner + 自动策略选择）**：
    1. **Offline Stage**：Automatic Analyzer 以模型超参数 + 网络硬件配置为输入 → 用 profiling 数据（不同 batch/seq len 的 compute/comm latency）校准理论模型 → 枚举所有满足 n_proc × n_node = d_TP × d_EP 的 (d_TP, d_EP, d_DP) 组合 → 在内存约束（式 8）下选最小化 TTFT/ITL 或最大化 Throughput 的策略 → 输出最优 (d_TP, d_EP, d_DP)。
    2. **Online Stage**：Partitioner 按最优策略切分 Attention weights（intra-node TP + inter-node DP）和 MoE weights（intra-node TP + inter-node EP）→ Weight Loader 加载对应 shards → 初始化 mixed parallel communication groups → 向 forward method 注入 RS/AG/A2A 通信算子。
    3. **DP-EP Trade-off 自动管理**：考虑延迟/吞吐需求 + 内存约束，自动选择 d_DP = / > / < d_EP 的最优配置。d_DP = d_EP 时最平衡（rank 一一对应），d_DP > d_EP 时 expert weights 冗余换高吞吐，d_DP < d_EP 时 hidden states 冗余但 effective dropping 降低通信开销（Fig. 6）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层（Fused AR-A2A Communication Algorithm）**：
    1. **解耦重组 AR/A2A**：将 AR 分解为 RS+AG，A2A 分解为 Dispatch+Combine → 重组为 RS→A2A→AG 三段式流程。hybrid TP-EP 使 TP group 和 EP group 分别映射到 intra-node 和 inter-node（d_TP = n_proc, d_EP = n_node），通信量从纯 EP 的 AR(bsh,n_proc) + 2×A2A(bshk,n_node) 降至 AR(bsh,n_proc) + AG(bshk/n_proc,n_proc) + 2×A2A(bshk/n_proc,n_node)（式 12-13）。
    2. **Fused RS-Combine（Alg 1）**：intra-node RS 与 inter-node A2A pairwise 异步重叠——每 node 内 TP rank 持有 hidden states 分片 → 发送到下一 node 同 TP rank（inter-node isend/irecv）→ 同时在本 node 内做 RS → 下一轮用接收的 hidden states 继续 → n_node rounds 后 intra-node AG 汇总。时间 O(n_node)，空间 O(bsh·n_proc)（临时存储）。
    3. **Fused AG-Dispatch（Alg 2）**：intra-node AG 与 inter-node Dispatch 异步重叠——local TP rank 做 expert routing → 发送到下一 node 同 TP rank → 同时在本 node 内做 AG → 下一轮继续 → n_node-1 rounds 后无需末轮通信（local shards 在 TP/EP group 内）。时间 O(n_node)，空间 O(1)。
  - **硬件架构层**：同 baseline。无硬件修改。MixServe 利用 intra-node（NVLink 900 GB/s / HCCS 480 Gbps）和 inter-node（InfiniBand 400 Gbps / RoCE 200 Gbps）之间的带宽层次差异，将 TP 通信限于 intra-node 高带宽域，EP 通信限于 inter-node，通过异步重叠隐藏低带宽的 inter-node 通信延迟。
  - 对比 baseline 的改进映射：
    - **经验策略选择 → Automatic Analyzer 理论建模 + 自动搜索**：vLLM/Tutel 需手动指定 TP/DP/EP degree → MixServe 通过 profiling 校准的理论模型（含 compute + communication + queuing latency）自动枚举并选择最优 (d_TP, d_EP, d_DP)。消融实验（Fig. 11）验证了不同硬件平台下最优 DP-EP 配置不同（Ascend 910B 上 d_DP = d_EP 最优，H20 上 d_DP < d_EP 最优），证明了自动选择的价值。
    - **通信统一处理、无法利用带宽层次 → Hybrid TP-EP 解耦重组**：纯 EP 策略中 inter-node A2A 通信量 O(bshk/n_node) 且需 n_node-1 轮 → hybrid TP-EP 将每轮通信量降至 O(bshk/(n_proc·n_node))，降低了通信规模和通信量（式 12→13）。同时 TP group（intra-node 高带宽）与 EP group（inter-node 低带宽）精确对齐硬件层次。
    - **串行通信、带宽利用率低 → Fused AR-A2A 异步重叠**：baseline 中 RS→AG→Dispatch→Combine 串行执行，总延迟 = sum(各算子延迟) → MixServe 将 intra-node RS/AG 与 inter-node A2A 重叠，总延迟 ≈ max(RS+A2A, AG+Dispatch) + O(n_node)。消融实验（Fig. 12）显示异步通信显著降低 TTFT 和 ITL 并提升吞吐量，加速效果约等于 inter-node 通信开销。
    - **实验结果**：DeepSeek-R1 上 TTFT 1.08×~3.80× 加速、ITL 1.03×~1.66× 加速、Throughput 5.2%~50.3% 提升。Ascend 910B 上 DeepSeek-R1 TTFT 2.67× vs vLLM TP+PP、1.70× vs vLLM DP+EP；H20 上 DeepSeek-R1 Throughput +50.3% vs vLLM TP+PP。

## MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

- baseline方法是什么？
  Baseline 为当前 GPU 集群中广泛使用的**静态电气互连架构**——scale-up 域使用 NVSwitch 全交叉（fully-connected crossbar），scale-out 域使用 Clos-style Fat-tree（或 Rail-optimized）拓扑。以典型 1024 GPU cluster（128 servers × 8 GPU）训练 Mixtral 8×22B（EP=8, TP=8, PP=8）为例的全栈执行路径：
  - **算法层（MoE 训练）**：每层 MoE block 包含 attention → gate unit → parallel expert FFN。Gate unit 做 per-token top-K routing 选择激活的 experts。EP 将不同 expert 分配至不同 GPU，每 training iteration 执行 4 次 all-to-all 通信（FP 2 次 dispatch+collect，BP 2 次）。
  - **系统框架层**：Megatron-LM 3D 并行（DP + TP + EP + PP）。DP 做全局 all-reduce 同步梯度，TP 做 intra-server 高带宽通信（NVSwitch），EP 做跨 server all-to-all 通信。
  - **编译框架层**：论文未明确说明（标准 PyTorch + NCCL）。
  - **kernel 调度层**：NCCL collective communication library 处理 all-to-all、all-reduce 等原语。通信拓扑固定，不支持训练过程中动态重配置。
  - **硬件架构层**：intra-server NVSwitch（900 GB/s 或 NVLink），inter-server Fat-tree/Rail-optimized EPS fabric（100G/200G/400G/800G Ethernet 或 InfiniBand）。全网使用 uniform full bisection bandwidth，拓扑在整个训练过程中保持静态不变。
  - **核心缺陷**：MoE 的 EP 通信具有**时间非确定性**（token-specific expert activation 导致通信矩阵在 iterations 间变化）和**空间非均匀性**（sparse all-to-all，仅部分 GPU 对间有大量通信），且存在**强局部性**（只有同一 MoE block 内的 expert 层需要 all-to-all）。但现有 Fat-tree/Rail-optimized 等静态拓扑使用过配的 full bisection bandwidth 来容纳这些变化，导致宝贵的网络带宽大部分时间处于 under-utilized 状态。OCS-based 方案（如 TopoOpt、Google Lightwave Fabrics）仅支持训练前的一次性重配置（one-shot），无法在训练过程中随 traffic pattern 变化动态调整拓扑，因此在 MoE 训练场景下性能显著劣于 Fat-tree。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MixNet 通过**区域可重构高带宽 OCS 域 + 混合光电 fabric + 训练中拓扑重配置**三层次设计解决上述缺陷。全栈执行路径（以 Mixtral 8×22B，1024 GPU cluster，400 Gbps links 为例）：
  - **算法层（MoE 训练不变）**：MixNet 不改变 MoE 的并行策略和训练算法（EP/TP/PP/DP 照旧），仅优化底层数据传输路径。MoE 训练 accuracy 不受影响。
  - **系统框架层 — Custom Collective Communication Runtime**：
    1. **Traffic Demand Characterization（§5.1）**：利用 MoE block 内 4 次 all-to-all 的对称性（2 次 FP + 2 次 BP 的 traffic matrix 相同或转置），配合 MixNet-Copilot 预测算法（SLSQP 估计条件概率转移矩阵），提前预测 traffic pattern 以支持主动重配置。预测准确度（Top-K accuracy）显著高于 random/uniform 方案。
    2. **Topology Reconfiguration（§5.2）**：Greedy Algorithm 1 迭代识别 bottleneck server pairs（完成时间最长），优先为这些 pairs 分配直接 OCS 电路。在 OCS NIC 端口用尽后停止。重配置是去中心化的——各 region 独立运行 topology controller，无需全局控制面。
    3. **Topology-Aware EP Routing（§5.3）**：5 步流程——(1) topology lookup 确定 delegation GPU → (2) intra-host gather via NVSwitch → (3) inter-host all-to-all via OCS 直连（优先）+ EPS fallback → (4) intra-host all-to-all via NVSwitch → (5) intra-host scatter。步骤 (3) 和 (4) overlap 执行。
    4. **DP Hierarchical All-Reduce**：intra-host NVSwitch reduction → inter-host EPS ring all-reduce → intra-host NVSwitch broadcast。多 EPS NIC 时 multi-ring 并行。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层 — RDMA-based Data Transfer**：
    - 自定义 collective communication runtime（C++ ~6K LoC）基于 FuseLink raw ibverbs library 实现 RDMA 高速传输。
    - 通信原语暴露为 Python 接口（mixnet.all_to_all, mixnet.all_reduce），集成入 Megatron-LM。
    - EPS 通信复用 NCCL 高性能原语，OCS 通信走 RDMA over RoCEv2。
  - **硬件架构层 — Regionally Reconfigurable OCS + EPS Hybrid**：
    1. **架构设计思想（§4）**：利用 MoE 通信的强局部性（同一 MoE block 内的 expert 才需要 all-to-all），将 OCS 划分为多个隔离的 region（而非 global OCS），各 region 独立处理局部 traffic。突破 OCS 技术中 reconfiguration speed 与 port count 的根本 trade-off——毫秒级 OCS（如 Polatis, MEMS）port count 仅数百，但每个 EP group 最多 64-128 GPU，足以被单一 OCS 覆盖。
    2. **每 server NIC 分配**：2 NIC → EPS Fat-tree（处理 DP、PP 全局通信），6 NIC → OCS（处理 EP 局部 all-to-all）。OCS 仅需连接同一 EP group 内的 GPU（最大 64 GPU），可由 500-port 级 commodity OCS 轻松支持。
    3. **重配置时机**：FP 第一个 all-to-all 阻塞网络等待 OCS 重配置（25ms），后续 BP 的 2 个 all-to-all 在 attention/expert computation 期间隐藏重配置延迟。总重配置次数：每 MoE layer 2 次（FP 一次 + BP 一次）。
  - 对比 baseline 的改进映射：
    - **静态 full bisection bandwidth → 按需分配 OCS 直连电路**：Fat-tree 对全部 GPU 对提供均等带宽，但 MoE 的 EP 通信是 sparse 且动态的 → MixNet 的 greedy algorithm 识别 traffic-intensive GPU 对并分配专用 OCS 电路，其他 pairs 走 EPS fallback。用更少的网络硬件（更低的 cost）达到与 Fat-tree 相当的性能。1024 GPU + 400 Gbps 下 cost-efficiency 提升 1.9×-2.3× vs Fat-tree。
    - **训练前一次性重配置（TopoOpt/Google Lightwave）→ 训练中动态重配置**：TopoOpt 假设 traffic pattern 在训练全程不变 → MixNet 每 iteration 根据实时 traffic demand 调整拓扑。仿真结果：MixNet 比 TopoOpt 快 1.3×-1.5×（因 TopoOpt 的静态拓扑无法适应 MoE 的动态 all-to-all）。
    - **Global OCS 的 scalability-reconfiguration trade-off → 区域可重构**：Global OCS 需要上千端口（scalability）但重配置慢（分钟级）→ MixNet 的 regional design 将 OCS 限制在 EP group 范围（<128 GPU），使毫秒级重配置成为可能。仿真验证 MixNet 可扩展至 32768 GPU（4096 servers），通过多个 region 并行工作。
    - **NVSwitch/NVLink 仅限 intra-server → OCS 扩展 scale-up 域边界**：前瞻分析（§8）显示，当 OCS 通过 co-packaged optics 直接连接到 GPU chip 时，MixNet 将 scale-up 域从 NVL72 的 72 GPU 扩展到整个 EP group，训练 DeepSeek-V3 时比 NVL72 快 1.3×。

## MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

- baseline方法是什么？
  Megatron-LM 是 baseline——attention 和 FFN 均使用 Tensor Parallelism (TP) 进行 intra-node 并行，expert parallelism 跨节点执行（因 TP 占满 intra-node 通信）。以训练 352B MoE / 8 GPU per node / PP=15 为例的全栈执行：
  - **算法层**：每层 MoE 包含 self-attention + expert FFN，所有组件用 TP 切分 hidden dimension。Attention 中 QKV projection 和 output projection 均经 TP 的 all-gather/reduce-scatter。FFN 中 expert 经 TP 切分 intermediate dimension，token dispatch 需跨节点 all-to-all。
  - **系统框架层**：Megatron-LM 3D 并行（DP + TP + PP），Interleaved 1F1B pipeline scheduling，依赖 torch.autograd 进行 backward 自动微分，通信-计算重叠仅限 DP 和 PP（来自 MegaScale），intra-layer TP 通信在 critical path 上。
  - **编译框架层**：论文未明确说明（标准 PyTorch + NCCL）。
  - **kernel 调度层**：FlashAttention 加速 self-attention，NCCL collectives（all-gather, reduce-scatter, all-to-all）处理通信，torch.scatter_add/gather 做 token dispatch/combine。无 fused communication-computation kernel。
  - **硬件架构层**：NVIDIA H800 GPU，intra-node NVLink 400 GB/s，inter-node RDMA/NIC ~50 GB/s。
  - **核心缺陷**：TP 通信量恒定为 2bsh(n-1)/n（与并行度 n 无关），随着 GPU 计算能力增长（H800 vs A100），通信时间占比持续上升（forward pass 中通信占 43.6%）；TP 切分 FFN intermediate dimension 降低 GEMM 效率（小矩阵乘法 GPU 利用率低）；DP 下 attention 激活内存 8× 膨胀导致 OOM；cross-node expert all-to-all 与 TP all-gather/reduce-scatter 叠加导致通信成为瓶颈；torch.autograd 的自动微分限制了 communication 与 computation 的灵活重排。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MegaScale-MoE 用三方面优化系统性解决 baseline 的通信瓶颈，全栈执行如下：
  - **算法层（通信高效并行策略）**：
    - Attention 改用 Sequence Parallelism (SP)：基于 DeepSpeed-Ulysses 的 all-to-all 风格，通信量降至 2bsh(n-1)/n × (2+2/m)/n，当 m=4 (GQA) 时约为 TP 的 1/4。SP 复制 attention weights 带来额外参数量，但因 MoE 中 expert 参数占绝对多数（>90%），额外内存仅 1.2-5.4%，DP 参数同步通信差异仅 0.3-3.1%。
    - FFN 改用 Expert Parallelism (EP)：通信量 2k/n × bsh(n-1)/n。top-k > n 时自适应切换 all-to-all → all-gather + reduce-scatter（避免 all-to-all 的全对全通信，改用高效环形通信）。EP 保持完整 expert 在单个 GPU 上，GEMM 效率远高于 TP 切分后的碎片化矩阵乘法。
    - 每个 MoE layer 限制在单 node 内（利用 NVLink 高带宽），跨 node 仅用 PP。
  - **系统框架层（Inter-operator overlap）**：
    - 不再依赖 torch.autograd，而是将每个 MoE layer 分解为独立的 GPU kernel 算子，在统一 macro module 中手动编排 forward/backward 的执行顺序。
    - Selective activation rematerialization：仅保留计算密集的激活（如 GroupedGEMM 输出），丢弃可由通信或轻量计算重新获得的激活（如 fc2_in 通过 recompute SiLU + fc3_out 获得，ffn_in 通过 re-perform RMSNorm + all-gather 获得），节省 ~50% 激活内存，重计算与反向通信重叠。
    - Holistic scheduling：backward 中将 activation recomputation 与 gradient communication 交织执行，所有非依赖算子异步在不同 CUDA stream 上并发。
  - **kernel 调度层（Intra-operator overlap）**：
    - 将通信 operator 与直接依赖的计算 operator 以 tile 粒度融合，使用 device memory barrier 实现 tile 级同步（消除 host CPU 干预的随机延迟）。
    - 四类 fused kernel：A2A+GEMM（all-to-all 数据到达即通知 GEMM 计算该 tile）、GEMM+A2A（GEMM tile 完成后直接发起 remote write）、AG+Scatter+GroupedGEMM（token 按 expert→rank 排序使 tile 依赖 rank 数最小化，scatter 内联为 row selection）、GroupedGEMM+Gather+RS（前述逆过程）。
    - Swizzling 重排 tile 通信/计算顺序，减少多 rank 同时读写同一 GPU 的 NVLink contention。
  - **编译框架层**：论文未明确说明（基于 CUDA kernel 实现，无编译器级修改）。
  - **硬件架构层**：NVIDIA H800 GPU，intra-node NVLink 400 GB/s。论文通过计算-通信比公式 R ≈ 3/2 × h_ffn × bandwidth/peak 论证了扩展性：只要 expert intermediate dimension 足够大，EP+SP 策略理论上可以 scaling beyond NVLink domain 到 RDMA 级别仍保持效率（通信量随 n 增大而减少，与 TP 恒定通信量不同）。
  - **通信压缩补充**：BF16 训练中将 FP32 reduce-scatter 替换为 BF16 all-to-all + FP32 本地归约（梯度通信量减半，且避免 BF16 环形累积精度流失）；FP8 训练中用 FP8 all-to-all 替代 BF16 reduce-scatter + per-token quantization（forward）/ per-channel group quantization（backward）。

## MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

- baseline方法是什么？
  Baseline 为 vLLM（tensor parallelism only）和 TensorRT-LLM（tensor parallelism + expert parallelism for expert layers），二者均为 attention 和 FFN 模块**共置部署**（colocated）。以 vLLM 为例说明全栈执行路径：
  - **算法层**：MoE 模型推理时，每层依次执行 attention → gating → expert FFN。Decoding 阶段 attention 为 memory-intensive（需访问每个请求的 KV cache），FFN 为 compute-intensive（weights 可跨 batch 共享）。但在 MoE 中，因 sparse activation（每个 token 仅激活 top-K expert），给定 batch size B 时每个 expert 仅处理 B × top-K / #experts 个 token。以 Mixtral 8×22B 为例，B=156 时每个 expert 仅得 ~39 tokens，理论 MFU 仅 25%（= top-K / #experts = 2/8）。
  - **系统框架层**：vLLM 使用 PagedAttention + continuous batching 管理 KV cache，通过 tensor parallelism 跨多个 GPU 切分权重矩阵。整个模型（含 attention 和 FFN）部署在同一组 GPU 上。Mixtral 8×22B / DBRX 需要最小 8 GPU 单节点，Scaled-MoE 需多节点（pipeline parallelism）。TensorRT-LLM 额外支持 expert parallelism（expert 分布到不同 GPU），但 attention 和 FFN 仍共置。
  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。
  - **kernel 调度层**：vLLM 使用 FlashAttention + custom CUDA kernels，TensorRT-LLM 使用 custom kernel optimizations。NCCL All-to-All 用于 MoE 层的 token dispatch/collect（expert parallelism 模式）。无 communication-computation overlap 设计。
  - **硬件架构层**：NVIDIA 80GB Ampere GPU（A800），200 Gbps InfiniBand inter-node，400 GB/s NVLink intra-node。
  Baseline 核心缺陷：
  1. **FFN GPU 利用率低**：MoE 的 sparsity（top-K / #experts）直接降低每个 expert 的有效 batch size，使 FFN 从 compute-intensive 退化为 memory-intensive，MFU 仅 25%（Mixtral 8×22B）或更低。
  2. **无法独立扩展**：Attention 和 FFN 共置部署，无法分别根据其 memory-intensive vs compute-intensive 特性独立优化并行策略和硬件选择。
  3. **通信开销无法隐藏**：All-to-All 通信与计算串行执行，GPU 在通信期间空闲。
  4. **同构部署浪费成本**：Attention（memory-intensive）和 FFN（compute-intensive）被迫使用相同 GPU 类型，无法利用 heterogeneous hardware 的性价比优势。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MegaScale-Infer 通过 **Disaggregated Expert Parallelism + Ping-Pong Pipeline Parallelism + High-Performance M2N Communication** 三层次设计解决上述缺陷。全栈执行路径（以 Mixtral 8×22B, tp_a=2, tp_e=1, n_a=4, E=8, m=3 为例）：
  - **算法层 — Disaggregated Expert Parallelism + Ping-Pong Pipeline**：
    1. **Module Disaggregation**：将每个 Transformer layer 的 attention 模块和 expert FFN 模块物理分离到不同 GPU 节点。Attention node 复制全部 attention 参数（data parallelism），expert node 按 expert parallelism 分布 experts（每 expert node 1-8 GPU，tensor parallelism within node）。
    2. **Independent Scaling**：n_a 个 attention node 聚合请求到 E 个 expert node，每个 expert 的有效 batch size = b_a × n_a × K/E。通过调节 n_a 可使 expert 从 memory-intensive 变为 compute-intensive（满足 b_e ≥ F/B，即 batch ≥ compute/memory bandwidth ratio）。
    3. **Ping-Pong Pipeline**：将 global batch B 拆分为 m 个 micro-batch。在 Layer ℓ：micro-batch 0 在 attention → expert → attention（Layer ℓ+1），micro-batch 1 在 attention → expert，micro-batch 2 在 attention。3 个 micro-batch 交替流动，使 attention 和 expert 在对方计算时持续 busy，且通信被计算覆盖（T_c < T_f 时）。
    4. **Deployment Plan Search（Algorithm 1）**：枚举 tp_a, tp_e, n_a, m，SIMULATE 函数通过性能模型（ROI-based GEMM timing + profiling 获取的 linear model coefficients + network bandwidth utilization profiling）binary search 最大 B 满足 SLO（T_iter ≤ 150ms TBT），选最大化 throughput per unit cost 的 plan。
    5. **Heterogeneous Hardware Selection**：attention node 选高 per-cost 内存带宽/容量 GPU（如 H20: 51.9 GB/$, 2214.1 GB/s/$），expert node 选高 per-cost 计算 GPU（如 L40S: 335.2 TFLOPS/$）。
  - **系统框架层 — M2N Communication + Fused Kernels**：
    1. M2N 通信库（PyTorch extension, ~4900 行 C/C++ + ~5000 行 Python）：替代 NCCL 的 peer-to-peer primitives，使用 GPUDirect + RDMA write with immediate 消除 GPU-to-CPU 拷贝，使用 CUDA event + cuStreamWaitValue32 消除 GPU synchronization，使用 GDRCopy flush 确保数据一致性。
    2. Flux-based kernel fusion：将 TP 的 all-gather 与 GEMM 融合为单 kernel。
    3. Sequential operator fusion：gating + top-k + token scatter 融合，减少 kernel launch 和 memory access。
    4. Expert load balancing：on-device redundancy based on expert popularity，greedy approximation 分配 experts 到 nodes。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层 — M2N Communication Optimization**：
    1. M2N Sender chain：cudaEventSynchronize → cuStreamWaitValue32(block stream) → RDMA write with immediate → poll CQ → shared memory flag wake stream。
    2. M2N Receiver chain：cudaEventSynchronize → cuStreamWaitValue32(block stream) → poll CQ → GDRCopy flush → shared memory flag wake stream。
    3. Traffic optimizations：高优先级 ACK queue 隔离 ACK 与 data 包；congestion control 微调减少 rate-limiting。
    4. 对比 NCCL：消除 batch-of-8 group op 限制、GPU sync instability、group init overhead；对比 DeepEP CPU vs GPU 通信 trade-off（本场景 ~256KB/pair 下 CPU single-thread 足以饱和带宽）。
  - **硬件架构层**：
    同构：8×NVIDIA 80GB Ampere GPU per node, 200 Gbps InfiniBand, 400 GB/s NVLink。
    异构：H20（96 GB, 4096 GB/s, 148 TFLOPS）+ L40S（48 GB, 864 GB/s, 362 TFLOPS）。无硬件修改。
  对比 baseline 的改进映射：
  - **FFN GPU 利用率低 → Disaggregated Expert Parallelism 增大有效 batch size**：Colocated baseline 下每个 expert 仅得 B×K/E tokens → disaggregated 下每个 expert 得 B×n_a×K/E tokens（n_a 个 attention node 的请求聚合）。以 Mixtral 8×22B 为例，n_a=4 时 expert batch size 增至 4×，FFN 从 25% MFU 提升至 compute-bound。端到端 decoding throughput per GPU 提升 2.56× vs vLLM，1.28× vs TensorRT-LLM。
  - **无法独立扩展 → Attention Replication + Expert Parallelism 独立配置**：Attention 使用 data parallelism（按需 n_a replica）、expert 使用 expert parallelism（按需 E expert nodes），各自独立选择 tp size 和 GPU 类型。Deployment plan search 自动找到平衡 T_a ≈ T_e 的配置，最大化 GPU 利用率。
  - **通信开销无法隐藏 → Ping-Pong Pipeline Parallelism**：m=1 时 attention/expert 互相等待（idle time）→ m=2 时双方同时 busy → m=3 时通信被计算完全覆盖（T_c < T_f 条件满足时），throughput 额外提升 1.10×–1.38×（模型越大、通信开销越大，m 增加收益越显著）。
  - **同构部署浪费成本 → Heterogeneous Deployment**：H20（高 per-$ bandwidth）为 attention + L40S（高 per-$ compute）为 expert。Decoding throughput per unit cost 提升 3.24× vs vLLM on H20，1.86× vs TensorRT-LLM on H20。同时吞吐 per unit power 提升 1.80×（H20 500W vs L40S 350W 的功耗-性能效率差异化利用）。

## M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

- baseline方法是什么？
  - Baseline 为单域单任务 MLP、多任务方法（ShBot-MTL, PLE-MTL, MMoE-MTL, AdaTT）、多域方法（STAR, ShBot-MDL, MMoE-MDL, PLE-MDL）、以及多域多任务方法（ShBot-MDMT, MMoE-MDMT, PLE-MDMT, M2M）。以 MMoE-MDMT 为例，它在每个域上复用 MMoE 的 shared expert + task-specific gate/tower 结构，所有 task gate 对所有 expert 输出进行加权求和后送入各 task tower 预测。该全栈执行路径为：
    - 算法层：输入特征 x_d → 共享 embedding 层 → N 个 shared expert（MLP + ReLU）→ 每个 task t 的 gate 对 expert 输出做 softmax 加权 → task-specific prediction tower（2 层 MLP + Sigmoid）→ y_hat_{d,t}。所有域共享同一套 expert 参数，gate 在每个 domain 独立运行但结构相同。
    - 系统框架层：论文未明确说明（标准 PyTorch 训练，无 Serving 改造）。
    - 编译框架层：论文未明确说明。
    - kernel 调度层：论文未明确说明。
    - 硬件架构层：论文未明确说明。
  - MMoE-MDMT 的核心缺陷在于：(1) 域间信息和任务间信息均通过同一套 shared expert 隐式学习，缺乏对 domain-specific 和 task-specific 模式的显式建模；(2) 融合方式单一（仅 gate 加权），无法精确控制域/任务/共享信息的贡献比例；(3) 出现 MDMT seesaw 现象——同一多域信息传递方法无法泛化到不同任务，同一多任务优化平衡策略无法泛化到不同域。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - M3oE 通过三个解耦的专家模块 + 两级自适应融合机制解决上述缺陷：
    - 算法层全栈执行路径：输入 x_d → **Domain Representation Extraction**（W_d ⊙ W_sh 元素乘捕获域特定+共享模式 → W_c 映射到统一空间 + f_DA 域无关映射）→ h_d → 并行送入三个模块：
      1. **Shared Expert Module S**：N 个 expert，每个 = ReLU(LayerNorm(W_e h_d + b_e))，D×T 个独立 gate 做 softmax 加权求和 → S_{d,t}(h_d)，捕获跨域跨任务共性。
      2. **Domain Expert Module D**：D 个 expert（每域一个），f_E^d(h_d) 为对应域 expert 输出，β_d 控制当前域 vs 其他域的加权融合 → D(h_d)，显式捕获域特定偏好。
      3. **Task Expert Module T**：T 个 expert（每任务一个），f_E^t(h_d) 为对应任务 expert 输出，β_t 控制当前任务 vs 其他任务的加权融合 → T(h_d)，显式捕获任务特定偏好。
      → **两级融合**：Level-1（β_d/β_t 控制域间/任务间融合）+ Level-2（α_d/α_t 控制域/任务/共享模块间的贡献平衡）→ h̄_d = S(h_d) + α_d·T(h_d) + α_t·D(h_d) → D×T 个独立 prediction tower → y_hat_{d,t}。
      → **AutoML Bi-Level Optimization**：α_d, α_t, β_d, β_t 由 Sigmoid(可训练标量 e_w) 生成，与模型参数交替优化（外层更新 W，内层更新 α/β），自适应确定每对 (d,t) 的最优融合权重。
    - 系统框架层：论文未明确说明（标准 PyTorch 训练）。
    - 编译框架层：论文未明确说明。
    - kernel 调度层：论文未明确说明。
    - 硬件架构层：论文未明确说明。
  - 对比 baseline 的改进映射：
    - MMoE 只有一个 shared expert 模块 → M3oE 新增 domain expert 和 task expert 模块，显式建模域/任务特定信息，解决"无法泛化到不同域/任务"的 MDMT seesaw。
    - MMoE 用统一 gate 融合 → M3oE 用两级融合（β 控制专家内部平衡 + α 控制模块间平衡），实现更精细的信息贡献控制。
    - MMoE 固定架构 → M3oE 通过 AutoML 自适应学习融合权重，无需人工为不同数据集调参，提升泛化能力。

## MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

- baseline方法是什么？
  - Baseline 为 GPTQ uniform bit-width quantization（"Uni"），即对所有 expert 使用相同位宽（如 2-bit）做 PTQ 量化，各 expert 重要性差异被忽略。另一对比 baseline 为 BSP（Block Score Predictor, Li et al. 2024），它基于 block 级别做混合精度分配（25% MoE layer 为 4-bit，其余 2-bit），但仍是 layer 粒度而非 expert 粒度。动态剪枝方面，baseline 为 weight-only pruning（Lu et al. 2024），仅基于 routing weight ratio w₁/w₀ 剪枝低分 expert，不保护重要 token。
  - 以 GPTQ 2-bit uniform quantization + weight-only pruning 为 baseline，全栈执行路径为：
    - **算法层**：MoE-LLM 推理时，对于每个 token t 在每个 MoE layer：Router G(t) 计算 softmax(W_g · t) 生成 N 个 expert 的 routing scores → 取 Top-2 expert {E₀, E₁} 及对应权重 {w₀, w₁} → 若 w₁/w₀ < μ 则仅计算 y = w₀·E₀(t)，否则 y = w₀·E₀(t) + w₁·E₁(t) → 所有 expert 权重 W 均被 2-bit uniform GPTQ 量化。expert 存储：每层 8 个 expert 均占用相同 ~2-bit/param。注意力模块保持 16-bit。
    - **系统框架层**：标准 PyTorch + HuggingFace Transformers 推理，无特殊 Serving 框架修改。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：使用 HQQ 保存量化权重和反量化，CUDA kernel 基于 HQQ 实现，无 expert-specific 优化。
    - **硬件架构层**：NVIDIA A100-80GB GPU，Mixtral 8×7b 需 2 卡（FP16 约 96.8 GB），量化后 ~13.6 GB 仍需 at least 1 卡 A100。
  - Baseline 核心缺陷：
    1. **忽略 expert 异质性**：不同 expert 的激活频率、routing weight、量化敏感度差异巨大，uniform 量化导致重要 expert 欠保护而冗余 expert 过保护，2-bit uniform 量化在 8 个 benchmark 上平均准确率下降 28.6%（71.29% → 42.67%）。
    2. **BSP 的 layer 粒度粗放**：仅区分 layer 不区分 expert，某些 layer 内仍有高重要性 expert 被 2-bit 量化损坏，某些低重要性 expert 浪费 4-bit 预算。
    3. **Weight-only pruning 引发 attention decay**：仅依赖 routing weight 剪枝时，某些关键 token 对应的 expert 被错误剪枝，导致后续层 attention map 畸变，PPL 从 ~5.9 升至 ~6.5（约 10% 相对退化），且 15% 剪枝率下 LM-Eval 降约 10%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MC 通过"expert 显著性驱动的静态混合精度量化（PMQ）+ token 重要性感知的动态剪枝（ODP）"双阶段设计解决上述缺陷：
  - 全栈执行路径（以 Mixtral 8×7b, k=2.54-bit 为例）：
    - **算法层 — PMQ 阶段（pre-loading）**：
      1. 在 C4 校准数据上对原始 16-bit Mixtral 8×7b 做一次前向推理，为每个 MoE layer 的每个 expert i 计算三维重要性向量：(a) 访问频率 ϕᵢ = nᵢ/N；(b) 激活权重和 wᵢ = Σσᵢʲ/N；(c) 各候选位宽 j∈{1,2,3} 下的量化重构 F-norm ϵᵢⱼ = ‖F(θ) − F(θ[eᵢ→Q(eᵢ,j)])‖_F。
      2. 构建 Integer Programming 模型：目标函数 MIN ΣᵢΣⱼ ϕᵢᵅ·wᵢᵝ·(ϵᵢⱼ·xᵢⱼ)ᵞ，约束为平均位宽 = k、每个 expert 唯一分配、至少 1 个 3-bit 和 1 个 2-bit expert。求解得到的 xᵢⱼ 给出每个 expert 的最优位宽 Bᵢ ∈ {1,2,3}。
      3. 按 Bᵢ 用 GPTQ 量化每个 expert：对 2/3-bit 使用线性量化 + Hessian 误差补偿；对 1-bit 使用 B̃ = (sign(W)+1)/2 映射到 {0,1} + scaling factor s = ‖W‖_ℓ₁/(d×m)。Attention/gating 等非 expert 参数统一 4-bit。量化耗时 ~90 分钟（Mixtral 8×7b）。
    - **算法层 — ODP 阶段（online inference）**：
      1. 对每个输入 token，在当前 MoE layer 之前，基于上一层 attention map A = softmax(KᵀQ/√dₖ) 计算 token importance：Iⱼ = ‖tⱼ‖₁ · (Σ_{i≥j} Aⱼᵢ)/(L−j)，结合 token 特征范数和被关注度。
      2. 对 top-2% 高重要性 token 启用完整保护：保留 Router 分配的 Top-2 expert 计算，不做剪枝。
      3. 对非保护 token，仍执行 routing-score-based pruning：当 w₁/w₀ < μ（μ 为 calibration 中位数）时仅保留 primary expert，跳过 secondary expert。
      4. 1-bit expert 反量化使用位运算加速：s · xB = s(Σ_{B̃ᵢⱼ=1} xⱼ − Σ_{B̃ᵢⱼ=0} xⱼ)，MACs 从 dm 降至 m。
    - **系统框架层**：使用 HQQ 工具保存混合精度量化权重并执行反量化，设计了 1-bit 权重紧凑存储格式（bit-change transformation B̃ ∈ {0,1}）。CUDA kernel 基于 HQQ 适配。未修改特定 Serving 框架。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：HQQ 提供的 CUDA kernel 处理不同位宽权重的反量化和矩阵乘法。1-bit 权重使用加法树替代乘累加。未引入新的 kernel 调度策略。
    - **硬件架构层**：单张 NVIDIA A100-80GB（或 RTX 3090）即可运行压缩后 Mixtral 8×7b，无需多卡。FP16 baseline 需 2×A100。
  - 对比 baseline 的改进映射：
    - **Uniform quantization 忽略 expert 异质性 → PMQ 三层 expert 重要性驱动位宽分配**：ϕᵢ（频率）× wᵢ（路由权重）× ϵᵢⱼ（量化误差）三维建模每个 expert 的真实重要性。Integer Programming 在平均位宽约束下自动将高位宽分配给关键 expert（如某些 layer 中高频高权 expert 获 3-bit），低位宽分配给冗余 expert（如几乎不被激活的 expert 获 1-bit）。结果：2.54-bit PMQ 在 8 benchmark 上 Avg 67.50%（仅降 3.8%），远超 BSP 的 49.07%（降 22.2%）和 Uni 2-bit 的 42.67%（降 28.6%）。
    - **BSP layer 粒度粗放 → PMQ expert 粒度精细**：BSP 在 layer 级别决策（25% layer 4-bit, 75% layer 2-bit），PMQ 在 expert 级别决策（每个 expert 独立 1/2/3-bit）。同一 layer 内不同 expert 可获不同位宽，精度-效率 trade-off 更优。
    ## MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism

- baseline方法是什么？
  - Baseline 为 **FastMoE**（primitive expert parallelism）和 **FasterMoE**（pipeline parallelism + expert shadowing）。以 FasterMoE 为例说明全栈执行路径：
    - **算法层**：MoE 训练时，对每个 mini-batch 的 All-to-All dispatch → Expert FFN（Linear1 + GeLU + Linear2）→ All-to-All collect 三个阶段**串行**执行，通信阶段 GPU 空闲等待，计算阶段网络带宽空闲。FasterMoE 引入了 pipeline parallelism，但有两个关键缺陷：(1) 按 **node 维度**切分 batch（而非 batch 维度），将 All-to-All 拆解为多组 P2P 通信，丧失 NCCL 的 All-to-All 优化能力（如 ring/tree topology 聚合），且在异构带宽下同步等待造成资源浪费；(2) pipeline granularity **固定**，无法适应动态变化的 batch size 和网络条件。同时 FasterMoE **未考虑内存优化**，activation tensors 和 temporary buffers 占据大量 GPU DRAM，限制了可训练的 batch size 和模型规模。
    - **系统框架层**：基于 PyTorch + NCCL + CUDA 实现，使用 NCCL 的 All-to-All 原语进行 token dispatching。FasterMoE 通过 NCCL group calls 将 All-to-All 拆分为多路 P2P 通信，每个 group 内独立执行。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：默认 PyTorch CUDA kernel launch，未对 communication/computation/memory copy 的 stream 并行做系统性优化，三种操作串行执行或简单重叠，不考虑资源竞争（带宽竞争、SM 竞争）导致的 slowdown。
    - **硬件架构层**：NVIDIA A100 40GB / V100 16GB GPU，200 Gbps / 56 Gbps HDR InfiniBand，第 3 代 / 第 2 代 NVLink。
  - FasterMoE 的缺陷映射：
    1. **通信效率低**：按 node 维度切分导致 NCCL All-to-All 退化，pipeline granularity 受限于 node 数（通常 2-8），无法精细调优。
    2. **自适应能力差**：pipeline granularity 固定，无法随 batch size、模型规模、集群规模变化而动态调整。
    3. **内存占用高**：activation tensors（Equation 2：M_act = 4*B*M + B*H）和 temporary buffers（Equation 3）未优化，如 1.5B GPT-2 在 batch size=32、seq len=1K 时需要约 60GB GPU 内存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MPMoE 通过三个核心机制联合解决上述缺陷：
  - **全栈执行路径**（以 MoE-GPT-XL 在 Valor 集群 n=6 为例）：
    - **算法层 — 自适应 Micro-Batch Pipeline**：
      1. 将 mini-batch T_I(N, B, M) 沿 batch 维度切分为 n 个 micro-batch T_I[i](N, B/n, M)，而非沿 node 维度。
      2. 对每个 micro-batch 执行 pipeline: S(i) → C(i) → R(i)，三个阶段重叠执行。S(i+1) 在 C(i) 启动后并发开始，R(i) 在 C(i) 完成后启动，同时 S(i+2) 开始。交替调度 S 和 R stage 以增强内存访问局部性（Figure 7）。
      3. 自适应确定最优 n：MPMoE-pb 通过 Algorithm 1 的 profile-based 搜索（利用"n 单调递增于 B"和"性能关于 n 呈抛物线"的两个假设减少搜索空间），MPMoE-pm 通过 piecewise 性能模型估算（图 8 的 3 种 paradigm + 图 9 的 piecewise 速度拟合 + Section 2.3 的 α 干扰因子）。
    - **算法层 — Memory Reuse**：
      1. 识别"memory bubbles"：不同 micro-batch 的 T_DI[i]、T_M[i]、T_DO[i] 在不同时间激活，可共享 buffer。n 个 partition 的 buffer 需求从 O(n) 降为 O(1)（Figure 6）。
      2. 4 种恢复策略（S1-S4, Table 2）：按需组合 CPU offload、通信重放、重计算三种机制恢复后向所需的被覆盖 tensors，根据当前 N（GPU 数量）和 B（batch size）选择开销最小的策略。
    - **系统框架层**：
      1. 基于 PyTorch 1.9 + CUDA 11.1 + NCCL 2.7 实现，修改了 MoE layer 的 forward/backward 实现。
      2. 沿 batch 维度切分保留原始 All-to-All 语义（不降级为 P2P），充分利用 NCCL 对 All-to-All 的优化（合并小消息、ring/tree topology 聚合）。
      3. 使用 Tensor Cores 加速 Expert FFN 中的矩阵乘法（Linear1 和 Linear2）。
    - **kernel 调度层**：
      1. 使用多个 CUDA stream 并行执行 computation（C）、communication（S/R）、memory copy（M/D2H/H2D），通过 α(y,x) slowdown 因子（Section 2.3, Figure 3）量化并行操作间干扰。
      2. 建立 3 种 pipeline paradigm（图 8）的性能模型：P0-P4 五阶段分解，每个阶段由瓶颈 stream 的执行时间决定。
      3. Profiling 微基准（图 9）获取 W_comp, W_comm, W_mem 的 piecewise 速度函数（区分小体积/大体积数据的不同硬件利用率）。
    - **硬件架构层**：
      1. Adira: 64×A100 40GB + 200 Gbps InfiniBand + NVLink 3.0。
      2. Valor: 16×V100 16GB + 56 Gbps InfiniBand + NVLink 2.0。
      3. 自适应配置考虑了不同集群的硬件特性差异：Adira 上 MPMoE-pb 优于 MPMoE-pm（因网络波动大导致性能模型精度下降），Valor 上两者性能相当（网络稳定）。
  - 对比 baseline 的改进映射：
    - **FasterMoE 按 node 切分 → MPMoE 按 batch 切分**：保留了 NCCL All-to-All 的集体通信优化能力，避免 P2P 拆解带来的 kernel launch 开销和同步等待，pipeline granularity n 不受 node 数限制（可从 2 到 8 灵活选择），micro-benchmark（Figure 13）验证了更低的 dispatch/recovery 通信延迟。
    - **固定 pipeline → 自适应 pipeline（profile-based + performance model）**：MPMoE-pb 通过 Algorithm 1 的动态搜索缓存（G 和 C）在运行时学习最优 n，搜索次数随训练进行收敛；MPMoE-pm 通过性能模型在零 profiling 开销下估计（<1% overhead, Figure 16），在稳定网络环境（Valor）下与 pb 性能可比。
    - **未考虑内存优化 → Memory Reuse + 自适应策略选择**：4 种策略可根据 N（GPU 数）和 bottleneck 类型（计算瓶颈 vs 通信瓶颈）自适应选择——N 小时 CPU offload 更优（S1, S2），N 大时 recompute 更优（S4，避免 PCIe/memory bandwidth 竞争）。最终实现最高 53% 内存节省（Figure 11），达理论上限的 ~95%（Figure 12），同时可增加 batch size 以提升 GPU 利用率（减少 bubble overhead）。：Iⱼ = ‖tⱼ‖₁ · mean_attention 同时考虑 token 自身特征强度和跨 token 注意力，保护仅 2% 的关键 token 即可将 PPL 从 6.46 降至 6.24（≈3.4% 改善），且激活参数压缩比仅从 15.1% 降至 14.8%（几乎无损）。同时可进一步剪枝低重要性 token 的所有 expert（2% token masking → CR 15.8%, PPL 6.35），实现效率与精度双赢。

## MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

- baseline方法是什么？
  - Baseline 为 **FastMoE**（primitive expert parallelism，All-to-All 与 expert 计算串行执行）和 **FasterMoE**（pipeline parallelism + expert shadowing）。以 FasterMoE 为代表，全栈执行路径为：
    - **算法层**：MoE 训练时对每个 mini-batch，gating network 做 top-1 routing → All-to-All dispatch → Expert FFN（Linear1 + GeLU + Linear2）→ All-to-All collect，三个阶段串行执行。FasterMoE 引入 pipeline parallelism，但沿 **node 维度**切分 tensor（将 All-to-All 拆解为多组 P2P 通信），pipeline granularity 受限于 node 数且**固定不变**，无法适应动态 batch size。同时 FasterMoE 的 dynamic shadowing 额外增加内存占用（比 FastMoE 更多），**未考虑 activation/temporary buffer 的内存优化**。
    - **系统框架层**：基于 PyTorch + NCCL 实现，FasterMoE 通过 NCCL group calls 将 All-to-All 降级为 P2P 通信，丧失 NCCL 内置的 All-to-All 优化（ring/tree topology 聚合、小消息合并）。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：默认 PyTorch CUDA stream 调度，computation 和 communication 串行或简单重叠，未系统性考虑并行 CUDA stream 间的资源竞争（memory bandwidth、SM 占用）。
    - **硬件架构层**：8×NVIDIA DGX A100 服务器，每节点 8×A100 SXM 40GB，200 Gbps HDR InfiniBand，NVLink 3.0。
  - FasterMoE 的核心缺陷：
    1. **Pipeline granularity 固定且粗放**：n 不随 B 变化，coarse-grained 时重叠不充分，fine-grained 时 kernel launch overhead 导致 GPU 利用率下降。
    2. **按 node 维度切分导致通信效率低**：All-to-All 降级为 P2P，在异构带宽下同步等待浪费资源，且 granularity 受 node 数限制（通常 2-8）。
    3. **内存占用高**：activation tensors（4*B*M + B*H）和 temporary buffers（B*M + B*H）随 batch size 线性增长，限制了大 batch size 训练（大 batch size 对 GPU 利用率至关重要）；FasterMoE 的 dynamic shadowing 进一步加剧内存压力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MPipeMoE 通过 **自适应 pipeline parallelism + memory reusing + 性能模型**联合优化解决上述缺陷。全栈执行路径（以 MoE-GPT-XL，n=4，S4 为例）：
    - **算法层 — Adaptive Pipeline Parallelism**：
      1. 将 mini-batch T_I(N, B, M) 沿 **batch 维度**切分为 n 个 micro-batch（而非 node 维度），每个 micro-batch 大小为 B/n，保留 NCCL All-to-All 集体通信语义。
      2. Pipeline 调度：S(i) → C(i) → R(i)，跨 3 条 CUDA stream 重叠——S(i+1) 与 C(i) 并发启动，R(i) 在 C(i) 完成后启动，同时 S(i+2) 启动。S 和 R 在通信 stream 上交替执行以增强 memory access locality。
      3. **Adaptive Granularity (Algorithm 1)**：基于"n 随 B 单调递增"假设，将 B 的值域划分为 disjoint 区间 {R_n}，映射到最优 n。用二分搜索树维护 (n, range) 映射集，O(log n) 查找。cache hit 直接返回；cache miss 时调用 searchBestGran(B) 做 trial profiling 搜索。搜索开销由多 epoch 训练摊销。
    - **算法层 — Memory Reusing**：
      1. Pipeline 中不同 micro-batch 的 T_DI[i]、T_M[i]、T_DO[i] 在不同时刻激活（"memory bubbles"），可共享同一 buffer。n 个 partition 的 activation buffer 从 m 降为 m/n。
      2. 为恢复 backward pass 所需的被覆写 tensors，设计 4 种策略：S1 (T_DI offload + T_M offload)、S2 (T_DI 通信恢复 + T_M offload)、S3 (T_DI offload + T_M 重计算)、S4 (T_DI 通信恢复 + T_M 重计算)。各策略在 forward/backward 中引入不同的 CUDA stream 操作组合（Table II：Q_fw/Q_bw 以及 μ/η 干扰因子不同）。
      3. 性能模型（Eq 10）：C(S) = (1/W_comp) * max(q1, q2*α/μ, q3*β/η)，其中 α=W_comp/W_comm, β=W_comp/W_mem。选择 C 最小的 S 作为运行时最优策略。
    - **系统框架层**：基于 PyTorch 1.9 + CUDA 11.1 + NCCL，实现为 Python 库 `pmoe`。通过 `pmoe.MoELayer(d_model=1024, pipeline=True, memory_reuse=True)` 启用优化。默认使用 top-1 gating 和 FFN expert（Linear1 + GeLU + Linear2）。
    - **kernel 调度层**：3 条 CUDA stream（comp / comm / mem copy）并行调度。通过 micro-benchmark 测量 W_comp(vol)、W_comm(vol)、W_mem(vol) 的 piecewise 速度（小 volume 线性增长、大 volume 饱和），以及 μ/σ/η 干扰因子（Figure 3）。计算几乎不受干扰（σ≈1），通信与计算重叠可行（μ_comm > 0.5），通信与 memory copy 因带宽竞争不宜并行。
    - **硬件架构层**：8×DGX A100 服务器（64×A100 40GB GPU），200 Gbps HDR InfiniBand + NVLink 3.0。无硬件修改。
  - 对比 baseline 的改进映射：
    - **FasterMoE 按 node 切分 → MPipeMoE 按 batch 切分**：保留 NCCL All-to-All 的集体通信优化（ring/tree topology 聚合、消息合并），pipeline granularity n 不再受 node 数限制（可在 2/4/8 间灵活选择），且避免异构带宽下 P2P 同步等待。PipeMoE 由此取得 2.26× avg speedup vs FasterMoE。
    - **固定 pipeline → Adaptive Granularity (Algorithm 1)**：n 随 B 单调递增假设将搜索复杂度从 O(B_domain) 降为 O(log n)。B<8k 选 n=2（GPU 利用率优先），8k-22k 选 n=4，>22k 选 n=8（重叠率优先）。自适应选 n 在所有 batch size 下均最优（Figure 12 dashed line）。
    - **无内存优化 → Memory Reusing + Perf Model 自适应选择**：activation buffer 从 n 份压缩为 1 份，节省 ΔM_act = B*(2M*(n-2)/n + H*(n-1)/n)（Eq 5）。N 小时 S1/S2 更优（offload 的 memory copy 开销可容忍，重计算因计算瓶颈而昂贵），N 大时 S4 更优（重计算开销被通信瓶颈掩盖，且避免 PCIe bandwidth 竞争）。结果：最高 47% 内存节省（vs FasterMoE）+ 2.8× speedup，实际节省达理论上限 ~95%（Figure 10）。

## Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

- baseline方法是什么？
  - Baseline 为传统数据驱动的 GNN 动态系统建模方法（以 EGNN 为代表），包括 Linear、Dynamic（物理匀速模型）、GNN（Kipf & Welling 2017a）、Radial Field（Köhler et al. 2019）、EGNN（Satorras et al. 2022）、EGNO（Xu et al. 2024）。以 EGNN 为例说明全栈执行路径：
    - **算法层**：给定初始状态 X⁽⁰⁾ 和交互图 G，EGNN 通过 L 层等变消息传递迭代更新 node representation hᵢ 和 coordinate xᵢ。每层中，ϕ 网络学习边交互（输入 hⱼ, xⱼ, hᵢ, xᵢ → 输出 eᵢⱼ），AGG 聚合邻居边信息，COM^H 和 COM^X 分别更新 node/coordinate（Eq. 1-3）。最终通过 Decoder 输出预测状态 X̂⁽ᵗ⁾ = Decoder(H^L)。训练时最小化 MSE loss。模型在训练数据上学习动力学规律，环境变化的应对完全依赖训练数据分布覆盖。
    - **系统框架层**：标准 PyTorch 训练和推理，无特殊 Serving 框架修改。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明（标准 PyTorch CUDA kernel，无自定义 kernel）。
    - **硬件架构层**：论文未明确说明（无特定硬件/模拟器要求）。
  - Baseline 核心缺陷：
    1. **分布偏移下泛化能力差**：EGNN/EGNO 等数据驱动方法在训练环境（如 Spring strength=1.0）下表现良好，但当环境参数变化（Hard: strength=1.10, Soft: strength=0.90, Temporal Shift: 不同起止状态）时，MSE 显著上升。原因是模型仅从数据中隐式学习动力学，无法利用显式的环境上下文信息来适应变化。
    2. **缺乏环境感知能力**：不同系统参数（如弹性系数 k、电荷量 q₁,q₂）产生不同的演化规律，但传统 GNN 将所有环境的输入统一处理，无法区分"当前处于什么环境"以及如何调整预测策略。
    3. **单一模型无法覆盖多模态动力学**：同一系统中不同环境可能对应本质上不同的动力学模式（如周期性振动的不同阶段），单个 GNN 模型难以同时拟合所有模式，出现模式平均（mode averaging）导致预测模糊。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - LEGO 通过"LLM as context-aware routing function + Graph MoE with diversity enhancement"三层设计解决上述缺陷。全栈执行路径（以 Spring 系统，K=5 experts，Llama 3.1 8B 为例）：
    - **算法层 — Hierarchical Prompt Engineering**：
      1. 从三个层次提取环境上下文转化为文本：(a) System level：系统参数和背景描述，如 "The force on the balls are significant, and forces between them result in strong accelerations" + 弹簧系数 k 等数值；(b) Object level：各物体初始位置 (x,y,z) 和速度 (vx,vy,vz) 向量，数值 digit 作为 token（遵循 Gruver et al. 2024）；(c) Edge level：边连接关系，如 "ball 2 connects ball 0, ball 1, ball 3"。三层 prompt 共同构成对当前环境的完整文本化描述。
    - **算法层 — Graph Mixture-of-Expert with Diversity Enhancement**：
      1. K 个独立的 EGNN experts（同架构不同参数 θ¹...θᴷ），每个 expert 对输入并行预测，生成 K 个候选状态 X̂⁽ᵗ⁾,¹...X̂⁽ᵗ⁾,ᴷ（Eq. 6：通过 one-hot routing 获得各 expert 独立预测）。
      2. Diversity-enhanced contrastive loss（Eq. 9-10）：最大化同 expert 内表征相似度、最小化不同 expert 间表征相似度，确保各 expert 学习不同的动力学模式（如某些 expert 擅长高能量模式、某些擅长低能量模式）。
    - **算法层 — LLM Judge for Context-Aware Routing**：
      1. 将 hierarchical prompt + K 个 candidate predictions 的描述送入预训练 LLM（Llama 3.1 8B），LLM 基于环境上下文推理选择最合适的 expert，而非直接生成数值预测（避免 LLM 在复杂张量生成上的不可靠性）。
      2. Label smoothing（Eq. 7）：选中 expert 权重 α，其余 (1-α)/(K-1)，避免错误路由的硬性惩罚积累。
      3. 交替优化（Algorithm 1）：LLM routing weights 每隔若干 epoch 更新一次（减少 LLM API 调用成本），graph expert 参数通过梯度下降持续优化。LLM 推理只需文本理解能力，无需微调。
    - **系统框架层**：标准 PyTorch 训练，LLM 通过 API 调用（论文未说明具体服务框架）。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明。
    - **硬件架构层**：论文未明确说明（无特定硬件/模拟器）。
  - 对比 baseline 的改进映射：
    - **分布偏移下泛化能力差 → LLM Judge 利用环境上下文自适应路由**：传统 EGNN 从数据中隐式学习、无法利用显式环境描述 → LEGO 将环境参数、物体状态、连接关系转化为三层 prompt，LLM 以 zero-shot 方式理解环境变化并选择最匹配的 expert。在 Charged 数据集上，EGNN+LEGO 相比 EGNN 在 Hard/Soft/Temporal Shift 三种 OOD 场景下分别取得 25.4%/27.3%/16.0% 的 MSE 降低（Table 1），验证了 LLM 对环境上下文的有效利用。
    - **缺乏环境感知能力 → Hierarchical Prompt 提供三层环境描述**：Ablation（Table 4）验证了多层 prompt 的必要性——V1（仅 system level）MSE=0.761，V2（system+edge）MSE=0.735，V3（完整三层）MSE=0.728，每增加一层信息均带来一致的性能提升。Edge level 信息（连接关系文本化）的贡献尤为显著（V1→V2 降幅 > V2→V3）。
    - **单一模型无法覆盖多模态动力学 → Graph MoE + Diversity Loss**：多个 experts 通过 contrastive diversity loss 被迫学习互补的动力学模式。LLM 根据环境选择合适的 expert，实现"不同环境用不同 expert"的 specialization。参数实验（Figure 3b）显示 K=5 时性能最佳，过少（K=3）覆盖不足、过多（K=15/20）LLM Judge 判断困难导致性能下降。
    - **LLM 直接生成不可靠 → LLM-as-Judge 而非 LLM-as-Predictor**：Table 5 显示 LLM Forecasting（直接生成预测）的 MSE=6.4201 远高于 LEGO 的 0.0072（约 890× 差距），且 LLM Forecasting 推理时间更长（1.270s vs 0.438s per sample），验证了"判断优于生成"的设计哲学。

## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

- baseline方法是什么？
  - Baseline 为现有 MoE 训练框架（以 Microsoft Tutel 和 Megatron-LM SwitchMLP 为代表），它们使用固定 expert capacity 和 token dropping/padding 机制来实现 MoE 层的高效计算。以 Tutel (Hwang et al. 2022) 为例说明全栈执行路径：
    - **算法层**：MoE 训练时，Router 通过 top-k greedy selection 分配 token 到 expert → 定义固定 expert capacity（capacity_factor × num_tokens/num_experts）→ Permutation 将 token 按 expert 分组 → 对超出 capacity 的 token **直接丢弃**（不参与 expert 计算），对不足 capacity 的 expert **padding 零填充** → 使用 **batched matrix multiplication**（所有 expert 共享相同 batch size = capacity）并行计算所有 expert → Un-permutation 恢复 token 顺序 → 输出缩放。
    - **系统框架层**：基于 Megatron-LM（Shoeybi et al. 2019）+ PyTorch 实现。MoE 层使用 NCCL All-to-All 进行跨设备 token dispatching。Tutel 在此基础上实现了动态 capacity factor（运行时自适应计算最小不丢 token 的 capacity factor）和通信隐藏优化。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：使用 cuBLAS batched GEMM 作为 expert 计算原语。batched GEMM 要求所有矩阵乘法的尺寸相同，这强制了 expert capacity 约束。无自定义 GPU kernel 处理稀疏/动态计算。
    - **硬件架构层**：NVIDIA A100 SXM4 80GB GPU + 200 Gbps InfiniBand。8-way expert model parallelism。
  - Baseline 核心缺陷：
    1. **Token dropping 降低模型质量**：即使使用 load balancing loss，token routing 仍然高度不均衡。capacity_factor=1 时丢 token 导致 validation loss 仅降低 0.15（vs dense baseline），而不丢 token 可降低 0.26（1.73× 改善），足以超越更大 dense model（图 2）。完全避免丢 token 需要 capacity_factor 高达 11（Hwang et al. 2022），导致 computation 增加超过 2×（图 2）。
    2. **Padding 浪费计算和内存**：为满足 batched GEMM 的形状约束，不足 capacity 的 expert batch 需要 zero-padding。padding 在 MoE 层显著增加 activation 存储需求，导致 Tutel 被迫使用 2×–8× 更小的 micro_batch_size（表 3），降低 GPU 利用率和硬件效率。
    3. **Capacity factor 超参数调优成本高**：capacity_factor 在模型质量和计算效率之间构成 trade-off，需要为每个模型和任务调优。大型模型训练成本可达数十万美元，这阻碍了 capacity factor 的充分探索（Artetxe et al. 2021; Clark et al. 2022 完全放弃了 capacity factor 调优）。
    4. **Sequential expert 计算退化**：Megatron-LM 的 SwitchMLP（逐个 expert 顺序计算）虽避免丢 token，但随 expert 数量增加性能急剧退化——num_experts=128 时比 MegaBlocks 慢 20×（图 10），因为单个 expert 的计算量不足以饱和 GPU，小矩阵乘法序列化执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MegaBlocks 通过"block-sparse 操作重表述 + 自定义 GPU block-sparse kernels"双层设计解决上述缺陷，实现真正的 dropless-MoE (dMoE)。全栈执行路径（以 MoE-Small 训练为例，8×A100）：
    - **算法层 — dMoE 的 block-sparse 重表述（§4, Figure 4）**：
      1. Router 分配 token 到 expert（与 baseline 相同：indices, weights = router(x)）。
      2. **make_topology(indices)**：构造图 3C 的 variable-size block diagonal matrix。每个 expert 的 token batch（size 可变）被分解为 ceil(num_tokens_expert/128) 个 128×128 固定 block，所有 expert 的 block 沿对角线排列，构成一个大的 block-sparse 矩阵。这一步取代了 baseline 的 capacity 约束——不再需要 token dropping 或 padding 到固定容量。
      3. **padded_gather**：按 expert 分组 token，仅 padding 到 128 的倍数（而非 padding 到固定 capacity），padding 量极小。
      4. **sdd(x, w1, topology)**：SDD（Sparse = Dense × Dense）操作。输出是 block-sparse 的 intermediate result——sparse output matrix 中只有分配给各 expert 的 token row 被计算（对应图 3C 的非零 block），未被分配的位置默认为零。等效于同时计算所有 expert 的第一层 FFN，但只计算实际需要的位置。
      5. **dsd(intermediate, w2)**：DSD（Dense = Sparse × Dense）操作。以 block-sparse intermediate 为左输入，计算所有 expert 的第二层 FFN，输出 dense tensor。
      6. **padded_scatter + scaling**：恢复 token 顺序并乘以 router probabilities。
    - **系统框架层**：
      1. 基于 Megatron-LM + PyTorch 构建，实现自定义 dMoE layer 替换标准 MoE layer。
      2. 支持 data parallelism 和 expert model parallelism（§5.3）。expert model parallelism 中先通信各设备将接收多少 token（避免 all-to-all 阶段的丢 token/padding）。
      3. 与 Megatron-LM 的 mixed-precision training（FP16 + FP32 accumulation）完全兼容。
    - **kernel 调度层 — 自定义 Block-Sparse GPU Kernels（§5.1）**：
      1. 基于 CUTLASS 2.5 扩展实现 SDD、DSD、DDS 三种 block-sparse GEMM kernel，支持所有 transposed/non-transposed 输入组合（满足前向+反向 6 种操作需求）。
      2. **Hybrid Blocked-CSR-COO 编码**：BCSR 作为主格式（高效行迭代），**额外物化行索引**使得 BCSR 兼具 BCOO 能力。SDD kernel 中每个 threadblock 通过 row_idxs[blockIdx.x] 和 column_idxs[blockIdx.x] 直接 O(1) 定位其 non-zero block 的坐标，无需搜索 row offsets。metadata 存储开销 <0.1%（128×128 block 含 16384 非零值仅需 1 个索引）。
      3. **Transpose Indices**：为支持向后传播中稀疏矩阵转置操作（SDD^T, DS^T D, DSD^T, DD^T S），构造转置元数据（等效 BCSC 编码：column offsets + 转置顺序的 non-zero block 偏移数组）。不显式转置非零数据，通过间接索引实现在转置顺序下迭代矩阵，避免 O(nnz) 的数据复制开销。
      4. **128×128 block size** 基于 CUTLASS tile benchmark 选择（图 5），实测在所有 tile dimension 配置中表现最优（与 cuBLAS 为 dense Transformer 选择的配置一致）。大 block size 提供足够的算术强度以充分利用 A100 Tensor Cores，同时摊销稀疏元数据开销。
      5. **Custom permutation kernel**：将 token padding（到 128 倍数）融合进 gather/scatter kernel，且在前向开始时一次性构造 block-sparse 和 transpose 元数据，摊销到后续 6 次矩阵乘法。
    - **硬件架构层**：NVIDIA A100 SXM4 80GB GPU × 8。CUDA 11.5 + CUTLASS 2.5。无硬件修改。
  - 对比 baseline 的改进映射：
    - **Token dropping 降低模型质量 → 无丢 token 的 dMoE（block-sparse 重表述）**：标准 MoE 需要固定 capacity → 超出 capacity 的 token 被丢弃。MegaBlocks 用 block-sparse 操作替代 batched GEMM，天然支持每个 expert 接收不同数量的 token（variable-size block），从根本上消除了 token dropping 的必要性。结果为更低的 loss（图 7）同时避免 capacity_factor 超参数调优。
    - **Padding 浪费计算和内存 → block-sparse 只计算实际需要的 tokens**：batched GEMM 要求所有 expert batch 等大小 → 大量 zero-padding 占用 activation 内存。block-sparse 操作只对实际分配的 token row 计算（sparse matrix 的非零 block），仅需将 token batch padding 到 128 的倍数而非固定 capacity。Tutel 因 padding 导致 micro_batch_size 被迫缩小 2×–8×，MegaBlocks 的 micro_batch_size 与模型大小自然匹配，端到端训练加速 1.38×–4.35×。
    - **Capacity factor 超参数调优成本高 → 无 capacity_factor 参数**：MegaBlocks 的 dMoE 从根本上不需要 capacity_factor——不需要在丢 token 和 wasteful computation 之间权衡。这不仅减少了 hyperparameter 搜索空间，还避免了"token dropping MoE 需要额外调优 capacity factor → 调优成本和训练成本叠加"的困境。图 8 显示即使与最优 capacity_factor 的 MoE 比较，dMoE 仍减少 1.18×–1.38× 训练时间。
    - **Sequential expert 计算退化 → block-sparse 并行计算所有 expert**：Megatron-LM 的 SwitchMLP 顺序计算 expert，随 expert 数量增加 GPU 利用率急剧下降。MegaBlocks 通过 block-sparse kernels 一次性并行计算所有 expert（单个 kernel launch），在 num_experts=128 时实现 20× 加速（图 10），在常规 num_experts=64 时端到端加速达 4.35×（图 7）。
    - **现有 block-sparse 库不适用 → 自定义 MoE-tailored kernels**：cuSPARSE blocked-ELL 格式要求所有 row 的非零数相同（与 MoE 冲突），Triton Blocksparse 假定稀疏拓扑在 iterations 间不变（与 MoE 的动态路由冲突）。MegaBlocks 的 custom kernels 专为动态稀疏拓扑设计，且 block-sparse 矩阵运算达到 cuBLAS 密集运算的 98.6% 吞吐量（图 9），实现了"sparse computation at dense throughput"的工程目标。

## MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

- baseline方法是什么？
  Baseline 为 **BTX (Branch-Train-Mix)** (Sukhbaatar et al. 2024)，它将多个从同一 ancestor model 分支出来的 dense expert 合并为 MoE 模型。全栈执行路径（以 4 个 1B expert 合并为例）：
  - **算法层**：将各 expert 的非 FFN 层（embedding、attention、norm、head）通过 **unweighted averaging** 逐参数相加取平均合并为一套共享参数；FFN 层保持独立作为 MoE expert；插入一个随机初始化的 MLP router 在 attention 和 FFN 之间做 token-level top-K routing。合并后 MoE 在全部数据源混合数据集上 fine-tune 约 40B tokens 来训练 router 并恢复因参数干扰导致的性能损失。
  - **系统框架层**：基于标准 PyTorch 分布式训练，fine-tuning 阶段因 MoE 的 expert parallelism 引入跨 GPU all-to-all 通信开销（论文引 BTX 原论文指出 "fine-tuning MoEs is expensive due to communication cost between GPUs"）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明（标准 PyTorch CUDA kernel）。
  - **硬件架构层**：论文未明确说明具体 GPU 型号和集群配置。
  - BTX 的核心缺陷：
    1. **参数干扰 (Parameter Interference)**：当 experts 的参数空间发散较大时（不同 ancestor、激进不同的训练数据），简单平均无法处理 sign conflict 和 magnitude disparity——大 magnitude 参数与小 magnitude 参数/符号冲突参数平均后输出接近零的小值，削弱 task vector 效果，导致合并后 MoE 性能下降，需要大量 fine-tuning 恢复。
    2. **Fine-tuning 成本高且不可行**：MoE fine-tuning 需要多 GPU（跨 expert 通信），且需要访问所有 expert 的训练数据。当 expert 训练数据不公开或计算资源受限时，fine-tuning 无法执行。
    3. **无法处理异构专家**：BTX 要求所有 expert 具有相同架构（相同层数、hidden dimension），无法合并如 CodeLlama + Olmo 等不同架构的 expert 模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MergeME 通过三类技术分层解决上述缺陷。全栈执行路径（以 4-expert MoE, Base-1B + Math + Code + Knowledge 为例）：
  - **算法层 — 同构合并：Dare/Ties 替代平均**：
    1. 计算 task vector τᵢ = θ_b - θᵢ（base 与 expert 参数差）。
    2. Ties merging: drop bottom (100-p)% 最小 magnitude 参数 → 确定每个位置主导符号方向 → 仅累加同符号 task vector 值，消去异符号冲突。
    3. Dare merging: 随机 drop (100-p)% 参数 → rescale τᵢ/(0.01p) 补偿丢弃 → 简单求和所有 τᵢ。
    4. 合并回 base: θ_m = θ_b + λ·τ_m（λ=1/3, p=80%）。
    5. 保留 FFN 独立 + 插入 MLP router + fine-tune。
  - **算法层 — 无 Fine-tuning：PPL 路由 + 分离 Attention**：
    1. 不合并 attention 层：各 expert 保留自己的 attention 参数，避免合并后 attention 受 l 个 task vector 影响而 FFN 仅受 K 个 task vector 影响的不一致性。
    2. PPL 路由：对推理输入 x_inf 在每个 expert 上计算 PPL(x_inf|θᵢ) = exp(−1/t·Σ log P(xⱼ|x_{<j},θᵢ))，取 1/PPL 为 confidence，SoftMax(top-K(confidence)) 作为路由权重。
    3. 仅需一次额外 forward pass（远少于 inference 时 generate 多 token 的 forward pass 次数），无 fine-tuning 开销。
  - **算法层 — 异构合并：Projector + Sequence-level Router**：
    1. 共享 embedding/head 层：各 expert 的 embedding/head 参数 padding 零对齐到最大维度 d_m 后取平均。
    2. Proj-inᵢ: R^{d_m}→R^{dᵢ} 和 Proj-outᵢ: R^{dᵢ}→R^{d_m}（随机初始化 MLP），为每个异构 expert 提供维度适配。
    3. Sequence-level routing：因异构 expert 的 attention 层不能合并，所有 token 必须路由到同一 expert。将全部 token embedding 平均 → avg_e = 1/t·Σ eⱼ → router θ_r·avg_e → SoftMax(top-K) 做序列级路由。
    4. Fine-tune 所有参数（含 projector + router）。
  - **系统框架层**：基于标准 PyTorch 实现，与 BTX 相同。同构合并 fine-tuning 阶段仍需多 GPU MoE 并行；无 fine-tuning 模式避免了分布式训练开销；异构合并因不合并 attention 导致总参数略多于 BTX（~4B vs ~3.7B），fine-tuning 成本相应增加。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明具体 GPU 型号。
  - 对比 baseline 的改进映射：
    - **参数干扰 → Dare/Ties 替代平均**：Ties 通过符号剪枝 + 主导方向选择消除 sign conflict，Dare 通过随机 drop + rescale 避免大 magnitude 被小值稀释。Table 1 显示 Dare merging 平均 12.86 vs BTX 11.72（+9.72% 相对提升），Ties merging 平均 12.52（+6.94%）。Figure 10 显示 Dare/Ties 在 fine-tuning 早期阶段优势更明显（早期 token 数少时性能差距大），随着 fine-tuning 进行差距缩小但始终优于 BTX。路由分析（Figure 5）表明 Dare/Ties 合并的 MoE 更准确地将 token 路由到领域专家（如 GSM8K 上 Math Expert 路由概率从 BTX 的 0.28 升至 Dare 的 0.46）。
    - **Fine-tuning 成本高且不可行 → PPL 路由 + 分离 Attention 实现无 fine-tuning MoE**：PPL 路由仅需一次额外 forward pass（overhead 极低，因 inference 时 forward pass 数 = generate token 数 >> 1），Table 2 显示 PPL 路由能有效将各 benchmark 的输入导向对应专家（GSM8K: Math 43%, Knowledge 32%；HumanEval: Code 43%, Math 45%）。分离 attention 解决了合并 attention 和 FFN 的 task vector 数量不一致问题，Table 3 显示 separate attention + PPL routing 平均 8.08 vs merge attention + PPL routing 平均 7.32（+10.4% 相对提升），且优于 SoTA dense merging（Dare Dense 7.11）。
    - **无法处理异构专家 → Projector + Sequence-level Router 实现异构合并**：Proj-in/Proj-out 提供维度桥接（类似 Roberts et al. 2024 的 dense 模型异构合并），sequence-level router 因异构架构 attention 不能合并而采用序列级路由。Table 4 显示 MoE w/ Math TinyLlama 平均 13.34 vs 3-expert MoE (same data) 10.54（+26.6%），MoE w/ Math Olmo 平均 11.17 vs 3-expert MoE 10.54（+6.0%），证明了异构合并的有效性。局限性：异构合并因 embedding 层平均导致 router 在 math benchmark 上不一定将最高路由概率给 math expert（Figure 6），论文建议未来添加 load balancing loss 解决。

## MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

- baseline方法是什么？
  Baseline 为 **M-SMoE**（Li et al. 2023），它从传统的"参数合并"视角出发压缩 MoE 模型。以 M-SMoE 压缩 Qwen1.5-MoE-A2.7B（60 experts → 30 experts per layer, layers 10-23）为例说明全栈执行：
  - **算法层**：M-SMoE 的核心流程：(1) 基于 expert 参数相似度聚类（如将 60 个 experts 聚为 30 组）；(2) 簇内对三个权重矩阵 W_D, W_G, W_U 分别做使用频率加权的参数平均，得到合并后的 expert 权重；(3) 路由权重取簇内原始路由权重之和。这一过程等价于 MergeMoE 框架下的 T_1 = [I; I; ...; I]（拼接单位矩阵）, T_2 = T_3 = 加权平均矩阵（式 4），但 T_1 直接从参数平均导出而非优化得到。
  - **系统框架层**：标准 PyTorch 实现，MoE layer 的 forward pass 不变（router → top-K expert selection → expert FFN → weighted sum）。合并后的模型以标准 HuggingFace 格式加载和推理。使用 DCLM 框架评估。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA）。
  - **kernel 调度层**：论文未明确说明。合并后的模型推理使用标准 PyTorch CUDA kernel，无自定义 kernel。
  - **硬件架构层**：NVIDIA H20 96GB GPU。合并和推理无特殊硬件要求。
  - M-SMoE 的核心缺陷在于 T_1, T_2, T_3 未经过量化优化——T_2/T_3 的加权平均权重基于经验选择（使用频率）而非最优性证明，T_1 直接从参数平均等价得到而非优化，导致压缩后的 expert 输出与原始 expert 输出的线性组合间存在可优化的残差。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MergeMoE 将 expert merging 重新解释为**输出合并**视角下的线性优化问题，通过理论分析 + 最小二乘法求解最优压缩矩阵，系统性减少逼近误差。全栈执行路径（以 Qwen1.5-MoE-A2.7B, 60→30 experts, layers 10-23 为例）：
  - **算法层 — 输出合并视角的优化框架**：
    1. **理论重构（§3.2）**：将 merging 过程重新表述为在前向计算中插入矩阵 A（求和矩阵，式 2）和 B（加权矩阵）及维度缩减矩阵 T_1, T_2, T_3（式 3）。合并后 expert 的输出 = W'_D T_1 (σ(T_2 W'_G X) ⊙ (T_3 W'_U X))，目标是最小化与原始 experts 输出线性组合的 Frobenius 误差。
    2. **权重最优性证明（Theorem 1）**：在假设 router logits 与 expert 输出独立的条件下，减少 experts 的 Y(BA-I_N) 误差下界，**严格证明**簇内使用相对频率 f_j / Σ f_k 作为合并权重是最优的——而 M-SMoE 仅凭经验选择此方案。
    3. **T_2/T_3 优化**：改用 W_U 和 W_G 的拼接相似度作为聚类度量（M-SMoE 用整体参数相似度），使 T_2/T_3 的加权平均在更相似的 W_G/W_U 间进行，减少非线性和 Hadamard 积引入的误差。
    4. **T_1 最小二乘优化（式 5-6）**：固定 T_2/T_3 后，通过采样输入 X̂ 计算 P = σ(T_2 W'_G X̂) ⊙ (T_3 W'_U X̂) 和 Q = σ(W'_G X̂) ⊙ (W'_U X̂)，对线性系统 T_1 P = Q 求 Moore-Penrose 伪逆闭式解 T_1 = Q P†。这是 M-SMoE 完全缺失的步骤——M-SMoE 等价于 T_1 = [I; I; ...; I]（仅做拼接，不做维度缩减优化）。
    5. **路由权重更新**：合并后路由权重 = A · 原始路由权重（与 M-SMoE 相同，因求和矩阵 A 由聚类唯一确定）。
  - **系统框架层**：基于 PyTorch，逐层反向遍历执行压缩。使用 torch hooks 获取中间激活 → GPU 内存中 BFloat16 最小二乘计算 → 释放内存。单一 H20 GPU 完成全流程，每层 <1 分钟。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。无自定义 GPU kernel。
  - **硬件架构层**：NVIDIA H20 96GB（合并用 1 卡，评估用 2 卡）。无硬件修改。
  - 对比 baseline 的改进映射（以 Qwen1.5-MoE 压缩为例，Table 2）：
    - **M-SMoE 的 T_1 未优化 → MergeMoE 的最小二乘 T_1**：M-SMoE 中 T_1 = [I; I]（无优化的拼接）→ MergeMoE 通过 QP† 在采样输入上最小化 T_1 P - Q 的残差。Table 2 结果：WinoGrande 70.48 vs 68.98 (+1.50), Hellaswag 71.58 vs 68.87 (+2.71), SQuAD 56.40 vs 54.99 (+1.41), MRPC 74.75 vs 72.30 (+2.45)，在所有 benchmark 上一致优于 M-SMoE。
    - **M-SMoE 的权重选择仅凭经验 → MergeMoE 的理论最优性证明（Theorem 1）**：同等聚类条件下，使用频率加权被证明是误差下界的最优解，使权重分配有理论保证。
    - **M-SMoE 的聚类度量不够精细 → MergeMoE 的 W_U||W_G 拼接距离**：聚类时仅关注与 T_2/T_3 直接相关的 W_U 和 W_G，减少 T_2/T_3 加权平均在非线性激活 σ 和 Hadamard 积 ⊙ 处引入的误差。消融实验（Table 5, w/o merging errors）验证了聚类误差和合并误差的分离。
    - **输入样本的敏感性**：MergeMoE 的最小二乘法存在样本数临界阈值（~32），低于阈值时性能崩溃（random guessing），但高于阈值后性能持续提升（Figure 4）。跨数据集泛化实验（Table 4）表明即使使用单一数据集采样，性能下降也很小（如 Hellaswag: 71.56 self-sourced vs 71.58，仅差 0.02）。
  - 局限性：MergeMoE 比 M-SMoE 慢（因最小二乘法），且在极低样本量下性能崩溃，需要保证足够的输入样本量。

## Making MoE-based LLM Inference Resilient with Tarragon

- baseline方法是什么？
  Baseline 为 **MegaScale-Infer**（解耦 attention-expert 部署）和 **vLLM**（单体部署）的粗粒度故障恢复机制。以 MegaScale-Infer 为例说明全栈执行路径：
  - **算法层**：MoE 推理执行 layer-wise synchronized 的前向传播。Decoding 阶段，每个 AW 对 layer ℓ 执行 attention 计算（更新 KV cache），gating network 选 top-k expert，通过 NCCL all-to-all 或 M2N 通信将 token embeddings 发送到对应 EW。EW 聚合同 layer 同 expert 的 tokens 成 batch 执行 FFN，返回结果给 AW。AW 等待所有 expert 返回后才进入 layer ℓ+1。此过程每层重复，形成严格的同步屏障。
  - **系统框架层**：MegaScale-Infer 使用 vLLM 作为 compute engine，AW 与 EW 之间通过 M2N 或 NCCL all-to-all 通信。Expert placement 是**静态**的——每个 logical expert 被永久绑定到一个物理 EW/GPU，路由固化在 datapath 中。当任一 worker（AW 或 EW）故障时：CCL/NCCL 将 worker set 视为静态通信组，任一 worker 不可用即 abort 整个 communicator，触发所有 worker 被 kill 并 restart。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：EW 使用标准 PyTorch CUDA kernel 执行 expert FFN（libtorch），按 layer-wise batch 聚合 tokens。AW 侧 attention kernel 为 vLLM 内置。无故障感知的 kernel 调度。
  - **硬件架构层**：GCP A3 Ultra 节点，8x H200 GPU (141 GB)，8x 400 Gbps ConnectX-7 RDMA，NVLink 3.6 Tbps。故障后整个 pipeline 停滞 64 秒（包含 worker restart ~18.5s + 重放 prefill + 重放 decoding）。
  - Baseline 核心缺陷：
    1. **粗粒度故障域**：单个 worker 故障导致整个推理任务被 tear down 并 restart，所有 in-flight 状态（KV cache、partial outputs）被丢弃。在 40 节点（320 GPU）部署中，任意时刻至少 1 个节点故障的概率约为 18.1%。
    2. **用户可见 stall**：故障恢复期间 pipeline 完全停止产生新 token（T_stall ~64s for MegaScale-Infer），对交互式 LLM 服务造成严重用户体验退化。
    3. **浪费的计算**：KV cache 和 expert outputs 被丢弃并从零重算。已 decoding 的 token 越多，GPU 计算浪费越大（与已 decoding token 数 i 呈线性增长）。
    4. **静态 expert placement**：expert 与物理 GPU 绑定，该 GPU 故障直接导致对应 expert 不可用，必须等待 replacement worker 启动。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TARRAGON 通过**可重构数据通路 + 双向自愈 + 异步 KV cache checkpointing**三层次设计解决上述缺陷。全栈执行路径（以 decoding 阶段 AW 故障为例，Mixtral-8×7B, 8 AWs + 8 EWs）：
  - **算法层 — 异步增量 KV Cache Checkpointing + Per-Request Restoration**：
    1. 每个 AW 在每层 attention 完成后，利用 AW-EW 通信间隙（link idle 时段，如图 8 的 bursty 流量模式所示），通过 one-sided RDMA write 将新产生的 KV cache segment（每 token 每层一个小 segment，大小为 C = 2×H_kv×(N_hidden_size/H_attn)×S_elem，对 Mixtral-8×7B 约为 expert traffic V 的 12.5%）异步写入 checkpoint store。使用 "async log + commit record" 设计——每个 RDMA write 携带单调递增的 work request ID 作为 sequence number 保证顺序。
    2. AW 故障时，Orchestrator 识别该 AW 上所有活跃请求及其最后 committed token。对于每个被重分配的请求，checkpoint store 通过 GPUDirect one-sided RDMA write 将完整 KV cache segments 直接注入替代 AW 的 GPU 显存。替代 AW 从 committed token+1 开始继续 decoding，无需重放任何 prefill/decoding。
  - **系统框架层 — 可重构数据通路 (REFE + ERT)**：
    1. REFE（Reconfigurable Forwarding Engine）是 AW 侧 C++ 扩展，通过双 QP 设计（control-plane QP for liveness probe + data-plane QP for token embeddings via GPUDirect RDMA）替代 NCCL/M2N 的静态通信模式。对外暴露 `expert_io(expert_id, layer_id, token_embeddings)` API。
    2. ERT（Expert Routing Table）将 logical expert ID 与 physical EW/GPU **解耦**，每个 AW 独立维护一份 ERT，由 Orchestrator 动态更新。故障时只需更新 ERT 条目即可实现流量重路由，无需全局重启。
    3. **AW 侧自愈**：REFE 对 EW 响应设超时→探测 liveness→立即重路由到健康 EW 或 shadow expert→重播 token embeddings。因 expert 计算 deterministic（纯函数），重播产生相同结果。
    4. **EW 侧自愈**：EW 不再等待所有 AW 输入。当收到"足够"健康 AW 的 tokens（或 batch 达最小区间）即开始 expert FFN，将未响应 AW 的 slots 从 batch 中省略。消除全局同步屏障。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层 — Shadow Experts**：
    1. 在每个 EW GPU 显存中预加载 inactive expert 副本（shadow expert），包含与 primary 相同的权重和计算 kernel。Primary 故障时立即激活，避免从存储重新加载权重（数百毫秒到秒级延迟）。
    2. Shadow expert 在无故障时不消耗任何 compute 资源（仅占 GPU 显存，约 2.5 GB per expert for DeepSeek-R1），不引入 kernel-level interference（单 expert 执行延迟与"shadow loaded + primary active"时完全一致，如图 14）。
  - **硬件架构层**：GCP A3 Ultra，3 节点（AWs + EWs + Checkpoint store 各 1 节点），8x H200 (141 GB) per node，ConnectX-7 RDMA with GPUDirect。故障后 stall 从 ~64s 降至 0.3s (EW) / 0.4s (AW)，稳态吞吐与 MegaScale-Infer 匹配（偏差 <2.8%）。
  - 对比 baseline 的改进映射：
    - **粗粒度故障域 → Worker 级故障域（REFE + ERT）**：将 expert identity 与物理 location 解耦，故障时仅更新 ERT 条目而非重启所有 worker。任何 AW 可向任何 EW 发送请求，数据通路完全可重构。AW 和 EW 形成独立故障域，互不强制全局重启。
    - **用户可见 stall → 双向自愈消除等待**：AW 侧自愈（EW 故障）：AW 本地超时重路由，不等待 orchestrator，其他 AW 继续前进。EW 侧自愈（AW 故障）：EW 不等待所有 AW，用部分输入开始 expert 计算。结果：stall 从 64s 降至 0.3-0.4s（160-213× improvement）。
    - **浪费的计算（KV cache 重建）→ 异步增量 checkpointing + Per-request 恢复**：AW 故障后无需重放 prefill 和全部 decoding。替代 AW 直接从 checkpoint store 注入 KV cache，从 committed token 继续。恢复延迟比 Sequential replay 低 1800×，传输流量低 8×，GPU 重计算消除。
    - **静态 expert placement → Shadow Expert 快速激活**：EW 故障后 shadow expert 立即接管（已在 GPU 显存中），避免 T_w（18.5s worker 初始化）的等待。后台 provisioning 并行进行，恢复容量时不影响在线推理。
    - **Spurious 无故障开销 → <3%**：异步 checkpointing 利用 link idle 间隙，不干扰正常 AW-EW 流量。Failure detection probing (10ms) 和 ERT remapping 的总 Steady-state overhead <3%（Ablation study, Fig. 15）。

## MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

- baseline方法是什么？
  MOE-style LoRA 方法（如 MOELoRA、LoRAMoE、MoCLE 等）是 baseline。这些方法将每个 LoRA 模块内部分解为多个 sub-rank experts（例如 MOELoRA 中将 r=32 的 LoRA 分解为 32 个 single-rank LoRA，每 4 个组成一个 expert，共 8 个 experts per LoRA module），通过 token-wise router 为每个生成 token 动态选择激活的 experts。以 MOELoRA + LLaMA-2 7B 在 multi-tenant serving 下的全栈执行：
  - **算法层**：每层 Transformer 包含 7 个 LoRA 模块（Q/K/V/O/G/U/D），每个模块内有 8 个 sub-rank experts + 1 个 router。生成每个新 token 时，7 个 router 各自计算 top-4 路由概率，调用 4 个激活的 experts 计算 LoRA 增量：x' = xW_m + x·Σ(e_i·W_m^{A,i}·W_m^{B,i})。总计：每 token 每层调用 7 个 router + 7×4=28 个 sub-expert forward。
  - **系统框架层**：HuggingFace Transformers + PEFT 库。LoRA 参数不 merge 回 backbone（multi-tenant 设置下每个 tenant 有自己的 LoRA weights）。每个 generation step 的 forward pass 中额外执行所有 LoRA modules + routers。
  - **编译框架层**：论文未明确说明（标准 PyTorch forward，无编译优化）。
  - **kernel 调度层**：论文未明确说明（标准 PyTorch linear kernel + softmax，无 custom kernel）。
  - **硬件架构层**：NVIDIA A40 GPU (48GB)。
  - **核心缺陷**：(1) 每 token 每层计算 7 个 router + 大量 sub-expert forward，产生显著推理延迟（tps 比 base model 降低约 20%）；(2) token-wise routing 在 multi-tenant 下每个 tenant 请求都需要独立计算 router，延迟随 tenant 数量线性增长；(3) 多任务学习中共享 LoRA parameters 导致任务间数据冲突（单任务 ST 到多任务 MT 性能下降 0.5-2.0%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MiLoRA 通过两个核心设计系统性解决 baseline 的延迟与效率问题，全栈执行如下：
  - **算法层 — Prompt-Aware Routing + Layer-Level Expert Selection**：
    - Expert 粒度提升：每个 LoRA 模块（而非其子结构）被定义为一个 expert，N_mod=7 个 experts per layer。每个 Transformer layer 仅激活 1 个 expert（k=1，通过 Top-k=3 的 softmax 概率分布实现），被选中模块用 LoRA 修正其 output，其余 6 个模块以原始 backbone 权重执行。
    - Router 从 token-wise 降为 prompt-wise：Router 仅在 input prompt 首次经过 backbone 时计算一次路由决策（before the first new token），后续所有 auto-regressive token 生成步骤均复用该决策。Router 计算流程：H^l (prompt hidden states) → SelfAttnPool(·) → Rational Activation Ra(·) → Softmax(W_r^l · h^l) → Top-k。
    - Load Balancing：训练时施加 auxiliary loss L_lb = N_mod · Σ f_i^l · p̂_i^l（λ_lb=1e-2），防止 experts 分配不均衡。
  - **算法层 — Learned Rational Activation Functions**：
    - 替代固定激活函数（ReLU/GeLU 统一用于所有层），使用有理函数 Ra(x) = Σ a_j x^j / (1 + ||Σ b_i x^i||)（m=6, n=5，可学习 a_j, b_i），每层 router 有独立参数。通过 DARTS 风格的 bi-level optimization 训练 activation params（architectural params Θ, lr=1e-6）和 LoRA params（Ω, lr=1e-4）。
    - 效果：不同深度的 Transformer layer 学习到不同的激活函数形态，较统一 ReLU 或 GeLU 有更好的 routing 质量和下游任务表现。
  - **系统框架层**：HuggingFace Transformers + PEFT 库。实现与 MOELoRA 等共用框架，但 adapter 结构更简单（per-layer 1 router + 1 activated LoRA module vs. 7 routers + 28 activated sub-experts）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。
  - **硬件架构层**：NVIDIA A40 GPU (48GB)。
  - 对比 baseline 的改进映射：
    - **token-wise routing → prompt-aware routing**：Router 从每 token 调用 7 次降为整个序列仅调用 1 次（per layer）。MoE 路由开销从 O(L × T × N_mod) 降为 O(L × N_mod)，在 L=32 层 × T=256 tokens 场景下，router 调用次数从 32×256×7=57344 降至 32×7=224。实测推理加速：beam=1 时 tps 43.7 vs MOELoRA 35.9（+21.7%），beam=3 时 tps 33.5 vs MOELoRA 28.4（+17.9%）。
    - **per-module multi-expert → per-layer single-expert activation**：Activated LoRA 参数量从 MOELoRA 的 30.1M（每 token）降至 25.2M（每 prompt 固定选择），且在 generation 阶段仅执行被选中的 1 个 LoRA module（而非 7 个 module 内的多个 experts），减少 memory bandwidth 和 compute 开销。
    - **固定激活函数 → 可学习激活函数**：每层 router 学习最适合其深度的激活函数形态，缓解了深层/浅层对路由敏感性不同的矛盾。Ablation 显示 learnable activation 在 BoolQ/PIQA/MMLU 上均优于固定 GeLU 或 ReLU/GeLU 混合方案。
    - **多任务学习性能保持**：MiLoRA 在 ST→MT 切换中性能几乎不下降（Avg. 75.4→75.2, Δ=-0.1%），而 LoRA/DoRA 分别下降 2.0%/2.2%。MoE routing 机制天然为不同任务/数据选择不同 expert，缓解数据冲突。

## MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

- baseline方法是什么？
  - Baseline 为 LoRA（r=80, alpha=160, 应用于 q,k,v,o + w1,w2,w3）和 DoRA（同配置，weight-decomposed LoRA 变体），均为标准 PEFT 方法，在单个下游任务上独立微调。
  - 全栈执行路径（以 LoRA + LLaMA-2 7B 在 BoolQ 上单任务微调为例）：
    - **算法层**：LoRA 将权重更新分解为低秩矩阵 B·A（B∈R^{d1×r}, A∈R^{r×d2}, r=80），前向计算 W' = W + B·A。单任务微调时，LoRA adapter 仅学习该任务的知识。多任务场景下，同一套 LoRA 参数在混合任务数据上训练，缺乏对不同任务模式的显式分离机制——所有 token 经过相同的一组 LoRA adapter 计算，无任务/ token 级差异化处理。
    - **系统框架层**：HuggingFace Transformers + PEFT 库。单模型训练/推理，无 MoE 路由开销。
    - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。
    - **kernel 调度层**：标准 PyTorch linear kernel，无自定义 CUDA kernel。
    - **硬件架构层**：24GB consumer GPU（RTX 3090/4090/A5000），half precision。
  - Baseline 核心缺陷：
    1. **多任务学习中性能退化**：LoRA 单任务→多任务切换中 average accuracy 下降 4.4%（69.9%→65.5%，Table 2），DoRA 下降 8.0%（74.3%→66.3%）。原因是有限的 trainable parameters 在混合任务上缺乏对不同任务模式的显式分离，导致 catastrophic forgetting 和跨任务干扰。
    2. **模型容量受限**：单套 LoRA adapter 的容量受限于 rank r，无法像 MoE 模型那样通过增加 expert 数量来扩展模型容量。LoRA/DoRA 在 LLaMA-3 8B（78.2%/78.5%）和 LLaMA-2 13B（81.5%/81.9%）之间存在显著差距（3.3%/3.4%），表明 model capacity 是性能瓶颈。
    3. **无 token 级差异化计算**：所有输入 token 经过相同的 LoRA adapter，无法像 MoE 那样根据 token/task 特性选择不同的计算路径。不同任务（如推理任务 ARC vs 知识任务 OpenBookQA）受益于不同的参数特化，但标准 LoRA 无法提供。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MixLoRA 通过"FFN 层 LoRA 专家化 + Top-K 路由 + Attention 层 LoRA 解耦 + 负载均衡"四层设计解决上述缺陷。
  - 全栈执行路径（以 MixLoRA + LLaMA-2 7B 多任务学习为例，K=8 experts, top-2 router, r=16）：
    - **算法层 — MoE 化 FFN 构造**：
      1. 每个 expert = 共享冻结 FFN 权重（W1,W2,W3）+ 独立 LoRA adapter（A^k_W1,B^k_W1, A^k_W2,B^k_W2, A^k_W3,B^k_W3），而非传统 LoRA-MoE 方法中将整个 LoRA 模块作为 expert。这使 MixLoRA 更接近 Mixtral 等预训练 MoE 模型的架构。
      2. Router 为线性层 W_r·x → Softmax → KeepTop-2，为每个 token 选择最优 2 个 expert。
      3. Expert 输出由 router probability 加权求和：h = Σ R(x)_k · (W·x + B_k·A_k·x)。
      4. Self-attention 层使用独立的 LoRA adapter（q,k,v,o 投影），不参与 MoE 路由——因为 ST-MoE 研究表明微调 attention 层可显著提升 MoE 模型性能。
      5. Auxiliary load balance loss（a=1e-2）确保 8 个 expert 负载均衡（平均 std dev 0.0223）。
    - **算法层 — 性能优化**：
      - **共享计算**：先计算 W1·x 和 W3·x，再按路由权重切片给各 expert 的 LoRA 计算，避免每个 expert 重复计算 FFN 骨干。W2 无法共享因其依赖 W1/W3 输出。
      - **多模型高吞吐**：多个 MixLoRA 模型的输入合并为一个 batch，共享预训练权重，各模型独立路由。
    - **系统框架层**：基于 HuggingFace Transformers + PEFT 库实现。优化使用了 m-LoRA 风格的多 LoRA 并行技术。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明（标准 PyTorch CUDA kernel，优化通过减少冗余计算而非自定义 kernel 实现）。
    - **硬件架构层**：24GB consumer GPU（RTX 3090/4090/A5000），half precision。
  - 对比 baseline 的改进映射：
    - **多任务学习中性能退化 → MoE 路由实现任务/token 级计算路径分离**：不同 token 被 Top-2 Router 分配到不同 expert 组合，每个 expert 通过 LoRA adapter 学习到不同的参数特化。MixLoRA 多任务学习 accuracy 仅下降 -0.6%（74.7%→75.3% 反而提升），而 LoRA 下降 -4.4%、DoRA 下降 -8.0%。MoE 路由机制天然缓解了多任务间的 data conflict 和 catastrophic forgetting——不同任务的数据倾向于被路由到不同的 expert 子集（Figure 5 显示 8 个 expert 负载均衡，各任务间分布均匀）。
    - **模型容量受限 → MoE 结构以低成本扩展模型容量**：8 个 expert 的 MixLoRA（r=16, 2.9% trainable params）在 LLaMA-3 8B 上取得 83.5% avg accuracy，超过 LLaMA-2 13B 上 LoRA 的 81.5%。每个 expert 通过独立 LoRA 提供不同的参数特化，总体可学习参数量虽与 baseline LoRA（r=80）相近，但通过 MoE 路由实现了条件计算——不同 token 使用不同的参数子集，有效扩展了模型的表征能力。
    - **无 token 级差异化计算 → Top-2 Router 实现 token-wise dynamic routing**：Router 为每个 token 独立计算 expert assignment，不同 token 被分配给不同 expert 对（top-2 from 8），实现细粒度的计算路径差异化。Ablation 显示 rank=16 且 8 experts 的 MixLoRA 优于 rank=32 单体的变体——说明条件计算带来的收益超过了单体 rank 增加。
    - **LoRA 容量 vs MoE 效率的 Pareto 改进**：朴素 MixLoRA token 计算延迟 535.2 µs（LoRA 的 218%），但共享计算优化后降至 462.5 µs（188%），同时 accuracy 远超 LoRA。多模型模式下 per-model GPU memory 从 15.1GB 降至 8.8GB（训练），为 consumer GPU（24GB）上同时微调多个模型提供了可能。
    - **DoRA 兼容性（MixDoRA）**：将 expert 基础单元从 LoRA 替换为 DoRA，MixDoRA 在部分场景下（如 Gemma 2B 单任务 71.6%）优于 MixLoRA（69.9%），但在 LLaMA-2 7B 多任务中 MixDoRA（74.9%）与 MixLoRA（75.3%）性能相近，且 MixDoRA 对负载均衡 loss coefficient 更不敏感。

## FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

- baseline方法是什么？
  Baseline 为 **Megatron-LM 原生的 Expert Parallelism (EP)**，以及基于 expert scheduling 的 SmartMoE 和 FlexMoE。以 Megatron-LM 的 vanilla EP 为例说明全栈执行路径（GPT 32×1.3B, DP=8, EP=4）：
  - **算法层**：每层 MoE 包含 self-attention（DP）→ gate network（top-2 routing）→ token dispatch（EP group 内 all-to-all）→ expert FFN（SwiGLU）→ token combine（all-to-all）→ residual。Token 的 GPU 分配完全由 gate network 决策决定——每个 token 被路由到 top-2 expert 所在的 GPU，无法调整。
  - **系统框架层**：Megatron-LM 3D 并行（DP + TP + PP），EP 将 experts 分布在 EP group 内（每 group 含每个 expert 恰好 1 个 replica）。All-to-all 通信限定在 EP group 内（size=4），通信时间与 expert 计算时间串行执行（通信期间 GPU 空闲）。
  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel + NCCL）。
  - **kernel 调度层**：NCCL all-to-all collective + PyTorch CUDA kernel（batched GEMM for expert FFN）。无自定义通信-计算重叠 kernel。
  - **硬件架构层**：NVIDIA H100 80GB SXM GPU, 900 GB/s NVLink intra-node, 400 Gbps InfiniBand inter-node。
  - **核心缺陷**：
    1. **Token-to-GPU 固定映射无调度空间**：EP group 内每个 expert 仅 1 个 replica，token 必须计算在 gate 选中的 expert 所在 GPU 上。GPU load 由 expert load 固定，无法通过调度调整。
    2. **Straggler 效应**：最重负载的 GPU 成为瓶颈（straggler），所有 GPU 等待其完成 all-to-all 同步。Expert load 分布动态变化且高度偏斜（training 初期尤其严重），每个 micro-batch 都产生 GPU 空闲。
    3. **Expert scheduling 粗粒度**：SmartMoE 和 FlexMoE 通过调整 expert-to-GPU placement 实现 load balancing，但（a）以 expert 为调度单元导致离散有限调度空间，无法实现最优均衡；（b）placement 调整需迁移大量 expert 参数，无法 per-micro-batch 适应动态 load 变化；（c）SmartMoE 基于长期 load 分布优化，面对 micro-batch 间波动时反而可能劣于 Megatron-LM。
    4. **DeepSpeed 的 padding 浪费**：DeepSpeed 将每个 expert 的 load padding 到最大 expert load，在 load 高度不均衡时浪费大量计算和内存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FineMoE 通过 **token scheduling（替代 expert scheduling）+ graph-theoretic expert placement** 双层设计解决上述缺陷。全栈执行路径（GPT 32×1.3B, DP=8, EP=4, d=2）：
  - **算法层 — Token Scheduling（FineEP, §5）**：
    1. **扩展调度空间**：合并 d=2 个 EP group 为 1 个 FineEP group（8 GPU），利用 Expert Data Parallelism——每个 expert 在多个 GPU 上有 replica（同一 EDP group），token 可选择任一 replica 计算。
    2. **Shuffle Expert Placement**：打乱 EP group 间的 expert placement（如图 3c），使不同 experts 的 EDP groups 交叉，扩大调度空间。例如 expert 0 在 GPU {0,2}，expert 1 在 GPU {0,1}（交叉），而非 expert 0 和 expert 1 都在 GPU {0,2}（无交叉）。
    3. **LPP 建模（§5.1）**：每 micro-batch 将 load balancing 建模为线性规划问题——变量 `x_e^g`（expert e 在 GPU g 的 replica load），约束 `Σ_g x_e^g = load_e`，目标 `min max_g Σ_e x_e^g`。使用 HiGHs 求解器在 CPU 上 warm-start 求解（变量数 O(|E|d)，~100 μs 到 <1 ms）。
    4. **Locality-Aware Routing（§5.2）**：优先将 token 路由到本地 GPU 上的 replica（减少 all-to-all 通信），再路由到 remote replica。
    5. **Distributed + Overlapped Scheduling（§5.3-5.4）**：所有 GPU all-gather 收集 load 信息 → 各 GPU 独立运行确定性调度 → CPU 调度与 GPU token permutation 重叠。
  - **算法层 — Graph-Theoretic Expert Placement（§6）**：
    1. **Symmetric Placement（§6.2）**：无先验 load 知识时，用 Cayley graphs 构造对称 placement（如 8 GPU + 8 experts → cycle graph），保证 max induced subgraph density 最小化。
    2. **Asymmetric Placement（§6.3）**：已知 load 分布时，greedy 确定各 expert 的 replica counts + Monte Carlo sampling 选择最优 placement graph（Equation 3: m = max_{G_max} (1/|G_max| · Σ load_e)）。
    3. **Adaptive Replacement（§6.4）**：后台监控 load → 时间序列预测 → Equation 3 评估 → 触发 placement 更新，token scheduling 负责 fine-grained 均衡，adaptive replacement 处理 coarse-grained 偏差。
  - **系统框架层**：
    1. 基于 Megatron-LM 实现：修改 MoELayer forward（插入 Token Dispatcher）→ 扩展 all-to-all 通信组大小 → 新增 Placement Manager（Python + C++ token scheduling）。
    2. 实现了 Distributed Scheduling 和 Overlapping（利用 Megatron-LM 的 token permutation 阶段）。
    3. 额外实现 SmartMoE 和 FlexMoE 在 Megatron-LM 上用于公平对比。
    4. 集成 DeepEP（high-performance all-to-all backend）和 Pipelining FineEP。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：HiGHs 求解器在 CPU 上执行（单 thread），LPP 求解和 token routing 结果与 GPU CUDA kernel 通过 overlapping 和 warm-start 降低开销。NCCL all-to-all 或 DeepEP 用于通信。无自定义 GPU kernel。
  - **硬件架构层**：4 节点，32×H100 80GB GPU，900 GB/s NVLink + 400 Gbps InfiniBand。

  - 对比 baseline 的改进映射：
    - **Token-to-GPU 固定映射 → Token Scheduling 提供细粒度调度空间**：Vanilla EP 的 token-GPU 映射固定 → FineEP 通过合并 EP groups + shuffle expert placement 创建交叉 EDP groups，使每个 expert 的 token 可在多个 GPU 间选择。调度空间从无到有（O(|E|d) 个变量），可实现 per-micro-batch 的细粒度负载均衡。
    - **Straggler 效应（GPU idle）→ LPP 最小化 max load**：LPP 1 的优化目标直接最小化最大 GPU load（straggler），在 load skewness s<1 时 FineMoE (w/o AR) 即能完美均衡所有 GPU（max_load/avg_load = 1.0）。端到端加速最多 47.6% vs Megatron-LM。
    - **Expert scheduling 粗粒度&动态性不足 → Token scheduling 细粒度&per-micro-batch**：Expert scheduling（SmartMoE/FlexMoE）以 expert replica 为调度单元（离散有限空间，placement 调整缓慢）→ token scheduling 以单个 token 为调度单元（连续优化空间，per-micro-batch 调整）。FlexMoE 的 replica count 调整需迁移参数 → FineMoE 的 token scheduling 仅需 all-gather load 信息（~数 us）+ LPP 求解（~100 μs）。结果：FineMoE 在所有条件下优于 FlexMoE 和 SmartMoE。
    - **DeepSpeed padding 浪费 → 根本无需 padding**：DeepSpeed 通过 padding 使 expert loads 相等 → FineMoE 通过 token scheduling 使 GPU loads 相等，不 padding，不浪费计算。
    - **Long-term 部署 → Graph theory 指导 placement**：Symmetric placement（Cayley graphs）为 unknown load 提供数学最优保证——max induced subgraph density 最小化，保证最坏情况下的均衡能力。Asymmetric placement + Adaptive Replacement 为 known/evolving loads 提供持续优化——greedy replica count 基于 load-per-replica + Monte Carlo 图密度最小化。

## Mixture of A Million Experts

- baseline方法是什么？
  Baseline 包括三种：**Dense FFW**（标准 Transformer 中每个 token 激活所有 FFW 参数）、**Coarse-grained MoE**（expert-choice routing, 128 experts, 每个 expert 大小等于原 dense FFW）、**PKM**（product key memory, 检索记忆向量而非可学习函数）。

  以 Dense FFW baseline 的全栈执行路径为例（一个 token 的推理）：
  - **算法层（Dense FFW）**：输入 x ∈ R^(d_model)，经过两层 MLP：h = W_1 x + b_1, y = W_2 σ(h) + b_2。所有 d_model × d_ffn × 2 个参数均参与计算，FLOPs = O(d_model × d_ffn)。参数数量 P 与计算量 C 线性耦合：C ∝ P。增加模型容量直接导致计算开销同比例增加。
  - **系统框架层**：标准 Transformer 训练/推理框架（如 JAX/FLAX 或 PyTorch），FFW 层作为密集矩阵乘法执行。batch 中的每个 token 独立执行相同的 FFW 计算，无稀疏激活。内存占用 = 模型参数 + 激活值 × batch_size × seq_len（激活内存随 batch/seq_len 线性增长）。
  - **编译框架层**：标准 XLA/TVM 编译，将 FFW 矩阵乘法映射为 GPU/TPU 上的 matmul kernel。无特殊编译优化。
  - **kernel 调度层**：使用高度优化的 BLAS 库（cuBLAS/TPU matmul）执行密集矩阵乘法。所有 d_model × d_ffn 权重在每个 token 上都被读取和计算，无 kernel 级稀疏优化。
  - **硬件架构层**：标准 GPU/TPU，矩阵乘法在 Tensor Core/TPU MXU 上执行。内存带宽需求 = 读取 W_1, W_2 + 写入激活值，随模型增大线性增长。
  - Baseline 核心缺陷：
    1. **计算与参数的线性耦合**：Dense FFW 的 FLOPs 和激活内存随 hidden width 线性增长。增大模型容量（更多参数 P）必然导致更高的计算开销。
    2. **Coarse-grained MoE 的专家数量受限**：token-choice 和 expert-choice routing 均需在 N×M 的 gating score 矩阵上执行 top-k 操作，路由复杂度至少 O(N)，限制专家数量通常 < 128。Fine-grained MoE scaling law 预测更高粒度（更多更小专家）可带来更好性能，但现有 MoE 无法扩展到百万级专家。
    3. **PKM 检索记忆向量而非函数**：PKM 检索的是静态记忆向量（不依赖输入变化），而非输入依赖的 expert 网络。记忆向量的表达能力远弱于可学习函数，无法利用输入信息动态调整输出。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PEER 通过 **Product Key 检索路由 + 单神经元 Expert + Multi-Head 检索** 三层设计解决上述缺陷。全栈执行路径（一个 token 通过 PEER 层，以 N=1024², h=8, k=16 为例）：

  - **算法层（Product Key + Singleton Expert）**：
    1. **Query 投影**：输入 x ∈ R^(d_model) 经 h=8 个独立 query network 映射为 8 个 query 向量 q^i(x)，每个 query 维度为 d（product key 维度）。
    2. **Product Key 检索**：每个 query 拆分为 q_1, q_2 ∈ R^(d/2)。在两个子密钥组 C, C'（各 1024 个 d/2 维向量）上分别 top-k=16 检索，获 16²=256 个候选 product keys。计算 q_1^T c_i + q_2^T c'_j 并再次 top-k=16 选出最终 expert 索引。**路由复杂度 O((√N + k²)d) = O((1024 + 256)d)，而 naive 方法为 O(10^6 × d)，加速约 1000×**。
    3. **Expert 计算**：每个 expert e_i(x) = σ(u_i^T x) v_i —— 单神经元（d_expert=1），仅需 2×d_model 个参数。通过 Embedding lookup 检索 u_i, v_i 权重。h×k = 128 个 active expert 共享 expert pool，动态组装成等效 128 神经元 MLP：f(x) = Σ_i Σ_j g_j(x) σ(u_j^T x) v_j。
    4. **Router 权重聚合**：scores 经 softmax 归一化后与 expert 输出相乘求和。总 active 参数 P_active = hk × 2d_model，总参数 P = N × 2d_model = 10^6 × 2d_model。
  - **系统框架层**：PEER 层可直接替换 Transformer 中任意 FFW 层，替换后在 JAX 中以 embedding lookup + einsum 操作执行。训练时 expert 权重存储在 Embedding 层中（类似大词表），通过索引检索而非全量矩阵乘法。batch 扩展时，只有 P_active（active 参数）随 batch_size × seq_len 增加激活内存，P（总参数）仅存储一份。
  - **编译框架层**：论文未明确说明（使用 JAX/XLA 的 embedding lookup + einsum 算子）。论文指出高效实现需要 specialized hardware kernels 加速 embedding lookup 与 einsum 的融合。
  - **kernel 调度层**：核心操作为 embedding lookup（类似大词表查表）和 batched einsum（小矩阵批量乘法）。与 dense FFW 的大矩阵乘法（matmul）不同，PEER 的 compute pattern 是大量独立的小 inner product + 加权求和，对 GPU/TPU 的 Tensor Core 利用率可能偏低。论文未实现专门的 kernel 优化。
  - **硬件架构层**：标准 GPU/TPU。总参数 P（10^6 × 2d_model）存储在高带宽内存中，每个 token 仅需读取 hk 个 expert 的权重（128 × 2d_model），推理时内存带宽需求远低于同参数量的 dense 模型。

  - 对比 baseline 的改进映射：
    - **Dense FFW 的计算-参数耦合 → PEER 解耦**：Dense: P_active = P = O(d_model × d_ffn)。PEER: P_active = hk × 2d_model = 128 × 2d_model（固定），P = N × 2d_model = 10^6 × 2d_model（独立扩展）。增加 N 仅增加参数存储（无额外计算），提高模型容量而不增加 FLOPs。
    - **Coarse-grained MoE 的 O(N) 路由 → PEER 的 O(√N) 路由**：token-choice/expert-choice 须在 N×M gating matrix 上 top-k（min O(N log k)），限制 N < 128。PEER 的 product key 将检索分解为两个 √N 候选集的笛卡尔积，O(√N + k²) 复杂度，支持 N ≥ 10^6。
    - **PKM 的静态记忆 → PEER 的输入依赖函数**：PKM 检索记忆向量 v_i（固定值），输出为 Σ g_i v_i。PEER 检索 expert 函数 e_i(x) = σ(u_i^T x) v_i，输出为 Σ g_i σ(u_i^T x) v_i —— 多了 u_i^T x 的非线性变换，使每个 expert 的输出依赖输入 x。**等价于从"检索记忆"升级为"检索可学习函数"**。
    - **Shared expert 参数 → 知识迁移和参数效率**：multi-head retrieval 共享同一 expert pool，不同 head 可以检索到相同或不同的 expert，隐式实现 expert 间 hidden neuron 共享，提升参数效率和知识迁移。
    - **Expert 负载不均衡 → Query BatchNorm**：直接使用产品密钥可能导致某些 expert 被频繁选中、其他闲置。在 query 上添加 BatchNorm 层使 query 分布更均匀，expert 使用率接近 100%（表 2），unevenness（KL 散度）显著降低。

## Mixture of LoRA Experts

- baseline方法是什么？
  Baseline 为 **Normalized Linear Arithmetic (NLA) Composition**（Eq.2）：对 N 个已训练的 LoRA 进行加权求和 $\hat{\boldsymbol{W}} = \boldsymbol{W} + \sum_{i=1}^{N} w_i \cdot \Delta \boldsymbol{W}_i$，其中 $\sum w_i = 1$。以 DreamBooth + Stable Diffusion V2.1 三概念多主体生成（Dog + Cat + Sunglasses）为例的 LoRA 组合全栈执行路径：
  - **算法层（LoRA 组合推理）**：对每个 Transformer block，所有 N 个 LoRA 的增量权重 $\Delta \boldsymbol{W}_i = A_i B_i$（rank decomposition）按全局统一标量 w_i 线性叠加到预训练权重 W 上，形成组合权重 $\hat{\boldsymbol{W}}$。单个 forward pass 实质等价于用 $\hat{\boldsymbol{W}}$ 进行一次标准推理。Attention 的 Q/K/V/O projection 和 FFN 的 fc1/fc2 均按相同的 {w_i} 权重组合。**权重组合是静态的**：w_i 在推理前确定、所有层共享，不存在层间差异。**无反向传播**：NLA 是纯前向算术操作，不涉及训练或梯度。
  - **系统框架层**：论文未明确说明。标准 Stable Diffusion pipeline（Diffusers/HuggingFace），LoRA 权重通过 PEFT 库加载和 merge（merge_and_unload 或 set_adapter 方式），无调度或并行策略定制。
  - **编译框架层**：论文未明确说明（标准 PyTorch）。
  - **kernel 调度层**：论文未明确说明。合并后的权重矩阵进行标准 GEMM 计算，无定制 kernel。
  - **硬件架构层**：论文未明确说明 GPU 平台。
  - Baseline 核心缺陷：
    1. **全局统一权重导致 LoRA 特性稀释**：所有层使用相同的组合权重 w_i，但不同层的 LoRA 参数编码了不同特征（如 Observation 2 所示：NLP 中 LoRA 的 0%-20% 层擅长 QNLI，80%-100% 层擅长 ANLI-R1）。当 N≥3 时，归一化将每个 LoRA 的 w_i 压缩到 1/N，导致关键层中的区分性特征被平均噪声淹没。
    2. **组合灵活性差**：一旦确定 {w_i}，无法在不重新计算所有权重的情况下增删 LoRA。若要排除某个 LoRA，需重新归一化剩余权重。
    3. **缺乏数据驱动的适应性**：NLA 的权重 w_i 由人工指定或启发式搜索（如 LoRAHub 的 gradient-free optimization），无法根据具体下游数据自适应调整。
    4. **直接算术组合（Eq.1 w/o 归一化）在 N 增大时破坏生成能力**：如 Fig. 3 I 所示，直接叠加 3 个以上 LoRA 会导致生成图像无意义输出；NLP 中 FLAN-T5 组合 4+ LoRA 输出混乱。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MOLE 通过 **hierarchical weight control（逐层可学习 gating function）+ gating balancing loss + 双推理模式** 解决上述缺陷。以 DreamBooth + Stable Diffusion V2.1 三概念生成（Dog + Cat + Sunglasses）为例的 MOLE 全栈执行路径：
  - **算法层（MOLE 逐层组合）**：
    1. 预训练 block 前向：$F_\theta(x) = \text{Attn}(\text{LN}(x|\theta)) + \text{FFN}(\text{LN}(\cdot|\theta))$（Eq.5-6）
    2. 每个 LoRA expert i 的独立前向：$E_{\Delta\theta_i}(x) = \text{Attn}(\text{LN}(x|\Delta\theta_i)) + \text{FFN}(\text{LN}(\cdot|\Delta\theta_i))$（Eq.7-8）—— 为每个 LoRA 单独计算 full block 输出（float32 精度，N 路并行或串行）。
    3. Gating 函数逐层计算权重：concat 所有 $E_{\Delta\theta_i}(x)$ → Normalization → flatten → dot-product 映射到 N 维（Eq.9-10）→ softmax 归一化（Eq.11, learnable temperature τ）→ gating values $\mathcal{G}_i$。
    4. 加权组合：$\tilde{E}_\Omega(x) = \sum_i \mathcal{G}_i \cdot E_{\Delta\theta_i}(x)$（Eq.12）。
    5. 残差融合：$O(x) = F_\theta(x) + \tilde{E}_\Omega(x)$（Eq.13）。
    6. **训练阶段**：仅优化 gating function 参数（e 和 τ），冻结所有 LoRA 和预训练模型权重。V&L 域使用 CLIP local+global guidance 作为无监督训练信号（L_CLIP），NLP 域使用 FLAN-T5 的 cross-entropy。同时施加 gating balancing loss $\mathcal{L}_{\text{balance}} = -\log(\prod_i q^{(i)})$（Eq.14-15）防止 gating 坍塌。
  - **系统框架层**：论文未明确说明。实现层面需在 PyTorch/PEFT 基础上，为每个 Transformer block 注入 gating 模块，并支持逐 LoRA 独立前向计算（memory 开销 = N × 单 LoRA 前向）。推理时两种模式：(1) 全专家模式——使用所有 LoRA + 已学习 gating weights；(2) mask 模式——手动排除某些 LoRA 后，gating 重新按比例分配剩余权重，无需重训练。
  - **编译框架层**：论文未明确说明（标准 PyTorch）。
  - **kernel 调度层**：论文未明确说明。每个 LoRA 的 $E_{\Delta\theta_i}(x)$ 计算可并行（batch N），但论文未讨论 kernel fusion 或内存优化。
  - **硬件架构层**：论文未明确说明 GPU 平台。
  - 对比 baseline 的改进映射：
    - **全局统一权重 → 逐层 learnable gating**：NLA 所有层共享 {w_i}（1 组 N 维标量）→ MOLE 每层独立学习 gating 分布（M 组 N 维 softmax 输出）。对应 Observation 2：不同层编码不同特征 → 不同层应有不同组合权重。Table 9 的 coarse-to-fine 分析验证了 layer-wise/block-wise gating（1-MoLE/b-MoLE）优于 network-wise（n-MoLE），证明逐层控制的必要。NLP 域中，gating 可视化（Fig. 7）显示 0%-20% 层对 LoRA A 的权重达 45%，80%-100% 层对 LoRA C 的权重达 52%——自动复现了 Observation 2 的层特异性规律。
    - **特征稀释 → gating 动态"增强/抑制"**：NLA 将每个 LoRA 的贡献强制均分（w_i ≈ 1/N）→ MOLE 的 gating 对期望特征赋予高权重（如 Dog LoRA 在"耳朵/鼻子"相关层权重 0.45）、对不期望特征赋予低权重（如 Dog LoRA 在"背景/风格"层权重 0.05）。V&L 域 Text-alignment 从 NLA 的 0.678 提升到 MOLE 的 0.759（+0.081），Image-alignment 从 0.694 提升到 0.757（+0.063，Table 1 平均）。
    - **组合灵活性差 → 双推理模式**：NLA 增删 LoRA 需重新计算所有权重 → MOLE 推理模式 2 通过 mask 排除特定 LoRA 后，gating 自动按比例重新分配权重（无需重训练）。Fig. 8 验证了从 3-LoRA MOLE → 2-LoRA MOLE 的平滑降级能力。
    - **无数据驱动适应性 → 下游数据微调 gating**：NLA 权重人工指定 → MOLE 用 domain-specific loss（V&L: CLIP guidance, NLP: task cross-entropy）微调 gating 参数。NLP 域泛化实验（Table 8）：NLI 任务训练的 gating → BBH 评估，MOLE 仍然优于 LoRAHub（+2.4），证明 gating 学习到的是结构性的组合策略而非过拟合到特定任务。
    - **Gating 坍塌 → gating balancing loss**：无约束 gating 会收敛到仅激活 1 个 LoRA（Fig. 5b: w/o L_balance 时 LoRA β 权重大 68%）→ L_balance 鼓励均匀分布，保持多 LoRA 利用。Table 7 消融：MOLE w/o L_balance（77.57）< MOLE（78.07），且仅调大 τ（温度上升）导致性能更差（76.35-77.45），因为高温使 softmax 过平坦、丧失区分能力。
    - **大规模 LoRA 组合**：NLP 域 128 LoRA 时 MOLE（38.5）远优于 LoRAHub（35.5），因 LoRAHub 的 gradient-free 优化常将多数 LoRA weight 置零，而 MOLE 的 gating balancing loss 保持较均匀分布。

## Mixture of Lookup Experts

- baseline方法是什么？
  MoE（Mixture-of-Experts）是 baseline，以 Mixtral 风格 MoE（top-2 routed experts，无共享 expert）为例说明全栈执行路径：
  - **算法层（MoE 推理）**：每层 Decoder 执行 Self-Attention → 中间特征 h → Router 计算 ArgTopK({h·r_j}) 选择 top-K expert → 加载对应 expert FFN 权重 → 计算 h' = Σ g_j·FFN_j(h) + h。每个 expert FFN 需要 h 作为输入执行 Standard MLP 计算（通常含 SwiGLU 激活）。Router 和 expert FFN 的输入都是中间特征（含上下文信息）。
  - **系统框架层**：HuggingFace Transformers（PyTorch）标准推理 pipeline。MoE 模型总参数大（如 Mixtral 8×7B: 46B 参数，仅 13B 激活），需多 GPU 或 expert offloading。
  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。
  - **kernel 调度层**：标准 cuBLAS GEMM 计算 expert FFN（W1·h, SiLU, W2·h 等）。若使用 expert offloading，需 GPU→CPU/disk 间 PCIe 传输完整 expert 权重（~176M/expert for Mixtral 8×7B）。
  - **硬件架构层**：NVIDIA V100 GPU（PCIe 4.0×16, 16 GB/s 带宽），expert offloading 到 CPU RAM 或 disk。
  - Baseline 核心缺陷：
    1. **VRAM 占用大**：虽然每 token 仅激活 top-K expert，但所有 expert 权重必须常驻 VRAM（因 Router 动态选择、无法预知哪个 expert 被激活）。Mixtral 8×7B 需 ≥92 GB VRAM (FP16)，单卡 80GB 无法容纳。
    2. **Expert offloading 延迟高**：将 expert 权重 offload 到 CPU RAM/disk → 每 inference step 加载 k 个 activated experts 到 VRAM。Mixtral 8×7B 中每 expert ~176M，k=2 时需传输 ~22.6B 参数/step（k=2 含 layer 数 32）。PCIe 4.0×16 下传输延迟 ~0.7s/step，disk 下 >10s/step，不可接受。
    3. **Batch generation 不友好**：不同样本在同一 step 可能选择不同 experts → batch size >1 时需加载所有被选中 experts（可能等于全部 N 个 experts），VRAM 使用量和通信延迟同步增加。
    4. **动态 routing 的不可预测性**：因 expert 选择由 Router 根据中间特征动态决定，prefetching 无法准确预测，CPU-GPU 通信无法被计算隐藏。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoLE 通过"embedding tokens 替代中间特征作为 expert 输入 + 全激活训练 + 推理前重参数化为 LUT"三个关键设计解决上述缺陷。全栈执行路径（以 MoLE-16E, 1B 激活参数为例）：

  - **算法层 — 训练阶段结构修改**：
    1. Routed experts 的输入从中间特征 h 改为 embedding tokens e = Embedding(input_ids)。因为 e 仅由 input_ids 决定，输入空间从无穷连续空间收缩为有限离散集 |V|（vocabulary size = 50k）。
    2. 所有 N 个 routed experts **同时激活**（不做 top-K 稀疏选择）。Router 输出 SoftMax({h·r_j}) 为全 N 维向量，对所有 expert 输出加权求和。共享 expert FFN_shared 保持标准中间特征输入 → SwiGLU 计算。
    3. 仅使用 LM cross-entropy loss 训练，无需 auxiliary loss（z-loss / load balance loss），因所有 experts 始终激活、梯度全程可微、无 collapse 风险。
    4. 前向：h' = Σ_{j=1}^N g_j·FFN_j(e) + FFN_shared(h) + h，其中 g = SoftMax(Router(h)), e = Embedding(i)。

  - **算法层 — 推理前重参数化（LUT 预计算）**：
    1. 以 embedding layer 权重 W_emb ∈ R^{|V|×d} 为输入，对每个 expert FFN_j 做一次 forward pass：v_j = FFN_j(W_emb) ∈ R^{|V|×d}。得到 |V| 个 token 对应的 expert 输出。
    2. LUT_l = {{v_j^i}_{j=1..N}}_{i=1..|V|}，大小为 N × |V| × d。
    3. LUT 整体 offload 到 CPU RAM/disk。与 MoE expert offloading 不同，LUT offload 不参与计算，仅存储。

  - **算法层 — 推理阶段（zero-computation experts）**：
    1. Lookup：根据 input_ids 从 offloaded LUT 检索 v_j^i（仅加载当前 batch 的 token 对应输出），传输量 = dN per token（与 |V| 无关）。
    2. Router 计算（同 MoE）：g = SoftMax(Router(h))。
    3. Expert 组合（无计算）：h' = Σ_j g_j·v_j^i + FFN_shared(h) + h。routed experts 仅需一次 lookup + weighted sum，零 FLOPs。
    4. Per-token 加载参数量：dN（如 1B MoLE-4E: d=2048, N=4 → 8KB），vs MoE expert offloading 的 2dkD_r（~537MB），小 60000× 以上。

  - **系统框架层**：HuggingFace Transformers + PyTorch。推理时 LUT 存储在 CPU/disk，通过 PCIe 传输 lookup results（dN 级别，<KB 量级）。共享 expert 和 attention 权重常驻 VRAM。

  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。

  - **kernel 调度层**：论文未明确说明。推理时 routed experts 无 compute kernel（仅 lookup + 加权求和），共享 expert 使用标准 cuBLAS GEMM。

  - **硬件架构层**：NVIDIA V100 GPU。CPU RAM 或 disk 作为 LUT 存储设备，通过 PCIe 4.0×16 传输。LUT 存储开销（7.4B 参数 for MoLE-16E 160M）虽比 MoE offloaded experts（1.0B）大 2.4-7.4×，但存储设备可扩展，且随模型增大（1B 激活参数），LUT/Expert 比例下降至可比较水平。

  - 对比 baseline 的改进映射：
    - **VRAM 占用大 → LUT offloading + 计算-free experts**：MoE 需常驻所有 expert 权重于 VRAM → MoLE 的 LUT 整体 offload 到 CPU/disk，VRAM 仅保留共享 expert + attention 权重。VRAM 使用等同于同激活参数量的 dense model。
    - **Expert offloading 延迟高 → Per-token 仅加载 LUT lookup results (dN)**：MoE 每 step 需加载 2dkD_r 完整 expert 参数（数十 MB 至数百 MB）→ MoLE 每 step 仅加载 dN lookup results（KB 级别），延迟可忽略（Figure 3 验证 MoLE latency ≈ Dense model latency）。通信开销降低 1000-2000×。
    - **Batch generation 不友好 → LUT offloading 通信量与 batch size 天然友好**：MoE 不同样本可能选择不同 experts → batch 增大时加载全部 expert → MoLE 加载的 LUT lookup results 是 per-token 的 dN，batch 增大仅线性增加 KB 级传输量，通信量仍可忽略。
    - **Dynamic routing 不可预测 → routing 与 LUT 解耦**：Router 仍动态运行在中间特征上（含上下文），但 expert output 已预计算为 LUT。router 只需输出 g_j 权重，LUT 存储所有可能 v_j^i，两者独立——避免了 prefetching 的预测难度。
    - **Router collapse 需 auxiliary loss → 全激活训练无需 auxiliary loss**：MoE 的 top-K sparsity 导致 router 需要 load balance loss 和 z-loss 防止 collapse → MoLE 所有 experts 始终激活并接收梯度，天然避免了 collapse。Ablation（Table 4）验证添加 auxiliary loss 反而降低性能。
    - **Embedding as input 的性能损失 → 全激活 + 更多 experts 补偿**：将 expert 输入从中间特征改为 embedding tokens 仅带来 0.7 point 性能下降（Table 7, 160M: 41.5 → 40.8），但全激活带来 1.5 point 提升（40.3 → 41.8），净收益 +0.5 point。更多 experts（N=16 vs 4）持续提升性能（Table 6: 39.7 → 42.3），证明了可扩展性。
    - **实验结果**：同 FLOPs 和 VRAM 下，MoLE 性能与 MoE 可比（MoLE-4E 1B: AVG 47.4 vs MoE-10E 1B: 46.6），推理速度与 dense model 相当，比 MoE expert offloading 快 1000× 以上。

## Mixture-of-Experts with Expert Choice Routing

- baseline方法是什么？
  **Token-Choice Routing（以 GShard Top-2 gating 为例）**：传统 MoE 路由策略中，每个 token 通过 softmax gating 独立选择得分最高的 top-k 个专家（通常 k=1 或 k=2）。全栈执行例子如下：
  - **算法层**：输入 token X ∈ R^{n×d} → 计算 gating score S = Softmax(X · W_g) ∈ R^{n×e} → 对每个 token 取 TopK(S, k)，k=2 → 每个 token 分配到 2 个专家 → 各专家独立计算 FFN → 加权合并输出。k 对所有 token 固定。
  - **系统框架层**：MoE 层通过 "shuffle" 阶段将 token 按 expert ID 聚集（all-to-all dispatch），FFN 计算后 "unshuffle" 回原始顺序。为缓解负载不均，引入 auxiliary load balancing loss（如 Switch Transformer 的 load balance loss），但该 loss 需仔细调权以不压倒主 loss。
  - **编译框架层**：论文未明确说明（标准 TPU XLA 编译 + GSPMD 2D sharding）。
  - **kernel 调度层**：einsum 操作执行 expert FFN 的批量矩阵乘法。Token dispatch/gather 通过 TPU 的 collective communication 原语实现。负载不均导致最繁忙 expert 的 step latency 成为瓶颈（step time 比 EC 慢约 20%）。
  - **硬件架构层**：Google TPU V4（512 chips for 8B/64E），利用 2D torus 拓扑做 GSPMD sharding。负载不均导致部分 TPU core 过载、部分闲置。
  - **Baseline 核心缺陷**：
    1. **负载不均（Load Imbalance）**：token-choice 独立路由导致某些专家接收远超容量的 token。auxiliary loss 无法保证均衡，尤其在训练早期，过容量比率可达 20%-40%，大量 token 被丢弃。
    2. **专家欠专业化（Under Specialization）**：过大的 auxiliary loss 倾向负载均衡但路由效果差，导致专家冗余或不够专精。在负载均衡与专业化之间取得平衡极为困难。
    3. **每个 token 固定计算量**：所有 token 精确分配 k 个专家，无论其复杂度。重要 token 和简单 token 获得相同计算资源，浪费且不灵活。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Expert Choice Routing**：反转路由方向，让每个专家独立选择 top-k 个 token（而非 token 选专家）。全栈执行路径如下：
  - **算法层（Expert Choice Routing）**：
    1. 计算 token-to-expert affinity S = Softmax(X · W_g) ∈ R^{n×e}（与 baseline 相同）
    2. 反转：对 S^T ∈ R^{e×n} 的每一行（每个专家）取 TopK，k = n×c/e（固定专家容量）→ G, I = TopK(S^T, k)
    3. 排列矩阵 P = OneHot(I) ∈ R^{e×k×n}，将 token 按专家分组：X_in = P · X ∈ R^{e×k×d}
    4. 各专家独立 FFN：X_e[i] = GeLU(X_in[i] · W_1[i]) · W_2[i]^T
    5. 反排列 + 门控加权：X_out[l,d] = Σ_{i,j} P[i,j,l] · G[i,j] · X_e[i,j,d]
    - 每个 expert 恰好处理 k 个 token，负载天然完美均衡。每个 token 可被 0~e 个专家选中，实现可变计算分配。
  - **系统框架层**：与 baseline 相同的 MoE 层结构（每两层 Transformer 替换一层 FFN 为 MoE），shuffle/unshuffle 阶段不变。关键区别：无需 auxiliary load balancing loss——负载均衡由算法本身保证。容量系数 c=2 匹配 GShard top-2 的 per-token 计算量。
  - **编译框架层**：论文未明确说明（标准 TPU XLA + GSPMD 2D sharding，与 baseline 一致）。
  - **kernel 调度层**：einsum 操作与 baseline 相同。因负载完美均衡，所有 expert 计算时间一致，step latency 由均匀负载决定（无 straggler expert），step 时间比 GShard top-2 快约 20%。
  - **硬件架构层**：Google TPU V4（512 chips for 8B/64E），与 baseline 一致。负载均衡使 TPU 利用率更高，无闲置 core。

  **缺陷 → 方法设计直接映射**：
  - **负载不均 → 专家选 token + 固定专家容量 k**：每个专家恰好接收 k 个 token（k = n×c/e），从设计上消除负载不均。无需 auxiliary loss，训练早期即保持均衡。效果：EC-CF2 收敛速度比 GShard top-2 快 2× 以上，且每步快 20%。
  - **欠专业化 → 学习到的 token-expert affinity 不受 load balance loss 干扰**：无 auxiliary loss 意味着 gating 网络的学习目标纯粹是最大化 token-expert affinity，自然产生更专业化的专家。效果：下游 GLUE/SuperGLUE 11 任务平均 accuracy 提升 2%+（8B/64E: EC-CF2 92.6 vs GS Top-2 90.3 vs ST Top-1 88.9）。
  - **固定计算量 → 可变数量专家 per token**：每个 token 可被 0~e 个专家选中（实际分布：约 77% tokens 被 1-2 个专家选中，23% 被 3-4 个，3% 被 >4 个）。Ablation 验证：限制每个 token 最多 2 个专家（EC-CAP2）导致 accuracy 下降 0.8 points，证实可变专家数有效。

  **可选约束扩展（EC-CAP）**：通过熵正则化线性规划 + Dykstra 交替投影算法限制每个 token 最多 b 个专家。EC-CAP3（b=3）达到与无约束 EC-CF2 相当的 accuracy，验证了可变分配的有效性同时提供了可控性。

## MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

- baseline方法是什么？
  Baseline 为 Megatron-Core 的标准 5D 混合并行（TP+EP+CP+DP+PP），其中 Attention 层和 MoE 层使用**相同的并行映射**——EP group 被限制在 DP 的子组内，最大 EP 度受 DP 度约束。以 Mixtral-8x22B 在 128 H100 GPU（TP=2, EP=4, PP=8）上的训练为例，全栈执行路径：
  - **算法层**：MoE Transformer 每层执行 Self-Attention（全序列密集计算，TP/CP 切分 hidden dim/sequence）→ Router top-K gating → Expert FFN（稀疏计算，仅激活部分 expert）。Attention 和 MoE 层共享同一套 (TP, CP/EP, DP, PP) 并行映射，即 EP=CP 且两者绑定。Token dispatching 使用 full-sequence-based token dropping（需跨 rank 收集 logits 保证一致性，引入额外通信开销）。
  - **系统框架层**：NVIDIA Megatron-Core（https://github.com/NVIDIA/Megatron-LM）。PyTorch 2.5.0 + CUDA 12.6，使用 NCCL 集合通信（All-to-All、AllGather、ReduceScatter）。
  - **编译框架层**：Megatron-Core 框架作为训练系统的核心引擎，负责并行组初始化和通信调度。Baseline 中 Attention 和 MoE 的并行组生成逻辑耦合（EP 从属于 DP），ranks 布局固定。
  - **kernel 调度层**：NCCL collective communication library 处理 EP 的 All-to-All（token dispatch/combine）和 TP 的 AllGather/ReduceScatter。通信算子与 GEMM 计算串行执行，且 EP 的 All-to-All 可能跨越节点间低带宽 InfiniBand（400 Gbps），与 Attention 层密集通信叠加。
  - **硬件架构层**：NVIDIA Eos 集群：DGX H100 节点（8×H100 GPU，NVLink 4th Gen 450 GB/s intra-node，InfiniBand 400 Gbps inter-node），最多 1024 GPU。
  - **核心缺陷**：(1) 统一并行映射导致 sub-optimal——Attention 需要 TP/CP（序列级通信），MoE 适合 EP（token 级通信），但 baseline 强制两者相同；(2) EP 受 DP 约束，最大 EP 度有限，scalability 受限；(3) 通信域不可折叠——当 EP group 跨越节点时，All-to-All 走低带宽 inter-node 链路，通信开销占主导；(4) token-dropping 需 full-sequence 通信收集 logits，额外开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE Parallel Folding** 策略：将 Attention 层和 MoE 层的并行映射**解耦**，允许各自独立配置最优并行度。Attention 层使用 TP×CP×DP×PP 四维组，MoE 层使用 TP(ETP)×EP×DP(EDP)×PP 四维组（仅 PP group 数量和成员保持一致）。同时设计统一的 flexible token dispatcher 支持 token-dropping（sub-sequence dropping 替代 full-sequence dropping）和 token-dropless 两种范式。

  MoE Parallel Folding 全栈执行路径（以 Mixtral 8x22B 在 128 H100 GPU 上的最优配置 TP=2, EP=8, ETP=1, PP=8 为例）：
  - **算法层（并行映射解耦 + Folding）**：
    - Attention 层：TP=2, CP=1, DP=8, PP=8 → 使用 TP 切分 hidden dim，DP 处理不同 micro-batch。
    - 转换（Attention → MoE）：仅需 reshape 操作（将 sequence/subsequence 展平为 batch of tokens），无显式通信开销。
    - MoE 层：ETP=1, EP=8, EDP=1, PP=8 → EP=8 将 8 个 expert 分布到 8 GPU，无需 ETP（即 MoE 层不做 tensor parallelism），最大化 GEMM 效率。
    - **Folding 效果**：Attention 的 TP(2)×DP(8)=16 个 GPU 被"折叠"到 MoE 的 EP=8 组。通过将 EP 组与 Attention 的 TP/DP 子组折叠，使 EP 的 All-to-All 通信尽可能在节点内 NVLink（450 GB/s）完成，避免跨节点 InfiniBand。
  - **系统框架层（Megatron-Core 修改）**：
    - 并行组生成：实现 generate_mappings() 函数，为 Attention 和 MoE 分别生成独立并行组。Attention ranks 布局 (attn_dp, pp, cp, tp)，MoE ranks 布局 (moe_dp, pp, ep, tp)。
    - Token Dispatcher：统一处理 ETP 和 EP 组合，forward 流程为 Router→Permutation→All-to-All-V(跨 EP)→AllGather-V(跨 ETP)→Expert GEMM→ReduceScatter-V(跨 ETP)→All-to-All-V→Unpermutation。backward 流程中 AG/RS 与 RS/AG 互换。
    - Sub-sequence dropping：基于本地 sub-sequence logits 做 token dropping 决策，无需跨 rank 通信收集 logits（经验验证不影响模型收敛）。
  - **编译框架层**：Megatron-Core 框架，代码开源在 https://github.com/NVIDIA/Megatron-LM。核心修改为并行组生成和 token dispatcher。
  - **kernel 调度层**：NCCL All-to-All-V、AllGather-V、ReduceScatter-V 作为通信原语。MoE Parallel Folding 将 EP 通信限制在更紧凑的组内（folding 使得 EP group ≤ NVLink domain），降低 All-to-All 的通信带宽需求。FP8 训练使用 Transformer Engine 的 delayed scaling。
  - **硬件架构层**：同 baseline（NVIDIA Eos 集群，H100 GPU）。MoE Parallel Folding 通过折叠并行组充分利用节点内 NVLink 高带宽（450 GB/s），减少跨节点 InfiniBand（400 Gbps）通信。FP8 实验在 H100 上达到 631.7 TFLOPS。

  **缺陷 → 方法设计直接映射**：
  - **统一并行映射 sub-optimal → MoE Parallel Folding 解耦 Attention/MoE 并行策略**：Attention 需要 TP/CP（序列级密集通信），MoE 需要 EP（token 级稀疏通信），分离后各自独立优化。Mixtral-8x22B: MCore baseline 46.3% MFU → MCore w/ Folding 49.3% MFU（+3.0pp）。Qwen2-57B-A14B: 35.3% → 39.0%（+3.7pp）。
  - **EP 受 DP 约束 → Folding 使 EP 可折叠到 Attention 任意子组**：baseline 中 EP group 被限制在 DP 子组内，最大 EP=DP。Folding 后 EP 可独立于 DP 扩展，例如 Mixtral-8x22B 使用 EP=8 同时 ETP=1（MoE 层不做 TP），而 Attention 层用 TP=2。
  - **跨节点通信开销大 → Folding 使通信域紧凑化**：当 EP×CP group 超过 8（跨越 NVLink domain），baseline 的 All-to-All 走低带宽 InfiniBand。Folding 将 CP 和 EP 折叠在一起，使 All-to-All 优先走 NVLink。Ablation（Figure 6）：无 Folding 时 CP×EP>8 导致延迟急剧上升，Folding 后保持稳定。
  - **Fine-grained MoE 通信瓶颈 → ETP 替换为 EP**：Fine-grained MoE（Mixtral-8x22B-G8T8）中 ETP 通信占比超 70%（因 expert hidden size 小导致 GEMM 效率低）。Folding 支持用 EP 替代 ETP，EP 的通信开销远低于 ETP（All-to-All vs AllGather+ReduceScatter）。G8T8: MCore baseline 17.1% MFU → MCore w/ Folding 28.8% MFU（+11.7pp，相对提升 68%）。
  - **Token dropping 通信开销 → Sub-sequence dropping**：baseline 的 full-sequence dropping 需跨 rank 收集 logits（额外 AllGather 通信）。Sub-sequence dropping 仅基于本地 logits 决策，零额外通信开销，且经验验证不影响模型收敛（附录 validation loss 曲线与 MCore v0.9 对齐）。

## MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

- baseline方法是什么？
  Vanilla MoE（Top-2 Routing）：每个 MoE 层包含 N 个结构相同的 FFN 专家和一个 Router G=Wx。每个 token 固定选择 Top-2 个 FFN 专家，加权聚合输出：y = Σ g_i * FFN_i(x)。所有 token（无论难易）都激活相同数量的 FFN 专家，导致简单 token（标点、词片段）浪费计算资源。
  全栈执行例子：输入 token x 经过 Router 计算 logits → Top-2 选择 2 个 FFN 专家 → 每个 FFN: x → Linear(D→4D) → GELU → Linear(4D→D) → 加权求和。训练时使用均匀负载均衡损失（所有专家相同 token 分配），推理时 FFN 专家分布在多 GPU 上通过 All-to-All 通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE++ 引入三种零计算专家（zero/computation expert）与 FFN 专家混合：
  1. **Zero Expert**（输出 0）：使 Top-2 退化为 Top-1，减少简单 token 的 FFN 计算。
  2. **Copy Expert**（输出 x）：允许 token 跳过当前 MoE 层（shortcut），对齐残差网络思想。
  3. **Constant Expert**（输出 α1*x + α2*v，可训练向量 v 和权重 W_c）：用少量参数调整输出。
  同时引入 Gating Residuals（W_g 将前一层路由分数融入当前层）和异构负载均衡（τ 参数控制 ZC/FFN token 分配比例）。

  全栈执行对比（以 τ=0.75, MoE++ 1B/(16+4)E 为例）：

  **算法层（对比核心）**：
  - Baseline：每个 token 固定激活 2 个 FFN。简单 token（如标点","）仍然消耗 2 个 FFN 的计算。
  - MoE++：Router 计算 logits = Wx + W_g * G_prev（加入前层路径信息）→ Top-2 选择。简单 token 可能被路由到 zero + copy expert（跳过该层，0 计算），或 zero + FFN（退化为 Top-1 FFN），或 constant + FFN（用可训练向量微调 FFN 输出）。挑战性 token（如动词"touch"）仍可用满 2 个 FFN。

  **系统框架层**：
  - Baseline：FFN 专家分布在多 GPU，All-to-All 通信同步 token，负载不均时某些 GPU 空闲。
  - MoE++：零计算专家参数极少（constant expert 仅 W_c∈R^{2×D} 和向量 v），可全部部署在每个 GPU 上，无需跨 GPU 通信。token 被路由到 ZC expert 时直接本地计算，消除对应 All-to-All 开销。Expert forward throughput: 1B 模型从 610.9ms → 500.3ms（提升 22.1%）。

  **编译框架层**：论文未明确说明（使用 Megatron 训练框架，未修改编译层）。

  **Kernel 调度层**：
  - Baseline：每个 token 的 2 个 FFN GEMM kernel 必须全部执行。
  - MoE++：ZC expert 无 GEMM（zero/copy 是 O(1) 操作，constant 仅 O(D) 标量操作），等效减少了 GEMM kernel 调用次数。计算复杂度从 O(T) 降至 O(τ*N_FFN*T / (τ*N_FFN + N_ZC))，τ=0.75 时约为 baseline 的 85.7%。

  **硬件架构层**：论文未明确说明（纯算法/软件层面改进）。

  **关键设计思路映射**：
  - **痛点：固定 Top-K 对简单 token 计算浪费** → **零计算专家**：提供 skip/discard/replace 三种低成本路径，token 按难度动态选择 FFN 数量。
  - **痛点：Router 独立决策无层间一致性** → **Gating Residuals**：W_g 矩阵连接前后层路由，稳定异构专家选择，减少 routing score 方差（Fig.6）。
  - **痛点：均匀负载均衡不适用于异构专家** → **异构 Load Balance Loss**：τ 参数控制 FFN vs ZC 的 token 分配比例，τ 越小 ZC 分配越多（throughput 越高但可能性能略降），τ=0.75 为默认平衡点。

## MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems

- baseline方法是什么？
  现有LLM评测基准（MLPerf, Open-LLM-Leaderboard, LLM-Perf, Artificial Analysis, TensorDock）使用vanilla MBU和MFU作为细粒度系统性能指标。以Qwen1.5-MoE-A2.7B在1×A100 PCIe上运行GSM8K为例说明全栈执行路径：
  - **算法层（指标计算）**：MBU = B_achieved / B_peak = ((S_model + S_KV) / TPOT) / B_peak，其中S_model使用全量14.3B参数。MFU = (T_token × F_token) / F_peak，F_token假设所有参数参与计算。**缺陷**：MoE模型中每个token仅激活top-k个expert（Qwen1.5-MoE为4 shared + 4 routed = 8个expert），vanilla MBU/MFU忽略稀疏激活导致资源高估——batch size=1时高估3×以上，即使有shared expert仍高估1.5×（见图2分析）。运维者据此过度采购GPU，造成严重资源浪费。
  - **系统框架层（评测工具与成本模型）**：MLPerf/LLM-Perf等使用固定harness仅在GPU上统计吞吐和延迟，成本模型仅按GPU使用时长计费。**缺陷**：(1) 未考虑CPU、DRAM、SSD等异构资源——MoE系统日益依赖CPU做expert offloading（如Fiddler将低负载expert迁移到CPU计算），DRAM/SSD做expert存储（如MoE-Infinity），忽略这些导致成本估算严重偏低；(2) 未建模PCIe/NVLink通信开销和多级存储功耗。
  - **硬件层（部署决策）**：基于高估的MBU/MFU值，运维者倾向采购高端GPU（如H100）满足"理论"带宽需求。**缺陷**：实际稀疏激活下，batch size=1时DeepSeek-R1带宽需求仅1040 GB/s，消费级RTX 4090配合offloading即可满足0.1s/token SLO——而vanilla MBU给出18,901 GB/s的误导性需求，迫使运维者走向昂贵的DGX-H100方案。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出MoE-CAP——专为MoE系统设计的CAP三维评测方法，核心创新：(1) Sparsity-aware指标S-MBU和S-MFU，仅计激活参数；(2) 覆盖全部异构资源的完整成本模型；(3) CAP雷达图可视化三维权衡；(4) 多约束决策矩阵指导系统选型。以Qwen1.5-MoE在MoE-CAP下的全栈执行路径对比：
  - **算法层（S-MBU/S-MFU公式）**：S-MBU = B_achieved / B_peak, B_achieved = (S_activated + S_KV) / TPOT。S_activated = n_layer × S_attn + Σ_l Σ_i 𝟙[l,i] × S_expert，通过在SGLang/HF的每层MoE router后植入probe追踪布尔变量𝟙[l,i]（expert i在layer l是否被当前batch激活）。S-MFU = (T_token × S-F_token) / F_peak，S-F_token = F_attn + 2N_router + 2k_expert × N_expert，k_expert从模型配置直接获取无需运行时追踪。dense模型作为特例（n_expert=1, ∀i 𝟙[l,i]=1）兼容。**直接解决**：vanilla MBU/MFU忽略稀疏激活的高估问题——实验证明Mixtral-8x7B上vanilla MBU高估>260%，S-MBU与profiler实测值误差<1%（图8）。S-MFU与profiler误差<0.05%（表2）。Qwen模型（含shared expert架构）的S-MBU同样精确（图9）。
  - **系统框架层（完整CAP评测流水线）**：在SGLang/HuggingFace中植入轻量级expert activation profiler（兼容CUDA graph，overhead <2.7%），构建支持6种框架（vLLM/SGLang/MoE-Infinity/K-Transformers/HF/Accelerate）+5个benchmark（MMLU/GSM8K/MATH/Arena-Hard/LongBench）+8种以上MoE模型的自动化流水线。成本模型C_hardware = C_GPU + C_CPU + C_Motherboard + C_DRAM + C_SSD，覆盖CPU计算、DRAM/SSD存储、PCIe/NVLink通信的全部异构资源。能耗C_energy = (P_GPU + P_CPU + P_C2M + P_PCIe + P_NVLink) × R。Per-token cost C_token = (C_hardware + C_energy × $/kWh) / (T_token × R)。**直接解决**：(1) 仅GPU的成本模型无法反映MoE offloading场景——AMD 777X CPU峰值功耗280W可与A6000 Ada（300W）相当，忽略CPU能耗导致成本预测过于乐观；(2) 缺乏统一的MoE系统横向对比工具。
  - **硬件层（CAP雷达图+决策矩阵）**：CAP雷达图将Cost（$/token/W）、Accuracy（exact match/F1/win rate）、Performance（TPOT/吞吐）三维归一化可视化。通过benchmarking不同CAP类型系统——PA型（SGLang/vLLM：低延迟高精度高成本）、PC型（K-Transformers量化：提速降精度降成本）、CA型（MoE-Infinity offloading：低保成本高精度低吞吐）——生成多约束决策矩阵（表7），根据hardware tier、batch size、primary/secondary constraint直接推荐系统方案。同时通过S-MBU分析batch size对sparsity的影响（图5/6），发现batch size=8时DeepSeek-R1仅激活18.44%参数，功耗仅450W的RTX 4090即可满足需求。**直接解决**：缺乏sparsity-aware硬件选型指导导致的盲目过度采购——batch size=1场景下，低功耗边缘设备（Apple M3 Max, NVIDIA Orin AGX/NX）配合offloading即可部署大MoE模型，无需昂贵数据中心GPU。

## MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

- baseline方法是什么？
  Baseline 为现有 MoE 模型的 expert 压缩策略，主要包括四类：
  1. **Expert Quantization**（如 MC-MoE、MoQE、QMoE、CMoE、MoE-MPTQS、HOBBIT、EdgeMoE）：将 expert 参数从浮点精度量化为低精度整数（1/2/3/4/8-bit），以约 4× 内存节省换取推理加速。然而低比特量化引入不可控和不可预测的误差，导致生成性能显著下降（如 QMoE 在 20× 压缩比下 accuracy drop 达 6.7%，CMoE 在 150× 压缩比下 accuracy drop 达 23.81%）。
  2. **Expert Distillation**（如 ExpertFlow）：将大型 MoE 模型的知识蒸馏到更小的模型/reduced expert set。
  3. **Expert Pruning**（如 Lu et al. 2024）：识别并移除贡献小的 expert。
  4. **Expert Decomposition**（如 MiLo）：使用低秩分解技术减少参数量。

  以 expert quantization（最常见的 offloading 场景压缩策略）为 baseline，全栈执行路径如下：
  - **算法层（Expert Quantization + Offloading）**：MoE 推理时，所有 expert 权重预先量化为低精度（如 2-bit 或 4-bit）并存储在 CPU 主内存中。Router 选择 top-K expert → 通过 PCIe 从主内存加载对应量化 expert 权重到 GPU 显存 → GPU 上反量化恢复浮点精度 → 执行 FFN 计算 → 输出加权聚合。**核心缺陷**：(1) 量化误差不可控——低比特量化 (1-2 bit) 的量化噪声分布无法保证 bounded error，且不同 expert 对量化误差的敏感度高度异质（shallow/middle/deep layer 的 sensitivity 差异巨大），uniform 位宽分配导致重要 expert 欠保护而冗余 expert 过保护；(2) 缺乏系统性的压缩误差敏感性分析——现有工作未回答"哪些 expert 对压缩误差更敏感"这一关键问题，导致 compression 策略盲目、无法针对性优化。
  - **系统框架层**：基于 HuggingFace Transformers 或类似推理框架，使用 GPU offloading 技术（如 MoE-Infinity、SwapMoE、Pre-gated MoE）。核心瓶颈为 PCIe 带宽（PCIe 4.0: 32 GB/s << GPU HBM: 300 GB/s），数据传输延迟无法被计算隐藏。量化压缩减少传输量但牺牲精度。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA）。
  - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM 或量化 kernel（如 HQQ）执行反量化+矩阵乘法。
  - **硬件架构层**：PCIe 4.0 GPU 服务器（GPU 内存有限，需 offloading 到 CPU 主内存），论文未指定具体 GPU 型号。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出用 **error-bounded lossy compression（SZ3/CuSZp）替代 quantization**，并首次系统性地从 7 个维度分析了压缩误差对不同 layer expert 的推理精度影响。核心贡献是**误差敏感性分析**而非完整的 compression system implementation（论文明确说明其聚焦于三步中的前两步：①高效压缩算法选择、②误差敏感性分析）。

  全栈执行路径（以 Moonlight 模型 + GSM8K dataset 为例）：

  - **算法层（Error-Bounded Compression + Layer-Aware Sensitivity Analysis）**：
    1. **Error-Bounded Lossy Compression 替代 Quantization**：不同于量化（误差不可控），SZ3（CPU）和 CuSZp（GPU）提供严格 error bound ê 保证。压缩后专家参数的最大误差不超过 ê，压缩误差分布近似 N~(0, ê)。这使误差可控——通过调整 ê 可精确 trade-off 压缩比与精度。论文通过模拟 normal distribution error N~(0, ê) 注入 expert 参数来研究误差影响，其中 ê 取 L1 范数平均值的 10%/30%/50%/80%。
    2. **层级化误差敏感性分析（7 个维度）**：
       - 维度 1（单 expert）：分析了 expert-0 in layer 1 在不同 ê 下的表现。结果：小误差不影响推理，但完全随机化参数→错误输出——证明即使是"不重要"的 expert 也 critical。
       - 维度 2（最高频 expert）：layer 1 的 expert-26（激活频率最高），ê=30%/50%/80%。结果：PIA 保持 0.95-0.96（仅降 0-0.01），ICA 从 0.86 降至 0.79——误差先影响指令遵循能力、后影响推理能力。
       - 维度 3（跨层最高频 expert）：layer 1/13/20/26 的各自最高频 expert，ê=80%。结果：ICA 呈非单调分布——layer 1: 0.79, layer 13: 0.75, layer 20: 0.89, layer 26: 0.96（**深层 ICA 反而超过 baseline 0.86**）。PIA 始终保持 ≥0.94。
       - 维度 4（Top-K expert）：layer 1 和 layer 26 的 top-6 highest-frequency experts，ê=80%。结果：layer 1 ICA 从 79%→74%（累积效应），layer 26 ICA=0.90 仍 > baseline 0.85。
       - 维度 5（全层 expert）：layer 1/13/20/26 全部 64 experts，ê=80%。结果：shallow layer ICA 骤降至 0.33，middle layer 13 ICA 最低 0.38（最敏感），deep layer 26 ICA=0.85（几乎不下降）。
       - 维度 6（跨层 group）：Group1 L1-L10 / Group2 L9-L18 / Group3 L17-L26，ê=30%/50%/80%。结果：ê=80% 时所有 group 模型完全失效（不输出），ê=50% 时 Group2 (middle) ICA 最低 0.69——中间层对误差最敏感。
       - 维度 7（跨数据集泛化）：在 MATH dataset 上重复维度 3 和 6。结果：更难的数据集上误差影响更显著（baseline PIA 本身仅 0.70，ê=80% on layer 13 时 PIA 降至 0.60），但深层误差仍可能带来增益（layer 26 ICA: 0.66 vs baseline 0.62）。
    3. **9 条关键结论（Takeaway）**指导实践：
       - 浅层专家（attention + token→vector 转换）：对 bounded error 最鲁棒，可激进压缩。
       - 中层专家（核心推理）：最敏感，需保守压缩/保护参数完整性。
       - 深层专家（指令遵循 + 输出整合）：可控误差可提升性能（隐式集成效应→多样化 ensemble），可作为优化策略。
       - 多 expert/多 layer 误差呈非线性级联放大效应，跨层 group 注入比单层影响大得多。

  - **系统框架层**：论文未实现完整的 offloading-compression 集成系统（明确说明此步骤留待 future work）。但给出了设计方向：在 MoE offloading 框架中，于 expert 从主内存传输前执行压缩（CPU 上 SZ3 或 GPU 上 CuSZp），传输后再解压。未来需设计 pipeline 算法 overlap compression/decompression 与 offloading 任务以隐藏延迟。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。使用了现有的 error-bounded lossy compressor（SZ3 for CPU, CuSZp for GPU），但未实现自定义 kernel 或将其集成到推理 pipeline 中。
  - **硬件架构层**：论文未明确说明具体 GPU。在 PCIe-limited 场景下（GPU HBM 有限 + 主内存 abundant），压缩-传输-解压流水线需要 CPU 或 GPU 上的 compressor 支持。

  **对比 baseline 的改进映射**：
  - **Quantization 的不可控误差 → Error-Bounded Compression 的可控误差**：量化（尤其是 1-2 bit）引入不可预测误差导致严重精度退化 → SZ3/CuSZp 提供严格 bounded error 保证（‖θ_compressed - θ_original‖_max ≤ ê），且误差分布可建模为 Normal 分布，使误差影响可预测、可控制。
  - **Uniform 位宽分配 → Layer-Aware 差异化压缩**：quantization 对所有 expert 使用相同位宽（或简单 heuristic）→ 论文的敏感性分析结果表明应以不同压缩力度处理不同层 expert：浅层可激进压缩、中层保守保护、深层可适当注入噪声。
  - **缺乏误差敏感性理解 → 7 维度系统分析 + 9 条实践指导**：这是本工作的首要贡献——不是提出新的 compression algorithm，而是回答"哪些 expert 的压缩误差对推理精度影响最大"这一基础性问题。结果直接指导 MoE compression 系统的设计（如 MC-MoE 的 expert 重要性驱动位宽分配可受益于本篇的 layer 级敏感性洞察）。
  - **深层误差增益的发现 → 隐式集成优化策略**：最反直觉的发现——深层 expert 注入可控噪声可提升 ICA（layer 26: 0.96 vs baseline 0.86, layer 20: 0.89 vs 0.86），揭示了一种无需训练的低成本模型鲁棒性增强方法：在推理时对深层 expert 参数添加微小随机扰动。

## MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

- baseline方法是什么？
  Baseline 是**缺乏系统性 MoE 推理 benchmark 的状态**——即研究者各自使用不同的硬件、框架、模型和配置进行实验验证，导致：(1) MoE 推理性能结论无法横向比较；(2) 关键超参数（FFN dim、#experts、#active experts）对吞吐量的影响缺乏定量研究；(3) 多种优化技术（量化、剪枝、投机解码、Fused MoE、并行策略）在统一硬件和框架下的相对效果未知；(4) LLM 和 VLM 两类 MoE 模型的推理特性差异未被系统对比。以 Mixtral-8x7B 在 4×H100 + vLLM 上的执行路径为例：
  - **算法层**：Standard Top-K routing（每层 8 experts, k=2）。每个 token 经 self-attention → router logits → softmax → select Top-2 → 两个 expert FFN 计算 → weighted sum。Baseline 无任何优化（FP16, 无剪枝, 无 Fused MoE, 无投机解码）。**痛点(1)**：激活 expert 越多（k=2→k=8），每个 token 要经过更多 FFN 计算和参数读取，吞吐量下降 50-80%——但当前缺乏"k 值 vs FFN dim vs #total experts"的 joint scaling 指导。
  - **系统框架层**：vLLM 推理框架。支持 TP/PP/EP 并行策略配置和 PagedAttention KV-cache 管理。**痛点(2)**：vLLM 提供了多种并行策略（TP, PP, EP, Hybrid），但这些策略在 H100 上对 MoE 模型的 relative effectiveness 未知——研究者选 TP 还是 EP 缺乏实验依据；(3) vLLM 内置 Fused MoE kernel 可用但实际收益在不同 batch size/sequence length 下未量化。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch + CUDA kernel。
  - **kernel调度层**：Fused MoE kernel（vLLM 内置）将 routing + FFN 融合，但未与其他优化（FP8、剪枝、投机解码）联合评估。
  - **硬件架构层**：NVIDIA H100 SXM5 80GB + NVLink。**痛点(4)**：H100 的 TP 扩展效率 vs EP 扩展效率缺乏实测数据；H100 vs Cerebras CS-3（wafer-scale）在 MoE 推理上的对比缺失。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE-Inference-Bench**——一个综合性 MoE 推理 benchmark 套件，不提出新算法，而是在统一硬件（H100）、统一框架（vLLM）下系统性地评估多种优化技术及其交互。以 Mixtral-8x7B 在 4×H100 + vLLM 上的全栈执行路径为例：
  - **算法层（系统评估多种优化技术）**：
    1. **FP8 Quantization**：将 MoE FFN 权重和激活从 FP16 量化为 FP8（H100 第四代 Tensor Core 原生支持 FP8），权重保留 FP8 master copy，计算时 dequantize → INT8 matmul on Tensor Core。吞吐量提升 20-30%（batch size=64 时差距最大），且在不同 sequence length 下保持稳定优势。**解决痛点(1)**：量化可在不增加 active experts 的前提下提升吞吐量，部分抵消多 expert activation 的性能代价。
    2. **MoE Pruning**：系统性评估 inter-expert pruning（移除整个 expert）和 intra-expert pruning（缩减 expert 内部 FFN dim）在 12.5%/25%/50% 三种比例下的效果。关键发现：(a) 50% 高比例剪枝反而显著改善吞吐量（因减少的总参数和计算量 > 负载不平衡损失），而 12.5%/25% 低比例剪枝因引入负载不均反而可能降低吞吐量；(b) OLMoE-1B-7B 对 intra-expert 剪枝容忍度高，Qwen1.5-MoE-A2.7B 更敏感。**解决痛点(1)**：提供剪枝比例与模型类型的选择指南。
    3. **Speculative Decoding**：Qwen3-30B-A3B (target) + Qwen3-1.7B (最优 draft) 的投机解码。Draft model 大小需要平衡——太小（0.6B）acceptance rate 低导致验证浪费，太大（8B）draft 开销抵消收益。最优 draft 大小（1.7B）在所有 input lengths 下吞吐量最高。**解决痛点(1)**：提供 draft model size 选择的实验指导。
  - **系统框架层（vLLM 并行策略 + Fused MoE）**：
    1. **并行策略对比**：TP-only 在 1→4 GPU 扩展时吞吐量 >2×（NVLink 高带宽掩盖 all-reduce 开销），TP+EP 扩展效率次之，PP+EP 几乎无加速，PP-only 持平。**解决痛点(2)**：明确建议 MoE 推理优先使用 TP，PP 和 EP 在单节点内收益有限。
    2. **Fused MoE 全面评估**：Fused MoE 在 batch size 增大时优势更明显（大 batch 时 15-20% throughput gain），因为更大 batch 下未经融合的 kernel launch 和中间显存传输开销更高。在不同 sequence length 下保持 12-18% 优势。**解决痛点(3)**：量化 Fused MoE 在不同 workload 下的收益。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Fused MoE kernel 将 token-to-expert dispatch → grouped GEMM → SiLU activation → weighted sum 融合为单 kernel。效果：减少中间 tensor 的 HBM 往返（每层节省约 3-4 次 HBM read/write），与 batch size 正相关。
  - **硬件架构层（H100 vs Cerebras CS-3 对比）**：
    - H100：在 context length >1024 时延迟急剧上升（受限于 HBM 带宽和 KV-cache 增长），吞吐量在长 context 下显著退化。
    - CS-3：WSE-3 的多数量级更高内存带宽 + 减少 inter-device 通信使其延迟增长平缓，在长 context 推理中优势明显。
    - **解决痛点(4)**：提供两类硬件的 quantitative 对比，指出 H100 适合短 context 高 batch 场景，CS-3 适合长 context 低延迟场景。
  
  **MoE 超参数 Scaling 的核心发现（指导部署设计）**：
  - FFN dim scaling：增大 FFN dim（1792→14336）导致吞吐量平均下降 50%，且大 FFN dim + 多 active experts 时降幅最大（~60%），原因是内存带宽饱和压倒计算并行优势。
  - Expert count scaling：小 FFN dim（1792/3584）时增加 expert 数量（8→64）可保持或略微提升吞吐量（5-15%），大 FFN dim 时增加 expert 数量反而受内存带宽限制无收益甚至 OOM。
  - Active expert scaling：active experts 从 1→8 吞吐量下降 50-80%，且下降幅度在 big FFN dim 时更大（60-80% vs 20-30%）。
  - **联合指导**：小 FFN（1792-3584）可灵活使用更多 active experts；大 FFN（7168-14336）必须使用保守的 activation 策略（1-2 active experts）以避免 OOM。

  **LLM vs VLM 的关键差异**：
  - VLM 的 latency 差距远大于 LLM（ITL 240% vs 100% gap，end-to-end 260% vs 120% gap），主要原因是视觉编码器的额外计算负载和多模态处理开销。

## Mixture of Diverse Size Experts

- baseline方法是什么？
  Baseline 是传统的 **Same-Size Expert MoE**（如 Switch Transformer、GShard、Mixtral 架构风格），即每个 MoE FFN 层内所有 N 个 expert 拥有完全相同的结构（相同的 hidden dimension h），仅通过 top-k gating 选择 k 个 expert 激活。以 300M×8（基于 Llama 2, dim=1536, n_layers=8, h=3840, N=8, k=2）在 NVIDIA A800 集群上的执行路径为例：
  - **算法层（MoE Routing + Expert FFN）**：Gating network 接收 self-attention 输出 x [B, S, 1536]，计算 logits = x·W_g → 加噪声 Softplus(RMSNorm(x·W_n)) → Softmax → KeepTopK(k=2)。每个 token 被路由到 top-2 experts。**每个 expert E_i 结构完全相同**：w1 [1536, 3840] → SiLU → w2 [3840, 1536]，参数量均为 2×1536×3840≈11.8M。8 个 expert 总参数量为 8×11.8M≈94.4M。**缺陷(1)**：所有 expert 能力相同，无法区分处理不同难度 token——容易预测的 token（如常见短语内部词）和困难 token（如跨领域知识推理）被路由到相同能力的 expert，导致"大材小用"或"小材大用"的资源浪费。困难 token 在相同尺寸 expert 中的预测准确度受限。
  - **系统框架层（ZeRO 分布式训练）**：基于 ZeRO 优化器状态分片，8 个 expert 均匀分布在多个 GPU 上。由于所有 expert 尺寸相同，天然负载均衡——每个 GPU 计算量相同。**缺陷(2)**：虽然 baseline 天然负载均衡，但这是通过牺牲 expert 的异构能力换来的（所有 expert 相同尺寸 = 无差异化预测能力）。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch 训练基础设施。
  - **kernel调度层**：论文未明确说明。标准 CUDA kernel 执行矩阵乘法（GEMM for expert FFN），每个 expert 的 GEMM 尺寸完全相同（1536×3840 和 3840×1536），无需特殊 kernel 调度。
  - **硬件架构层**：NVIDIA A800（80GB），NVLink + NVSwitch 节点内互联，ZeRO 模式下跨节点通信通过 InfiniBand（论文未明确说明 inter-node 互联方式）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoDSE（Mixture of Diverse Size Experts）**，核心是在同一 MoE layer 内设置不同 hidden dimension 的 expert，并通过 expert-pair allocation 策略保持 GPU 负载均衡。以 300M×8 模型在 A800 集群上的执行路径为例（expert 尺寸：[4.5×, 0.5×, 4.0×, 1.0×, 3.0×, 2.0×, 2.5×, 2.5×]，对应 h_i [6912, 768, 6144, 1536, 4608, 3072, 3840, 3840]）：
  - **算法层（Diverse Size Expert Routing + Expert Pair）**：
    1. **Diverse Size Experts**：Gating 逻辑不变（同 baseline top-k routing），但 expert 的 FFN hidden dimension 多样化——大专家 h_i > h（如 6912 vs baseline 3840），提供更强预测能力；小专家 h_i < h（如 768），处理简单 token。Experts 按对组织：每对 (i_k^1, i_k^2) 满足 h_{i1} + h_{i2} = 2h = 7680，总参数量与 baseline 一致（每个 pair: 2×1536×7680 = 2×1536×2×3840）。**解决缺陷(1)**：困难 token 可以被路由到大专家获得更强预测能力（Section 4.3 分析显示 CE>2.0 的 180 个高难度 token 中，6215 次选择大专家 vs 3085 次选择小专家，比例为 2:1），容易 token 路由到小专家节省计算。MoDSE 在 700M×8 所有 9 个 benchmark 上均超越 baseline（如 MMLU: 29.9 vs 26.5, SIQA: 60.9 vs 42.9）。
    2. **Auxiliary Load Balance Loss**：沿用 Switch Transformer 的辅助均衡损失 $L_a = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$，f_i 是路由到 expert i 的 token 比例，P_i 是 router 对该 expert 的平均概率。训练后期 token 分布趋于均匀（last epoch: max/min ratio 从早期 >3.0 降至 <3.0，多数在 1.5-3.0 之间）。
  - **系统框架层（Expert-Pair Allocation Strategy）**：每对 expert $(\hat{E}_{i_k^1}, \hat{E}_{i_k^2})$ 放置在同一 GPU 上。由于每对 expert 的参数量之和等于 baseline 两个 expert 的参数量（h_{i1}+h_{i2}=2h），每个 GPU 的总参数量和计算量保持均衡。在 8 expert 4 对×4 GPU 的配置下，每个 GPU 的参数量与 baseline 完全相同。**解决缺陷(2)**：在保持专家异构性的同时实现 GPU 负载均衡，MoDSE 的推理速度与 baseline 几乎相同（如 MMLU: 3min27s vs 3min26s, GSM8K: 20min43s vs 20min26s），无额外推理开销。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch 训练基础设施。
  - **kernel调度层**：论文未明确说明。不同 expert 的 GEMM 尺寸不同（6912 vs 768 宽度），但通过 expert-pair 分配在各 GPU 上对称分布，每个 GPU 处理相同的 expert 尺寸集合，无需特殊调度。
  - **硬件架构层**：同 baseline（NVIDIA A800, NVLink+NVSwitch）。MoDSE 的 expert-pair allocation 确保每个 GPU 的总参数和工作量一致，利用 NVLink 实现 GPU 内多 expert 并行计算，跨节点通过分布式数据并行（ZeRO）同步梯度。论文验证了即使 expert 尺寸多样化，训练和推理速度与 baseline 保持可比（Table 4）。

  核心设计直觉：预训练中 token 预测难度差异巨大——同一短语内 token 极易预测（靠局部上下文即可），跨领域知识推理 token 极难预测（需综合多种知识）。Same-size expert 抹平了这种差异，MoDSE 的 diverse-size expert 让不同能力的 expert "各司其职"：大专家专注高难度推理，小专家高效处理模式化预测，从而实现相同参数预算下更好的 loss 收敛和下游任务表现。

## MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving

- baseline方法是什么？
  Baseline 是现有 MoE 推理 offloading 系统中的专家缓存策略，核心分为三类：
  1. **基于依赖的预取**（DeepSpeed-Inference, HuggingFace TGI）：按计算图执行顺序预测下一层的 expert，将所有可能的 expert 都预取到 GPU，不考虑激活稀疏性，导致 PCIe 链路上大量无效数据传输，GPU 频繁因等 expert 而空闲（GPU idle time 513ms）。
  2. **基于计数的缓存**（BrainStorm, DeepUM）：用全局频率计数器追踪每个 expert 的历史使用次数，假设高频 expert 未来也会被使用。但在跨请求场景下 expert 使用趋于均匀分布，计数方法无法区分请求内的偏斜重用模式，BrainStorm 甚至比按需取 expert 的 vLLM 更差（934ms vs 485ms TPOT）。
  3. **LRU/LFU 局部性缓存**（vLLM, Llama.cpp/Ollama, Mixtral-Offloading）：按最近/最不频繁使用淘汰，仅考虑单个 expert 的访问局部性，不感知同一请求内 expert 间的协同激活关系（grouped activation），导致缓存命中率低。

  全栈执行例子（以 DeepSpeed-Inference 处理 "What is AI?" prompt，DeepSeek-V2-Lite 64×2.4B，A5000 GPU 为例）：
  - **算法层**：Router top-k gating 为每个 token 选择 top-6 experts → dispatch token 到对应 expert FFN
  - **系统框架层**：DeepSpeed-Inference 按计算图依赖顺序预取——处理 layer i 时预取 layer i+1 的所有 64 个 expert，不区分哪些会实际激活。每层需传输大量无用参数 → PCIe 4.0 32GB/s 下仅传输就需数百毫秒 → GPU 大量时间空闲等待
  - **编译框架层**：论文未明确说明。DeepSpeed 使用 PyTorch 原生执行。
  - **kernel调度层**：论文未明确说明。DeepSpeed 使用 cuBLAS GEMM 执行 expert FFN，数据传输用 cudaMemcpy。
  - **硬件架构层**：NVIDIA RTX A5000 (24GB)，CPU host memory ↔ GPU 通过 PCIe 4.0 ×16 (32GB/s)，GPU 计算单元在等待 DMA 传输期间空闲。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE-Infinity 提出了 **Sparsity-Aware Expert Cache**，核心洞察是：batch_size=1 时，同一请求内 expert 激活具有高度稀疏性（<5% experts 被重复使用）和偏斜重用模式（skewed reuse），且相似请求共享相似的 expert 激活组（可被 K-means 聚类为 10-30 组）。但不同请求间的 expert 激活组转换不可预测（Markov 转移概率 <0.3）。因此，预测应通过**匹配请求级激活模式**而非学习跨请求转移规律来实现。

  具体设计：
  1. **EAMC（Expert Activation Matrix Collection）+ 余弦距离匹配**：追踪每个请求的 request-level EAM（L×E 矩阵），在 EAMC 中保存历史 rEAM。新请求的当前 iEAM 与 EAMC 中历史 rEAM 做余弦距离匹配，找到最相似的历史激活模式。**解决缺陷**：相比全局计数（不区分请求）和依赖预取（不感知稀疏性），EAMC 在请求级别捕获专家协同激活模式（S: sparsity 和 G: grouped activation 均满足）。
  2. **PredictEAM + Layer Proximity Decay**：对匹配到的历史 rEAM 进行聚合、行归一化，施加层邻近衰减（1-(i-l)/L），生成 pEAM。**解决缺陷**：考虑了重用预测（R: reuse 属性），并利用 MoE 逐层执行特性——越远的层预测置信度越低，避免过早预取不准确的 expert。
  3. **Probability-aware Cache Eviction**：淘汰 priority score 最小的 expert，score = n_token / ((pEAM + ε) × (1 - layer_idx/L))。综合考虑三个因素：expert 历史激活频率（n_token）、pEAM 中的预测概率、层位置（浅层优先保留，因浅层预取预测置信度低）。**解决缺陷**：LRU 不看未来、计数方法在均匀分布下失效，而 EAMC 匹配利用请求级偏斜模式准确识别哪些 expert 应被淘汰。
  4. **Prefetching 与计算重叠**：根据 pEAM 预取下一层高概率 expert，与当前层 GPU 计算重叠。每个 GPU 独立 I/O 线程使用 pinned memory + DMA 传输。**解决缺陷**：依赖预取全部 expert 导致 PCIe 拥堵阻塞按需取 expert（DeepSpeed），而 MoE-Infinity 只预取少量高概率 expert，保证按需取 expert 的 PCIe 带宽不被占用。

  全栈执行例子（MoE-Infinity 处理同一 "What is AI?" prompt，DeepSeek-V2-Lite 64×2.4B，A5000 GPU）：
  - **算法层**：Router top-k gating 同 baseline（选择 top-6 experts/token），不修改模型结构和路由算法
  - **系统框架层**：Prefill 阶段累积 rEAM → EAMC 余弦匹配（CPU 执行，21μs/query @1K EAMs）。Decode 每次迭代：Layer i Router dispatch → Cache lookup → Hit 直接用 / Miss FetchOnDemand；同时 PredictEAM → 仅预取 layer i+1 中 pEAM 概率 top-k expert（非全部 64 个）；Cache 满时按 probability-aware priority 淘汰。GPU idle time 从 513ms 降至 51ms（3.1–16.7× 加速）
  - **编译框架层**：论文未明确说明。基于 PyTorch 推理运行时，无 graph capture 或 custom compilation pass。
  - **kernel调度层**：每个 GPU 独立 I/O 线程用 pinned memory + DMA 做 CPU→GPU 传输，PCIe 4.0 单线程即可打满 32GB/s。集成 FlashAttention 优化 attention kernel。MoE FFN 用标准 GEMM 执行。
  - **硬件架构层**：同 baseline 硬件（A5000 + PCIe 4.0）。关键优化不在硬件层面，而在减少无效 PCIe 传输量——通过 EAMC 匹配将传输量从"全 expert 预取"降至"仅高概率 expert 预取"，PCIe 带宽用于真正需要的 expert 传输，GPU 闲置率从 baseline 的 51-80% 大幅降低。

## MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

- baseline方法是什么？
  **FlexGen** [42] 是 state-of-the-art 的 memory-constrained 高吞吐批量推理系统。采用 zigzag 计算顺序（逐层加载 weights → GPU 计算 → 卸载输出），支持两种调度模式：
  - **$S_4$ (GPU Attention 模式)**：KV cache 从 CPU H2D 传输到 GPU 做 attention，与 weights transfer 竞争同向 PCIe 带宽。FlexGen 倾向于小 batch size，导致 GPU compute 和 I/O 利用率不足。需要将所有请求 padding 到最大 prompt length。
  - **$S_3$ (CPU Attention 模式)**：attention 在 CPU 执行，但整层 weights 一次性 H2D 传输会长时间阻塞后续微批次的 hidden states H2D，产生大量 I/O 气泡，实际吞吐可能低于 $S_4$。
  
  FlexGen 的 policy 搜索基于 extensive offline data fitting（耗时数小时/天），固定 hardware-model-workload 映射，不考虑 bottleneck resource 随 workload 变化而变化。

  **全栈执行例子（FlexGen $S_4$ — Mixtral 8x7B on T4 16GB, MTBench）**：
  - **算法Pipeline 层**：Mixtral 8x7B 标准 MoE，每层 Top-2 expert gating + 8 expert FFNs + GQA attention。无额外算法优化（无量化/稀疏化/蒸馏）。
  - **系统框架层**：FlexGen 逐层串行执行——(1) 加载所有 experts weights CPU→GPU（一次性传输，占用 PCIe 带宽），(2) GPU 上执行 Attention（KV cache CPU→GPU H2D）+ MoE FFN，(3) 卸载中间结果 GPU→CPU。所有微批次共享同一轮 weights 加载（amortize I/O overhead）。无跨层 I/O 重叠。
  - **编译框架层**：论文未明确说明。PyTorch eager mode，无自定义编译器。
  - **kernel调度层**：GPU attention kernel（FlashInfer/PyTorch SDPA），KV cache D2H/D2D。MoE FFN 使用 PyTorch cuBLAS GEMM。CPU 端仅做 KV cache malloc/free 和 host-device synchronization。
  - **硬件架构层**：T4 GPU (16GB HBM, 65T FP16 FLOPS, 320GB/s BW) + Intel Xeon 24-core (192GB DRAM, ~200GB/s BW) + PCIe Gen3 (~16GB/s BW)。GPU 计算受限于 HBM capacity（仅能容纳少量层的 weights），大量时间消耗在 PCIe weight transfer 和 KV cache H2D 等待。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE-Lightning 提出 **CGOPipe**（高效 GPU-CPU-I/O 流水线调度 + CPU attention + weights paging）和 **HRM**（瓶颈感知的 performance model）两大组件，系统性地解决 FlexGen 的资源利用率不足和次优 policy 问题：

  **CGOPipe 解决 I/O 调度缺陷**：
  - FlexGen $S_4$ 的 KV cache transfer 与 weight transfer 竞争 PCIe 带宽 → CGOPipe 将 decode attention 完全放在 CPU 执行（基于 HRM 分析：attention 的 operational intensity 低于 $P_1$ turning point，传 KV cache 不如在 CPU 算），仅传 hidden states（比 KV cache 小 3-4×），释放 PCIe 带宽给 weight transfer。
  - FlexGen $S_3$ 整层 weights 一次性传输阻塞 hidden states H2D → CGOPipe 的 **weights paging** 将每层 weights 分 n 页（n = 微批次数），在 PostAttn(i,j) 执行时仅传输第 j 页 weights，交错传输 hidden states H2D，消除 I/O 气泡。
  - FlexGen 缺少跨层 pipeline → CGOPipe 通过 **两步超前的 CPU attention**（Algorithm 1，PreAttn(i, j+2) 和 CPUAttn(i, j+2) 提前两个微批次），确保 GPU 始终有可执行的 PostAttn 任务，减少 GPU idle。

  **HRM 解决次优 policy 问题**：
  - FlexGen data fitting 不考虑 bottleneck 变化 → HRM 扩展 Roofline Model 到多层内存层次，明确定义 **turning points**：$P_1$（低于此前不值得跨层传输数据计算）、$P_2$（低于此前吞吐受限于 PCIe 带宽），以及 **balance point**（Eq. 11，GPU BW × $I_{GPU}$ = PCIe BW × $I_{CPU}$，此时达到资源利用均衡点）。MILP 搜索出的策略可精确匹配当前 H/W 和工作负载的瓶颈资源。
  - FlexGen 倾向小 batch size → HRM 识别 GPU memory capacity 决定 throughput 上界，指导尽可能增大 batch size（在满足 CPU memory 约束下）直到达到 balance point。这使得 MoE-Lightning 可以用更少的 CPU memory 达到更高的 throughput（Fig. 1）。

  **全栈执行例子（MoE-Lightning — Mixtral 8x7B on T4 16GB, MTBench, gen_len=128）**：
  - **算法Pipeline 层**：同 baseline（Mixtral 8x7B 标准 MoE）。论文未明确说明对算法层的修改。策略搜索中 $A_g=0, F_g=1$ 表明 attention 全在 CPU 执行、MoE FFN 全在 GPU 执行。
  - **系统框架层**：CGOPipe 执行 Algorithm 1——(1) Prologue 预热前两个微批次，(2) Main Pipeline 逐层执行 GPU PostAttn(i,j) + PreAttn(i,j+2) 与 CPU CPUAttn(i,j+2) 重叠。Weights 分 14 页（14 微批次）交错 H2D。最终 MoE-Lightning (p) 达到 30.12 tokens/s（vs FlexGen 9.5 tokens/s, 3.17×），即使 batch size 减半（504 vs 1112）吞吐仍翻倍，因为消除了 KV cache H2D 竞争和 I/O 气泡。
  - **编译框架层**：论文未明确说明。无自定义编译器 pass，依赖 PyTorch eager execution。
  - **kernel调度层**：GPU 端 PostAttn——O projection (GEMM [μ, h1×h1]) + MoE FFN（gate routing: Top-2 selection + 2 expert FFN GEMM [μ, h1×h2]×2），通过 page table 访问 paged weights。CPU 端——MKL GQA kernel (QK dot + softmax + AV weighted sum)。Weight transfer——CPU→pinned (memcpy) || Pinned→GPU (cudaMemcpyAsync)，pages 间流水线化。
  - **硬件架构层**：T4 GPU (16GB HBM) + Intel Xeon 24-core (192GB DRAM) + PCIe Gen3。CGOPipe 使四个资源并行：GPU SM (GEMM via Tensor Cores)、CPU cores (MKL attention)、PCIe bus (weight pages H2D)、CPU memory controller (KV cache R/W + weight CPU→pinned)。达到 balance point 时 GPU memory capacity 为 throughput 上界——因此添加 TP (multi-GPU) 获得 super-linear scaling（S6→S7: 2.77-3.38×）。

## MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

- baseline方法是什么？
  Baseline 是 MegaBlocks（Gale et al., 2023），将 MoE 计算重新表述为 block-sparse 操作以避免 padding 和 token dropping。MegaBlocks 的 token dispatch 依赖 **基于排序的方法**：将所有 token 的 top-k 选择展平为 (expert_id, token_id) 元组 → 按 expert_id 做 multi-pass radix sort 分组 → index recovery 重建 token 顺序并计算 per-expert range。此方法存在两个核心瓶颈：(1) **激活内存膨胀**——分配 per-expert materialized token buffer（大小 L×K×d，在 DeepSeek 规模下约 94GB），以及 compact 后的 FFN 中间激活（约 98GB），总 activation memory 可达数百 GB；(2) **dispatch 开销**——sorting 需要多次 global memory passes（radix sort 的 pass 数与 key width 成正比），强制 multi-kernel dispatch pipeline（multi-pass sort + segmented scan + index recovery），kernel launch latency 高且 GPU 资源利用率低。

  全栈执行例子（MegaBlocks + DeepSeek 配置，单 H100）：
  - **算法Pipeline 层**：Gate = softmax(W_g · x) → TopK → 为每个 expert e 分配容量 C ≈ γ·LK/E 的固定 buffer → 按 gate score 排序 token 并打包入 buffer（超出容量的 token drop 或路由到 residual path）。
  - **系统框架层**：PyTorch + CUDA custom kernels。Token dispatch 通过 [CUB radix sort](https://nvlabs.github.io/cub/) 实现——flatten top-k 结果 → sort by expert_id → compute offsets → bucketize tokens。
  - **编译框架层**：论文未明确说明。MegaBlocks 使用自定义 CUDA kernel（block-sparse matrix multiplication），无编译器框架修改。
  - **kernel调度层**：dispatch kernel 流程：(1) radix sort kernel（≈4 passes for 16-bit expert_id key）→ 每次 pass 需 read + write L×K 个 (expert_id, token_id) pair，即 O(LK) global memory traffic； (2) segmented scan kernel 计算 per-expert offsets；(3) scatter kernel 将 token 写入 per-expert buffer。FFN kernel：(1) 加载 materialized routed token buffer → W1 GEMM；(2) 存储中间激活 (L×h) → activation function → W2 GEMM → 输出 buffer。激活内存 = L×K×d（routed token buffer）+ L×h（FFN 中间激活），对 SwiGLU 额外 ×2。
  - **硬件架构层**：NVIDIA H100 GPU。Memory bandwidth bound 是主要瓶颈——activation function 为 point-wise 操作，在 tall-and-skinny 矩阵（L≫d）下 memory bandwidth bound；radix sort 的多次 global memory pass 受限于 HBM bandwidth（3.35 TB/s）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoEBlaze 通过三个协同设计打破 memory wall：(1) **索引替换 materialized buffer**——用四组轻量级索引数据结构（expert_token_indices、expert_token_offsets、token_expert_indices、token_index_map，各 L×K 个 int32，总计 4×L×K×4 bytes vs MegaBlocks 的 L×K×d×2 bytes）替代 per-expert materialized token buffer；(2) **atomic-free 并行 dispatch 构建**——3-step kernel（dense map 构建 → warp-level reduction 计计数 → tile-level scan + location map 写入）替代 multi-pass radix sort，利用 shared memory prefix sum 和 warp-level reduction，避免 atomics 和多次 global memory pass；(3) **kernel fusion + activation checkpoint**——fused SwiGLU kernel 将 W1/W2 两个 GEMM + SiLU + element-wise multiply 融合为单 kernel，forward 中仅保存 a, b, y_swi（SiLU(a) 不保存），backward 时 recompute SiLU。

  全栈执行例子（MoEBlaze 对比 MegaBlocks）：
  - **算法Pipeline 层**：
    - **MegaBlocks**：Gate → sort-based dispatch → 写入 materialized routed buffer → W1 → 存储中间激活 → act → W2 → 输出 → 反向需要 (L,d)→(L×k,d) 展开。
    - **MoEBlaze**：Gate → 构建 4 组索引数据结构（仅 int32 IDs） → on-the-fly gather x[token_ids] → fused W1+W2+SwiGLU（单 kernel） → 仅保存 a, b, y_swi → W3 → on-the-fly reduction via token_index_map → 反向通过 scatter（逆向索引）直接映射梯度，无需展开。
    - 关键差异：MoEBlaze 不分配 L×K×d 的 routed token buffer（节省 ~94GB for DeepSeek scale），仅分配 4×L×K 个 int32 索引（~16MB for L=2M, K=4）。FFN 计算中 recompute SiLU 进一步节省 L×h 的激活存储。
  - **系统框架层**：PyTorch 2.0.1 + CUDA 12.1 自定义 kernel。替换 MegaBlocks 的 sort-based dispatch pipeline 和 block-sparse FFN 为 MoEBlaze 的 index-based dispatch + fused FFN kernel。触发时机相同（每个 MoE layer 的 forward/backward），但 memory footprint 和 kernel launch chain 大幅缩短。
  - **编译框架层**：论文未明确说明。所有优化在 CUDA kernel 层面，无编译器框架修改。
  - **kernel调度层**：
    - **MegaBlocks dispatch**：radix sort（≈4 global memory passes）× 每次 O(LK) + segmented scan + scatter → kernel launch chain length ≈ 6-8，总 global memory traffic ≈ 8×L×K×(8+4) bytes（sort key + value）。
    - **MoEBlaze dispatch**：3-step kernel chain：(1) dense map fill（1 pass, L×E writes）；(2) warp-level count（1 pass, E reductions）；(3) tile-level scan + location write（1 pass, L×K writes）。Kernel launch chain length ≈ 3，总 global memory traffic ≈ L×E + L×K（int32 writes），无 sort 的多次 full pass。
    - **MegaBlocks FFN**：W1 GEMM → store a, b → SiLU compute（load a, compute, store SiLU(a)）→ element-wise multiply（load SiLU(a), b, compute, store y_swi）→ W3 GEMM → backward 需要 a, b, σ(a), SiLU(a), y_swi 全部在 HBM 中。
    - **MoEBlaze FFN**：fused kernel 内：load x once → stream through W1 GEMM, W2 GEMM simultaneously → compute SiLU(a) in register → y_swi = SiLU(a)⊙b in register → store a, b, y_swi only → W3 GEMM。Backward：recompute SiLU(a) from a（element-wise O(L×h)，memory bandwidth bound → recompute cost ≈ 直接读 HBM 的 cost）。SwiGLU 下节省 5 个中间 tensor 的 HBM write/read（a, b, σ(a), SiLU(a), y_swi_product），内存节省最显著（最高 4× reduction，conf3 从 40GB→10GB）。
  - **硬件架构层**：NVIDIA H100 GPU。利用 WGMMA（warp-group matrix multiplication）和 TMA（Tensor Memory Accelerator）加速 fused GEMM。Fused kernel 将 computation 从 memory-bound domain 推向 compute-bound domain——原本 activation function 受限于 HBM bandwidth（3.35 TB/s），融合后 point-wise ops 在 register/shared memory 完成，仅需读 x 一次（vs 两次）且无中间 global write。Speedup 在 SwiGLU 下更显著（2×–6.2× vs 1.4×–3.7× for SiLU），因为 SwiGLU 的中间激活更复杂、记忆节省更大。

## MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing

- baseline方法是什么？
  Baseline 是 Megatron-LM 的默认 expert parallelism，采用 contiguous block 专家放置策略：将每层的 E 个 expert 按索引顺序均匀分配给 G 个 GPU（如 8 experts / 4 GPUs → GPU0 分配 experts 0-1、GPU1 分配 experts 2-3...）。此策略仅考虑内存均衡（每个 GPU 等量 expert 参数），不考虑：**(1) Token 处理负载不均衡**——某些 expert 被激活频率远高于其他 expert（如 layer 14 中 experts 0-1 处理 64% token，layer 23 中 experts 6-7 处理 69% token），导致 hosting GPU 处理时间远超其他 GPU，产生计算尾延迟；**(2) 跨 GPU 通信倾斜**——all-to-all token dispatching 中不同 GPU pair 间通信量严重不均，某些 pair 通信量远超其他，带宽利用不均导致通信尾延迟。在 Mixtral-8x7B 中 all-to-all 通信占端到端推理时间的 35.7%。
  全栈执行例子（Baseline - Megatron-LM contiguous placement）：
  - **算法层**：Mixtral-8x7B，32 层 MoE decoder，每层 8 experts + top-2 routing → 每个 token 在每层由 router 选择 top-2 expert → 输出为两个 expert 的加权和。
  - **Serving/框架层**：Megatron-LM 初始化 → expert parallel size=4 → 每层 experts {0,1}→GPU0, {2,3}→GPU1, {4,5}→GPU2, {6,7}→GPU3（contiguous block）→ 每个 micro-batch 到达 → token dispatch: 每个 GPU 上的 token 按 router 决策 → all-to-all send 到目标 expert 所在 GPU → expert FFN 计算 → all-to-all send 结果回原 GPU。
  - **通信执行**：All-to-all 通过 NCCL group 实现 → intra-node 走 NVLink（900GB/s）、inter-node 走 IB（400/800Gbps）→ layer 14 中 expert 0-1 处理 64% token → GPU0 接收大量 remote token、GPU1-3 接收少 → GPU1-3 完成通信后等待 GPU0 → 通信尾延迟。
  - **计算执行**：GPU0 上 experts 0-1 的 token 处理量远大于 GPU1-3 → GPU0 GEMM 执行时间长 → GPU1-3 完成计算后 idle 等待 GPU0 → 计算尾延迟。
  - **Kernel/硬件层**：论文未明确说明 kernel 细节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoETuner**，一个基于 ILP 的专家放置优化框架，通过两阶段 ILP 求解最优 expert-to-GPU mapping，打破 contiguous block 限制。

  **对应缺陷 1（Token 处理负载不均衡）→ ILP 1: Load-Balanced Expert Clustering**
  - 利用 token routing profiling 收集的 P_{e,l}（每个 expert 的 token 处理量）→ 决策变量 x_{c,e,l}（expert e 是否归入 cluster c）→ 目标 min Σ|T_{c,l} - T̄_l|（最小化 cluster 负载与层均值的偏差）→ 约束每个 cluster 至少一个 expert。
  - **效果**：将高频和低频 expert 混合分配到同一 GPU cluster，使各 GPU 处理的 token 总量接近。例如，layer 14 中 experts 0-1 不再同属一个 GPU，而是与低频 expert 搭配。单节点减少 token 处理尾延迟 36%，平均延迟 34.8%。

  **对应缺陷 2（跨 GPU 通信倾斜）→ ILP 2: Cluster-to-GPU Assignment**
  - 利用跨层 token 路由依赖 R_{e_1,e_2,l} 预计算 cluster 间通信成本 C_{c_1,c_2,l} → 目标 min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2})（最小化每层跨 GPU pair 的通信 tail）→ 约束每个 GPU 等量 expert、cluster-GPU 一对一映射。
  - **核心洞察**：token 在相邻层间存在路由亲和性——若 token 在 layer l 路由到 expert e_1，在 layer l+1 很可能路由到特定少数 expert → 将频繁跨层通信的 expert 放在同一 GPU，消除跨 GPU 通信。
  - **效果**：单节点减少 all-to-all tail 延迟 36.3%，平均延迟 35.4%；多节点减少 tail 30.5%、平均 24.7%。

  全栈执行例子（MoETuner）：
  - **Profiling 阶段**：在 WikiText-103 采样子集上运行 Megatron-LM 推理 → 逐 token 记录路由路径 → 构造 P_{e,l} 和 R_{e_1,e_2,l} 表 → 验证小样本路由统计可近似全数据集模式。
  - **ILP 1 求解**（Gurobi 12.0.0，tolerance 0.025）：输入 P_{e,l}（如 layer 14: expert 0→300 tokens, expert 1→280, expert 2→80, ...）→ 将 8 experts 聚类到 4 clusters → 输出 x_{c,e,l}：cluster 0={expert 0, expert 3}（高+低）、cluster 1={expert 1, expert 6}... → 确保各 cluster token 总量约等于 T̄_l。
  - **ILP 2 求解**：用 x_{c,e,l} 计算 C_{c_1,c_2,l} = Σ R_{e_1,e_2,l} · x_{c_1,e_1,l} · x_{c_2,e_2,l} → 在 B_{g_1,g_2}（NVLink 900GB/s intra-node, IB 400Gbps inter-node）约束下 → 输出 y_{c,g,l}：layer l 中 cluster 0→GPU1, cluster 1→GPU2, ...；layer l+1 中 cluster 0→GPU3, cluster 1→GPU1, ... → 最大化跨层同 GPU expert 对。
  - **部署阶段**：Megatron-LM 加载 Mixtral-8x7B → 读取 expert-to-GPU mapping tensor → 初始化 expert parallel 布局 → 推理时 token dispatch 按优化后 layout 执行 all-to-all → 例如 layer 4 中 expert 7 和 layer 5 中 expert 6 同放 GPU-a → 大比例 token 在层间无需跨 GPU 通信 → 端到端加速 9.3%（单节点 8×H100）和 17.5%（多节点 16×H200）。
  - **Kernel/硬件层**：论文未明确说明 kernel 细节。

## MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

- baseline方法是什么？
  Baseline 是标准确定性 Top-K 路由的 MoE 推理（greedy decoding）。每层 MoE 中，每个 token 仅激活 k 个 expert（router logits → softmax → top-k → FFN），其余 E−k 个 expert 处于闲置状态。全栈执行例子：
  - **算法 Pipeline**：输入 token → embedding → 逐层 Attention + MoE FFN。MoE 层中 router 计算 $\mathbf{R} \in \mathbb{R}^E$ → softmax → TopK 选择 k 个 expert → 各 expert SiLU(W_gate·h) ⊙ (W_up·h) → W_down 投影 → 加权求和 → 残差连接 → 最终 lm_head logits → argmax 取下一 token。整个推理过程完全确定性，每 token 只走一条内部计算路径。
  - **系统框架/Serving调度**：论文未明确说明 baseline 使用的 serving 框架。实验使用标准 PyTorch（HuggingFace Transformers）单 batch 推理，无特殊调度优化。
  - **编译框架/Kernel调度/硬件架构/芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  Baseline 的核心缺陷：标准 MoE 推理每 token 仅激活 k 个 expert，大量已训练好的 expert（E−k 个）在推理时闲置，模型内部知识未被充分利用。增加 top-k 的 active expert 数量不能奏效，因为模型训练时只见过 k-expert 聚合模式，直接增加 k 会导致训练-推理不匹配。这限制了 MoE 模型的"潜力天花板"——模型参数远多于每 token 实际使用的参数。

  RoE 通过三项设计解决：

  **对应缺陷 1（Expert 利用率低）→ Gumbel-Top-K 随机路由**
  - 在 router logits 上注入可控 Gumbel 噪声：$\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$
  - Gumbel-Max 性质保证这是一个从 router 定义的 categorical 分布中无放回采样的过程——高 logit 的 expert 仍更可能被选中，但低 logit 的 expert 也有机会被激活。
  - 每个 token 运行 n 次独立采样 → n 条不同的内部计算路径 → n 个候选 logits → 概率平均聚合。
  - 效果：以概率方式探索了训练期间见过的各种 expert 组合，充分释放 MoE 的"潜力"。

  **对应缺陷 2（计算开销）→ Batched Inference + Clean Cache**
  - 将 n 次独立 forward 合并为单次 batched call，利用 GPU 的 sub-linear batch scaling 减少 wall-clock 时间。
  - Clean Cache：batch 中第一个样本使用确定性路由（τ=0）产生共享 KV-cache，其余样本复用此 cache，无需维护 n 份 KV-cache。内存开销与单样本完全相同。

  全栈执行例子（对比 baseline）：
  - **算法 Pipeline**：输入 token → embedding → 逐层 Attention（共享 KV-cache，sample 0 计算一次，其余复用）→ MoE 层中 router 计算 R ∈ ℝ^E → 对 batch 中 n 个样本分别采样 Gumbel 噪声（sample 0 用 τ=0 确定性路由，sample 1..n-1 用调优后的 τ_l）→ TopK 选择不同的 expert 组合 → 并行 FFN 计算 → 各样本独立残差连接 → 最终 n 组 lm_head logits → softmax 后概率平均 → argmax 下一 token。
  - **系统框架**：论文实现了 custom batched inference（HuggingFace Transformers 级别），非标准 serving 框架修改。
  - **温度搜索**：Optuna TPE 逐任务搜索逐层温度 τ_l，搜索空间受两个启发式约束：仅中段层参与搜索（首尾层 τ=0），温度上限 0.5。
  - **编译框架/Kernel调度/硬件架构/芯片设计**：论文未明确说明。

  **关键量化结果**：OLMoE-7B + RoE (K=32) 达到 10.5B 标准 MoE 的性能水平，同时内存开销减少 25%，每 token 延迟减少 30%。使用 64 samples 时 GPU 内存仅增加 12%，功耗增加 20%。


## MoH: Multi-Head Attention as Mixture-of-Head Attention

- baseline方法是什么？
  Baseline 是标准的 multi-head attention（MHA），将输入投影到 h 个低维子空间，每个 head 独立做 scaled dot-product attention，最后将所有 head 输出求和（summation form）：MultiHead(X, X') = Σ_{i=1}^{h} H^i · W_O^i。所有 head 对所有 token 均等激活，不存在 token 级别的选择性。

  全栈执行例子（以 ViT 图像分类为例）：
  - **算法pipeline**：一张 224×224 图像 → patch embedding 为 T 个 token (T×d_in) → 每层 multi-head attention：所有 h 个 head 并行计算 Q_i/K_i/V_i → Scaled Dot-Product Attention → 求和拼接 → 输出投影 → FFN → 下一层。标准 MHA 中所有 h head 对所有 T token 全激活。
  - **系统框架**：训练时使用 TransNeXt 框架在 8 GPUs 上做数据并行；推理时标准 PyTorch 前向传播，无特殊调度。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 cuBLAS batch GEMM 实现 multi-head attention，所有 head 的 QKV 投影和 attention 计算均为 dense 矩阵乘法。论文 Inference Time 实验（Tab.7）中，head dim=64, seq=256 时 MHA 耗时 0.360ms，seq=512 时 1.376ms。
  - **硬件架构**：论文未明确说明具体 GPU 型号。
  核心缺陷：（1）**注意力头冗余**——多个 head 可能学习相似特征，许多 head 可被剪枝而不影响精度（Voita et al. 2019, Michel et al. 2019）；（2）**token 级无差别计算**——所有 token 经过所有 head，但不同 token 可能需要不同 head 的关注模式，造成计算浪费；（3）**求和缺乏灵活性**——标准 MHA 对所有 head 等权求和，无法根据 token 内容动态调整各 head 贡献。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoH（Mixture-of-Head Attention），将 attention head 视为 MoE expert，引入 router 为每个 token 动态选择 Top-K head。核心设计：（1）**Heads as Experts + Router**——通过可学习的 W_s/W_r 将 token routing 到不同的 attention head，仅激活 Top-K 个 routed head（+ 所有 shared head），显著减少激活 head 数（50%~90%）；（2）**Shared Heads**——前 h_s 个 head 始终激活，捕获通用知识（如语法规则），剩余 head 动态路由处理 token-specific 信息，减少冗余；（3）**Two-Stage Routing**——W_h 产生 α_1/α_2 系数动态平衡 shared 和 routed head 权重，实现加权求和替代标准等权求和；（4）**Load Balance Loss**——防止 routing collapse，确保所有 routed head 得到充分训练。

  全栈执行例子（以 MoH-ViT 为例）：
  - **算法pipeline**：一张 224×224 图像 → patch embedding 为 T 个 token → 每层 MoH attention：token x_t 输入 router → W_s/W_r 计算 routing score → shared head 全部激活 + Top-K routed head 激活 → 仅激活的 head 计算 Q_i/K_i/V_i 和 Scaled Dot-Product Attention → 以 g_i 加权求和 → 输出投影 → FFN → 下一层。激活 head 的预算在各层不均匀分布——浅层激活少、深层激活多。
  - **系统框架**：训练基于 TransNeXt（ViT）、DiT（扩散模型）、Megatron（LLM）框架；LLaMA3-8B 的 continue-tuning 分两阶段——第一阶段 300B tokens 做数据分布适配，第二阶段 100B tokens 转为 MoH 模型（含参数无关 router + straight-through estimator 量化 routing score 保持输出分布稳定）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：论文 Inference Time 实验（Tab.7）展示：将 Q/K/V 特征通过 router mask 转为稀疏矩阵，用稀疏矩阵乘法替代 dense 矩阵乘法。head dim=64, seq=256 时：90%激活 0.352ms，75%激活 0.321ms，50%激活 0.225ms（分别比 MHA 的 0.360ms 快 2.2%/10.8%/37.5%）。seq=512 时加速更显著（50%激活 0.863ms vs MHA 1.376ms，快 37.3%）。
  - **硬件架构**：论文未明确说明。

  关键效果：MoH-ViT-B 用 75% head 达 84.9% Top-1 Acc（TransNeXt 100% head 84.8%）；MoH-DiT-XL/2 用 90% head 达 FID 2.94（DiT-XL/2 100% head FID 3.22）；MoH-LLaMA3-8B 用 75% head 在 14 benchmark 上平均 64.0%（LLaMA3-8B 61.6%），仅需 100B continue-tuning tokens。

## MoLA: MoE LoRA with Layer-wise Expert Allocation

- baseline方法是什么？
  Baseline 是 **标准 LoRA-MoE（即 MoLA-□ / MoLA Rectangle）**——将 LoRA 与 MoE 结合，在 Transformer 每层使用**相同数量**的 LoRA expert，通过 router 做 top-K 路由选择。典型代表如 MoELoRA (Liu et al., 2023)、LoRA-MoE (Dou et al., 2023)、MoLORA (Zadouri et al., 2023)。

  核心缺陷：
  1. **不考虑层级差异**：所有 Transformer 层分配相同数量的 expert。但不同层处理的信息粒度不同——底层处理 token-level 特征（词义、语法），中层学习有效表示，高层处理抽象推理。等量分配忽略了这一层级差异，导致底层 expert 冗余、中高层 expert 不足。
  2. **底层 expert 高度冗余**：底层的 LoRA expert 学习的低秩矩阵彼此非常相似（Frobenius Norm 小），多个 expert 产生重叠表示，浪费参数预算。
  3. **中高层 expert 能力受限**：中高层需要处理多样化的抽象特征和任务特定模式，但固定的 expert 数量限制了其 fitting 能力，无法充分学习 fine-grained task-specific 模式。

  全栈执行例子（以 LLaMA-2-7B MoLA-□(5555) 微调为例）：
  - **算法pipeline**：输入 token 序列 x → 第 j 层（j=1..32）self-attention：对每个 token，W_q/W_k/W_v/W_o 各创建 5 个 LoRA expert（A_i,B_i 低秩对, r=8），router W_r 计算 5-dim softmax → top-2 选择 → 两个 expert 的 A_iB_i x 加权求和加到原始 W_0 x 上 → MLP 同理（W_gate/W_down/W_up 各 5 expert）。第 1 层和第 32 层用同样 5 个 expert → 底层 expert 间 Frobenius Norm 差异小（冗余大），高层 expert 间差异大但 expert 数不足。
  - **系统框架**：Hugging Face Transformers 训练循环，PyTorch 数据并行在 8×A100-40G 上，AdamW 优化器，仅训练 LoRA expert 和 router 参数，预训练权重 W_0 冻结。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch 矩阵运算，无自定义 kernel。各 linear module 的 LoRA expert 计算为 batch 低秩矩阵乘加操作。
  - **硬件架构**：A100-40G GPU + A6000 GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoLA（MoE-LoRA with Layer-wise Expert Allocation）**，核心创新是**为不同 Transformer 层分配不同数量的 LoRA expert**。关键设计：
  1. **层级别灵活 expert 分配**：每层 j 分配 N_j 个 expert，ΣN_j 为总 expert 预算。不是所有层用相同数量，而是根据层级重要性差异化分配。
  2. **五种层级别配置假设**：MoLA-△（底层多）、MoLA-▽(2468, 高层多）、MoLA-▷◁(8228, hourglass，两端多）、MoLA-✸(2882, diamond，中层多）、MoLA-□(5555, rectangle，等量 baseline）。
  3. **全 dense weight 覆盖**：与 LoRA-MoE (Dou et al., 2023) 仅对 FFN 应用不同，MoLA 对 attention 的 W_q/W_k/W_v/W_o 和 MLP 的 W_gate/W_down/W_up 全部应用 LoRA expert。
  4. **专家冗余的定量分析**：通过 Frobenius Norm 量化各层 expert 间相似度，发现底层 expert 差异更小（更冗余），高层 expert 差异更大。

  对应缺陷的解决：
  - **缺陷 1（忽略层级差异）→ 层级别灵活分配**：不再强制所有层相同 expert 数量。实验证明 MoLA-▽ (2468) 和 MoLA-✸ (2882) ——即中层/高层分配更多 expert ——在三个 base model（LLaMA-2、Mistral、Gemma）上均优于等量分配，即使总参数量相同甚至更少。
  - **缺陷 2（底层冗余）→ 减少底层 expert**：Frobenius Norm 分析证实底层 expert 最相似。极端配置实验（10-2-2-2 vs 2-2-2-10）显示底层过多 expert 平均性能最低。MoLA-▽ 将底层 expert 从 5 减至 2，将节省的参数分配给中高层，性能反而提升。
  - **缺陷 3（中高层能力受限）→ 增加中高层 expert**：中高层分配更多 expert 可增强 fitting 能力。MoLA-▽ (2468) 以仅 62.5% 的参数量（vs MoLA-□ 8888）在部分 benchmark 上取得更好或相当性能，证明了参数效率。

  全栈执行例子（以 LLaMA-2-7B MoLA-▽ (2468) 微调为例）：
  - **算法pipeline**：输入 token 序列 x → 第 1-8 层（底层）：每层仅 2 个 LoRA expert（总 16 expert），router 选择 top-2 → 第 9-16 层：每层 4 expert（总 32） → 第 17-24 层：每层 6 expert（总 48） → 第 25-32 层（高层）：每层 8 expert（总 64）。总 expert 数 = 8×(2+4+6+8) = 160，与 MoLA-□(5555) 的 32×5=160 总 expert 数相同但性能更优。router 的 top-2 选择 + load balancing loss 确保所有 expert 被充分训练。Frobenius Norm 分析显示：底层 expert 间差异 ~0.1，高层 ~0.6（避免冗余最大化利用度）。
  - **系统框架**：Hugging Face Transformers + PyTorch，与 baseline 相同训练框架，仅在模型结构中为每层配置不同 expert 数量。训练循环无额外计算开销（每 token 的激活 expert 数 = K×7 个 linear module = 2×7 = 14 个 LoRA 前向，与 MoLA-□ 相同）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：无特殊 kernel 优化，标准 PyTorch 矩阵运算。每层 expert 数的差异不影响 kernel 执行路径——所有 expert 的 A_i B_i x 均为独立低秩矩阵乘法。
  - **硬件架构**：A100-40G / A6000 GPU，与 baseline 相同硬件，无需定制硬件支持。

  关键效果：MoLA-▽ (2468) 在 LLaMA-2 上以 105.6M 可训练参数（1.5% of 7B）超越 LoRA（159.9M）和 MoLA-□(8888)（169M），在 CommonsenseQA 上达 78.95%（vs LoRA 75.51%, MoLA-□5555 78.13%）。Continuous Learning 中 MoLA-▽ 的 performance drop 仅 -0.47%（vs LoRA -2.17%），显示优越的抗遗忘能力。

## Muon is Scalable for LLM Training

- baseline方法是什么？
  Baseline 是 AdamW 优化器，作为当前大规模 LLM 训练的事实标准。AdamW 的核心机制：(1) 维护两个动量 buffer（一阶动量 m_t 和二阶动量 v_t）；(2) 通过自适应学习率 η_t / (√(v_t) + ε) 进行逐元素更新；(3) 从 steepest descent 视角看，AdamW 是 Max-of-Max norm 约束下的最陡下降，其 norm constraint 动态变化。
  
  Baseline 缺陷：
  (1) **计算效率不足**：AdamW 的逐元素自适应更新虽然稳定，但计算效率受限于二阶矩估计和 element-wise 操作。相比矩阵级正交化更新，AdamW 在相同计算预算下达到的 loss 更高——scaling law 拟合显示 AdamW 的 loss-C 曲线为 2.608 × C^(-0.054)，高于 Muon 的 2.506 × C^(-0.052)。
  (2) **优化方向多样性不足**：AdamW 的逐元素更新缺乏对矩阵整体结构的考虑。从 steepest descent 角度看，AdamW 使用的 norm constraint 是动态变化的 Max-of-Max norm，而非更合理的 operator norm（spectral norm），导致权重矩阵在低维主导方向上过拟合。
  (3) **对非矩阵参数无特别优势**：对 RMSNorm、embedding 等非矩阵参数，AdamW 的逐元素更新是合理的。但对于矩阵参数（attention 投影、FFN），缺乏矩阵级结构利用。

  全栈执行例子（AdamW 训练 Llama 密集模型）：token → embedding → 逐层 attention（QKV 投影 [H,H] 矩阵 × 输入 → attention → output 投影 [H,H]）→ FFN（[H, 2.6H] up/gate → SwiGLU → [2.6H, H] down）→ RMSNorm → LM head → cross-entropy loss → 反向传播得到各矩阵梯度 G → AdamW：m_t = β₁ m_{t-1} + (1-β₁) G, v_t = β₂ v_{t-1} + (1-β₂) G² → 更新 ΔW = -η m̂_t / (√(v̂_t) + ε) + λW → 每个矩阵元素独立更新，无矩阵级正交性约束 → 权重矩阵的奇异值分布逐渐集中在少数主导方向 → SVD entropy 偏低 → 模型容量利用不充分。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出扩展 Muon 优化器进行大规模 LLM 训练，核心理念：**对矩阵参数使用矩阵正交化更新替代逐元素自适应更新**。Muon 将 momentum 矩阵通过 Newton-Schulz 迭代进行近似正交化（≈ (M M^T)^(-1/2) M = U V^T），使得更新矩阵的奇异值全部为 1，确保在所有方向上均匀更新。从 steepest descent 角度，Muon 提供的是 spectral norm（operator norm）约束，对矩阵参数而言比 AdamW 的动态 Max-of-Max norm 更合理。三项关键技术保证可扩展性：

  **对应缺陷 1（计算效率不足）→ 矩阵正交化 + Consistent Update RMS**
  - Muon 的正交化更新迫使参数在所有奇异向量方向上等强度学习，避免在少数主导方向过拟合，使相同 FLOPs 下的有效学习更充分。Scaling law 显示 Muon 仅需 ~52% FLOPs 即可匹配 AdamW 性能。
  - Consistent Update RMS：Lemma 1 证明 shape [A,B] 矩阵的 Muon 理论更新 RMS = √(1/max(A,B))，导致不同 shape 矩阵更新尺度不一致。通过缩放因子 0.2·√(max(A,B)) 统一所有矩阵参数的更新 RMS，鲁棒的训练行为消除了针对不同 shape 矩阵的手动调参需求。

  **对应缺陷 2（优化方向多样性不足）→ Spectral Norm 约束 + Weight Decay**
  - Muon 的 spectral norm 约束（当 Newton-Schulz 精确计算时）比 AdamW 的 Max-of-Max norm 更匹配权重矩阵作为 operator 的数学本质。SVD entropy 实验证实：Muon 训练的权重矩阵在 90%+ 情况下 SVD entropy 高于 AdamW，singular value 分布更平坦，意味着模型在学习更丰富的特征方向。
  - Weight decay 解决原始 Muon 在 long-training regime 中权重 RMS 持续增长问题：vanilla Muon 初期收敛快但长期权重发散超出 bf16 范围，加入 λW 项后 Muon 在过训练区间持续优于 AdamW。

  **对应缺陷 3（分布式兼容性）→ Distributed Muon + ZeRO-1**
  - 分布式 Muon（Algorithm 1）：在 Megatron-LM 的 ZeRO-1 框架下，增加 bf16 DP Gather 操作从分片梯度恢复到全梯度矩阵进行 Newton-Schulz 迭代，计算完成后丢弃非本地分片。额外通信开销仅为 Distributed AdamW 的 0~25%（在多个 DP 组下几乎无感知）。
  - Muon 仅需 1 个动量 buffer（vs AdamW 的 2 个），内存消耗减半。
  - 非矩阵参数（RMSNorm、embedding、LM head）继续用 AdamW 处理，两优化器共享 lr 和 weight decay，无缝集成。

  全栈执行例子（Muon 训练 Moonlight MoE 模型）：token → embedding (AdamW) → 逐层 attention：QKV 投影矩阵 [H,H] → Muon 正交化更新使 QKV 学习多样化 query/key/value 子空间 → attention → output 投影 [H,H] (Muon) → FFN experts：各 expert 的 up/gate/down 矩阵 (Muon) → router → top-k 选择 (AdamW 对 router 权重) → shared expert (Muon) → RMSNorm (AdamW) → 反向传播得到各矩阵梯度 G → Muon: reduce-scatter(G) → momentum → gather → Newton-Schulz 5 步迭代 → 0.2·√(max(A,B))·O_t + 0.1·W → 结果：所有矩阵更新奇异值均匀（SVD entropy 高），router 权重多样性显著提升（专家选择更差异化），最终在 5.7T tokens 后 MMLU=70.0, GSM8K=77.4, HumanEval=48.1。

## NetMoE: Accelerating MoE Training through Dynamic Sample Placement

- baseline方法是什么？
  Baseline 是现有 MoE 分布式训练框架（FastMoE、FasterMoE、SmartMoE），它们采用静态的样本放置策略：每个 GPU 上的训练样本在迭代内固定不变，All-to-All 通信时 tokens 按 routing 结果发送到对应 expert 所在 GPU，计算结果再按原路径返回。FasterMoE 通过动态 expert placement 减少通信量（将热门 expert 复制到多个 device），但调整 expert 位置需要传输大量 expert 参数（expert size 远大于 sample size），因此无法每迭代调整，导致次优。SmartMoE 在 FasterMoE 基础上增加负载均衡，但主要针对计算负载而非通信效率。

  全栈执行例子（FastMoE baseline, MoE-GPT-S, 2 nodes × 8 A800 GPUs）：
  - 算法 Pipeline：GPT-2 backbone MoE 模型，每层 MoE layer 有 gating network（softmax top-K routing, K=2）+ E=16 experts（FFN）。token → embedding → attention → RMSNorm → gating network 产生 `route ∈ N^{I×L×K}` → tokens 按 routing 分发到各 expert 所在 GPU → experts FFN 计算 → 结果按原 token 位置 gather 回来 → + residual → 下一层。
  - 系统框架（训练框架）：FastMoE/FasterMoE/SmartMoE on PyTorch。样本在迭代开始时固定分配到各 GPU（data parallelism 方式），每个 GPU 持有 I/J 个样本。All-to-All scatter + gather 操作通过 NCCL 通信原语在 NVLink（intra-node 400 GB/s）和 InfiniBand（inter-node 100 GB/s）上执行。
  - 编译框架：论文未明确说明（使用 PyTorch 默认 JIT，无自定义编译器）。
  - Kernel 调度：论文未明确说明（使用标准 NCCL All-to-All kernel + cuBLAS GEMM for expert FFN）。
  - 硬件架构：NVIDIA A800-SXM4-40GB GPU，NVLink 400 GB/s intra-node，InfiniBand 100 GB/s inter-node。baseline 下 inter-node 通信量约占总通信量的 50-75%（取决于 expert 分布），而 inter-node 带宽仅为 intra-node 的 1/4，导致 inter-node 通信成为主要瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  NetMoE 提出**动态样本放置（Dynamic Sample Placement）**：在每个 MoE layer 的 All-to-All gather 阶段，不将 tokens 返回原 GPU，而是按优化后的目标位置发送——将原本需要通过 inter-node 传输的 tokens 改为在 intra-node 甚至 intra-device 传输。

  **对应缺陷 1（样本固定 → 跨节点通信浪费）→ Stage 1 跨节点二分图匹配**
  - Baseline 中样本位置固定不变，tokens 的 routing 目标 expert 的分布是动态且有偏的（同一 sample 的 tokens 倾向于路由到相同 expert），导致大量 tokens 需要跨节点传输。
  - NetMoE Stage 1 将"每个 sample 分配到哪个 node"建模为加权二分图最小权完美匹配问题：左侧 P 为 I 个 samples，右侧 Q 为 N 个 nodes（每个 node 容纳 I/N samples），边权重 `W_{i,n} = c_{i,n}^{(l,gather)} + c_{i,n}^{(l+1,scatter)}` 表示 sample i 放在 node n 时产生的跨节点通信量（tokens 数），使用 Kuhn-Munkres (KM) 算法 O(I^3) 求解，使跨节点通信总量最小化。

  **对应缺陷 2（仅靠节点级优化不够 → Stage 2 节点内二分图匹配）**
  - Stage 1 只确定 sample 到 node 的映射，node 内多 GPU 间的各 sample 放置仍有优化空间。
  - NetMoE Stage 2 在每个 node 内独立求解第二个二分图匹配问题：左侧为分配到该 node 的 I/N 个 samples，右侧为该 node 上的 J/N 个 GPUs（每 GPU 复制 I/J 次），KM 算法求解最小化 intra-node 通信量的样本-GPU 分配。N 个 node 的 Stage 2 并行求解。

  **对应缺陷 3（求解效率不足 → 求解全过程被通信+计算隐藏）**
  - KM 算法在 CPU 上后台线程执行，求解时间与 All-to-All scatter + expert computation 重叠（Table 4 显示 KM 时间 0.48ms < scatter+computation 7.13ms），实现零额外开销。
  - Expert Residual Inlining：将 `output = input + expert_output` 从 gather 之后移到 scatter 之后、gather 之前执行，保证样本位置变化后计算逻辑等价。
  - 下一层 routing 预测：将当前层 input 传入下一层 router 提前获取路由结果，为 Stage 1/2 求解提供 `c^{(l+1,scatter)}` 信息。

  全栈执行例子（NetMoE, MoE-GPT-S, 2 nodes × 8 A800 GPUs）：
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

## Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

- **baseline方法是什么？**
  Baseline 是未经任何 expert 剪枝的原始 Mixtral 8x7B/Instruct MoE LLM（每层 8 个 expert，top-2 routing）。部署时整模型 bf16 加载需 2 块 A100-80G GPU（总参数 47B，expert 占 96%/45B），每 token 固定激活 2 个 expert 计算 SwiGLU FFN。现有 weight pruning 方法（Wanda/SparseGPT 的 2:4 结构化稀疏）虽能减少参数量，但需专用硬件（FPGA/N:M sparse tensor core）支持 plug-and-play 部署。

  全栈执行例子：
  - 算法 Pipeline：输入 token x → Router(logits l ∈ R^8) → Softmax → top-2 选择 e0, e1 → SwiGLU FFN：E_i(x) = W_down·(SiLU(W_gate·x) ⊙ W_up·x) → 输出 z = w̃_{e0}·E_{e0}(x) + w̃_{e1}·E_{e1}(x)。8 个 expert 各含 3 个权重大矩阵，每 token 仅激活 2 个。
  - 系统框架：HuggingFace Transformers 加载完整 47B 模型 → 2×A100-80G GPU 推理 → GPU 间通信（expert 分布在跨 GPU）。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明（标准 PyTorch linear/SwiGLU kernel）。
  - 硬件架构：NVIDIA A100-80G GPU。

- **论文方法是什么？如何对应解决Baseline的缺陷？**

  **(1) Expert Pruning** 解决 Baseline 的 **内存占用过大** 缺陷：Baseline 8 个 expert 需 2 GPU 加载，实际每 token 仅用 2 个 expert，大量参数闲置。论文方法逐层枚举 expert 组合，以最小化校准集上的 token 重建损失（||F'(x,C) − F(x)||_F）为标准，永久丢弃 n−r 个不重要 expert。Prune 2 个 expert (r=6) 后只需 1 块 GPU 加载，内存从 89,926MB 降至 68,383MB（↓24%）；prune 4 个 (r=4) 降至 46,879MB（↓48%），减少 GPU 间通信开销从而实现 1.27× 加速。

  **(2) Dynamic Expert Skipping** 解决 Baseline 的 **运行时 FLOPs 不减** 缺陷：Pruning 后每 token 仍激活 k=2 个 expert，FLOPs 不变。论文方法利用 routing weight 比值 w_{e1}/w_{e0} < β 在推理时动态决定跳过权重较小的 expert，减少实际激活 expert 数。β 取校准集每层 w_{e1}/w_{e0} 的中位数，使跳过概率 ~50%。此方法与 expert pruning 正交组合：r=6 + skipping 可达 1.23× 加速，同时精度（62.91）显著高于 r=4 纯 pruning（59.57）。

  **(3) Task-Specific 校准** 解决 Baseline/通用 Pruning 的 **domain 迁移差** 缺陷：C4 通用校准集 prune 的模型在数学任务上表现差（GSM8K 从 58.61 降至 41.02）。论文将校准集从 C4 切换到 MATH training set，使 prune 后 GSM8K 5-shot 从 41.02 升至 51.25（r=6），经 MetaMathQA fine-tune 后可达 79.53，接近原 8-expert 模型的 81.35。Fig.4 显示 C4 与 MATH 校准后的 expert 选择仅在 4/32 层相同，说明 domain 对 expert 重要性分布有显著影响。

  论文方法全栈执行例子：
  - 算法 Pipeline：**Pruning 阶段**：校准集 token x → 原始 MoE 层 F(x) → 缓存 (x, Y) → 枚举 C⊆{expert_0,...,expert_7}, |C|=r → 计算 F'(x,C) → 选 min||F'(x,C)−Y||_F 的组合 → 逐层拼接得到 r-expert 模型。**Inference 阶段**：token x → Router(仅 r 个 expert 的 weight) → top-2 → 若 w_{e1} < β·w_{e0} 则仅 E_{e0}(x) 否则 E_{e0}+E_{e1}。
  - 系统框架：HuggingFace Transformers → 修改 model config（expert 数 8→r）→ 加载 pruned checkpoint → 1×A100-80G 推理（r≤6），无需跨 GPU 通信。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明。
  - 硬件架构：NVIDIA A100-80G GPU。论文方法不依赖专用硬件，是 plug-and-play 的算法层面稀疏化技术。

## Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- baseline方法是什么？
  Baseline 是现有 MoE 模型的 expert offloading 系统（如 SwapMoE、MoE-Infinity、EdgeMoE），它们依赖观察到的"局部 expert 激活重复"现象来设计 expert cache 策略（LRU/LFU），但缺乏系统性的量化指标来衡量不同模型适合 offloading 的程度。这些系统面临的挑战是：并非所有 MoE 模型都能从 expert offloading 中受益——频繁的 cache miss 导致 CPU offload 或 on-demand loading，显著拖慢推理速度。

  Baseline 全栈执行例子：
  - 算法 Pipeline：token x ∈ R^d → Router Softmax + TopK → 选择 top-k experts → 若 expert 不在 GPU cache → CPU 加载 (memory copy overhead) 或 LRU eviction + PCIe 传输 → FFN 计算 → output。问题：如果模型局部路由一致性低，相邻 token 激活完全不同 expert，cache miss 频繁。
  - 系统框架：现有 expert offloading 系统（如 SwapMoE）采用通用 cache 策略（LRU/LFU），不做模型感知的差异化配置。无论何种 MoE 架构，均使用同一 cache policy。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明。
  - 硬件架构：GPU（快速但有限内存）+ CPU（大容量慢速内存）异构架构。缺少对特定 MoE 架构特性与硬件适配的系统性理解。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是提出两个理论化的**模型级度量指标**——SRP 和 SCH，量化 MoE 模型的"局部路由一致性"，使开发者能够在部署前评估模型对 expert offloading 的友好程度，并给出优化后的部署配置建议。

  论文方法全栈执行例子：
  - 算法 Pipeline（两阶段）：
    **阶段 1（离线分析）**：对候选 MoE 模型在 22,528 样本语料上收集 router decisions → 统计 expert 在 segment 内的激活频率 f(e,T,p,m) → 计算 SRP（最大化 F1 的最优 threshold α_e^m）→ 计算 SCH（oracle segment cache 在缓存比 ρ 下的 hit rate）→ 生成 SCH-vs-ρ 曲线 → 识别拐点 ρ*（通常为 2× 激活 expert 数）→ 筛选高 SRP 模型部署。
    **阶段 2（部署决策）**：根据 SCH 分析选择模型 → 配置 expert cache size = ρ*×k×expert_size → 运行时用 LRU/LFU cache。

  - 系统框架层面的改进：论文提出的分析表明：（1）**无 shared experts** 的 MoE 模型（如 OLMoE、LLaMA-MoE-v2）有更高局部路由一致性——指导架构设计时避免或少用 shared experts；（2）**domain-specialized experts** 而非 vocabulary-specialized experts 贡献了主要的路由一致性——意味着针对特定领域（如代码、数学）优化时效果更好；（3）**cache size ≈ 2x 激活 expert 数** 是大多数模型的 sweet spot，超出此比例 hit rate 增益递减。这些见解可直接指导 expert offloading 系统在模型选择、缓存配置、领域适配方面的设计，而非对所有模型统一处理。

  - 编译框架：论文未明确说明。

  - Kernel 调度：论文验证了 LRU/LFU cache 与 SCH 的强相关性（m=16 时 Pearson r=90.43/88.70），证明命中率上限由模型路由行为决定。实际 offloading 系统可将 SCH 作为理论命中率上界指导 kernel 层面的 expert 调度。

  - 硬件架构：GPU 内存（A100 80GB）+ CPU（大容量）异构。结论适用于 memory-constrained 边缘设备场景。论文 insight：decoding 阶段 overhead 与局部路由一致性负相关（r≈−0.3），而 prefilling 阶段正相关（r≈0.2），因为 prefilling 瓶颈在 expert-level load balance 而非跨 token 的路由一致性。

  核心创新：将 expert offloading 系统的性能分析从**系统实现层面**（如何优化 cache/cache policy）下沉到**模型路由行为层面**（哪些模型天然适合 offloading），实现了模型设计与部署系统之间的双向指导。论文发现的关键 trade-off——局部路由一致性 vs. 局部负载均衡——解释了为什么一些模型可在保持全局负载均衡的同时获得高局部路由一致性（通过 domain-specialized experts 实现：特定领域上下文中集中激活某些 expert → 高局部一致性；不同领域激活不同 expert → 全局均衡）。

## Optimizing Dynamic Neural Networks with Brainstorm

- baseline方法是什么？
  Baseline 是现有的 DL 框架（主要 PyTorch eager mode）执行动态神经网络。PyTorch 的 tensor-centric 编程模型只能表达 tensor 级别的静态数据流图（DFG），无法理解 sub-tensor 级别的 dynamism（如 token、patch、pixel 的动态路由）。具体来说：
  - Router 的路由逻辑（如 MoE top-k gate、patch-based super-resolution branching）用 Python 原生 control-flow + 数据搬运算子（如 einsum）实现，与计算逻辑耦合。
  - 编译器无法追踪 sub-tensor 级别的数据流：不知道 "token" 是什么、如何在不同 expert branch 间分发、跨层 expert 之间如何关联。
  - 无运行时 profile 收集能力：不知道 branch 激活的统计分布、token 在 expert 间的不均匀分配模式。
  - 全栈执行例子（SwitchTransformer with MoE, batch=8, 128 tokens/sentence, 256 experts, single A100 GPU）：
    - **算法层**：SwitchTransformer MoE layer，每 token 通过 softmax gating 路由到 top-1 expert，每个 expert 有 capacity=64 tokens。
    - **系统框架层**：PyTorch eager mode。Router 用 Python 实现（linear gate + argmax），tokens 通过 einsum 操作重新排列并按 expert 分组，各 expert FFN 串行执行（for loop 遍历 256 个 expert），每个 expert 计算其收到的 token subset。
    - **编译框架层**：PyTorch 仅做基础的 vertical operator fusion（如 Conv+BN+ReLU），无 sub-tensor 级别分析或优化能力。
    - **Kernel层**：每个 expert 的 FFN 为一组 GEMM kernel launch，256 个 expert 产生 256×k 次 kernel launch。对于每个 expert 仅收到少量 token 的常见情况，GPU CU 利用率极低。
    - **硬件架构层**：单 A100 GPU，108 SMs，每个 SM 可并行执行多 warp。串行 expert 执行时，仅少数 SM 被使用，其余空闲。
  - Baseline 缺陷：
    1. **细粒度 dynamism 不可追踪**：tensor-centric DFG 无法表达 token/patch 级别的数据流，无法收集 branch 负载、激活频率等关键 profile 信息。
    2. **串行执行 branch 导致 GPU 利用率低**：每个 branch 独立 launch kernel，GPU SM 闲置，kernel launch overhead 累积。
    3. **无负载感知优化**：all-to-all 通信对 uneven token distribution 做大量 padding 冗余传输；无法基于 expert 相关性优化多 GPU 放置。
    4. **Router 计算开销不可隐藏**：Router 涉及 CPU-GPU 同步和 control-flow，在 MSDNet/DynamicRouting 中占 44%~65% 延迟，无法跳过或预取。
    5. **动态 branch 无法做 weight preloading**：现有 memory swapping 方案（如 SwapAdvisor, Capuchin）依赖静态执行顺序，动态网络无法提前知道下一个激活的 branch。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Brainstorm 框架，通过 Cell 和 Router 两大核心抽象统一动态网络的表达，使 sub-tensor 级别的 dynamism 可追踪，进而基于运行时 profile 实施四项动态优化。

  1. **Cell 抽象**：让开发者标注 tensor 中 dynamism 发生的粒度和维度（如 token=(0), patch=(0,1)），编译器通过符号执行推导 Cell 在静态算子间的传播关系（三种类型：保持、重排、混合）。
  2. **Router 抽象**：统一的路由接口（router_fn 定义规则 → Router 负责高效执行），解耦控制流与数据流，使编译器无需理解路由逻辑本身，只需收集 Routes 的统计分布。Router 的执行后端是高效的 GPU kernel（Cell rearrangement + sparse communication）。
  3. **四项动态优化**（基于 Router 统计 profile）：
     - **Dynamic Horizontal Fusion**：根据 branch Cell 负载分布编译多 shape tuned kernel，运行时按实际负载选择最优 kernel 并水平融合执行。
     - **Profile-Guided Model Placement**：分析跨层 expert co-activation 相关性，重新排列 expert 以减少 inter-GPU 通信。
     - **Speculative Routing**：预测高概率 branch 跳过 router_fn，错误时 unroll。
     - **Speculative Weight Preloading**：预测高概率 branch 权重提前加载，减少 GPU memory 占用 43.5%。

  - 对比 baseline 全栈执行例子（SwitchTransformer MoE, batch=8, 128 tokens/sentence, 256 experts, brain A100 GPU）：
    - **算法层**：未修改模型算法。仅通过 12 行代码改动接入 Brainstorm：用 `brt.annotate_cell` 标注 token 粒度，用 `Router(router_fn)` 包装 top-k gate。
    - **系统框架层**：Brainstorm 的 torch.fx 优化 Pass 将 256 个串行 expert 替换为一个 fused horizontal kernel，内含多个 tuned kernel variant（如 8/32/64/128-token shape）。Router 的 GPU kernel 高效完成 token→expert 的 scatter-gather 操作，无 einsum 开销。
    - **编译框架层**：AOT 静态 Cell-level 符号执行分析 Self-Attention → MoE → Self-Attention 的跨层 Cell 依赖关系，得出 Self-Attention 的 Cross-Cell mixing 约束（所有 token 需聚合到同一 GPU）。JIT Profiler 收集 256 个 expert 的 token 负载分布（发现 P50/P90/P100 分别对应不同 percentile 的负载），确定 tuned kernel 的 shape 集合。通过 TVM 对每个 shape auto-tune。
    - **Kernel层**：Fused kernel 一次 GPU launch 并发执行所有激活的 expert（而非 256 次 launch），根据每个 expert 实际收到的 token 数（如 4/8/27/64）选择 nearest tuned kernel 并 minimal padding。对于仅收到 0 token 的 expert，直接跳过。对比 Tutel（BatchMatmul 方式，需将所有 expert 的 token 数 pad 到 max），Brainstorm 因 token 分布不均（图 2a）而大幅减少冗余计算和显存占用——Tutel 在 256 expert 时甚至 OOM。
    - **硬件架构层**：单 A100 GPU。Fused kernel 使 108 个 SM 全部参与计算，CU utilization 显著提高。运行时 overhead 仅 12.3μs（branch 少时），可忽略。
  - **关键设计应对 Baseline 缺陷**：
    - 缺陷1（不可追踪）→ Cell/Router 抽象：开发者显式标注 dynamism 粒度，编译器通过符号执行 + JIT profiling 获得 sub-tensor 级数据流全貌。
    - 缺陷2（GPU 利用率低）→ Dynamic Horizontal Fusion：用 profile 决定 tuned kernel shapes，运行时选择最小 padding kernel，并发执行激活的 branch。SwitchTransformer 加速 3.63× vs PyTorch，3.33× vs Tutel。
    - 缺陷3（无负载感知优化）→ Sparse All-to-All + Profile-Guided Placement：TaskMoE 减少 42~87% inter-GPU 通信，SwinV2-MoE 加速最高 5.04× vs DeepSpeed。
    - 缺陷4（Router 开销大）→ Speculative Routing：预测 90~95% 准确，DynamicRouting 加速 1.7×，MSDNet 加速 8.44×（combined with horizontal fusion: 11.7×）。
    - 缺陷5（无预加载能力）→ Speculative Weight Preloading：DynamicRouting 加速 1.97×，GPU 内存减少 43.5%。

## Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

- baseline方法是什么？
  - Baseline：Switch Transformer（token-level MoE routing）。每个 token 的嵌入 t_t ∈ R^d 通过线性 gate W_g ∈ R^{N×d} 独立投影到 N 个专家分数空间：g(t_t) = softmax(W_g · t_t)，然后选择 top-k 个专家。由于 token embedding 被 token-identity 特征主导（如图 2 左：不同 token ID 形成不同 cluster），相邻 token 即使语义相似也会被路由到不同专家，造成极高的 inter-token expert activation variation（CSD_token 大），在内存受限的边缘设备上导致频繁的 expert swapping（50%-85% 总延迟来自 I/O）。
  - Baseline 全栈执行例子（Switch Transformer, 729M, 8 MoE 层 × 16 专家, Jetson Xavier NX 8GiB GPU, 单用户请求, batch=1）：
    - **算法层**：每个生成的 token 独立通过 W_g 矩阵乘法做 top-1 routing → 相邻 token 的 W_g · t_t 投影在高维空间中方差极大 → 几乎每 2 个连续 token 就切换一次专家。
    - **系统框架层**：load-on-demand 策略（FIFO/LRU/SwapMoE）管理有限 GPU 内存中的专家 → 因为 token-level routing 变化频繁，几乎所有专家都需要在内存中轮转 → SwapMoE 离线统计专家频率但仍无法应对稀疏、不一致的激活模式。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：标准 Transformer FFN kernel（矩阵乘法 + SwiGLU），专家计算延迟 L_compute 固定，但 I/O 延迟 L_swap 主导总延迟（占 50%-85% 甚至 >99%）。
    - **硬件架构层**：Jetson Xavier NX 384 核 Volta GPU，8GiB 显存。完整 729M 模型需全部显存；50% 内存预算时仅能驻留约一半专家。Switch Transformer + FIFO/LRU/SwapMoE 在 50% 内存下的延迟是 full-model 的 15-30 倍。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Oracle-MoE 用基于 Oracle Space 的路由替代 token-level routing。核心洞察：连续 token 具有语义局部性（semantic locality），但 token embedding 被 token-identity 特征主导，掩盖了高层语义的局部性。Oracle-MoE 通过以下步骤提取高层语义并保持路由一致性：
    1. **语义组划分**：利用注意力分数矩阵（Q·K^T 内积）发现高层语义相关性——a_ij > ε 的 token 归入同一语义组。
    2. **Oracle Space 构建**：语义组嵌入 z_S = mean(token embeddings in S)，保留高层语义、压制 token-identity 噪声（理论证明 Var(z_S) = (Σ_s + Σ_j)/n < Var(t_t)）。
    3. **Oracle Space 路由**：在 Oracle Space 上做 K-means 聚类（k = 专家数），每个聚类中心对应一个专家。连续 token 的高层语义平滑变化（图 4），因此路由到相同专家的概率大幅提高。Theorem 1 证明 CSD_oracle < CSD_token 以高概率成立。
    4. **Expert Prediction 优化**：利用第一层 embedding 预测深层专家激活（准确率 85%-95%），预加载专家，进一步减少 10%-15% latency。
  - 对比 baseline 全栈执行例子（Oracle-MoE, 729M, 8 MoE 层 × 16 专家, Jetson Xavier NX 8GiB GPU, 单用户请求）：
    - **算法层**：不再用 token embedding 做 gate，而是：注意力分数矩阵 → 贪心语义组划分 → 组嵌入平均 → SVD 降维 → K-means 聚类中心最近邻 → 同一语义组内所有 token 路由到同一专家。因为一个序列（1024 token）通常只有不到 5 个语义组且同一序列的语义组往往属于同一聚类，连续数百个 token 可不切换专家（图 6）。
    - **系统框架层**：load-on-demand + 低 expert swapping 需求 → 只需在语义组切换时才加载新专家。因为 Oracle-MoE 的 CSD_oracle 极低，不同 swapping 策略（FIFO/LRU/SwapMoE）对延迟影响很小，论文以三种策略的平均值报告结果。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：路由计算从 W_g · t_t（token 级矩阵乘法）替换为 z_reduced = W_svd · z_S（组嵌入降维）+ ||z_reduced - c_k|| 欧氏距离计算（聚类中心最近邻）。降维后的低维空间使计算开销可忽略（2.5e-4s vs token-level 的 1e-4s，相比单次 forward-backward pass 的 3.5s 可忽略）。
    - **硬件架构层**：Jetson Xavier NX 384 核 Volta GPU，8GiB 显存。Oracle-MoE 在 25% 内存预算下（仅驻留 1/4 专家）仅比 full-model inference 多 3s 延迟；Switch Transformer 延迟增加高达 2000%。50% 内存预算时 Oracle-MoE 几乎无额外延迟。First token latency：Oracle-MoE 4.910s vs Switch+FIFO 22.395s / Switch+LRU 23.428s / Switch+SwapMoE 12.767s。下游任务性能：Oracle-MoE 平均持平甚至略优于 Switch Transformer（如 729M: Ours 36.35 vs Switch 35.86 avg score）。
  - **关键设计应对 Baseline 缺陷**：
    - 缺陷1（token-level routing 受 token-identity 主导 → 高 CSD → 频繁 expert swapping）→ Oracle Space Routing：用语义组嵌入（压制 token-identity 噪声，保留高层语义）替代 token embedding 做路由，CSD_oracle 远小于 CSD_token（理论保证 Theorem 1 + 实验验证激活不一致性从 53-82 降至 4-6 per 100 tokens）。
    - 缺陷2（load-on-demand 策略无法应对不可预测的激活模式）→ Semantic Locality Preservation：连续 token 的高层语义在 Oracle Space 中平滑缓慢变化（图 4），同一语义组内所有 token 路由到同一专家。即使话题切换的跨数据集 scenario，Oracle-MoE 仍仅每 100 token 换 12.2 次 vs Switch 的 90.54 次。
    - 缺陷3（Prefill 阶段也需频繁 swapping → 高 first token latency）→ Oracle-MoE 在 prefill 阶段对一个输入仅激活 1-2 个专家，仅需一次加载。First token latency 降至 4.910s（Switch+FIFO 22.395s）。
    - 缺陷4（无法预测深层专家激活 → 无法预加载）→ Expert Prediction：用第一层 embedding 预测深层专家激活准确率 85%-95%，进一步减少 10%-15% latency。此可预测性源自 Oracle-MoE 路由与高层语义的强关联。

## Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

- baseline方法是什么？
  - Baseline 方法：(1) **传统 MoE serving 系统**——采用 *system-centric* 方法论，针对特定平台（CPU-GPU、multi-GPU、ML accelerator）设计平台特定的优化策略（如 expert migration、pipeline、caching），其洞察仅适用于该平台，无法泛化到其他 serving 架构；(2) **现有 GPU Command Processor**——无视 SM 物理位置和 data placement，将所有 SM 视为均等资源，按 uniform 方式分配任务，导致大量不必要的 die-to-die 通信；(3) **现有 GPU HBM 管理**——将所有 HBM die 视为 uniform memory space，不区分 local vs remote HBM，无法利用 local HBM 缓存减少 cross-die traffic；(4) **现有 Expert Placement（如 EPLB）**——依赖 periodically collected profiling data（每 3000+ steps 触发），在初始 ~1000 decode tokens 没有 profiling data 可用时无法有效设置 expert placement。
  - 全栈执行例子（以现有 multi-GPU MoE serving with 8×H100 为例）：
    - **算法层**：MoE 模型（如 DeepSeek V3 671B, 256 experts per layer, top-8 selection），Gate 网络使用 top-k routing 为每个 token 选择 expert。Expert placement 为默认连续布局（experts 0-15 on GPU-0, 16-31 on GPU-1, etc.）。
    - **系统框架层**：SGLang serving framework。使用 Expert Parallelism (EP8)，通过 DeepEP backend 做 all-to-all 通信。Expert placement 初始化后保持不变直到 EPLB 收集到足够 profiling data（~3000 decode steps 后触发调整）。在初始 decode 阶段（~1000 tokens），系统缺乏 expert 选择的运行时统计，无法优化 placement → 热门 expert 可能集中在少数 GPU，导致严重负载不均（max/min execution-time ratio ≈ 1.3× at EP8，更大会更严重）。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch eager mode + CUDA graphs for decode）。
    - **Kernel层**：MoE 计算使用 3 个 GEMM operations（gate_proj, up_proj, down_proj），通过 DeepEP all-to-all 通信库在 GPU 间传输 tokens。GPU Command Processor 按 uniform 方式分配 SM 计算任务，不考虑数据位置。
    - **硬件架构层**：Wafer-scale GPU 场景——Global CP 将 MoE kernel 均匀分配到各 die 的 SM，不区分 location。SM 读取 expert 权重时：本地 HBM（300ns）正常读取；远程 HBM 需通过 D2D XY routing 多跳传输（200ns/hop × N hops + remote DRAM latency），产生大量 inter-die traffic。由于 expert selection 的 skewness（部分 expert 被选中的频率是平均值的 16 倍以上），热门 expert 所在 die 可能成为瓶颈，而其他 die 空闲等待。
  - Baseline 缺陷：
    1. **System-centric 方法论局限性**：不同平台的优化策略不可泛化，已有洞察是 MoE 数据模式的片面视角。
    2. **Expert selection 的"随机性"未被理解**：大规模 MoE（200B+）的 expert selection 模式未被系统研究，数据移动优化缺乏理论基础。
    3. **Workload imbalance**：Expert selection 的高度 skewness 导致严重的负载不均——热门 expert 所在 die/GPU 过载而冷门 expert 所在 die/GPU 空闲。默认 uniform placement 和 EP 均无法同时解决 D2D traffic 和 load imbalance。
    4. **Prefill 阶段信息未被利用**：Prefill 和 decode 的 expert selection 模式高度相似，但现有系统未利用 prefill traces 指导 decode 阶段的 expert placement。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：(1) **Model-centric 方法论**——对所有 serving 平台进行 system-independent profiling，提取 *system-agnostic* insights，6 条关键洞察（Insight 1-6）可指导任何规模、任何平台的 MoE serving 系统设计；(2) **Case Study 1: Wafer-scale GPU 架构**——基于 Insight 1-3 提出两级 Command Processor（Global CP + Local CP）+ hardware-managed HBM（ATU + PDU），运行 Task Allocation Algorithm 和 Data-Driven Predictor，在单个 wafer-scale GPU 上实现 6.6× throughput 提升；(3) **Case Study 2: Prefill-guided Expert Placement**——基于 Insight 1 提出 Remap-based 和 Duplication-based 两种 placement 算法，利用 prefill traces 预测 decode 阶段 expert 选择并优化 placement，在 8×H100 上实现 up to 1.25× speedup。
  - 全栈执行例子（以 Wafer-scale GPU + Task Allocation + Predictor 为例，DeepSeek V3 在 Dojo 5×5 上）：
    - **算法层**：MoE 模型本身不变（DeepSeek V3 671B, 256 experts, top-8 selection）。论文方法在算法层之下做 profiling 驱动 + 架构级优化。6 条 Insight 中的 temporal insights（Insight 1: prefill-decode correlation, Insight 2: cross-hierarchy memory management）和 spatial insights（Insight 3: expert-placement-aware workload distribution, Insight 4: popular expert decentralization, Insight 5: expert-pair separation, Insight 6: workload-aware serving）直接来自对 >24,000 requests 的 expert selection traces 的 profiling 分析。
    - **系统框架层**：论文采用 single-GPU-like programming model——整个 wafer 作为统一 GPU 暴露给软件，硬件层完全抽象 multi-die topology 和 data placement。这与当前 multi-chiplet GPU（Blackwell, Rubin）的编程模型一致。
    - **Kernel层**：Global CP 在 kernel launch 时运行：(a) **Task Allocation Algorithm**——对每个 expert（按请求数升序遍历），候选 die = 存有该 expert 的本地 die + 距离 1 的邻居 die → cost model（DRAM access + compute + D2D comm）评估 → block size 50 贪心分配到最优 die → 合并为 allocation plan。关键创新：允许将请求分配到邻居 die 而非仅本地 die，在 workload balance 和 D2D traffic 之间 trade-off（Insight 3）；(b) **Data-Driven Predictor**——利用 cross-token heatmap（Insight 2 的 token-level correlation），从当前 kernel 的 expert selection 预测下一 token 的热门 expert → 生成 cp_en bits 配置 PDU prediction table → 在后续 remote data access 时自动触发 local caching。
    - **硬件架构层**：(i) **Global CP (A76 class)**——维护 Expert Distribution Table（每 expert 的 die ID + n-bit 分布 bitmask）和 Cross-token Heatmap Cache（0.5 MB on-chip SRAM），运行 allocation + prediction 算法；(ii) **Local CP (A72 class, 每 die)**——接收 Global CP 的子 kernel + prediction 配置，分配任务到 SM；(iii) **ATU (4.25 KB SRAM per die)**——当 SM 请求远程 expert 数据但该 expert 已被复制到本地 HBM 时，将远程地址翻译为本地地址；(iv) **PDU (128 B register file per die)**——存储 prediction table (cp_en + is_local bits)，在远程数据返回时检查 cp_en 决定是否缓存到本地 HBM。总 area/power overhead < 0.04%。
    - **对比 Baseline 的改进**：(a) Global CP 的 Task Allocation 考虑 expert placement（Expert Distribution Table）→ 将大多数任务分配到本地 die，hop count 降低 142×，消除 D2D communication 瓶颈；(b) Predictor + Hardware-managed HBM 自动缓存热门远程 expert → 进一步将 remote DRAM reads 转化为 local reads；(c) 从 system-centric 转为 model-centric → insights 同样适用于多 GPU 集群 (DGX, NVL72)、CXL/CPU memory disaggregation、Flash-based multi-tier、PIM 等架构。

## MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

- baseline方法是什么？
  - Baseline 方法包括三种 PTQ 方案：
    - **RTN（Round-to-Nearest）**：直接对权重做 per-channel 对称均匀量化，Q(W) = clamp(⌈W/s⌋, q_min, q_max)，无任何误差补偿。
    - **AWQ（Activation-aware Weight Quantization）**：利用激活分布选择平滑系数和剪枝权重，量化损失为 L(W_hat) = ||WX - W_hat X||_F^2，通过最小化输出误差指导量化。
    - **GPTQ**：基于 OBQ 的 Hessian 误差补偿方法，Hessian = X X^T，逐列量化和补偿误差，是当前最强的 LLM PTQ 方法。
    - 核心缺陷：这些方法均为 layer-wise 量化，忽略了 MoE 架构的两个关键特性：(1) 校准集中不同 expert 负载极不均衡——使用 WikiText2 或 C4 作为校准集时，部分 expert 被大量 token 路由到而其他 expert 收到的 token 极少，导致欠载 expert 校准不足；(2) gating network 为不同 token 分配不同的 expert 亲和力 c_i，但传统量化对所有 token 一视同仁（每个 token 的量化误差贡献相同权重），导致高亲和力 token 的量化误差被低估。
  - 全栈执行例子（以 GPTQ 在 Qwen-MoE-14B 上的 4-bit 量化为例）：
    - **算法层**：从 WikiText2 取 128 条 512-token 序列作为校准集 → 逐层 forward，收集每层每个 expert 的输入激活 X → 对每个 expert 的每个线性层（W_gate, W_up, W_down），计算 Hessian = X X^T → 逐列量化 W 的每一列，用 Hessian 逆矩阵补偿剩余列的量化误差。**问题**：WikiText2 校准集下，某些 expert 可能只收到不到 5% 的 token，Hessian 估计严重不足；gating weight c_i 被完全忽略，所有 token 对 Hessian 的贡献等权。
    - **系统框架层**：论文未明确说明具体推理框架。量化后模型通过标准 PyTorch/HuggingFace Transformers 加载推理。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。量化后使用标准 INT4 dequant + FP16 matmul kernel。
    - **硬件架构层**：NVIDIA A6000 GPU（48GB）。4-bit 量化后内存从 27.88GB 降至 8.51GB（Qwen-MoE-14B），3.28x 节省；解码速度从 8.35 提升至 10.60 tokens/s（1.27x 加速）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MoEQuant 的核心设计**：通过两个插件式模块 EBSS 和 AGQ，分别解决 inter-expert 和 intra-expert 的不均衡问题，可无缝集成到 GPTQ/AWQ 等现有 PTQ 方法中。
  - **解决 baseline 的两个缺陷**：
    1. **Inter-expert 校准不均衡 → EBSS（Expert-Balanced Self-Sampling）**：不再依赖固定的校准集（如 WikiText2），而是利用 LLM 自身能力自采样生成校准数据。从词汇表 V 开始，维护 w 个 beam，每步用 score(S^t||v) = (-1/(i+1))(R_S + log P(v|S)) + σ(M, S)/τ 对候选 token 排序，保留 top-w。此过程同时优化 perplexity（保证与预训练分布一致）和 expert balance（σ 即 expert 使用频率的 std），将搜索复杂度从 O(m^n) 降至 O(wn)。EBSS 生成的校准集中各 expert 分配到的 token 数基本均衡（参见 Figure 2），确保每个 expert 都有足够的校准样本。
    2. **Intra-expert 亲和力缺失 → AGQ（Affinity-Guided Quantization）**：将 token-expert 亲和力（即 gating coefficient c_i）纳入量化过程。传统量化损失 L = Σ_i ||W x_i - W_hat x_i||_F^2，AGQ 重定义为 L = Σ_i c_i · ||W x_i - W_hat x_i||_F^2，使高亲和力 token 的量化误差惩罚更大。对 Hessian-based 方法，改进 Hessian 为 H = (X ⊙ √c)(X ⊙ √c)^T = (X ⊙ c)X^T，物理含义是 token i 对 Hessian 的贡献按其 gating weight c_i 缩放，使得 router 更信任的 token 在误差补偿时占据更大权重。
  - 全栈执行例子（以 MoEQuant++（基于 GPTQ）在 Qwen-MoE-14B 上的 4-bit 量化为例）：
    - **算法层**：EBSS 以 w=4 branches、τ=1.2、sequence length n=512 自采样生成 expert-balanced 校准集 D* → 将 D* 输入模型，逐层 forward 收集每个 expert 的输入激活 X 和 gating coefficient c → AGQ 计算带亲和力权重的 Hessian H = (X ⊙ c)X^T → 对每个 expert 的每个线性层执行标准 GPTQ 逐列量化+误差补偿，但使用 AGQ 改进的 Hessian。量化参数：per-channel 对称均匀量化，4-bit（q_min=-8, q_max=7）。
    - **系统框架层**：基于 GPTQ 和 AWQ 官方仓库修改，集成 EBSS 校准集生成模块和 AGQ Hessian 计算模块。评估使用 lm-evaluation-harness v0.4.4（zero-shot 任务）和 MMLU/GSM8K/HumanEval 官方仓库（复杂推理任务）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。量化后推理使用标准 INT4 矩阵乘 kernel。
    - **硬件架构层**：NVIDIA A6000 GPU。MoEQuant++ 在 Qwen-MoE-14B 上 4-bit 量化后平均分 49.59（vs GPTQ 49.00，+0.59），在 DeepSeek-MoE-16B 上 40.01（vs GPTQ 39.01，+1.00），在 Mixtral-8x7B 上 55.58（vs GPTQ 53.42，+2.16）。HumanEval 上 DeepSeek-MoE-16B 4-bit 下 GPTQ 得分 22.56，MoEQuant++ 提升至 25.00（+10.8%）。instruction-tuned 模型上 MoEQuant++ 效果更显著：Qwen-MoE-14B-Chat 上 HumanEval 从 GPTQ 的 15.24 提升至 21.95（+44%）。

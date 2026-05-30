## TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation

- baseline方法是什么？
  Baseline 是标准 **Top-K (K=2) 路由的稀疏 MoE 训练**（Fedus et al., 2022; Zoph et al., 2022）。在 Baseline 中：
  - **模型推理算法层**：输入 token x 通过路由器计算 Softmax(Wx) 得到 N 个专家的概率分布 π，TopK 选出 K 个最高概率的专家，MoE 输出为 $y = \sum_{i \in A} \pi_i E_i(x)$。反向传播时，由于 Top-K 是不可微操作，使用 Straight-Through Estimator 绕过，理论上需要所有 N 个专家的输出来计算稠密梯度 $\partial y/\partial \pi = [E_1(x), E_2(x), \dots, E_N(x)]$，但实际只有 K 个专家有输出，导致只有 K/N 比例的专家参数和路由嵌入被更新。
  - **系统框架层**：基于 GPT-NeoX 训练框架 + Megablocks 稀疏专家库，支持数据并行训练，通过 NCCL 进行通信。每个 token 只激活 K 个专家，其余 (N-K) 个专家的参数和路由嵌入对该 token 无梯度更新。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准的 MoE 前向/反向 kernel（Megablocks 提供），包含 expert matmul、all-to-all 通信、路由器 softmax + TopK 等操作。
  - **硬件架构层**：标准 NVIDIA GPU 集群，论文未明确说明具体型号。

  Baseline 的核心缺陷：(1) **路由器接收稀疏梯度**——只有被 Top-K 选中的 K/N 比例的专家嵌入行获得梯度，未被选中的专家对应的路由嵌入 $W_i$ 得不到更新，导致路由器无法学习到所有专家的路由分布；(2) **专家负载不均衡**——稀疏梯度导致部分专家（hot experts）被过度使用，部分专家（cold experts）几乎空闲，资源利用率低；(3) **专家参数更新稀疏**——每个 token 只更新 K 个专家的参数，(N-K) 个专家对该 token 无任何更新，参数利用效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Expert Group Approximation（专家组近似）** 方法，核心思路是在反向传播中用已有专家输出来近似未激活专家的输出，从而为路由器和专家参数提供稠密梯度信号：

  - **模型推理算法层**：不改变前向传播（保持推断时的稀疏性），仅在反向传播中通过以下机制引入稠密梯度：
    (1) 将所有 token 按路由决策 $R(x)$ 分组（共 $\binom{N}{K}$ 组）；
    (2) 对于每组 $X_R$ 中的 token x 和每个未激活专家 i ∉ R，用同时被路由到 i 和 x 的某个激活专家 j ∈ R 的其他 token 的 $E_i$ 输出来近似 $E_i(x)$：
    $\hat{E}_i(x) = \frac{1}{K} \sum_{j \in R} \frac{1}{|X_{\{i,j,\cdot\}}|} \sum_{x' \in X_{\{i,j,\cdot\}}} E_i(x')$
    (3) 通过 stop-gradient 将近似注入计算图：$y := y + y' - \operatorname{sg}(y')$，前向不变，反向有完整梯度；
    (4) 同时更新路由器（所有 N 行嵌入接收梯度）和专家参数（所有专家对近似产生贡献的 token 接收额外梯度更新），两者通过 all-reduce 跨数据并行 worker 聚合。
  - **系统框架层**：基于 GPT-NeoX + Megablocks，在 MoE 层插入 Expert Group Approximation 计算和梯度注入逻辑。通过数据并行 all-reduce 聚合近似梯度，利用大全局 batch size 提升近似的统计质量。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 Triton 实现自定义 kernel（"Router backward" kernel 等）高效执行 token 分组、近似构造和梯度聚合。随 hidden size 增大（1024→4096），方法 overhead 从 13.32% 降至 1.57%（Table 4）。在 multi-node 训练中通信 overhead 主导时，方法 overhead 趋近于零。
  - **硬件架构层**：NVIDIA GPU（具体型号论文未明确说明）。

  **设计思路核心映射**：
  - 缺陷(1) "稀疏路由器梯度" → 方案：通过 Expert Group Approximation 用 N² 个组近似填充所有 N 个专家的梯度分量，使 $\partial y/\partial \pi$ 从稀疏 [K 个非零] 变为稠密 [N 个非零]
  - 缺陷(2) "负载不均衡" → 方案：路由器接收稠密梯度后能更好地学习 token-专家匹配分布，实验证明最大负载不均衡显著降低（Figure 7）
  - 缺陷(3) "专家参数更新稀疏" → 方案：近似梯度同时回传给参与近似的专家参数，使每个 token 对 (N-K)/N 比例的额外专家产生梯度贡献，专家参数利用率从 K/N 提升至接近 N/N

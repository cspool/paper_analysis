## Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Aurora 系统，通过三个维度的联合优化来最小化 MoE 推理时间：(1) **通信调度**：为 all-to-all 通信中的 token 传输确定最优顺序，避免接收端 GPU 的带宽竞争（Theorem 4.2, Alg.1）；(2) **专家共置**：将来自**不同模型**的专家放置在同一 GPU 上（而非同模型的多专家），使计算和通信可以完全交错，打破同步 all-to-all 通信限制；(3) **GPU 分配**：在异构集群中将热门专家分配到高性能 GPU（Theorem 5.1）。
  - 四种场景的理论分析：Exclusive+Homogeneous（§4）、Exclusive+Heterogeneous（§5）、Colocating+Homogeneous（§6，转化为瓶颈匹配问题求解）、Colocating+Heterogeneous（§7，3维匹配 NP-hard，通过解耦为两个二分图匹配得到次优解，仅偏离最优 1.07×）。
  - 实验比较：(1) 四种场景下的推理时间对比（vs. SJF、RCS、RGA、Lina、REC）；(2) Colocating 场景下的 GPU 利用率对比（vs. Lina）；(3) Colocating+Heterogeneous 场景下与暴力搜索最优解的差距；(4) 不精确 traffic 输入下（0%-75% 噪声）的性能鲁棒性。

- 硬件平台是什么，配置是什么。
  - 模拟环境（仿真评估）而非真实硬件。
  - 同构集群：网络带宽 100 Gbps。
  - 异构集群：4 种 GPU 类型，带宽分别为 100 Gbps、80 Gbps、50 Gbps、40 Gbps（从高到低性能排列），各类型 GPU 数量相同。
  - 所有 GPU 通过 big switch 模型（无阻塞网络）互联，如 Fig. 4(a) 所示。

- 开源Serving框架是什么。修改了什么。
  - Aurora 不基于现有开源 serving 框架，而是提出了一套理论驱动的优化方法。实现方式为仿真模拟。
  - 修改/设计内容：
    1. **通信调度算法（Alg. 1）**：基于 traffic matrix D，识别瓶颈 GPU（最大流量），确定 token 传输顺序以避免接收端带宽竞争。核心原理：通过添加非负人工 traffic matrix X 将原始 traffic matrix 转换为每行/列均为 b_max 的规整矩阵，再用 Farkas' Lemma 证明 X 的存在性，因此通信时间可压缩至 b_max。
    2. **GPU 分配策略（Theorem 5.1）**：在异构集群中按 expert 处理的 token 数量降序排列，从高到低性能 GPU 分配。
    3. **专家共置策略（§6.2）**：Case I（每 GPU 发送=接收流量）：交替选择热门和冷门 expert（Theorem 6.2）；Case II（发送≠接收）：转化为瓶颈匹配问题，使用二分搜索 + Hopcroft-Karp 算法（复杂度 O(n²√n log n)）。
  - 通信调度可通过在计算操作的 buffer 层调用 NCCL 等通信集体库按所需顺序实现（论文 §3）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未明确说明开源情况。Aurora 为模拟评估，未提及开源代码仓库或开源 link。
  - Aurora 的输入到执行全过程（以 Colocating+Heterogeneous 场景为例）：
    1. **输入**：两个 MoE 模型（各含 n 个 expert）的历史统计信息，包括 traffic matrix D_N（第一个 all-to-all 通信的 token 分布）、D_C（第二个 all-to-all 通信）、以及 Gate/FFN/Aggregation 在各 GPU 上的计算时间。
    2. **优化阶段**：Aurora 接收输入后，依次求解：(a) 专家共置——通过瓶颈匹配将 Model a 和 Model b 的 expert 配对，最小化聚合通信时间的最大列/行和；(b) GPU 分配——将共置后的 expert 对按 token 负载降序分配给高性能 GPU；(c) 通信调度——为每个 GPU 确定 token 传输顺序，确保任何时刻各 GPU 只从单一源接收数据。
    3. **推理执行**：每个 MoE layer 上，Gate 网络计算后触发第一个 all-to-all 通信（按 Aurora 的调度顺序发送 token）→ 各 GPU 上的 FFN 处理到达的 token → 第二个 all-to-all 通信（反向传输）→ Aggregation。两个模型的计算和通信因 expert 来自不同模型而完全交错：当 Model a 做 FFN 计算时，Model b 可同时进行 all-to-all 通信。
    4. **输出**：推理时间 t = E_{A^b} + |G^a|（Eqn. 4），GPU 利用率 = 计算时间/推理时间。

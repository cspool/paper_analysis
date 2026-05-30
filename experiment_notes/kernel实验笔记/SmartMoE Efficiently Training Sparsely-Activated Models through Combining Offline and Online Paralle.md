## SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SmartMoE 是面向 MoE 模型分布式训练的两阶段自动并行化系统，核心运行时调度包含：(1) **Offline Pool Construction（离线池构建）**：基于 workload-aware 性能模型，在训练前搜索最优混合并行策略组合（Data + Tensor + Pipeline + Expert Parallelism），以及同构策略组合中仅 expert placement 可变的执行计划池（pool）。Pool 中所有候选执行计划有相同 expert slot 配置，保证运行时切换开销只涉及参数交换（无内存分配/释放）；(2) **Online Adaptive Parallelization（在线自适应并行化）**：在训练运行时，根据 gating network 实际输出的 per-expert token 负载，周期性（默认每 10 iteration）执行三种轻量级 expert placement 搜索算法——Greedy（O(NE) 复杂度）、Dynamic Programming（O(N×4^E) 最优解）、Hybrid（Greedy+DP 组合，两阶段：先将 E 个 expert 用贪心分配到 M 个虚拟设备，再用 DP 将 M 个虚拟设备分配到 N 个物理设备，总复杂度 O(ME + N×4^M)，M 可调）；(3) **切换开销控制**：设置切换阈值过滤掉性能提升有限的 placement plan，利用 expert selection 的时间局部性（相邻 iteration 分布变化小）降低搜索频率，只在搜索前少数 iteration 收集历史统计避免无用开销。
  - 实验比较：(a) 端到端训练加速比——3 个 GPU 集群上 GPT-MoE（NLP）和 Swin-MoE（CV）模型，SmartMoE vs FasterMoE baseline 最高 1.88× speedup，平均 1.53×（inky A100 cluster）、1.17×（pinky V100 SXM cluster）、1.14×（blinky V100 PCIe cluster）；(b) 离线并行化消融——vs Alpa 推荐的 data-insensitive 并行方案，data-sensitive 方案达到 2.67× speedup（vs 2.36×）；(c) 在线并行化消融——16 层 MoE 在 64 V100 上 per-layer 平均 1.16× 加速，最高 1.43×；(d) 性能模型精度——R² > 0.5 for all configurations；(e) Overhead 分析——搜索 <1ms（1024 expert 时 <50ms），切换 ~20ms，而 Alpa 搜索需 825s。

- 后端平台是什么，配置是什么。
  - Cluster 1 (blinky)：8× NVIDIA V100 PCIe per node，max 32 GPUs，50Gb/s InfiniBand
  - Cluster 2 (pinky)：4× NVIDIA V100 SXM per node，max 64 GPUs，100Gb/s InfiniBand
  - Cluster 3 (inky)：8× NVIDIA A100 SXM per node，max 32 GPUs，200Gb/s InfiniBand
  - 软件：PyTorch（基于 FastMoE 框架），支持集成 Megatron-LM 和 DeepSpeed

- 评估性能的软件/脚本是什么。修改了什么。
  - 基础框架：FastMoE（Tsinghua 自研 MoE 训练框架，https://github.com/laekov/fastmoe）
  - SmartMoE 代码（https://github.com/zms1999/SmartMoE），Artifact Evaluation repo（https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE）
  - 修改内容：(1) 新增 expert slot 抽象——支持任意组合的 DP/TP/PP/EP 混合并行策略表达；(2) 新增 workload-aware 性能模型——基于 gating network 语义（capacity factor / topology-aware gate）估算 per-expert 负载上界，无需实际训练数据；(3) 新增 offline pool search——对候选池穷举搜索，使用性能模型预测最优池；(4) 新增 online 轻量级搜索算法——Greedy（O(NE)）+ DP（最优）+ Hybrid（两阶段可调复杂度）的 expert placement 搜索；(5) 新增 runtime 策略切换机制——包括切换开销阈值控制和搜索频率自适应；(6) 新增 MoE gating history 采集与同步模块
  - 快速复现（无需 GPU）：`./RUNME-a.sh`（处理预录 trace 数据生成图表，约 2 分钟）；深度复现：需 16× V100，约 2 小时

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/zms1999/SmartMoE（主代码），Artifact https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE（完整 AE）
  - 安装：基于 FastMoE，`cd src/fastmoe && USE_NCCL=1 python setup.py install --user`
  - 评估原理：测量端到端训练 latency（forward + backward + gradient sync + optimizer 全流程），使用真实数据集（OpenWebText for GPT-MoE，ImageNet for Swin-MoE）而非随机输入，确保 MoE 动态负载真实。Micro-benchmark 仅测量 MoE 层 forward/backward 时间。使用综合 expert selection 数据集（真实训练过程收集的不同模型结构和 gating 方法）
  - 运行时专家放置调度全过程（以 4 GPUs, 16 experts, GShard gate为例）：
    ```
    输入：token batch 分布在 4 GPUs（Expert Parallelism）
    
    [Offline - 训练前，一次性执行]
    Step 1 - 枚举混合并行候选池:
             遍历 DP/TP/PP/EP 组合 + expert slot 配置
             如：DP=2×TP=2, expert slots per GPU=4
    Step 2 - Workload-Aware 性能预测:
             基于 capacity factor=2.4 估算上界:
               max_tokens_per_expert = (capacity_factor × batch_tokens) / num_experts
             基于 topology-aware gate 估算通信量:
               intra-node expert 分配优先 → 估算 cross-node all-to-all 量
    Step 3 - 穷举搜索最优池:
             对候选池按预测性能排序 → 选最优池
             输出：固定 DP/TP/PP 策略，保留 expert placement 可变
    
    [Online - 每 iteration，周期性执行]
    Step 4 - Gate Network Forward (GPU):
             x → Linear(W_gate) → Top-K softmax → per-token expert indices
             
    Step 5 - Expert Selection History (CPU):
             收集 per-GPU 的 {expert_i: token_count_i}
             通过 All-Gather 聚合到 CPU scheduler
             
    Step 6 - Light-weight Expert Placement Search (<1ms):
             输入: C[16] = {E0:512t, E1:480t, E2:501t, ..., E15:490t}, N=4 GPUs
             
             方案A - Greedy (O(16×4)=64 ops):
               Sort experts by C_i descending: E5(520), E0(512), E3(508), ...
               For each expert in sorted order:
                 Pick GPU with min(samples[j]) AND experts[j] < E/N=4
                 Place expert on that GPU, update samples[j] and experts[j]
               结果: GPU_0={E5(520),E7(240),E12(245),E1(240)} → load=1245
                     GPU_1={E0(512),E9(250),E15(248),E2(242)} → load=1252
                     GPU_2={E3(508),E11(252),E6(249),E10(238)} → load=1247
                     GPU_3={E4(505),E13(248),E14(247),E8(245)} → load=1245
                     Imbalance = (1252-1245)/1245 = 0.56% (极低)
             
             方案B - Hybrid (Greedy + DP):
               先 Greedy: 16 experts → M=4 virtual devices（per node）
               再 DP: 4 virtual devices → N=4 physical devices（per node内最优）
               M=8（per GPU in node）时 → DP state: 2^8=256, 复杂度 O(4×4^8)
               
    Step 7 - 切换决策:
             计算 Δ = (当前plan延迟 - 新plan延迟) / 当前plan延迟
             若 Δ > threshold → 执行切换
             若 Δ ≤ threshold → 保持当前plan（避免微小改进引入通信开销）
             
    Step 8 - Expert Parameter Exchange (如触发切换):
             比较新旧 placement，确定需移动的 expert 参数
             All-to-All 交换参数（~20ms for 16 experts on 16×V100）
             无内存分配/释放（slot 配置不变）
             
    Step 9 - Expert FFN Computation:
             各 GPU 按新 placement 计算分配的 expert tokens
             与前 iteration 相比：优化后的 placement 消除负载热点
    
    性能输出：
    ├─ Per-iteration latency (ms): forward+backward total
    ├─ Speedup = T_baseline / T_SmartMoE
    ├─ 搜索开销 <1ms（搜索频率每10 iterations）
    ├─ 切换开销 ~20ms（被后续 iterations 的性能提升摊销）
    └─ End-to-end elapsed time with dynamic plan switching
    ```
  - 关键技术点：(1) Expert slot 是核心抽象——通过 capacity/#slots/#layers 三个属性统一表示所有并行策略，使得各种混合策略可互相比对和转换；(2) Workload-aware 性能模型的关键是利用 gating 超参数（capacity factor / topology-aware constraints）估算负载上界，而不是依赖实际运行时统计；(3) Pool 设计的核心洞察——固定 hybrid 策略、可变 expert placement 是在"优化空间"和"切换成本"之间的最佳平衡点。

## ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在DeepSpeed分布式训练框架上实现三项运行时通信与调度优化：(1) **Adaptive All-to-All Communication**：运行时监控每个GPU上per-expert的token选择计数，通过all-gather聚合各GPU的计数信息，计算精确的输入/输出slice大小，消除传统zero padding带来的不必要数据传输（训练早期zero ratio 88%，后期升至98%）；(2) **Dynamic Expert Clustering**：运行时profiling每个token在各MoE层的expert选择历史（使用<batchID, sequenceID, tokenIndex, tokenName>唯一标识，12B开销可忽略），基于K-means聚类（距离函数 = 序列长度 - 重叠expert选择数）将相似expert选择模式的token分组，复制热门expert到本地GPU HBM、将冷门expert offload到host pinned memory，重新映射expert到GPU位置以减少跨设备通信；(3) **Topology-aware Expert Remapping**：构建coverage matrix（C×C，cluster间expert覆盖度）和bandwidth matrix（GPU对间点对点网络带宽），使用遗传算法（fitness function = Σ((b·s - CM[SV[i]][SV[j]]·h) / BM[i][j])）搜索近最优cluster-to-GPU映射，在异构网络中最小化跨节点通信延迟。CPU侧聚类和remapping操作通过superbatch机制与GPU迭代overlap执行（overhead从12.48%降至0.001%）。
  - 实验比较：(a) 端到端迭代时间对比：Baseline(Tutel) vs +ADPT vs +ADPT+DEC vs ScaleMoE（全优化），在homogeneous和heterogeneous网络下评估MoE-BERT和MoE-GPT；(b) 性能随时间分析：epoch 1-21的speedup变化、all-to-all通信时间、通信量变化；(c) 灵敏度分析：MoE layer ratio (4/12, 6/12, 12/12)、k:Ne ratio (1:16, 1:32, 1:64)、superbatch size (1-400)、expert replica数量 (0-31)；(d) overhead breakdown：各操作延迟分解及overlap效果。

- 后端平台是什么，配置是什么。
  - 硬件：Amazon EC2 p4d.24xlarge实例 × 4节点，每节点8× NVIDIA A100 40GB GPU（共32 GPUs）
  - 节点内互联：NVLink 3.0 (600 GB/s)
  - 节点间互联：Ultra Ethernet (100 Gbps)；heterogeneous配置中限制一个节点至50 Gbps模拟云环境网络异构（带宽差2×）
  - 软件：PyTorch v2.0、DeepSpeed（基础分布式训练框架）、Tutel（baseline，2DH All-to-All配置）
  - 模型：BERT-MoE（encoder-only）、GPT-MoE（decoder-only），12层Transformer，hidden dim=768，sequence length=128，batch size=512
  - 变体参数：MoE layers={4, 6, 12}，experts Ne={32, 64, 128}，k:Ne ratio={1:16, 1:32, 1:64}

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：在DeepSpeed+Tutel框架上集成ScaleMoE的三个优化模块（Python package形式），测量平均迭代时间（average iteration time），计算speedup = baseline_iteration_time / ScaleMoE_iteration_time
  - 修改内容：在DeepSpeed的MoE层all-to-all通信路径中hook入adaptive all-to-all逻辑（替换原有zero-padded all-to-all dispatcher），聚合per-expert选择计数后使用精确slice size的NCCL all-to-all；在训练循环中加入dynamic expert clustering和topology-aware expert remapping模块（CPU执行），通过superbatch机制与GPU迭代overlap；修改expert-to-GPU内存布局以支持expert replication（热门expert）和offload（冷门expert到host pinned memory）
  - 基线：Tutel (built on DeepSpeed) with 2DH All-to-All configuration，不包含ScaleMoE的任何优化

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/SKKU-IDEAL/ScaleMoE
  - 评估原理：测量每个training iteration的wall-clock时间（从forward开始到backward结束），计算speedup vs Tutel baseline；单独profiling all-to-all通信时间、通信量（bytes）、expert selection分布
  - 运行时通信调度全过程（以4 GPUs, 4 experts, 10 tokens/GPU的MoE层forward pass为例）：
    ```
    输入：40 tokens (batch×seq) 分布在4 GPUs上，hidden dim=768

    [GPU侧 - 每个iteration]
    Step 1 - Router: GPU-i本地对每个token执行 gate(x·W_g) → top-k expert indices
               e.g., GPU-1 tokens选择: E1×4, E2×1, E3×3, E4×2
    Step 2 - 监控: GPU-i统计 per-expert dispatch counts
               GPU-1: {E1:4, E2:1, E3:3, E4:2}
    Step 3 - All-gather counts: 4 GPUs交换counts → 每个GPU计算全局input/output slice sizes
               dispatch: GPU-i 的第j列 = 发给GPU-j的token数
               combine: GPU-i 的第j行 = 从GPU-j接收的output数
    Step 4 - Adaptive All-to-All (dispatch): NCCL alltoallv仅发送有效token数据（无zero padding）
               token embedding (768 floats) → GPU(对应expert)
    Step 5 - Expert FFN: 每个GPU对local experts执行 FFN(assigned tokens)
    Step 6 - Adaptive All-to-All (combine): FFN outputs按slice size返回原始GPU
    Step 7 - Reorder: tokens按原始sequence顺序重排

    [CPU侧 - 每superbatch=100 iterations，与GPU overlap]
    Step A - Profiling: 收集per-token expert选择历史（前一个epoch）
    Step B - K-means Clustering: 按expert选择模式聚类tokens → C个cluster
    Step C - Topology-aware Remapping: 遗传算法搜索最优 cluster→GPU 映射
              Fitness = Σ((b·s - coverage[i][j]·h) / bandwidth[i][j])
    Step D - Expert Redistribution: 热门expert复制到本地HBM，冷门expert offload

    输出：FFN输出tokens（重排后）→ 下一Transformer层
    性能输出：iteration_time → speedup = baseline_time / ScaleMoE_time
              all-to-all通信量 (MB) → 减少比例
              GPU 负载均衡度
    ```

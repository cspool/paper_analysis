## TorchGT（拓扑诱导稀疏注意力图 Transformer 训练框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TorchGT 是 SC 2024 提出的面向大规模图 Transformer 训练的整体系统（Zhang et al., "TORCHGT: A Holistic System for Large-scale Graph Transformer Training"），核心用拓扑诱导的稀疏注意力（topology-induced sparse attention）与 cluster-aware 图并行降低 O(N²) 训练成本。TAGT 论文把 TorchGT 作为 SOTA GPU 软件 baseline（跑在 NVIDIA Tesla A100，6,912 cores、80GB HBM），并指出其三个局限：(1) 优化依赖严格拓扑前提——Hamiltonian path（NP-complete 验证、现实图常不满足）；(2) 前提失败被迫回退 O(N²) 全局注意力；(3) 选择性注意力导致明显准确率损失。TAGT-S（TDS 稀疏化的 DGL 软件实现）在 A100 上比 TorchGT 快 1.8×–2.5×，且准确率高于 TorchGT（Table VI：TorchGT 在 Reddit 上 GT 准确率 93.98% vs TAGT 97.11% 等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TorchGT 的稀疏注意力依赖图结构满足 Hamiltonian path（一条经过所有顶点一次的路径）以定义规则的稀疏注意力模式；对不满足的图回退全对注意力。执行管线与通用 GT 相同（Q/K/V 投影 → 稀疏/全对 QK^T → softmax → PV → FFN），区别只在注意力模式与并行策略（cluster-aware graph partitioning）。
- TAGT 的对照价值：TAGT-S 用 TDS（确定性、无 Hamiltonian 前提的稀疏结构）取代 TorchGT 的稀疏模式，在真实图数据集上持续优于 TorchGT 且精度接近全注意力参考。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：TorchGT 开源（https://github.com/hengruizhang98/torchgt）。TAGT 论文以其在 A100 上的执行时间/带宽利用/准确率为对比基准（Fig.3 profiling：off-chip 访问 60.5%、SM 利用 <25%、60.3% 数据冗余、cache line 利用 18.27%）。
- 使用：作为 GT 训练/推理的 GPU baseline；与 DGL-CPU（全注意力）、TAGT-S（TDS 稀疏化）、TAGT（FPGA 硬件）对比评估。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

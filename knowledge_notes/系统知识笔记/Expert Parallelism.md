## Expert Parallelism

术语解释
Expert Parallelism（EP）是MoE模型特有的分布式并行策略，将不同的expert分布到不同设备上，利用MoE稀疏激活特性使每个token仅访问部分设备上的expert，从而实现模型并行。

术语是什么？
Expert Parallelism的核心机制：
- 每个设备持有部分expert（可能1个或多个）+ 全部非expert参数（attention、layer norm等）
- 执行MoE层时，先在各设备本地完成attention和router计算
- 通过All-to-All通信将token重新分发到持有对应expert的设备
- 各设备计算其expert，再通过All-to-All通信将结果传回原始设备
- 执行时间主要由计算和通信两个阶段主导

EP通常与其他并行策略（Data Parallelism、Tensor Parallelism、Pipeline Parallelism）结合使用：
- 3D混合并行（Data + Tensor + Expert）：DeepSpeed-TED
- Intra-operator + Inter-operator重新分类：Alpa
- MoDa（MoE Parallism + Data Parallelism）：BaGuaLu

从系统架构角度拆解术语。
以Tutel的自适应并行策略为例，一个MoE层的端到端执行：
1. **输入准备**：batch tokens分布在各GPU上，每GPU持有全部非expert参数
2. **本地计算**：每GPU对本地token执行Attention + Router → 得到每个token的expert选择
3. **All-to-All Dispatch**：Router输出决定token的目标GPU → NCCL All-to-All通信重分发token
4. **Expert计算**：GPU收到token后对本地expert执行FFN
5. **All-to-All Combine**：将expert输出传回原始GPU
6. **输出聚合**：加权合并各expert输出

系统优化方向：
- 分层All-to-All（intra-node + inter-node分离）
- Expert放置优化（Prophet的贪心搜索、FlexMoE的细粒度复制）
- 负载均衡（Lazarus的expert副本分配、Brainstorm的历史分配数据）
- 通信与计算重叠（ScMoE的shortcut架构、EPS-MoE的动态kernel选择）
- 减少通信操作数（ExFlow将两次All-to-All减少为一次）

术语一般如何实现？如何使用？
- 基于DeepSpeed-MoE、Tutel、FasterMoE等分布式MoE训练/推理框架
- 基于NCCL或专用通信库（如DeepEP）的GPU间通信
- 关键配置参数：expert并行度（EP size）、expert副本数、容量因子（token buffer大小）
- 推理场景下，EP通常与vLLM等serving框架结合

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Accelerating Distributed MoE Training and Inference with Lina
- Accelerating MoE Model Inference with Expert Sharding
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA 训练使用 expert parallelism + ZeRO-1 data parallelism 组合，未使用 tensor parallelism 以减少通信开销；基于修改版 Megatron 框架，66 experts 分布在多个 GPU 上，attention 参数和 shared experts 在所有 GPU 上复制，routed expert FFN 参数通过 EP 分片；训练时 visual encoder 参数在所有 GPU 上复制（data parallel））
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts（Comet 在 Megatron-LM 的 EP 基础上用 NVSHMEM 替代 NCCL all-to-all 进行 token 级通信，通过 shared tensor dependency resolving + thread block specialization 实现 kernel 内通信-计算重叠，hide 86.5% 通信延迟。EP 配置下每 GPU 持有 1 至多个 expert，token 通过 NVSHMEM fine-grained get/put 而非 NCCL all-to-all 在 GPU 间路由）

**Comet 对 Expert Parallelism 的优化**：
Comet 不改动 EP 的 expert-to-device 映射关系，而是在通信机制层面将 NCCL all-to-all（coarse-grained, 完整大 tensor）替换为 NVSHMEM（fine-grained, token 级 intra-kernel I/O）。这使得：(1) 每个 token 到达后立即可被计算消费，不再等待整批 all-to-all 完成；(2) 通信和计算融合在单一 GPU kernel 中，消除 CPU 端 kernel launch/scheduling overhead；(3) 不同 EP/TP 组合下通过 adaptive thread block assignment 自动选择最优 SM 资源分配。在 H800 (NVLink) 上 1.96× 单层加速、1.71× 端到端加速；在 L20 (PCIe 25 GB/s) 上 1.19-1.46× 加速。

**Lina 对 Expert Parallelism 的扩展**：
Lina 在标准 Expert Parallelism（1 device = 1 expert）基础上引入两种动态调整机制：
1. **Expert Packing (Training)**: 当 FFN micro-op time << All-to-All micro-op time 时，动态 pack 多个 experts 到同一 device（2^n 递增），使 FFN 总时间对齐 All-to-All time，消除 pipeline bubble。Packing factor 调整频率: 每 10 steps。使用 DRAM-offloading 处理 GPU memory 不足。
2. **Dynamic Expert-Device Mapping (Inference)**: 不再固定 1:1 mapping。基于 expert popularity estimation，将 popular experts 复制到多 device，unpopular experts 打包到少 device。使用 first-fit-decreasing heuristic 最小化总 device 数。Scheduling 在每 MoE 层执行（two-phase）。

**MoEShard 对 Expert Parallelism 的替代**：
MoEShard 从根本上放弃了 Expert Parallelism（完整 expert 放置到不同 GPU），转而使用 Expert Sharding（将每个 expert 的矩阵沿 tensor 维度切分到所有 GPU）。区别：
- EP: GPU_i 持有完整 expert E_i, E_{i+1}, ... → 通过 all-to-all scatter/gather 路由 token → load imbalance 由 routing skew 决定
- MoEShard: 每个 GPU 持有所有 expert 的 1/|G| shard → 所有 token 全复制到所有 GPU → 每 GPU 计算全部 token 的 partial output → perfect load balancing，无需 all-to-all scatter/gather

**Task-MoE 对 Expert Parallelism 的消除（Kudugunta et al., EMNLP 2021）**：
Task-MoE 通过 task-level routing 从根本上消除了解码时的 Expert Parallelism 需求。因为同一 task 的所有 token 路由到相同的 experts，每个 task 仅需将 K=2 个 experts 加载到单加速器，无需跨设备 all-to-all 通信。Token-MoE 解码时 26.9%-36% step time 用于跨设备通信，Task-MoE 仅 0.0%-0.2%。不同 task 可在不同设备上独立并行解码，实现天然的多 task 并行。

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

**Capacity-Aware Inference 对 Expert Parallelism 的优化**：
Capacity-Aware Inference 识别了 EP 推理中的 Straggler Effect：MoE 层延迟 L ∝ max({N_i})，由最高负载 expert 决定。EP 下每 GPU 托管 n_l 个 expert，straggler expert 延迟通过 All-to-All barrier 传播到所有 GPU。Capacity-Aware Token Drop 和 Expanded Drop 在 All-to-All 通信前施加容量约束 C=γN̄，减少 straggler expert 负载。效果与 per-GPU expert 数负相关：Mixtral (1-2E/GPU) 获 1.85-1.87× 加速，OLMoE (8E/GPU) 加速显著降低。论文建议分配更多 GPU 做 EP 以增强容量约束效果。

---

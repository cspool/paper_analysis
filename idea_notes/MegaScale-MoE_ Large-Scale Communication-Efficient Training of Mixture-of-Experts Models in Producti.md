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

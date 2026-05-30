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

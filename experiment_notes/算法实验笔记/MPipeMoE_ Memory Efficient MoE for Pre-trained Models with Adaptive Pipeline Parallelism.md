## MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MPipeMoE，基于 PyTorch 1.9 + CUDA 11.1 的 MoE 训练库，包含三个核心实现：
    - **Adaptive Pipeline Parallelism**：将 mini-batch 的 tokens 沿 batch 维度切分为 n 个 micro-batch，对 MoE 层的三个阶段（S: All-to-All dispatch → C: Expert FFN 计算 → R: All-to-All collect）进行 pipeline 并行，重叠通信与计算。与 FasterMoE 沿 node 维度切分不同，MPipeMoE 沿 batch 维度切分，保留 NCCL All-to-All 集体通信优化能力。
    - **Adaptive Granularity Configuration (Algorithm 1)**：基于"n 随 B 单调递增"的假设，将 B 的值域划分为不相交区间{R_n}，每个区间一对一映射到最优 n。通过二分搜索树维护 (n, R_n) 映射集，以 O(log n) 复杂度查找。当 cache miss 时调用 searchBestGran(B) 做 trial-based 搜索。
    - **Memory Reusing Strategies (S1-S4)**：识别 pipeline parallelism 中的 "memory bubbles"——不同 micro-batch 的 T_DI、T_M、T_DO 在不同时刻激活，可共享同一 buffer。n 个 partition 的 activation buffer 从 O(n) 压缩为 O(1)，节省 ΔM_act = B*(2M*(n-2)/n + H*(n-1)/n)。为恢复 backward pass 所需的被覆写 tensors，设计 4 种策略：S1 (T_DI/T_M 均 CPU offload)、S2 (T_DI 通信重发 + T_M offload)、S3 (T_DI offload + T_M 重计算)、S4 (T_DI 通信重发 + T_M 重计算)，通过性能模型在运行时选择最优策略。
  - 实验比较：
    - 端到端训练速度：PipeMoE vs FastMoE vs FasterMoE，MPipeMoE vs PipeMoE vs FastMoE vs FasterMoE（Figure 8, 9）。
    - 内存占用：MPipeMoE vs FastMoE vs FasterMoE，归一化到 FastMoE（Figure 9）。
    - 理论内存节省 bound vs 实际节省（Figure 10），n=2/4/8 及 B=4k-32k。
    - Pipeline granularity 有效性：不同 n（1/2/4/8/16）在不同 B（2k-32k）下的性能（Figure 12）。
    - 内存复用策略开销：S1-S4 在不同 (N, B) 下的表现（Figure 13）。
    - 性能分解（Figure 11）：memory-time 坐标系下各方法对比。
  - 变体：*PipeMoE* 仅含 pipeline parallelism（无 memory reuse），*MPipeMoE* = PipeMoE + memory reuse。

- 硬件平台是什么，配置是什么。
  - 8 台 NVIDIA DGX A100 服务器，每节点 8×A100 SXM 40GB GPU，200 Gbps HDR InfiniBand 互联，96×第 2 代 AMD EPYC CPU 核，1.9 TiB 内存。节点内第 3 代 NVLink + NVSwitch。跨节点 1,600 Gbps InfiniBand 自适应路由。
  - 软件栈：PyTorch 1.9.0、CUDA 11.1、NCCL（版本论文未明确说明）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：3 种 MoE 配置（Table III）：
    - MoE-GPT3-S: d_model=768, d_hidden=3072, #experts=64
    - MoE-GPT3-XL: d_model=2048, d_hidden=8192, #experts=64
    - MoE-BERT-L: d_model=1024, d_hidden=4096, #experts=64
    - Expert 为 FFN（Linear1 → GeLU in-place → Linear2），gating 为 top-1 routing。
  - 数据集：Dummy dataset（随机生成 tokens），评估目标是训练系统的 throughput 和 memory footprint。
  - 优化器：Adam（momentum + variance 各占参数量内存）。
  - 评估指标：平均训练时间（计算 speedup）、峰值内存占用（peak memory footprint）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/whuzhangzheng/MPipeMoE
  - 算法 pipeline 伪代码：

```
输入: MoE layer input T_I, shape (N, B, M)
      N=device数, B=batch size, M=model dim
      n=pipeline granularity, S∈{S1,S2,S3,S4}=memory reuse策略
输出: MoE layer output T_O

// 1. Adaptive Granularity Search (Algorithm 1)
if B in cache_table:
    n = cache_table[B]
else:
    在集合S = {(n, range(B_lower, B_upper))} 中查找B所属区间
    if 未找到:
        n = searchBestGran(B)  // trial-based搜索
        插入新的(n, range(B,B))到S
    cache_table[B] = n

// 2. Micro-Batch Pipeline (按batch维度切分)
将T_I沿dim=1切为n个micro-batch: T_I[0..n-1], each (N, B/n, M)
Pipeline调度 (交替S和R以增强memory locality):
  stream_comm: S(0)→S(1)→R(0)→S(2)→R(1)→...
  stream_comp:         C(0)→C(1)→C(2)→...
  // S: All-to-All dispatch, C: Expert FFN, R: All-to-All collect
  // C(i) = Linear2(GeLU(Linear1(T_DI[i])))

// 3. Memory Reuse (共享buffer)
Buffer_DI = alloc(B/n * M * sizeof(fp16))  // n个partition共享
Buffer_M  = alloc(B/n * H * sizeof(fp16))
Buffer_DO = alloc(B/n * M * sizeof(fp16))
// 前向: 各micro-batch依次复用同一buffer, 后写入覆盖前写入

// 4. Backward Tensor Recovery (按策略S)
switch S:
  case S1:  // T_DI, T_M 均CPU offload
    fwd: D2H_copy(T_DI[i], T_M[i])
    bwd: H2D_copy(T_M[i], T_DI[i]), 计算梯度
  case S2:  // T_DI通信恢复, T_M offload
    fwd: D2H_copy(T_M[i])
    bwd: H2D_copy(T_M[i]), T_DI[i]=AlltoAll_replay(T_I[i]), 计算梯度
  case S3:  // T_DI offload, T_M重计算
    fwd: D2H_copy(T_DI[i])
    bwd: H2D_copy(T_DI[i]), T_M[i]=FFN_fwd(T_DI[i]), 计算梯度
  case S4:  // T_DI通信恢复, T_M重计算
    fwd: (无额外操作)
    bwd: T_DI[i]=AlltoAll_replay(T_I[i]), T_M[i]=FFN_fwd(T_DI[i]), 计算梯度

// 5. 性能模型选择最优S (Eq 10)
v0_comp = b*H*M, v0_comm = b*M, v0_mem = b*M, b=B/n
C(S) = max(q1*v0_comp/(σ*W_comp), q2*v0_comm/(μ*W_comm), q3*v0_mem/(η*W_mem))
// Q_fw=[q1,q2,q3]见表II, μ/σ/η为干扰slowdown因子, 选C最小的S

// 6. Memory Footprint
M_act^pipe = M_buf^pipe = 4*B*M + B*H          // Eq 4
ΔM_act = ΔM_buf = B*(2M*(n-2)/n + H*(n-1)/n)  // Eq 5
φ = (ΔM_act+ΔM_buf)/(M_ms+M_act^pipe+M_buf^pipe) // Eq 6, M_ms = 4*(E*M+2*H*M)
```

  - 关键结果：
    - PipeMoE 平均 2.26× speedup vs FasterMoE，最高 3.4× vs FasterMoE，最高 3.7× vs FastMoE（Figure 8）。
    - MPipeMoE 内存节省：平均 23%（最高 40%）vs FastMoE，平均 27%（最高 47%）vs FasterMoE，同时 3.1× speedup（Figure 9）。
    - 实际内存节省约达理论上限的 95%（Figure 10）。
    - Pipeline granularity: B<8k 时 n=2 最优, 8k-22k 时 n=4, >22k 时 n=8（Figure 12）。
    - 内存复用策略选择：N 小时 S1/S2 更优（I/O bound 容忍），N 大时 S4 更优（避免 memory bandwidth 竞争）（Figure 13）。
    - 策略 S1/S2 在小 N（如 8 GPU）表现好，S4 在 N=32/64 时表现好（通信瓶颈场景）。

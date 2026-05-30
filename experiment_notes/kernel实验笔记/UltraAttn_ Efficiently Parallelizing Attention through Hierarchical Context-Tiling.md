## UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  UltraAttn 在 kernel 调度/运行时计算层面的核心实现包括：(i) **Device-Level Context-Tiling**——将 attention workload 沿 Q 和 KV 两个维度划分为 $P \times P$ grid，通过 ILP 求解器（Gurobi）在计算负载均衡约束（$DLI_{P,CP} \le \theta_{DLI}$）下最小化每个 GPU 的通信量（MCV），输出每个 block $B_{r,c}$ 到设备 $U_g$ 的分配方案 $x_{r,c,g}$，同时推导出 Q/KV/O 的入站/出站流量（变量 A/B/C/D），生成 peer-to-peer 通信计划；(ii) **Node-Level Context-Tiling**——将跨节点通信异构性问题解耦，先在节点级别做 context-tiling（将每个节点视为集成设备，节点间使用 groupwise peer-to-peer 充分利用所有 NIC 带宽），再在设备级别做 context-tiling（节点内 NVLink peer-to-peer），两层方法类比统一；(iii) **Kernel-Level Context-Tiling**——在 node/device-level tiling 生成的 parallel dependency graph（DAG）上通过三种 substitution（computation kernel batching、peer-to-peer communication batching、collective communication batching）进行图变换，使用贪心策略按 transformation gain 降序逐个应用变换，通过 ILP runtime 评估执行时间决定保留与否，自适应选择最优 kernel granularity 以平衡 kernel overlap 和单 kernel device utilization；(iv) **ILP-based Runtime**——将 parallel dependency graph 的 kernel 调度形式化为 ILP：每个 kernel v 的 start time $S_v$（实变量）+ duration $D_v$（profiling 获取），同一 CUDA stream 内的 kernel 通过 $Order_{uv}$ 布尔变量控制互斥执行（Stream Exclusivity Constraints），kernel 依赖通过 DAG 边约束 $S_u + D_u \le S_v$ 保证，目标最小化 $End\_Time$。求解后按 $S_v$ 排序得到每 CUDA stream 的最优 kernel 执行顺序。Context remap 技术（$\phi: T \to CR$ 映射）作为离线预处理步骤，用于增强 workload locality（如 strided attention 用 $\phi(t_i) = \lfloor i \cdot 16/S \rfloor \mod 4$），仅在 attention 模块内影响性能，不影响其他 LLM 模块。

  实验比较：(1) 端到端训练/推理速度对比（Llama2-7B, S=512K, CP=8/64, Nh=1/32, 64 GPUs）vs ring attention、striped attention、zigzag ring attention baseline，UltraAttn 平均 $2.2\times$（Nh=1）和 $3.4\times$（Nh=32）端到端加速；(2) 分布式 attention 模块加速比——6 种 attention pattern：dense（full/causal）+ block sparse（strided/global+local/star/streaming），CP=2-64, Nh=1/32，平均加速从 $10.2\times$（strided training, Nh=1, CP=64）到 $1.9\times$（streaming inference, Nh=32, CP=8）；(3) 消融实验——逐步叠加 Node Tile → Node+Device Tile → Node+Device+Kernel Tile → +ILP Runtime，每步独立性能增益；(4) 强可扩展性——固定 S=512K，CP 从 16 到 64 的 MFU 变化，对比 baseline；(5) 性能预测准确度——intra-node $R^2=0.9932$，inter-node $R^2=0.9181$；(6) ILP 搜索开销——node/device-level tiling ILP 0.07ms（strided, CP=16）到 3672ms（causal, CP=64），runtime ILP 0.10ms（strided, CP=16）到 1073ms（causal, CP=32）。

- 后端平台是什么，配置是什么。
  8 节点集群，每节点 8× NVIDIA H100-NVLink-80GB GPU（共 64 GPUs），96 CPU cores，2 CPU sockets。节点内 NVLink 双向带宽 450GB/s。节点间 8× 400Gb/s InfiniBand EDR，每 GPU 与 1 NIC 有 affinity，PCIe-5.0 连接。软件：PyTorch 2.6.0、NCCL 2.21、FlashAttn 2.5.7、Gurobi ILP solver。模型：Llama2-7B，batch_size=1，FP16/BF16。Attention patterns：full attention、causal attention、strided attention、global+local attention、star attention、streaming attention。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + NCCL + FlashAttn 的自研 context parallelism 系统（~10K 行 Python）。直接使用 NCCL C-level API（非 torch.distributed）以获得更精细的通信 kernel 调度。修改/新增内容：
  - **ILP context-tiling 模块**：将 attention workload 建模为 ILP（9 个变量类型，10 个约束组），通过 Gurobi 求解器输出最优 workload-to-device 分配。变量包括 $x_{r,c,g}$（block 分配）、$H_{g,r}$（是否需要 $Q_r$）、$V_{g,c}$（是否需要 $KV_c$）、A/B/C/D（四种流量方向）、$Cin_g/Cout_g$（总通信量）、MCV（最大通信量）。10 个约束组：Allocate Uniqueness、Definition of H/V/A/B/C/D、Inbound/Outbound Traffic、Computation Balance、Minimization Objective
  - **Context remap 预处理**：对某些 pattern（如 strided attention）手动定义 $\phi$ 映射增强 locality。$\phi(t_i) = \lfloor i \cdot CP/S \rfloor$ 为默认序贯映射，$\phi(t_i) = \lfloor i \cdot 16/S \rfloor \mod 4$ 用于 strided attention 增强 node-level locality
  - **Parallel dependency graph 构建**：从 tiling 结果生成 DAG，节点为 computation kernel（矩形）、receive kernel（椭圆）、send kernel（菱形），边表示依赖
  - **Greedy kernel-level tiling**：在图变换候选集（三种 substitution）中按 gain（融合后减少的时间）降序排序，贪心选择
  - **ILP runtime**：将 kernel 调度建模为 ILP，stream exclusivity + dependency constraints + end time minimization
  - **CUDA graph 集成**：cudagraph 消除 CPU kernel launch overhead

  评估脚本 workflow：
  1. 编译 NCCL from source（`third_party/comm_test/third_party/nccl`, target sm_90）: `make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"`
  2. 集群 profiling：`third_party/kernel_profiler/scripts/bench_ops_m2_py.sh` 采集 FlashAttn kernel 性能 → `third_party/comm_test/scripts/wrapper_conflict_bench_hamming.sh 8` 和 `16` 分别采集 intra-node 和 inter-node NCCL 通信性能
  3. 创建 database 并复制 profiling 数据到 `database/m_configs/`
  4. 分布式 attention 评估：`scripts/schedule/task1_BSA_hamming.sh bsa_train`（单节点 8 GPU 生成执行计划 + 评估 intra-node）→ `scripts/schedule/task2_BSA_hamming.sh bsa_train`（8 节点评估 inter-node），结果缓存到 `database/inter_bsa_exe_plans_profile.json`
  5. Baseline：`third_party/UltraAttn_baseline/scripts/runtime/run_exp.sh`，结果手动复制到 database
  6. 端到端：Megatron-LM 上 `scripts/ultraattn_e2e.sh`（Llama2-7B），结果存 `results/UltraAttn_E2E/hamming/`
  7. 绘图：`python plot/da_bsa_training_pick.py` 等生成 Figure 7-11

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/oliverYoung2001/UltraAttn，Zenodo DOI: 10.5281/zenodo.15301789。Artifact 提供 Makefile 一键复现：`make spack_packages` → `make prepare_conda_env` → `make compile` → `make cluster_profile` → `make figure7`（约 6h）/ `make figure8`（约 9h）/ `make figure9`（约 10min）/ `make figure6`（约 1h）/ `make figure10` / `make figure11`。编译 NCCL 需 GCC，建议使用 Spack 管理依赖。

  评估原理与流程（以 strided attention training, CP=64, S=512K, Nh=1, 64 GPU 为例）：

  1. **Cluster Profiling（前置步骤）**：
     - Attention kernel profiler：在不同 (M, N, K) shape 组合下测量 FlashAttn kernel 执行时间 → 构建 kernel 性能 lookup table（$D_v$ duration 的来源）
     - Communication profiler：测量 intra-node NVLink 和 inter-node InfiniBand peer-to-peer 在不同 message size 下的带宽/延迟 → 构建通信性能 lookup table
     - 这些数据构成 ILP 性能预测模型的底层输入

  2. **Adaptive Workload Partition**：
     - 输入：strided attention pattern（Figure 2(c) 的 diagonal stripes），Q/KV context length = 512K
     - 计算 partition degree P：从公式 $DLI_{P,CP} = \lceil COMP/CP \rceil / (COMP/CP) - 1$ 递增 P 直到 DLI 低于阈值 $\theta_{DLI}$
     - 将 attention workload 划分为 $P \times P$ grid，每 cell 标注 Full/Causal/Empty（strided pattern 仅 diagonal band 内 cell 非空）

  3. **Node-Level Context-Tiling ILP**（CP > 8 时启用）：
     - 输入：$P \times P$ grid block table，FB/CB/EB 集合，CP=8（8 nodes），profiling 数据
     - ILP 变量：$x_{r,c,g}$（binary, $B_{r,c}$ 是否分配给 node g），$H_{g,r}$（binary, node g 是否需要 $Q_r$），$V_{g,c}$（binary, node g 是否需要 $KV_c$），A/B/C/D（integer, 四种流量），$Cin_g/Cout_g$（integer, 总通信量），MCV（integer, 最大通信量）
     - 约束：Allocate Uniqueness（$\sum_g x_{r,c,g}=1$）+ H/V definition（$H_{g,r} \ge x_{r,c,g}$）+ A/B/C/D definition（含 Cmap 映射）+ Inbound/Outbound Traffic（$Cin_g = A_g \times 1 + B_g \times 2 + C_g \times 1$，系数为 Q/KV/O 的 per-token 数据量比）+ Computation Balance（$\sum_{FB} x_{r,c,g} \times 1 + \sum_{CB} x_{r,c,g} \times 0.5 \le \tau$，$\tau = \lceil (|FB| \times 1 + |CB| \times 0.5) / CP \rceil$）
     - 目标：minimize MCV（$\ge \max\{Cin_g, Cout_g\}; \forall g$）
     - 输出：每个 node 分配的 workload blocks + node 间 groupwise peer-to-peer 通信计划（需 Q/KV/O 的 source→destination）

  4. **Device-Level Context-Tiling ILP**：
     - 在每个 node 内（8 GPU），对分配的 block subset 再次求解相同 formulation 的 ILP（CP=8, peer-to-peer 替代 groupwise peer-to-peer）
     - 输出：每个 GPU 分配的 workload blocks + GPU 间 peer-to-peer 通信计划
     - ILP 时间随 attention pattern 密度和 P 增长：strided P=2 仅 0.07ms，causal P=8 达 3672ms

  5. **Parallel Dependency Graph Construction**：
     - 从 device-level tiling 结果生成 DAG：每个 GPU 的 computation kernel（FlashAttn 对分配的 blocks 执行 attention）+ receive kernel（NCCL recv: 接收来自其他 GPU 的 Q/KV）+ send kernel（NCCL send: 发送本地 Q/KV 到其他 GPU）
     - 依赖关系：receive → computation → send（数据流依赖），以及跨 GPU 的通信依赖
     - 示例（Figure 5b, GPU1）：A0 → Q3(send to GPU3) → KV3(recv from GPU3) → A1 → A2

  6. **Greedy Kernel-Level Tiling**：
     - 生成三种 substitution 的 transformation candidates：
       a. Computation kernel batching：相邻 computation kernel 合并为大 kernel（受 FlashAttn backend 的 attention shape 支持范围限制）
       b. Peer-to-peer communication batching：同一 (src, dst) 对的多个 send/recv 合并
       c. Collective communication batching：多个 peer-to-peer 合并为 collective（如 all-to-all）
     - 按 transformation gain（融合后减少的时间）降序排序
     - 贪心遍历：检查 candidate 的 kernel 是否未被之前变换修改 → 应用到 DAG → 通过 ILP runtime 评估执行时间 → 若改善则保留

  7. **ILP Runtime Kernel Scheduling**：
     - 将 DAG kernel 按共享带宽分组：同一输出带宽的 send kernel → 同一 CUDA stream；同一输入带宽的 recv kernel → 同一 stream
     - 在每 stream 内求解 ILP：
       - 变量：$S_v$（start time），$Order_{uv}$（boolean, u 是否在 v 之前），$End\_Time$
       - Stream Exclusivity：$(S_u + D_u \le S_v + (1 - Order_{uv})Ub) \land (S_v + D_v \le S_u + Order_{uv}Ub)$
       - Dependency：$S_u + D_u \le S_v$ for $\forall (u,v) \in E$
       - 目标：minimize $End\_Time$
     - 求解后按 $S_v$ 排序得最优 kernel 执行顺序 → 转换为 CUDA stream graph
     - ILP 时间：从 0.10ms（strided, CP=16）到 1073ms（causal, CP=32）

  8. **CUDA Graph 执行**：
     - 将 CUDA stream graph 编译为 CUDA graph
     - GPU 执行：各 CUDA stream 并行执行，stream 内按 ILP 顺序串行
     - 实际时间线：computation kernel（FlashAttn forward）与 communication kernel（NCCL send/recv）交错，最大化 computation-communication overlap

  9. **Performance Measurement**：
     - CUDA event timing 测量 distributed attention 模块 wall-clock time
     - Speedup = baseline_time / UltraAttn_time
     - MFU = achieved TFLOPS / peak TFLOPS（H100 FP16: 989 TFLOPS × 64 = 63,296 TFLOPS 理论峰值）
     - 性能预测准确度：对比 predicted time vs actual time，计算 $R^2$ 和 relative error
     - Results：intra-node 仅 3.0% cases 超过 30% relative error，inter-node 约 5.8% 超过 50%

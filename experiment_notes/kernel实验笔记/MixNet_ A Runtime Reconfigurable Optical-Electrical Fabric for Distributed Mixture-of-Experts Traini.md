## MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现自定义 **collective communication runtime**（约 6K 行 C++），基于 NCCL 扩展，支持 **in-training topology reconfiguration**。核心组件：
    1. **Traffic Monitor**：运行时追踪 EP 的 traffic demands，利用 MoE 训练框架已有的 token dispatch 信息收集机制（如 Megatron-LM token_dispatcher.py）预测后续 all-to-all 通信模式。
    2. **Topology Controller**：去中心化 greedy 算法（Algorithm 1）——根据预测的 traffic demands，迭代识别 bottleneck links（最长完成时间的 GPU 对），优先为这些 pairs 分配直接 OCS 电路，生成 NUMA-optimized NIC 映射。
    3. **Topology-Aware EP Routing**（5 步流程）：(1) 各 GPU 查拓扑确定 intra-server delegation GPU → (2) intra-host gather via NVSwitch（数据汇聚到 delegation GPU）→ (3) inter-host all-to-all via OCS（优先）+ EPS（fallback）→ (4) intra-host all-to-all among local experts via NVSwitch → (5) delegation GPU scatter 数据到最终目标。步骤 (3) 和 (4) 通过 overlap 减少完成时间。
    4. **DP Hierarchical All-Reduce**：intra-host reduction via NVSwitch → inter-host ring all-reduce via EPS → intra-host broadcast via NVSwitch。多 EPS NIC 时使用 multi-ring all-reduce。
    5. **Traffic Demand Prediction (MixNet-Copilot, §B.1)**：使用 Sequential Least Squares Programming (SLSQP via scipy.optimize) 估计 conditional probability matrix P（前一层 expert load → 当前层 expert load），基于最近 k 次迭代的加权平均。利用预测结果提前重配置 OCS 拓扑以处理 FP 第一个 all-to-all。
  - 实验比较：
    - 原型：MixNet custom runtime + OCS reconfiguration vs 4×100G EPS baseline（NCCL all-to-all + all-reduce），训练 3 个 MoE 模型。
    - 仿真：MixNet topology-aware EP routing vs Fat-tree / Rail-optimized / TopoOpt 的 collective communication（NCCL-based）。

- 后端平台是什么，配置是什么。
  - 原型：4 台 server，每台 8×NVIDIA A100 GPU + 4×Mellanox ConnectX-6 100G NICs。3 NIC per server 接 Polatis OCS（RoCEv2），1 NIC 接 SN3700 Ethernet switch。每 server 内部 4×NVLink（相邻 GPU 对互联）。RDMA 通信使用 FuseLink raw ibverbs library。
  - 仿真：每 server 8 GPU（NVSwitch 900 GB/s）+ 8 NIC（带宽 B）。EPS fabric 用 fat-tree 拓扑。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **NCCL**（NVIDIA Collective Communications Library）+ **Megatron-LM** 构建。修改内容：
    1. 实现自定义 collective communication runtime（C++ ~6K LoC），通过 RDMA raw ibverbs（FuseLink）进行高速数据传输。
    2. 将 MixNet runtime 移植到 Python 以集成 Megatron-LM，实现通信原语：`mixnet.all_to_all` 和 `mixnet.all_reduce`（类似 torch.dist 接口）。
    3. DP 和 PP 通信复用了 NCCL 的高性能 intra-host 和 inter-host all-reduce/point-to-point 通信。
  - 对比 baseline：4×100G EPS 配置（全部 NIC 经 Ethernet switch），使用标准 NCCL。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://mixnet-project.github.io/
  - 通信 runtime 全栈执行流程（以 EP all-to-all，4 servers × 8 GPUs, EP degree=16 为例）：
    ```
    // ===== Step 1: Topology Lookup =====
    // 每个 GPU 查重配置后的 OCS 拓扑，确定 intra-server delegation GPU
    // delegation_gpu[src_server][dst_server] = 与 dst_server 有直接 OCS 链路的本地 GPU
    // 优先选择 OCS 直接连接的 GPU 对，否则 fallback 到 EPS
    
    // ===== Step 2: Intra-host Gather (NVSwitch) =====
    // 每 server 内部：各 GPU 通过 NVSwitch gather 出站数据到 delegation GPU
    // 数据按目标 server 分组，发送到对应的 delegation GPU
    // NUMA balancing: topology controller 确保多 OCS NIC 分布在多个 NUMA node
    
    // ===== Step 3a: Inter-host All-to-All via OCS (RDMA) =====
    // delegation GPU 之间通过 OCS 直连电路执行 RDMA write
    // 使用 FuseLink raw ibverbs: ibv_post_send(qp, wr, &bad_wr)
    // OCS 电路提供专用高带宽路径，无 packet switching overhead
    
    // ===== Step 3b: Inter-host All-to-All via EPS (RDMA fallback) =====
    // 无 OCS 直连的 GPU 对通过 EPS fabric 传输
    // 使用标准 RDMA over Ethernet
    
    // ===== Step 4: Intra-host All-to-All (NVSwitch) =====
    // 每 server 内部：接收到的 remote expert data 通过 NVSwitch
    // all-to-all 分发给本地 expert 对应的 GPU
    // Step 3 和 Step 4 通过 CUDA stream overlap 并行执行
    
    // ===== Step 5: Intra-host Scatter (NVSwitch) =====
    // delegation GPU 将收到的 all-to-all 数据 scatter 到最终目标 GPU
    
    // ===== DP Hierarchical All-Reduce 流程 =====
    // Stage 1 (intra-host, NVSwitch): GPU 内部 reduce → gateway DP GPU
    // Stage 2 (inter-host, EPS): ring all-reduce among gateway GPUs
    //   - 多 EPS NIC 时使用 multi-ring all-reduce 充分用满带宽
    // Stage 3 (intra-host, NVSwitch): gateway GPU broadcast → 所有 GPU
    
    // ===== OCS Topology Reconfiguration Algorithm (Algorithm 1) =====
    // 输入: E (expert all-to-all communication demands), α (optical degree),
    //       N (#servers), V (server node set)
    // 输出: S (NIC-level mapping in OCS)
    //
    // 1. D = CALCULATE_SERVER_DEMAND(E)
    //    // 将 expert-level traffic matrix 映射到 server-level demand
    //    // TX+RX 合并为 upper triangular matrix
    // 2. while True:
    //      (i,j) = FINDBOTTLENECKLINK(T, C, V)
    //      // 找完成时间最长的 server pair（T[i][j] = D[i][j] / C[i][j]）
    //      if avail_ocs[i] > 0 and avail_ocs[j] > 0:
    //        C[i][j]++, C[j][i]++  // 分配 OCS 链路
    //        avail_ocs[i]--, avail_ocs[j]--
    //      else: break  // 所有 NIC 端口已分配完
    //      T[i][j] = D[i][j] / C[i][j]  // 更新完成时间矩阵
    // 3. S = GetNICMapping(C)
    //    // 将链路分配矩阵 C 转化为 TX/RX NIC 配对
    // 4. S = permuteLinks(S)
    //    // NUMA-aware permutation 避免 intra-host congestion
    // 5. RECONFIGUREOCS(S)
    //    // 向 OCS 发送 TL1 commands 执行物理重配置
    ```

  - 关键性能结果（原型）：
    - MixNet（12 optical + 4 electrical ports）达到与 4×100G EPS baseline（16 electrical ports）相当的训练性能，证明用更少的总端口数（低成本）即可匹配全电气方案。
    - Reconfiguration turnaround time: 平均 41-47ms（1-16 pairs），99th percentile <70ms，NIC activation time 平均 ~5.67s（受限于 commodity transceiver 未针对快速重配置优化——论文排除此时间）。

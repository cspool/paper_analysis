## Multi-Die Task Allocation for MoE (多Die MoE任务分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Die Task Allocation 是运行在 wafer-scale GPU Global CP 上的启发式算法，将 MoE kernel 计算按 expert 拆分为 per-die 子任务，基于 expert 在各 die 上的分布信息和 cost model 将子任务分配到最优 die。核心创新是引入 **Candidate Mechanism**（扩展候选 die 到邻居 die，而不限于 expert 所在 die）和 **Block-Granularity Distribution**（以 block size=50 为单位分配请求，在效率和精度间折中）。该算法是 Insight 3（expert-placement-aware workload distribution）的实现——在 single-GPU-like programming model 下，算法对软件完全透明，运行在 Global CP 硬件上。

从kernel调度角度拆解术语：
算法（Algorithm 1 in paper）的完整流程：

```
Input:  expert_reqs_dict = {expert_id: num_requests}
        expert_die_map = {expert_id: [die_ids where expert resides]}
Output: allocation_plan = [(expert_id, target_die, num_requests)]

1. 初始化 load_per_die[d] = 0 对所有 die d

2. Sort experts by req_num ascending  // 先处理冷门 expert

3. for each (expert_id, req_num) in sorted experts:
     
     // Candidate Mechanism: 候选 die = 存有该 expert 的 die + 邻居 die
     candi_list = GenCandidateList(expert_id, dis=1)
     // 按当前负载排序候选 die
     candi_list = Sort(candi_list, key=lambda i: load_per_die[i])
     // 限制候选数 max_split_num ∝ req_num
     candi_list = candi_list[:max_split_num]
     
     // Block-Granularity Distribution
     while req_num > 0:
         req_blk = min(50, req_num)  // block size = 50
         costs = CostModel(candi_list, req_blk)
         // CostModel = f(DRAM_access, compute, D2D_comm)
         target_die = Argmin(costs)
         allocation_plan.append((expert_id, target_die, req_blk))
         load_per_die[target_die] += req_blk
         req_num -= req_blk

4. allocation_plan = MergeTasks(allocation_plan)
   // 合并在同一 die 上的相同 expert 的任务

5. return allocation_plan

Function GenCandidateList(expert_id, dis):
    local_dies = expert_die_map[expert_id]
    remote_dies = FindNearDies(local_dies, dis)  // Manhattan distance ≤ dis
    return local_dies + remote_dies
```

**Cost Model** 考虑三个维度：
- $C_{\text{DRAM}}$: 读取 expert 权重的 HBM access time（local=300ns, remote=300ns + hops×200ns）
- $C_{\text{compute}}$: 基于 die 的 FP16 TFLOPS 和请求数估算的 GEMM 执行时间
- $C_{\text{D2D}}$: 从远程 die 读取 expert 权重跨 D2D links 的通信时间（bandwidth contention 通过 central resource manager 建模）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 运行时特征：(1) 每 MoE layer 的 kernel launch 时执行一次（非 per-token）；(2) 输入 expert_reqs_dict 来自上一层的 Gate 网络输出（已确定每个 token 的 expert 选择）；(3) expert_die_map 由 Global CP 的 Expert Distribution Table 动态维护（expert migration/replication 后更新）。
- 两个关键 heuristic 的设计动机：(1) Candidate Mechanism——允许将请求分配到邻居 die（而非仅本地 die），在 workload balance 和 D2D traffic 之间 trade-off（传统 EP 将所有请求分配到本地 die 避免 D2D 但负载严重不均）；(2) Block-Granularity——split_num 和 block_size 的 trade-off（split 越多越平衡但 overhead 越大）。
- 效果：Allo Only (仅 task allocation) 降低 hop count 142×（vs Base），实现 6.3× throughput。大部分 performance gain 来自 allocation 使得绝大多数请求分配到本地 die。Host CPU 实现 overhead：Dojo 上 5.2-14.2%，Dojo-Enhanced 上 19.3-51.6%。
- 开源：https://github.com/zhongkaiyu/waferscale_gpu_moe_sim（Python 实现为 simulator 的一部分；论文讨论若 future programming model 变为 multi-GPU-like，此算法可在 host CPU 软件层实现而无需硬件修改）。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

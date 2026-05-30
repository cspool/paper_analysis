## Water-Filling Rebalancing（注水式重均衡）

术语是什么？
Water-Filling Rebalancing 是 PROBE 在 expert 复制后的 token 重分配策略，借鉴信息论中 water-filling 功率分配的思想。当 hotspot expert e* 从 rank r_src 复制到 rank r_dst 后，token 的重新分配遵循 locality-first + 贪心 leveling 原则：本地产生的 token 保持 pin 到本地 replica（消除不必要的网络传输），remote token 按 water-filling 逻辑从 r_src 动态分流到 r_dst，直到 r_src 的负载降至集群平均值或可转移 token 池耗尽。

从系统架构角度拆解术语：
```
// e* 已复制到 r_src 和 r_dst
target_load = mean(L)  // 集群平均负载

// Locality-First: 本地 token 不迁移
for token t routed to e* originating on r_src:
    assign t to r_src:e*          // pin 到本地

// Water-Filling: remote token 贪心分流
for token t routed to e* originating on r_other:
    if Load(r_src) > target_load:
        assign t to r_dst:e*      // 分流到新 replica
        Load(r_dst) += 1
        Load(r_src) -= 1
    else:
        assign t to r_src:e*      // r_src 已达到 target
```
不同于严格等分（strict equal split），water-filling 避免了对 token 的细粒度切分和额外同步，以 rank 级粒度操作。

术语一般如何实现？如何使用？
在 single-SM CUDA planner kernel 中实现，作为 token assignment 更新步骤的一部分。核心计算仅为 rank 级负载的加减操作，开销极低。适用于在线推理场景中需要毫秒级完成负载均衡决策的系统。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

## Pipeline Degree Optimization (Adaptive Pipelining)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pipeline Degree Optimization 是 FSMoE 的核心调度技术——将 MoE 训练中的通信和计算任务按 pipeline degree r 切分为 r 个等大数据块，在流水线中重叠执行。FSMoE 将调度场景按瓶颈来源分为 4 种 Case，通过线性性能模型和 SLSQP 求解器分别优化前向（r_fwd）和反向（r_bwd）的最优度。实验发现 912/1458 配置下前反向最优度不同。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FSMoE 的 4 种 Case 分类和优化目标：

```
Case 1 (节点间通信主导): t_1 = 2r·t_a2a + t_gar
  → min f_1(r) = 2r·α_a2a + 2n_a2a·β_a2a + t_gar

Case 2 (专家计算主导): t_2 = 2t_a2a + t_ag + t_rs + r·t_exp
  → min f_2(r) = 2α_a2a+2n_a2a·β_a2a/r + α_ag+n_ag·β_ag/r
                  + α_rs+n_rs·β_rs/r + r·α_exp+n_exp·β_exp

Case 3 (AlltoAll 通信主导): t_3 = 2r·t_a2a + t_ag + t_rs
  → min f_3(r) = 2r·α_a2a+2n_a2a·β_a2a + α_ag+n_ag·β_ag/r + α_rs+n_rs·β_rs/r

Case 4 (节点内通信主导): t_4 = 2t_a2a + r·t_ag + r·t_rs
  → min f_4(r) = 2α_a2a+2n_a2a·β_a2a/r + r·α_ag+n_ag·β_ag + r·α_rs+n_rs·β_rs
```

Algorithm 1: SLSQP 求解 4 个 Case，选 min(t_1,t_2,t_3,t_4) 对应的 r。前向 t_gar=0，反向 t_gar 由自适应梯度分区确定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 训练前通过微基准（nccl-tests + torch.matmul）测量 α/β 参数，最小二乘拟合 <10ms，SLSQP 求解平均 193ms per config，训练前执行一次。换集群时仅需重新 profiling。对比 Tutel（前反向统一 r），FSMoE 分别调度在 912/1458 配置下获得更优解。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

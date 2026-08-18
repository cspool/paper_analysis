## 性能成本模型（Performance Cost Model，XGBoost 预测器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
性能成本模型（cost model）是自动调优编译器中用机器学习预测程序执行性能的模块，替代/辅助真机测量来快速筛选候选实现。QiMeng-Tensify（ISCA'26）用 XGBoost（梯度提升树回归器）作为 cost model：离线阶段用 7 个代表子图的 12,500 次"程序 sketch-执行延迟"数据对预训练；在线 Simulation 阶段用它预测并排序随机采样的参数化程序 P，取 TopK 真机测量并回填更新模型，再对 top 候选的 Manhattan 邻域做局部搜索（PickByThreshold 选预测性能接近的邻居测量）。奖励 R = flops(p*)/t_best，作为 MCTS 回传信号。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2（Fine-Grained Parameter Specification）的运转流程：
```
P ← SamplePrograms(Sketches)                  # 从 MCTS sketch 随机采样参数化程序
P_k ← TopKSelection(P, cost_model)             # XGBoost 预测排序取 TopK
T_k ← MeasurePerformance(P_k, D)               # 真机测量（D=代码-性能数据集）
M ← UpdateModel(cost_model, D)                 # 在线更新
for (p_i,t_i) in (P_k,T_k):
    t_best ← t_i
    while 时间 ≤ Tuning_time:
        P_neigh ← GenerateNeighbors(p_i)        # Manhattan 距离邻域
        P_thres ← PickByThreshold(P_neigh, cost_model)
        T_thres ← MeasurePerformance(P_thres, D)
        t ← PickBestPerformance(...); 若无更好则 break
p* ← BestProgram(D); R* ← EstimateReward(p*)
```
作用：减少真机测量次数（只用 cost model 排名，只有 TopK/邻域候选上真机），加速收敛。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：XGBoost（Chen & Guestrin KDD'16）训练集为 (IR 特征, 执行时间) 对；在线时持续用新测量数据增量更新。使用方式：与 LLM prior 互补——LLM prior 引导"探索哪里"（结构/规则级），cost model 决定"候选怎么评估"（参数级）；消融显示离线预训练 + 在线更新的组合优于纯在线从零学习。相关术语注意：knowledge_repo 已有 "Coupled Performance Predictor（耦合性能预测器，XGBoost cost model）"（ATiM 论文，PIM 编译器），是结构特征+性能预测耦合的专用预测器，与本条通用 XGBoost cost model 属不同实现，不合并。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
